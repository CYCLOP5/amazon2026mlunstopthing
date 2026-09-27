'Bounded residual/decoder search against the exact .98805 submission'
import gc
import json
import os
from pathlib import Path
import shutil
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import export
from final_repair.pipeline import changes
from latest_fusion.calibration import adjust, curves, histogram
from latest_fusion.pipeline import log, logit, partition, sha256, sigmoid, write_json
from latest_fusion.training_weights import sample_weights
from latest_fusion.tuning import (
    apply_policy, fit_population, promotion, select_policy, split_search_check, top_pairs,
)


def predict_residual(model, frame, features):
    if model.num_feature() != len(features):
        raise ValueError('Model feature count differs from prepared schema')
    res = np.empty(frame.height, dtype=np.float32)
    for start in range(0, frame.height, 500_000):
        end = min(start + 500_000, frame.height)
        res[start:end] = model.predict(
            frame.slice(start, end-start).select(features).to_numpy(), raw_score=True,
            num_threads=int(os.environ.get('ER_THREADS', '48')),
        )
        if end == frame.height or end % 2_000_000 == 0:
            log(f'residual scoring {end:,}/{frame.height:,} pairs')
    return res


def compact(frame, raw, recipe):
    p = adjust(raw, frame['co'].to_numpy(), frame['seg'].to_numpy(), recipe)
    columns = ['qid', 'tid', 'co'] + (['y'] if 'y' in frame else [])
    return frame.select(columns).with_columns(pl.Series('raw', raw), pl.Series('p', p))


def anchor_metrics(accepted, anchors):
    return {
        'overall': decode.score(accepted, anchors),
        'country': {co: decode.score(accepted, anchors.filter(pl.col('co') == co))
                    for co in sorted(anchors['co'].unique())},
    }


def error_slices(frame, winners, accepted, anchors):
    'separate blocking losses, ownership losses and rejection losses'
    truth = frame.filter((pl.col('y') == 1) & pl.col('qid').is_in(anchors['qid'].implode()))
    evidence = truth.select('qid', 'tid', 'co', 'target_address_empty', 'house_conflict',
                            'name_core_exact', 'newest_present').join(
        winners.select('tid', pl.col('qid').alias('winner')), on='tid', how='left', validate='1:1',
    ).join(accepted.select('qid', 'tid').with_columns(pl.lit(True).alias('accepted')),
           on=['qid', 'tid'], how='left', validate='1:1').with_columns(
        pl.when(pl.col('accepted').fill_null(False)).then(pl.lit('accepted'))
        .when(pl.col('qid') == pl.col('winner')).then(pl.lit('correct_owner_rejected'))
        .otherwise(pl.lit('wrong_owner')).alias('outcome'),
    )
    return {
        'true_pairs': int(anchors['deg'].sum()),
        'covered_true_pairs': truth.height,
        'missing_true_pairs': int(anchors['deg'].sum())-truth.height,
        'candidate_oracle_macro_f05': decode.score(truth.select('qid', 'tid', 'y'), anchors)['macro_f05'],
        'outcomes_by_country': evidence.group_by('co', 'outcome').len().sort('co', 'outcome').to_dicts(),
        'slices': {c: evidence.group_by(c, 'outcome').len().sort(c, 'outcome').to_dicts()
                   for c in ('target_address_empty', 'house_conflict', 'name_core_exact', 'newest_present')},
    }


def prepare_domain_weights(heads, source_frame, test_path, output):
    'fit one unlabeled density-ratio model only when the bounded arm requests it'
    if not any(head.get('weight_mode', 'uniform') == 'domain_ratio' for head in heads):
        return None, None
    from latest_fusion.domain_adaptation import fit_weights
    log('learning domain weights from source and unlabeled test covariates')
    weights, report = fit_weights(source_frame, test_path, output)
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (source_frame.height,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Domain weights must be finite, positive and aligned to source rows')
    write_json(Path(output)/'domain_adaptation.json', report)
    for country, diagnostics in report.get('countries', {}).items():
        log(f'domain {country}: AUC={diagnostics.get("auc")}, weighted={diagnostics.get("used_weights")}, ESS fraction={diagnostics.get("ess_fraction")}, fallback={diagnostics.get("fallback_reason")}')
    return weights, report


def supported_domain_heads(heads, domain_report):
    'avoid fitting duplicate unit-weight controls when no shift is supported'
    if domain_report is not None and domain_report.get('weighted_source_rows', 0) == 0:
        return [], 'No supported covariate shift: every country used unit-weight fallback.'
    return heads, None


def candidate_policy(cfg, top, search, incumbent_policy, feasibility=None):
    'allow controlled method comparison with the incumbent decoder held fixed'
    if cfg.get('fixed_incumbent_policy', False):
        policy = dict(incumbent_policy)
        return policy, decode.score(apply_policy(top, policy), search)
    return select_policy(top, search, expected_f_floors=tuple(cfg['expected_f_floors']),
        feasibility=feasibility if cfg.get('feasible_policy_search', False) else None)


def run(data, train, test, incumbent, metadata, config, output):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    incumbent = Path(incumbent)
    cfg = json.loads(Path(config).read_text())
    prev = json.loads((incumbent/'report.json').read_text())
    method = cfg['incumbent_method']
    if (prev['selected'] != method or prev['matching_sha256'] != cfg['incumbent_matching_sha256'] or
            sha256(incumbent/'output'/'matching_results.tsv') != cfg['incumbent_matching_sha256']):
        raise ValueError('Wrong incumbent artifact; refusing benchmark substitution')
    refs = load_refs(str(data), 'train')
    fitids, tuneids = fit_population(refs)
    search, check = split_search_check(refs, tuneids)
    all_tune = pl.concat([search, check]).sort('qid')
    ff = json.loads(Path(train).with_name('features-train.json').read_text())
    if ff != json.loads(Path(test).with_name('features-test.json').read_text()):
        raise ValueError('Train/test feature schema mismatch')
    tr = pl.read_parquet(train)
    if tr.height != prev['coverage']['train']['union_pairs']:
        raise ValueError('Prepared train candidate pool changed')
    fitframe = tr.filter(pl.col('qid').is_in(pl.Series(fitids).implode()) &
                         ((pl.col('own') < 0) | pl.col('own').is_in(pl.Series(fitids).cast(pl.Int64).implode())))
    if fitframe.height != prev['fit_pairs'] or int(fitframe['y'].sum()) != prev['fit_positive']:
        raise ValueError('Incumbent residual training population changed')
    training_population = None
    if cfg.get('training_protocol') == 'target_group_competition':
        from latest_fusion.competition_training import select_target_group_train
        calids = refs.filter((pl.col('fold') == 0) & pl.Series(partition(refs['rid'].to_numpy()) == 1))['rid']
        protected = np.unique(np.concatenate([calids.to_numpy(), search['qid'].to_numpy(), check['qid'].to_numpy()]))
        fitframe, training_population = select_target_group_train(
            tr, refs, fitids, protected, folds=3, orphan_fraction=cfg.get('orphan_fraction', 1/3))
        write_json(output/'training_population.json', training_population)
        log(f'competition-preserving residual fit: {fitframe.height:,} pairs, {int(fitframe["y"].sum()):,} positives')
    write_json(output/'search_protocol.json', {
        'config': cfg, 'fit_owners': len(fitids), 'search_owners': search.height, 'check_owners': check.height,
        'fit_owner_sha256': __import__('hashlib').sha256(np.sort(fitids).tobytes()).hexdigest(),
        'search_owner_sha256': __import__('hashlib').sha256(search['qid'].sort().to_numpy().tobytes()).hexdigest(),
        'check_owner_sha256': __import__('hashlib').sha256(check['qid'].sort().to_numpy().tobytes()).hexdigest(),
        'scope': 'Check owners excluded from this search, but prior model selection used this population. Not a blind pipeline score.',
    })
    log(f'owners: {len(fitids):,} fit / {search.height:,} search / {check.height:,} check')
    base_logit = logit(tr['newest'].fill_null(1e-4).to_numpy())
    inc_model = lgb.Booster(model_file=str(incumbent/'residual_model.txt'))
    log('replaying the exact incumbent residual model')
    inc_residual = predict_residual(inc_model, tr, ff)
    inc_raw = sigmoid(base_logit + .5*inc_residual).astype(np.float32)
    saved_cal = prev['calibration']['methods'][method]
    inc_top = top_pairs(compact(tr, inc_raw, saved_cal['train']))
    inc_policy = {'rule': 'top1_threshold', 'threshold': prev['methods'][method]['tune']['threshold']}
    inc_accepted = apply_policy(inc_top, inc_policy)
    replay = decode.score(inc_accepted, all_tune)
    if abs(replay['macro_f05'] - prev['methods'][method]['tune']['macro_f05']) > 1e-10:
        raise ValueError(f'Incumbent tune replay differs: {replay}')
    log(f'incumbent replay PASS: tune F0.5 {replay["macro_f05"]:.9f}')
    calibration_refs = refs.filter((pl.col('fold') == 0) & pl.Series(partition(refs['rid'].to_numpy()) == 1))
    ncal = dict(calibration_refs.group_by('co').len().rows())
    calmask = (tr['fold'].to_numpy() == 0) & (partition(tr['qid'].to_numpy()) == 1)

    edges = np.asarray(json.loads((Path(metadata).parent/'latest_fusion_calibration.json').read_text())['edges'])
    co, seg, y = (tr[c].to_numpy() for c in ('co', 'seg', 'y'))

    def calibrator(raw):
        h = histogram(raw, co, seg, y, edges, calmask)
        return h, {'curves': curves(h, h, ncal, ncal, edges, transfer=False)}

    inc_hist, _ = calibrator(inc_raw)
    search_baseline = anchor_metrics(inc_accepted, search)
    best = None
    trials = []

    def search_checks(accepted, anchors):
        metrics = anchor_metrics(accepted, anchors)
        checks = {'overall_gain': metrics['overall']['macro_f05'] >=
                  search_baseline['overall']['macro_f05'] + cfg['minimum_gain']}
        for region in ['overall'] + list(metrics['country']):
            now = metrics['overall'] if region == 'overall' else metrics['country'][region]
            old = search_baseline['overall'] if region == 'overall' else search_baseline['country'][region]
            checks[region+'_macro_f05'] = now['macro_f05'] >= old['macro_f05']
            checks[region+'_precision'] = now['pair_precision'] >= max(
                cfg.get('minimum_precision', 0), old['pair_precision']-cfg['max_precision_loss'])
            checks[region+'_recall'] = now['pair_recall'] >= old['pair_recall']-cfg['max_recall_loss']
        return checks

    def consider(name, model_name, strength, residual):
        nonlocal best
        raw = sigmoid(base_logit + strength*residual).astype(np.float32)
        hist, recipe = calibrator(raw)
        if model_name == 'incumbent' and strength == .5:
            recipe = saved_cal['train']
            top = inc_top
        else:
            top = top_pairs(compact(tr, raw, recipe))
        policy, metrics = candidate_policy(cfg, top, search, inc_policy,
            feasibility=lambda accepted, anchors: all(search_checks(accepted, anchors).values()))
        accepted_search = apply_policy(top.filter(pl.col('qid').is_in(search['qid'].implode())), policy)
        country = anchor_metrics(accepted_search, search)
        gain = country['overall']['macro_f05'] - search_baseline['overall']['macro_f05']
        checks = search_checks(accepted_search, search)
        eligible = all(checks.values())
        record = {'name': name, 'model': model_name, 'strength': strength, 'policy': policy,
                  'search': country, 'gain': gain, 'eligible': bool(eligible), 'search_checks': checks}
        trials.append(record)
        log(f'trial {name}: search F0.5 {country["overall"]["macro_f05"]:.9f}, gain {gain:+.8f}, eligible {eligible}')
        write_json(output/'search_progress.json', {'incumbent': search_baseline, 'trials': trials})
        if eligible and (best is None or gain > best['record']['gain']):
            best = {'record': record, 'raw': raw, 'top': top, 'hist': hist, 'recipe': recipe}

    for strength in cfg.get('incumbent_strengths', cfg['strengths']):
        consider(f'incumbent_{strength}', 'incumbent', strength, inc_residual)
    del inc_residual
    domain_weights, domain_report = prepare_domain_weights(cfg['heads'], fitframe, test, output)
    params = dict(json.loads(Path(metadata).read_text())['params'])
    params['num_threads'] = int(os.environ.get('ER_THREADS', '48'))
    heads_to_train, experiment_rejection = supported_domain_heads(cfg['heads'], domain_report)
    if experiment_rejection:
        log('domain experiment rejected before head fitting: '+experiment_rejection)
    for head in heads_to_train:
        log(f'training {head["name"]}: {fitframe.height:,} pairs, {head["rounds"]} rounds')
        weights = domain_weights if head.get('weight_mode', 'uniform') == 'domain_ratio' else sample_weights(
            fitframe, head.get('weight_mode', 'uniform'), head.get('missing_address_multiplier', 4.0))
        model = lgb.train({**params, **head['params']}, lgb.Dataset(
            fitframe.select(ff).to_numpy(), label=fitframe['y'].to_numpy(),
            init_score=logit(fitframe['newest'].fill_null(1e-4).to_numpy()), weight=weights),
            num_boost_round=head['rounds'])
        write_json(output/(head['name']+'_training.json'), {
            'head': head, 'pairs': fitframe.height, 'positive': int(fitframe['y'].sum()),
            'weight_min': float(weights.min()) if weights is not None else 1.,
            'weight_max': float(weights.max()) if weights is not None else 1.,
            'weight_mean': float(weights.mean()) if weights is not None else 1.,
            'trees': model.num_trees(), 'model_parameters': model.params,
        })
        model.save_model(str(output/(head['name']+'_model.txt')))
        log(f'{head["name"]} fit complete; scoring frozen candidate pool')
        residual = predict_residual(model, tr, ff)
        for strength in cfg['strengths']:
            consider(f'{head["name"]}_{strength}', head['name'], strength, residual)
        del residual, model
        gc.collect()

    write_json(output/'locked_candidate.json', None if best is None else best['record'])
    log('search finished; evaluating only the locked candidate on the owner check')
    report = {
        'incumbent_job': cfg['incumbent_job'], 'incumbent_leaderboard_user_reported': cfg['incumbent_leaderboard_user_reported'],
        'incumbent_tune_replay': replay, 'incumbent_search': search_baseline, 'trials': trials,
        'training_population': training_population, 'domain_adaptation': domain_report,
        'experiment_rejection': experiment_rejection,
        'selected_weight_mode': None if best is None else next((head.get('weight_mode', 'uniform') for head in cfg['heads'] if head['name'] == best['record']['model']), 'incumbent'),
        'attribution': ('Covariate-shift weighted residual head' if best is not None and any(head['name'] == best['record']['model'] and head.get('weight_mode') == 'domain_ratio' for head in cfg['heads']) else 'Uniform control or incumbent; no domain-weighting improvement demonstrated'),
        'search_entities': search.height, 'check_entities': check.height,
        'limitations': ['Check was not used for this search, but has historical upstream/model-selection reuse.',
                        'No French labels. France predictions exactly preserve the .98805 submission.',
                        'Local F0.5 does not establish a leaderboard gain.'],
        'baseline_error_slices': error_slices(tr, inc_top, inc_accepted, all_tune),
    }
    promoted = False
    if best is not None:
        chosen = apply_policy(best['top'], best['record']['policy'])
        gate = promotion(chosen, inc_accepted, check, min_gain=cfg['minimum_gain'],
                         max_precision_loss=cfg['max_precision_loss'], max_recall_loss=cfg['max_recall_loss'],
                         family_size=cfg.get('promotion_family_size', 1),
                         confidence_alpha=cfg.get('confidence_alpha', .05))
        if cfg.get('minimum_precision', 0):
            check_metrics = anchor_metrics(chosen, check)
            for region, metric in [('overall', check_metrics['overall']), *check_metrics['country'].items()]:
                gate['checks'][region+'_absolute_precision'] = metric['pair_precision'] >= cfg['minimum_precision']
            gate['passed'] = all(gate['checks'].values())
        if cfg.get('owner_country_guards', False):
            candidate_metrics = anchor_metrics(chosen, check)
            baseline_metrics = anchor_metrics(inc_accepted, check)
            for country, now in candidate_metrics['country'].items():
                old = baseline_metrics['country'][country]
                gate['checks'][country+'_precision_nonregression'] = now['pair_precision'] >= old['pair_precision']-cfg['max_precision_loss']
                gate['checks'][country+'_recall_nonregression'] = now['pair_recall'] >= old['pair_recall']-cfg['max_recall_loss']
                gate['checks'][country+'_macro_nonregression'] = now['macro_f05'] >= old['macro_f05']
            gate['passed'] = all(gate['checks'].values())
        report['check'] = gate


        target_mask = tr['qid'].is_in(check['qid'].implode()).to_numpy()
        ncheck = dict(check.group_by('co').len().rows())

        def density_check(raw, h, policy):
            live = histogram(raw, co, seg, None, edges, target_mask)
            recipe = {'curves': curves(h, live, ncal, ncheck, edges, transfer=True)}
            top = top_pairs(compact(tr, raw, recipe))
            return apply_policy(top, policy)

        inc_density = density_check(inc_raw, inc_hist, inc_policy)
        new_density = density_check(best['raw'], best['hist'], best['record']['policy'])
        density_before = anchor_metrics(inc_density, check)
        density_after = anchor_metrics(new_density, check)
        density_checks = {'overall_macro_f05': density_after['overall']['macro_f05'] >= density_before['overall']['macro_f05']}
        if cfg.get('density_country_guards', False):
            for region in ['overall'] + list(density_after['country']):
                now = density_after['overall'] if region == 'overall' else density_after['country'][region]
                old = density_before['overall'] if region == 'overall' else density_before['country'][region]
                density_checks[region+'_macro_f05'] = now['macro_f05'] >= old['macro_f05']
                density_checks[region+'_precision'] = now['pair_precision'] >= max(
                    cfg.get('minimum_precision', 0), old['pair_precision']-cfg['max_precision_loss'])
                density_checks[region+'_recall'] = now['pair_recall'] >= old['pair_recall']-cfg['max_recall_loss']
        density_ok = all(density_checks.values())
        report['density_replay_check'] = {'incumbent': density_before, 'candidate': density_after,
                                          'checks': density_checks, 'passed': density_ok}
        promoted = bool(gate['passed'] and density_ok)
        report['locked_candidate'] = best['record']
        report['candidate_all_previous_tune_descriptive'] = anchor_metrics(chosen, all_tune)
        report['candidate_error_slices'] = error_slices(tr, best['top'], chosen, all_tune)
        log(f'owner check passed {gate["passed"]}; density replay passed {density_ok}; promoted {promoted}')
        if promoted:
            best['top'].write_parquet(output/'validation_winners.parquet')
        del chosen, inc_density, new_density
    if domain_report is not None:
        controls = [trial for trial in trials if any(head['name'] == trial['model'] and head.get('weight_mode', 'uniform') == 'uniform' for head in cfg['heads'])]
        report['uniform_control_search_trials'] = controls
        if best is not None:
            matched = next((trial for trial in controls if trial['strength'] == best['record']['strength']), None)
            report['matched_uniform_control_search'] = matched
            report['selected_minus_matched_uniform_search_macro_f05'] = (None if matched is None else
                best['record']['search']['overall']['macro_f05'] - matched['search']['overall']['macro_f05'])
    report['promoted'] = promoted
    write_json(output/'selection.json', report)
    del tr, fitframe, inc_top, inc_accepted, inc_raw, base_logit, co, seg, y
    if best is not None:
        best.pop('top', None)
        best.pop('raw', None)
    gc.collect()

    if not promoted:

        log('no verified improvement; preserving exact .98805 TSV')
        (output/'output').mkdir(exist_ok=True)
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            shutil.copyfile(incumbent/'output'/name, output/'output'/name)
        shutil.copyfile(incumbent/'accepted_test.parquet', output/'accepted_test.parquet')
        if sha256(output/'output/matching_results.tsv') != cfg['incumbent_matching_sha256']:
            raise ValueError('Exact fallback checksum mismatch')
        report['submission_recommendation'] = 'Do not spend a submission: output is identical to the .98805 benchmark.'
        report['changes'] = {'added_pairs': 0, 'removed_pairs': 0}
        report['submission_changed'] = False
    else:
        log('exporting selected model with exact incumbent France matches')
        te = pl.read_parquet(test)
        rt, tg = load_refs(str(data), 'test'), load_targets(str(data), 'test')
        if te.height != prev['coverage']['test']['union_pairs']:
            raise ValueError('Prepared test candidate pool changed')
        row = best['record']
        model_path = incumbent/'residual_model.txt' if row['model'] == 'incumbent' else output/(row['model']+'_model.txt')
        model = lgb.Booster(model_file=str(model_path))
        residual = predict_residual(model, te, ff)
        raw = sigmoid(logit(te['newest'].fill_null(1e-4).to_numpy()) + row['strength']*residual).astype(np.float32)
        if row['model'] == 'incumbent' and row['strength'] == .5:
            target_recipe = saved_cal['test']
        else:
            cot, segt = te['co'].to_numpy(), te['seg'].to_numpy()
            known = np.isin(cot, list(ncal))
            live = histogram(raw, cot, segt, None, edges, known)
            target_recipe = {'curves': curves(best['hist'], live, ncal, dict(rt.group_by('co').len().rows()), edges, transfer=True)}
            target_recipe['curves'].update({k: v for k, v in saved_cal['test']['curves'].items() if k.rsplit('|', 1)[0] not in ncal})
        known_frame = te.filter(pl.col('co').is_in(list(ncal)))
        known_mask = te['co'].is_in(list(ncal)).to_numpy()
        scored = compact(known_frame, raw[known_mask], target_recipe)
        accepted = apply_policy(top_pairs(scored), row['policy']).select('qid', 'tid', 'p')
        previous_acc = pl.read_parquet(incumbent/'accepted_test.parquet').select('qid', 'tid', 'p')
        france = rt.filter(~pl.col('co').is_in(list(ncal)))['rid']
        french = previous_acc.filter(pl.col('qid').is_in(france.implode()))
        accepted = pl.concat([accepted, french], how='vertical_relaxed')
        if accepted['tid'].n_unique() != accepted.height:
            raise ValueError('A target was assigned to multiple owners')
        report['changes'] = changes(accepted, previous_acc, rt)
        for change in ('added_by_country', 'removed_by_country'):
            if any(v for k, v in report['changes'][change].items() if k not in ncal):
                raise ValueError('Unseen-country policy changed')

        pairs = te.select('qid', 'tid').with_columns(pl.Series('p', raw))
        report['export'] = export(pairs, 'p', row['policy']['rule'], row['policy'].get('threshold', 0), rt, tg, str(output/'output'), accepted=accepted)
        accepted.write_parquet(output/'accepted_test.parquet')
        scored.write_parquet(output/'known_country_test_predictions.parquet')
        write_json(output/'selected_calibration.json', {'train': best['recipe'], 'test': target_recipe})
        write_json(output/'selected_policy.json', row)
        shutil.copyfile(model_path, output/'selected_model.txt')
        report['submission_changed'] = bool(report['changes']['added_pairs'] or report['changes']['removed_pairs'])
        report['submission_recommendation'] = (
            'New candidate passed the predeclared checks; leaderboard gain remains unknown.'
            if report['submission_changed'] else
            'Do not spend a submission: model changed locally but final matching decisions are identical.')
    from validate_outputs import validate
    report['validation'] = validate(Path(data), output/'output')
    report['matching_sha256'] = sha256(output/'output/matching_results.tsv')
    report['elapsed_seconds'] = time.monotonic()-started
    write_json(output/'report.json', report)
    log(f'completed: promoted={promoted}, matches={report["validation"]["matches"]:,}; validator PASS')
    return report

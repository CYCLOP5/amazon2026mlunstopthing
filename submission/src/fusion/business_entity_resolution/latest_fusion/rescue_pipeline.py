'additive recall using ownership consensus and country-wide name specificity'
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
from latest_fusion.calibration import adjust
from latest_fusion.pipeline import log, logit, partition, sha256, sigmoid, write_json
from latest_fusion.rescue_features import build_rescue_features
from latest_fusion.rescue_policy import apply_additions, select_additions, promotion_additions
from latest_fusion.tune_pipeline import predict_residual
from latest_fusion.tuning import fit_population, split_search_check, top_pairs, apply_policy

KEYS = ['qid', 'tid']


def frozen_scores(frame, model, features, recipe):
    residual = predict_residual(model, frame, features)
    raw = sigmoid(logit(frame['newest'].fill_null(1e-4).to_numpy()) + .5*residual).astype(np.float32)
    p = adjust(raw, frame['co'].to_numpy(), frame['seg'].to_numpy(), recipe)
    return frame.with_columns(pl.Series('inc_raw', raw), pl.Series('inc_p', p))


def proposal_pool(frame, accepted, countries, floor):
    'eligibility depends on scores only; retain all owners for each kept target'
    free = frame.filter(pl.col('co').is_in(countries)).join(accepted.select('tid'), on='tid', how='anti')
    scores = ['inc_p', 'newest', 'gate', 'neural', 'hybrid', 'graph', 'friend']
    promising = free.filter(pl.max_horizontal([pl.col(c).fill_null(0) for c in scores]) >= floor)['tid'].unique()
    return free.filter(pl.col('tid').is_in(promising.implode()))


def fit_heads(pool, fitids, features, cfg, output):
    'each head withholds both candidate owners and true owners for its fold'
    fit = pool.filter(pl.col('qid').is_in(pl.Series(fitids).implode()) &
                      ((pl.col('own') < 0) | pl.col('own').is_in(pl.Series(fitids).cast(pl.Int64).implode())))
    summary = {'pairs': fit.height, 'positive': int(fit['y'].sum()), 'folds': []}
    if summary['positive'] < cfg['minimum_fit_positive'] or fit.height-summary['positive'] < cfg['minimum_fit_negative']:
        return [], {**summary, 'reason': 'Insufficient isolated rescue training examples'}
    owners = pl.DataFrame({'qid': np.asarray(fitids, dtype=np.uint32)}).with_columns(
        (pl.col('qid').hash(1409) % cfg['folds']).alias('_fold'))
    fit = fit.join(owners, on='qid').join(owners.select(pl.col('qid').cast(pl.Int64).alias('own'),
        pl.col('_fold').alias('_true_fold')), on='own', how='left')
    params = {**cfg['lgbm'], 'num_threads': int(os.environ.get('ER_THREADS', '48'))}
    models = []
    for fold in range(cfg['folds']):
        train = fit.filter((pl.col('_fold') != fold) & ((pl.col('own') < 0) | (pl.col('_true_fold') != fold)))
        valid = fit.filter((pl.col('_fold') == fold) & ((pl.col('own') < 0) | (pl.col('_true_fold') == fold)))
        if train['y'].n_unique() < 2 or valid['y'].n_unique() < 2:
            return [], {**summary, 'reason': 'Insufficient positive and negative examples in owner CV'}
        log(f'rescue head {fold+1}/{cfg["folds"]}: {train.height:,} fit pairs, {int(train["y"].sum()):,} positives')
        dataset = lgb.Dataset(train.select(features).to_numpy(), label=train['y'].to_numpy(), feature_name=features)
        held = lgb.Dataset(valid.select(features).to_numpy(), label=valid['y'].to_numpy(), reference=dataset)
        model = lgb.train({**params, 'seed': 1801+fold}, dataset, num_boost_round=cfg['rounds'],
                          valid_sets=[held], callbacks=[lgb.early_stopping(cfg['early_stopping'], verbose=False)])
        model.save_model(str(output/f'rescue_model_{fold}.txt'))
        summary['folds'].append({'fold': fold, 'fit_pairs': train.height, 'valid_pairs': valid.height,
                                 'iteration': model.best_iteration, 'validation_logloss': model.best_score['valid_0']['binary_logloss']})
        models.append(model)
        log(f'rescue head {fold+1} complete: iteration {model.best_iteration}')
    return models, summary


def score_actions(pool, features, models, return_margins=False):
    'require all heads to prefer the same owner; use their lowest confidence'
    from latest_fusion.decoder import winners
    if not models or not pool.height:
        empty = pool.select(*KEYS, 'co', *(['y'] if 'y' in pool else [])).head(0).with_columns(pl.lit(0., pl.Float32).alias('score'))
        return empty.with_columns(pl.lit(0., pl.Float32).alias('min_head_margin')) if return_margins else empty
    pred = []
    for i, model in enumerate(models):
        if hasattr(model, 'feature_name') and model.feature_name() != list(features):
            raise ValueError('Rescue model feature order differs from inference schema')
        p = np.empty(pool.height, dtype=np.float32)
        for start in range(0, pool.height, 250_000):
            end = min(start+250_000, pool.height)
            p[start:end] = model.predict(pool.slice(start, end-start).select(features).to_numpy(),
                                        num_threads=int(os.environ.get('ER_THREADS', '48')))
        pred.append(p)
        log(f'rescue head {i+1}: scored {pool.height:,} candidate pairs')
    minimum = np.minimum.reduce(pred)
    qid, tid = pool['qid'].to_numpy(), pool['tid'].to_numpy()
    best = winners(qid, tid, minimum)
    votes = np.zeros(pool.height, dtype=np.uint8)
    min_margin = np.full(pool.height, np.inf, dtype=np.float32)
    for p in pred:
        order = np.lexsort((qid, -p, tid))
        starts = np.flatnonzero(np.r_[True, tid[order][1:] != tid[order][:-1]])
        next_positions = np.minimum(starts+1, len(order)-1)
        has_second = (starts+1 < len(order)) & (tid[order[starts]] == tid[order[next_positions]])
        certain = ~has_second | (p[order[starts]] > p[order[next_positions]])
        votes[order[starts[certain]]] += 1
        margins = np.full(pool.height, -1., dtype=np.float32)
        margins[order[starts]] = p[order[starts]] - np.where(has_second, p[order[next_positions]], 0.)
        min_margin = np.minimum(min_margin, margins)
    scores = pool.select(*KEYS, 'co', *(['y'] if 'y' in pool else [])).with_columns(
        pl.Series('score', minimum), pl.Series('_votes', votes))
    if return_margins:
        scores = scores.with_columns(pl.Series('min_head_margin', min_margin))

    tied = scores.group_by('tid').agg(pl.col('score').max().alias('_max'),
                                      pl.col('score').top_k(2).sort(descending=True).alias('_two'))
    tied = tied.filter((pl.col('_two').list.len() > 1) &
                      (pl.col('_two').list.get(0, null_on_oob=True) == pl.col('_two').list.get(1, null_on_oob=True)))
    return scores[best].filter(pl.col('_votes') == len(models)).drop('_votes').join(tied.select('tid'), on='tid', how='anti')


def risk_floor(actions, calibration_anchors):
    'fit the risk cutoff using calibration labels, never check labels'
    cal = actions.join(calibration_anchors.select('qid'), on='qid', how='inner')
    false = cal.filter(pl.col('y') == 0)
    floor = float(np.nextafter(np.float32(false['score'].max()), np.float32(np.inf))) if false.height else 0.
    return floor, {'entities': calibration_anchors.height, 'actions': cal.height,
                   'positive': int(cal['y'].sum()), 'false': false.height, 'minimum_score': floor}


def explain_actions(actions, pool, refs, targets, limit=80):
    sample = actions.sort('score', descending=True).head(limit)
    if not sample.height:
        return []
    columns = [c for c in pool.columns if c.startswith('rescue_') and
               ('fraction' in c or 'idf' in c or 'advantage' in c)]
    return sample.join(pool.select(*KEYS, 'inc_p', *columns), on=KEYS, validate='1:1').join(
        refs.select(pl.col('rid').alias('qid'), pl.col('nm').alias('reference_name'), pl.col('ad').alias('reference_address')),
        on='qid').join(targets.select(pl.col('rid').alias('tid'), pl.col('nm').alias('target_name'),
                                     pl.col('ad').alias('target_address')), on='tid').sort('score', descending=True).to_dicts()


def export_frozen_additions(data, prepared_test, incumbent, output, additions):
    'export only new matches; preserve every baseline pair and full candidate pool'
    base = pl.read_parquet(Path(incumbent)/'accepted_test.parquet').select(*KEYS, 'p')
    final = apply_additions(base, additions, 0.)
    if base.select(KEYS).join(final.select(KEYS), on=KEYS, how='anti').height:
        raise ValueError('Rescue removed a baseline match')
    rt, targets = load_refs(str(data), 'test'), load_targets(str(data), 'test')
    delta = changes(final, base, rt)
    if delta['removed_pairs'] or delta['changed_target_owners'] or delta['added_by_country'].get('france', 0):
        raise ValueError('Rescue altered protected baseline/France assignments')

    pairs = pl.read_parquet(prepared_test, columns=KEYS).with_columns(pl.lit(0., pl.Float32).alias('p'))
    details = export(pairs, 'p', 'additive_rescue', 0., rt, targets, str(output/'output'), accepted=final)
    final.write_parquet(output/'accepted_test.parquet')
    additions.write_parquet(output/'added_test.parquet')
    return details, delta


def run(data, train, test, incumbent, config, output):
    started = time.monotonic()
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(config).read_text())
    competition = cfg.get('training_protocol') == 'target_group_competition'
    if competition:
        from latest_fusion.competition_training import fit_competition_heads
        from latest_fusion.competition_policy import evidence_actions, lock_family_policy, apply_family_policy
    prev = json.loads((incumbent/'report.json').read_text())
    if prev['matching_sha256'] != cfg['incumbent_matching_sha256'] or prev['selected'] != 'residual_0.5':
        raise ValueError('Wrong .98805 incumbent artifact')
    refs, targets = load_refs(str(data), 'train'), load_targets(str(data), 'train')
    fitids, tuneids = fit_population(refs)
    search, check = split_search_check(refs, tuneids)
    calibration = refs.filter((pl.col('fold') == 0) & pl.Series(partition(refs['rid'].to_numpy()) == 1)).select(
        pl.col('rid').alias('qid'), 'deg', 'co')
    ff = json.loads(Path(train).with_name('features-train.json').read_text())
    if ff != json.loads(Path(test).with_name('features-test.json').read_text()):
        raise ValueError('Prepared feature schema mismatch')
    model = lgb.Booster(model_file=str(incumbent/'residual_model.txt'))
    recipes = prev['calibration']['methods']['residual_0.5']
    tr = pl.read_parquet(train)
    if tr.height != prev['coverage']['train']['union_pairs']:
        raise ValueError('Prepared training pool changed')
    log('replaying exact baseline before constructing unassigned-target rescue pool')
    tr = frozen_scores(tr, model, ff, recipes['train'])
    top = top_pairs(tr.select(*KEYS, 'co', 'y', pl.col('inc_p').alias('p'), pl.col('inc_raw').alias('raw')))
    base = apply_policy(top, {'rule': 'top1_threshold', 'threshold': prev['methods']['residual_0.5']['tune']['threshold']})
    all_tune = pl.concat([search, check])
    replay = decode.score(base, all_tune)
    if abs(replay['macro_f05']-prev['methods']['residual_0.5']['tune']['macro_f05']) > 1e-10:
        raise ValueError('Baseline tune replay mismatch')
    pool = proposal_pool(tr, base, cfg['countries'], cfg['pool_score_floor'])
    del tr, top
    gc.collect()
    report = {'baseline_tune_replay': replay, 'train_rescue_pairs': pool.height,
              'train_rescue_targets': pool['tid'].n_unique(), 'promoted': False, 'config': cfg,
              'limitations': ['Precision nonregression is measured on labeled partitions, not guaranteed on hidden test.',
                              'Historical upstream selection and aggregate diagnostics exposed evaluation owners.',
                              'Correlated neural members are evidence sources, not independent statistical votes.',
                              'No French labels; all French matches remain exactly frozen.']}
    write_json(output/'progress.json', report)
    models = []
    if pool.height:
        log(f'building consensus/rarity features for {pool.height:,} unassigned-target pairs')
        pool, rescue_features = build_rescue_features(pool, refs, targets, ff+['inc_p', 'inc_raw'])
        if set(rescue_features) & {'qid', 'tid', 'own', 'y', 'fold', 'deg', 'co'}:
            raise ValueError('Label or identifier entered rescue feature list')
        write_json(output/'features.json', rescue_features)
        if competition:
            protected = pl.concat([calibration, search, check])['qid'].unique().to_numpy()
            models, report['training'] = fit_competition_heads(pool, refs, fitids, protected, rescue_features, cfg, output)
            report['limitations'].append('Targets owned by fitting references can also compete against held-out references; historical transductive exposure remains.')
        else:
            models, report['training'] = fit_heads(pool, fitids, rescue_features, cfg, output)
    if models:
        actions = score_actions(pool, rescue_features, models, return_margins=competition)
        if competition:
            all_actions = actions
            actions = evidence_actions(actions, pool, refs, targets, min_head_margin=cfg['minimum_head_margin'])
            report['evidence_gate'] = {'unanimous_actions': all_actions.height, 'eligible_actions': actions.height,
                'families': actions.group_by('evidence_family').len().to_dicts()}
            policy, report['family_selection'] = lock_family_policy(actions, base, calibration, search,
                                                                   min_actions=cfg['minimum_family_actions'])
            sel = apply_family_policy(actions, policy)
            sel.write_parquet(output/'locked_train_additions.parquet')
            log(f'locked evidence-family rescue: enabled={policy["enabled"]}, selected={sel.height:,}')
        else:
            floor, report['risk_calibration'] = risk_floor(actions, calibration)
            eligible = actions.filter(pl.col('score') >= floor)
            policy, report['search'] = select_additions(eligible, base, search, cfg['minimum_actions'])
            policy['calibration_floor'] = floor
            sel = eligible.filter(pl.col('score') >= policy['threshold']) if policy['enabled'] else eligible.head(0)
            log(f'locked rescue policy: {policy}; search additions {report["search"]["added_actions"]}')
        write_json(output/'locked_policy.json', policy)
        if policy['enabled']:

            report['calibration_check'] = promotion_additions(sel, base, calibration, cfg['minimum_actions'])
            report['check'] = promotion_additions(sel, base, check, cfg['minimum_actions'])
            report['promoted'] = report['calibration_check']['passed'] and report['check']['passed']
            report['candidate_previous_tune_descriptive'] = decode.score(apply_additions(base, sel, 0.), all_tune)
            sel.write_parquet(output/'proposed_train_additions.parquet')
        write_json(output/'action_examples.json', {
            'proposed': explain_actions(sel, pool, refs, targets),
            'high_score_false': explain_actions(actions.filter(pl.col('y') == 0), pool, refs, targets, 40),
            'high_score_true': explain_actions(actions.filter(pl.col('y') == 1), pool, refs, targets, 40),
        })
    else:
        report['reason'] = 'No eligible rescue population or insufficient isolated training examples'
    write_json(output/'selection.json', report)
    log(f'rescue promotion: {report["promoted"]}')
    del pool, refs, targets, base
    gc.collect()
    if report['promoted']:
        rt, tt = load_refs(str(data), 'test'), load_targets(str(data), 'test')
        te = pl.read_parquet(test)
        if te.height != prev['coverage']['test']['union_pairs']:
            raise ValueError('Prepared test pool changed')

        te = te.filter(pl.col('co').is_in(cfg['countries']))
        te = frozen_scores(te, model, ff, recipes['test'])
        old = pl.read_parquet(incumbent/'accepted_test.parquet').select(*KEYS, 'p')
        test_pool = proposal_pool(te, old, cfg['countries'], cfg['pool_score_floor'])
        del te
        if test_pool.height:
            test_pool, test_features = build_rescue_features(test_pool, rt, tt, ff+['inc_p', 'inc_raw'])
            if test_features != rescue_features:
                raise ValueError('Rescue train/test feature order changed')
            test_actions = score_actions(test_pool, test_features, models, return_margins=competition)
            if competition:
                test_actions = evidence_actions(test_actions, test_pool, rt, tt, min_head_margin=cfg['minimum_head_margin'])
                additions = apply_family_policy(test_actions, policy)
            else:
                additions = test_actions.filter(pl.col('score') >= max(policy['threshold'], floor))
            write_json(output/'test_addition_examples.json', explain_actions(additions, test_pool, rt, tt))
            del test_pool
        else:
            additions = old.head(0).select(*KEYS).with_columns(pl.lit(0., pl.Float32).alias('score'))
        report['export'], report['changes'] = export_frozen_additions(data, test, incumbent, output, additions)
        report['submission_changed'] = report['changes']['added_pairs'] > 0
    else:
        (output/'output').mkdir(exist_ok=True)
        for name in ('matching_results.tsv', 'candidate_pairs.tsv'):
            shutil.copyfile(incumbent/'output'/name, output/'output'/name)
        shutil.copyfile(incumbent/'accepted_test.parquet', output/'accepted_test.parquet')
        report['submission_changed'] = False
        report['changes'] = {'added_pairs': 0, 'removed_pairs': 0, 'changed_target_owners': 0}
    from validate_outputs import validate
    report['validation'] = validate(Path(data), output/'output')
    report['matching_sha256'] = sha256(output/'output/matching_results.tsv')
    if not report['promoted'] and report['matching_sha256'] != cfg['incumbent_matching_sha256']:
        raise ValueError('Exact fallback checksum mismatch')
    report['submission_recommendation'] = ('Changed candidate passed measured precision/recall checks; hidden-test quality is unknown.'
        if report['submission_changed'] else 'Do not resubmit: matching decisions are unchanged from .98805.')
    report['elapsed_seconds'] = time.monotonic()-started
    write_json(output/'report.json', report)
    log(f'completed additive rescue: changed={report["submission_changed"]}, {report["validation"]["matches"]:,} matches; validator PASS')
    return report

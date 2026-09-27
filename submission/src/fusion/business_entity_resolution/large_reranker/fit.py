'fit an isolated shallow reranker and select country policies on tune only'
import gc
import json
from pathlib import Path

import numpy as np
import polars as pl

from er.safe import THREADS
from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import tuning_anchors, export, lgb_params
from final_hybrid.fit import weighted_partitions
from graph_resolution.pipeline import subset_for_tuning, compare_audit
from boosted_hybrid.pipeline import constrained_curve, error_changes, country_summary
from innovation.common import log, logit
from large_reranker.prepare import KEYS, checked, load_predictions

MIN_GAIN = .00005
BASES = ('hybrid', 'friend', 'friend_graph')


def friend_decoder(friend):
    report = json.loads((Path(friend)/'report.json').read_text())
    methods = report.get('methods', {})
    cands = [v['tune'] for v in methods.values() if v.get('score') == 'p2']
    if cands:
        return {k: cands[0][k] for k in ('rule', 'threshold')}
    sel = report.get('decision', report.get('selected_decoder', {}))
    if 'rule' in sel and 'threshold' in sel:
        return {k: sel[k] for k in ('rule', 'threshold')}
    raise ValueError('Friend report lacks raw p2 tune decoder')


def labeled_pool(data, hybrid_result, friend, graph, split):
    d = load_predictions(hybrid_result, friend, graph, split)
    refs = load_refs(data, split)
    d = d.join(refs.select(pl.col('rid').alias('qid'), 'co'), on='qid', how='left', validate='m:1')
    if d['co'].null_count():
        raise ValueError('Prediction owner absent from prepared references')
    if split == 'train':
        owners = pl.concat([pl.read_parquet(Path(data)/split/f's{s}.parquet', columns=['rid', 'own'])
                            for s in (2, 3)]).select(pl.col('rid').cast(pl.UInt32).alias('tid'), 'own')
        d = d.join(owners, on='tid', how='left', validate='m:1')
        if d['own'].null_count():
            raise ValueError('Prediction target absent from prepared records')
        d = d.with_columns((pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.UInt8).alias('y')).drop('own')
    return d


def base_predictions(pool, base):


    return pool.select(*KEYS, pl.col(base+'_p').fill_null(0.).alias('p'),
                       *(['y'] if 'y' in pool else []))


def read_scored(prepared, scored, split, pool):
    if not (Path(prepared)/'_SUCCESS').exists() or not (Path(scored)/'_SUCCESS').exists():
        raise RuntimeError('Prepare or GPU scoring stage incomplete')
    frame = checked(pl.read_parquet(Path(prepared)/split/'features.parquet'), 'selected features')
    paths = sorted((Path(scored)/split).glob('part_*.parquet'))
    if not paths and frame.height:
        raise ValueError('No large reranker score shards')
    values = checked(pl.concat([pl.read_parquet(p) for p in paths], how='vertical_relaxed'), 'large scores')\
        if paths else frame.select(KEYS).with_columns(pl.lit(None, pl.Float32).alias('large_lg'))
    if values.height != frame.height or values.select(KEYS).join(frame.select(KEYS), on=KEYS, how='anti').height:
        raise ValueError('Large scorer did not cover exact selected cached candidates')
    overlap = [c for c in values.columns if c not in KEYS and c in frame]
    frame = frame.drop(overlap).join(values, on=KEYS, how='left', validate='1:1', maintain_order='left')
    if frame['large_lg'].null_count() or not frame['large_lg'].is_finite().all():
        raise ValueError('Missing/nonfinite large model logit')
    stats = pool.group_by('tid').agg(pl.col('friend_graph_p').max().alias('base_best'),
                                   pl.len().alias('full_candidate_count'))
    d = frame.join(stats, on='tid', how='left', validate='m:1').with_columns(
        pl.col('large_lg').rank('min', descending=True).over('tid').cast(pl.Float32).alias('large_rank'),
        (pl.col('large_lg')-pl.col('large_lg').max().over('tid')).alias('large_best_gap'),
        pl.col('large_lg').top_k(2).over('tid', mapping_strategy='join').alias('_large_top'),
        (pl.col('friend_graph_p')-pl.col('base_best')).alias('base_competition_gap'))
    d = d.with_columns((pl.col('_large_top').list.get(0)-pl.col('_large_top').list.get(
        1, null_on_oob=True).fill_null(pl.col('_large_top').list.get(0))).alias('large_target_margin')).drop('_large_top')
    for old in ('expert_lg', 'ce_base_lg', 'frozen_lg'):
        if old in d:
            d = d.with_columns((pl.col('large_lg')-pl.col(old)).alias('large_minus_'+old))
    return d.with_columns(pl.Series('large_minus_base', d['large_lg'].to_numpy()-logit(d['friend_graph_p'].to_numpy())))


def feature_columns(frame):
    exclude = {'qid', 'tid', 'y', 'own', 'fold', 'co', 'eid', 'rid'}
    return [c for c, dtype in frame.schema.items() if c not in exclude and
            not c.startswith('_') and dtype.is_numeric()]


def fit_head(frame, refs, columns, rounds, bundle):
    import lightgbm as lgb
    fit, groups, weights, fractions = weighted_partitions(frame, refs)
    X = frame.select(columns).cast(pl.Float32).to_numpy()
    y = frame['y'].to_numpy()
    params = lgb_params({'lgbm': {'learning_rate': .035, 'num_leaves': 15,
        'max_depth': 4, 'min_data_in_leaf': 80, 'lambda_l2': 20.,
        'feature_fraction': .9, 'deterministic': True, 'force_col_wise': True, 'seed': 8043}})
    raw, models, history = np.zeros(frame.height), [], []
    for k in range(3):
        tr, va = fit & (groups != k), fit & (groups == k)
        if not va.any() or np.unique(y[tr]).size != 2 or np.unique(y[va]).size != 2:
            raise ValueError('Shallow head requires both classes in true-owner CV')
        model = lgb.train(params, lgb.Dataset(X[tr], label=y[tr], weight=weights[tr], feature_name=columns),
            num_boost_round=rounds, valid_sets=[lgb.Dataset(X[va], label=y[va], weight=weights[va], feature_name=columns)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
        raw[va] = model.predict(X[va], raw_score=True, num_threads=THREADS)
        raw[~fit] += model.predict(X[~fit], raw_score=True, num_threads=THREADS)/3
        model.save_model(str(bundle/f'head_{k}.txt'))
        models.append(model)
        history.append({'iteration': model.best_iteration, 'loss': model.best_score['valid_0']['binary_logloss']})

    transfer = {}
    countries = frame['co'].unique().sort().to_list()
    for country in countries:
        source = fit & (frame['co'].to_numpy() == country)
        tr, va = source & (groups != 0), source & (groups == 0)
        if not va.any() or np.unique(y[tr]).size != 2 or np.unique(y[va]).size != 2:
            transfer[str(country)] = None
            continue
        model = lgb.train(params, lgb.Dataset(X[tr], label=y[tr], weight=weights[tr], feature_name=columns),
            num_boost_round=rounds, valid_sets=[lgb.Dataset(X[va], label=y[va], weight=weights[va], feature_name=columns)],
            callbacks=[lgb.early_stopping(50, verbose=False)])
        transfer[str(country)] = model.predict(X, raw_score=True, num_threads=THREADS)
        model.save_model(str(bundle/f'transfer_{country}.txt'))
    info = {'fit_rows': int(fit.sum()), 'owner_cv': history, 'country_fit_fractions': fractions,
        'importance': sorted(zip(columns, np.mean([m.feature_importance('gain') for m in models], axis=0).tolist()),
                             key=lambda x: -x[1])[:40],
        'partition': 'fold0 fit only; candidate AND true owner excluded from tune/audit; true-owner CV',
        'calibration': 'none; weighted independent binary classifier, no audit-derived calibration'}
    return raw, models, transfer, info


def patch(base, frame, raw, weight):
    if len(raw) != frame.height or not np.isfinite(raw).all():
        raise ValueError('Invalid shallow prediction')
    prev = frame.select(KEYS).join(base.select(*KEYS, pl.col('p').alias('_base')),
        on=KEYS, how='left', validate='1:1', maintain_order='left')['_base']
    if prev.null_count():
        raise ValueError('Selected candidates absent from full base')
    p = 1/(1+np.exp(-np.clip((1-weight)*logit(prev.to_numpy())+weight*np.asarray(raw), -30, 30)))
    update = frame.select(KEYS).with_columns(pl.Series('_new', p.astype(np.float32)))
    return base.join(update, on=KEYS, how='left', validate='1:1', maintain_order='left').with_columns(
        pl.coalesce('_new', 'p').alias('p')).drop('_new')


def tune_policy(pairs, anchors, minimum=None):
    choices = []
    if minimum is None:
        decision = decode.tune(pairs, anchors, rules=('top1_threshold', 'expected_f'), floors=(.05, .1, .3, .5))
        return {'rule': decision['rule'], 'threshold': decision['threshold'],
                **decode.score(decode.apply(decision['rule'], pairs, decision['threshold']), anchors)}
    top = constrained_curve(decode.top1(pairs), anchors,
                            minimum['pair_precision']-1e-12, minimum['pair_recall']-1e-12)
    if top is not None:
        choices.append(top)
    for floor in (.05, .1, .3, .5):
        metric = decode.score(decode.expected_f(pairs, floor), anchors)
        if metric['pair_precision'] >= minimum['pair_precision']-1e-12 and metric['pair_recall'] >= minimum['pair_recall']-1e-12:
            choices.append({'rule': 'expected_f', 'threshold': floor, **metric})
    return max(choices, key=lambda x: (x['macro_f05'], x['pair_recall'])) if choices else None


def accept_countries(pred, policies, refs, pool=None):
    parts = []
    for country, policy in policies.items():
        sub = pred.join(refs.filter(pl.col('co') == country).select(pl.col('rid').alias('qid')), on='qid')
        if policy.get('friend_only'):
            sub = sub.join(pool.filter(pl.col('has_friend_p') == 1).select(KEYS), on=KEYS)
        parts.append(decode.apply(policy['rule'], sub, policy['threshold']))
    accepted = pl.concat(parts, how='vertical_relaxed')
    if accepted['tid'].n_unique() != accepted.height:
        raise ValueError('Country decoders assigned multiple owners to target')
    return accepted


def routed_predictions(pool, policies, frame, raw):
    parts = []
    for country, policy in policies.items():
        base = base_predictions(pool.filter(pl.col('co') == country), policy['base'])
        if policy.get('weight', 0):
            mask = frame['co'].to_numpy() == country
            base = patch(base, frame.filter(pl.Series(mask)), np.asarray(raw)[mask], policy['weight'])
        parts.append(base)
    return pl.concat(parts, how='vertical_relaxed')


def transfer_report(pool, frame, transfers, tune_co, baselines, baseline_accepted):
    countries = tune_co['co'].unique().sort().to_list()
    report = {'scope': 'new shallow head only; frozen large encoder and upstream encoders have seen both countries'}
    if len(countries) != 2:
        report['status'] = 'requires two labeled countries'
        return report
    for weight in (.25, .5, 1.):
        directions = []
        for source in countries:
            dest = next(c for c in countries if c != source)
            raw = transfers.get(str(source))
            if raw is None:
                directions.append({'source': source, 'target': dest, 'passes': False, 'reason': 'insufficient fit CV classes'})
                continue
            src = tune_co.filter(pl.col('co') == source).select('qid', 'deg')
            dst = tune_co.filter(pl.col('co') == dest).select('qid', 'deg')
            source_base = base_predictions(pool, baselines[str(source)]['base'])
            src_dec = tune_policy(patch(source_base, frame, raw, weight), src,
                                  decode.score(baseline_accepted, src))
            item = {'source': source, 'target': dest, 'source_decision': src_dec, 'passes': False}
            if src_dec:
                dest_base = base_predictions(pool, baselines[str(dest)]['base'])
                accepted = decode.apply(src_dec['rule'], patch(dest_base, frame, raw, weight), src_dec['threshold'])
                metrics, old = decode.score(accepted, dst), decode.score(baseline_accepted, dst)
                gain = metrics['macro_f05']-old['macro_f05']
                item.update(target_metrics=metrics, target_baseline=old, gain=gain,
                    passes=gain >= MIN_GAIN and metrics['pair_precision'] >= old['pair_precision']-1e-12 and
                    metrics['pair_recall'] >= old['pair_recall']-1e-12)
            directions.append(item)
        report[str(weight)] = {'directions': directions, 'both_pass': all(d['passes'] for d in directions)}
    return report


def run(data, hybrid_result, features_train, features_test, friend, graph, prepared, scored,
        output, rounds=500):


    output = Path(output)
    bundle = output/'bundle'
    bundle.mkdir(parents=True, exist_ok=True)
    refs = load_refs(data, 'train')
    anchors = refs.select(pl.col('rid').alias('qid'), 'deg', 'fold', 'co')
    tune = tuning_anchors(anchors, {'seed': 42, 'fit_fold': 0, 'tune_holdout_buckets': 5})
    tune_co = tune.join(anchors.select('qid', 'co'), on='qid')
    audit = anchors.filter(pl.col('fold') == 1).select('qid', 'deg')
    pool = labeled_pool(data, hybrid_result, friend, graph, 'train')
    frame = read_scored(prepared, scored, 'train', pool)
    columns = feature_columns(frame)
    raw, models, transfers, training = fit_head(frame, refs, columns, rounds, bundle)
    trials, baselines, policies = {}, {}, {}
    hybrid_report = json.loads((Path(hybrid_result)/'report.json').read_text())
    frozen = hybrid_report['decision']
    frozen_hybrid = base_predictions(pool, 'hybrid')
    frozen_accept = decode.apply(frozen['rule'], frozen_hybrid, frozen['threshold'])
    countries = tune_co['co'].unique().sort().to_list()
    small_pool = pool.join(subset_for_tuning(base_predictions(pool, 'hybrid'), tune).select(KEYS), on=KEYS)
    for country in countries:
        subanchors = tune_co.filter(pl.col('co') == country).select('qid', 'deg')
        minimum = decode.score(frozen_accept, subanchors)
        country_trials = {'hybrid_frozen': {'base': 'hybrid', 'weight': 0.,
            'rule': frozen['rule'], 'threshold': frozen['threshold'], **minimum}}
        for base in BASES:
            pred = base_predictions(small_pool, base)
            decision = tune_policy(pred, subanchors, minimum)
            country_trials[base] = {'base': base, 'weight': 0., 'eligible': decision is not None,
                                   'decision': decision}
            if decision:
                country_trials[base].update(decision)
        chosen = max((k for k, v in country_trials.items() if v.get('eligible', True)),
            key=lambda k: (country_trials[k]['macro_f05'], country_trials[k]['pair_recall']))
        baseline = dict(country_trials[chosen])
        baselines[str(country)] = baseline
        for weight in (.25, .5, 1.):
            pred = patch(base_predictions(pool, baseline['base']), frame, raw, weight)
            pred = subset_for_tuning(pred, tune)
            decision = tune_policy(pred, subanchors, baseline)
            country_trials[f'large_{weight:g}'] = {'base': baseline['base'], 'weight': weight,
                'eligible': bool(decision and decision['macro_f05'] > baseline['macro_f05']), 'decision': decision}
            if decision and decision['macro_f05'] > baseline['macro_f05']:
                country_trials[f'large_{weight:g}'].update(decision)
        eligible = [v for v in country_trials.values() if v.get('eligible', v.get('weight') == 0)]
        policies[str(country)] = dict(max(eligible, key=lambda v: (v['macro_f05'], v['pair_recall'])))
        trials[str(country)] = country_trials
    base_pred = routed_predictions(pool, baselines, frame, raw)
    base_accept = accept_countries(base_pred, baselines, refs)
    chosen_pred = routed_predictions(pool, policies, frame, raw)
    chosen_accept = accept_countries(chosen_pred, policies, refs)
    gain = decode.score(chosen_accept, tune)['macro_f05']-decode.score(base_accept, tune)['macro_f05']
    if gain < MIN_GAIN:
        policies = {c: dict(v) for c, v in baselines.items()}
        chosen_pred, chosen_accept = base_pred, base_accept
    stress = transfer_report(pool, frame, transfers, tune_co, baselines, base_accept)
    france = {'base': 'friend', 'weight': 0., 'friend_only': True, **friend_decoder(friend)}
    friend_policies = {str(c): {'base': 'friend', 'weight': 0.,
        **tune_policy(base_predictions(small_pool, 'friend'), tune_co.filter(pl.col('co') == c).select('qid', 'deg'))}
        for c in countries}
    friend_base_accept = accept_countries(base_predictions(pool, 'friend'), friend_policies, refs)
    france_stress = transfer_report(pool, frame, transfers, tune_co, friend_policies, friend_base_accept)



    french_decision = tune_policy(patch(base_predictions(pool, 'friend'), frame, raw, .25), tune,
                                 decode.score(friend_base_accept, tune))
    french_gain = (french_decision['macro_f05']-decode.score(friend_base_accept, tune)['macro_f05'])\
        if french_decision else None
    france_promoted = bool(france_stress.get('0.25', {}).get('both_pass') and
                          french_decision and french_gain >= MIN_GAIN)
    if france_promoted:
        france = {'base': 'friend', 'weight': .25, 'friend_only': True,
                  'rule': french_decision['rule'], 'threshold': french_decision['threshold']}
    paired = compare_audit(chosen_accept, base_accept, audit)
    paired.pop('historical_audit_reference', None)
    paired['reference'] = 'country-selected pure base alternatives on identical historical anchors'
    report = {'country_policies': policies, 'pure_base_policies': baselines, 'tuning': trials,
        'baseline_tune': decode.score(base_accept, tune), 'tune': decode.score(chosen_accept, tune),
        'audit': decode.score(chosen_accept, audit), 'baseline_audit': decode.score(base_accept, audit),
        'frozen_hybrid_tune': decode.score(frozen_accept, tune),
        'frozen_hybrid_audit': decode.score(frozen_accept, audit),
        'comparison': paired, 'audit_changes': error_changes(chosen_accept, base_accept, audit),
        'audit_by_country': {str(c): {'baseline': decode.score(base_accept, a),
            'selected': decode.score(chosen_accept, a), 'changes': error_changes(chosen_accept, base_accept, a)}
            for c in countries for a in [anchors.filter((pl.col('fold') == 1) & (pl.col('co') == c)).select('qid', 'deg')]},
        'transfer_stress': stress, 'france_transfer_stress': france_stress, 'france_policy': france,
        'france_gate': {'predeclared_weight': .25, 'global_source_tune_decision': french_decision,
            'global_source_tune_gain': french_gain, 'promoted': france_promoted, 'french_accuracy': None,
            'scope': 'head-only transfer proxy conditional on both-country pretrained encoders'},
        'promotion_policy': {'minimum_global_tune_gain': MIN_GAIN, 'precision_and_recall_must_not_decrease': True,
            'audit_used_for_selection': False,
            'france': 'predeclared .25 friend blend requires both source-only frozen-threshold gains and global source tune gain; otherwise raw p2'},
        'training': training, 'selection': json.loads((Path(prepared)/'selection.json').read_text()),
        'provenance': {'features_train': str(features_train), 'features_test': str(features_test)},
        'limits': ['Historical audit has influenced prior research and upstream models; it is not a fresh blind whole-pipeline benchmark.',
            'Large encoder adaptation uses isolated fold2; frozen upstream features may retain historical leakage.',
            'Source-country transfer measures only the new shallow head, conditional on both-country encoder training.',
            'France accuracy is unmeasured; promotion uses only a head transfer proxy, otherwise raw friend baseline.',
            'Sampling weights approximate owner/decoy inclusion within selected targets; full-population calibration is unverified.']}
    chosen_pred.write_parquet(output/'validation_predictions.parquet')
    del pool, base_pred, chosen_pred, chosen_accept, base_accept, frame, raw, transfers, small_pool
    gc.collect()
    pool = labeled_pool(data, hybrid_result, friend, graph, 'test')
    frame = read_scored(prepared, scored, 'test', pool)
    if set(columns)-set(frame.columns):
        raise ValueError('Test features lack trained columns')
    raw = np.zeros(frame.height)
    for start in range(0, frame.height, 100000):
        X = frame.slice(start, 100000).select(columns).cast(pl.Float32).to_numpy()
        raw[start:start+len(X)] = np.mean([m.predict(X, raw_score=True, num_threads=THREADS) for m in models], axis=0)
    refs, targets = load_refs(data, 'test'), load_targets(data, 'test')
    for c in refs['co'].unique():
        if str(c) not in policies:
            policies[str(c)] = dict(france)
            baselines[str(c)] = {'base': 'friend', 'weight': 0., 'friend_only': True,
                                 **friend_decoder(friend)}
    res = routed_predictions(pool, policies, frame, raw)
    accepted = accept_countries(res, policies, refs, pool)
    country_check = accepted.select(KEYS).join(refs.select(pl.col('rid').alias('qid'),
        pl.col('co').alias('_ref_country')), on='qid', how='left', validate='m:1').join(
        targets.select(pl.col('rid').alias('tid'), pl.col('co').alias('_target_country')),
        on='tid', how='left', validate='m:1')
    if country_check.filter(pl.col('_ref_country').is_null() | pl.col('_target_country').is_null() |
                            (pl.col('_ref_country') != pl.col('_target_country'))).height:
        raise ValueError('Accepted predictions contain missing or cross-country records')
    prev = routed_predictions(pool, baselines, frame, raw)
    previous_accept = accept_countries(prev, baselines, refs, pool)
    report['test_by_country'] = country_summary(res, accepted, previous_accept, refs, targets, frame)
    report['country_policies'] = policies
    res.write_parquet(output/'test_predictions.parquet')
    accepted.write_parquet(output/'accepted.parquet')
    report['export'] = export(res, 'p', 'top1_threshold', .5, refs, targets, str(output/'output'), accepted=accepted)
    report['export']['decoder'] = 'explicit country_policies; one owner per target; France friend-only'
    (bundle/'features.json').write_text(json.dumps(columns, indent=2))
    (bundle/'decision.json').write_text(json.dumps(policies, indent=2))
    (output/'report.json').write_text(json.dumps(report, indent=2))
    (output/'_SUCCESS').write_text('complete\n')
    log(f'Large reranker completed: audit={report["audit"]}; changes={report["audit_changes"]}')
    return report

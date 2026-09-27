'target-isolated, fixed-budget lexical action experiment with frozen replay'
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import directional_policy as policy
from . import directional_swaps as swaps
from .structured_protocol import WARNING

KEYS = ['qid', 'tid']
COUNTRIES = ['us', 'india']
MIN_CLASS = 50
MIN_ACTIONS = 25
ROUNDS = 180
DISABLED = {'enabled': False, 'drop_threshold': None, 'add_threshold': None}
PARAMS = dict(objective='multiclass', num_class=3, metric='multi_logloss',
              learning_rate=.04, num_leaves=15, max_depth=5,
              min_data_in_leaf=50, lambda_l2=40., verbosity=-1,
              seed=2709, deterministic=True, force_col_wise=True)


def _log(message):
    print(time.strftime('%H:%M:%S') + ' directional: ' + message, flush=True)


def _anchors(refs):
    return refs.rename({'rid': 'qid'}) if 'rid' in refs.columns else refs


def _country(refs, countries):
    return _anchors(refs).filter(pl.col('co').is_in(countries))


def _features(core_features):
    core = list(dict.fromkeys(core_features))
    forbidden = set(KEYS + ['rid', 'eid', 'y', 'own', 'deg', 'fold', 'co', 'nm', 'ad'])
    bad = forbidden.intersection(core) | {c for c in core if c.startswith('ds_')}
    if bad:
        raise ValueError(f'Forbidden control features: {sorted(bad)}')
    if not core:
        raise ValueError('The control requires at least one core feature')
    return core


def _class_counts(frame):
    return {'alias': frame.filter(pl.col('y') == 1).height,
            'decoy': frame.filter((pl.col('y') == 0) & (pl.col('own') < 0)).height,
            'other_owned': frame.filter((pl.col('y') == 0) & (pl.col('own') >= 0)).height}


def _validate_fit(train_frame, fit_frame, cal_refs, check_refs, fit_refs):
    if fit_frame.select(KEYS).unique().height != fit_frame.height:
        raise ValueError('Duplicate fitting candidate pairs')
    if fit_frame.select(KEYS).join(train_frame.select(KEYS), on=KEYS, how='anti').height:
        raise ValueError('Fitting rows outside full training candidate pool')
    cal, check = _anchors(cal_refs).select('qid'), _anchors(check_refs).select('qid')
    if cal['qid'].n_unique() != cal.height or check['qid'].n_unique() != check.height:
        raise ValueError('Duplicate calibration/check references')
    if cal.join(check, on='qid').height:
        raise ValueError('Calibration/check reference overlap')
    cal_targets = train_frame.join(cal, on='qid', how='semi').select('tid').unique()
    check_targets = train_frame.join(check, on='qid', how='semi').select('tid').unique()
    if cal_targets.join(check_targets, on='tid').height:
        raise ValueError('Calibration/check candidate target overlap')
    held = pl.concat([cal, check])
    held_targets = train_frame.join(held, on='qid', how='semi').select('tid').unique()
    if fit_frame.select('tid').join(held_targets, on='tid').height:
        raise ValueError('Fitting targets overlap calibration/check candidate targets')
    if fit_frame.filter(pl.col('y').is_null() | pl.col('own').is_null() |
                        ~pl.col('y').is_in([0, 1]) |
                        (pl.col('y') != (pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.Int8))).height:
        raise ValueError('Fitting labels disagree with target ownership')
    original = train_frame.select(*KEYS, pl.col('y').alias('_y'), pl.col('own').alias('_own'))
    if fit_frame.select(*KEYS, 'y', 'own').join(original, on=KEYS).filter(
            (pl.col('y') != pl.col('_y')) | (pl.col('own') != pl.col('_own'))).height:
        raise ValueError('Fitting labels differ from full candidate pool')
    if fit_refs is not None:
        owners = _anchors(fit_refs).select(pl.col('qid').cast(fit_frame.schema['own']).alias('own'))
        if fit_frame.filter(pl.col('own') >= 0).select('own').unique().join(owners, on='own', how='anti').height:
            raise ValueError('Fitting true owners outside supplied fit references')


def _fit(frame, features, path):
    labels = np.where(frame['y'].to_numpy() == 1, 0,
                      np.where(frame['own'].to_numpy() < 0, 1, 2)).astype(np.int32)
    model = lgb.train({**PARAMS, 'num_threads': int(os.environ.get('ER_THREADS', '48'))},
                      lgb.Dataset(frame.select(features).to_numpy(), label=labels,
                                  feature_name=features), num_boost_round=ROUNDS)
    model.save_model(str(path))
    return model


def _score(model, rows, features):
    pred = []
    for chunk in rows.iter_slices(250_000):
        p = np.asarray(model.predict(chunk.select(features).to_numpy(),
                                    num_threads=int(os.environ.get('ER_THREADS', '48'))))
        if p.shape != (chunk.height, 3) or not np.isfinite(p).all():
            raise ValueError('Invalid three-class directional predictions')
        pred.append(p)
    p = np.concatenate(pred) if pred else np.empty((0, 3))
    scored = rows.with_columns(*[pl.Series(name, p[:, i]) for i, name in
                                enumerate(['p_alias', 'p_decoy', 'p_other'])])
    if 'p' not in scored.columns:
        scored = scored.with_columns((pl.col('baseline_p') if 'baseline_p' in scored.columns
                                      else pl.col('p_alias')).alias('p'))
    return scored


def _locked(rows, baseline, anchors, rule):
    if not anchors.height:
        return {'passed': False, 'reason': 'No check references', 'after': {'macro_f05': 0.}}
    drops, adds = policy.propose_actions(rows, baseline, anchors, rule)
    return policy._guard(policy.evaluate_actions(baseline, drops, adds, anchors), MIN_ACTIONS)


def _select(rows, baseline, anchors):
    if not anchors.height:
        return dict(DISABLED), {'passed': False, 'reason': 'No calibration references'}
    return policy.select_policy(rows, baseline, anchors, min_actions=MIN_ACTIONS)


def _head(name, fit, sidecar, features, baseline, cal, check, output):
    counts = _class_counts(fit)
    _log(f'{name}: {fit.height:,} fit rows; class counts {counts}')
    details = {'features': features, 'fit_rows': fit.height, 'class_counts': counts,
               'rounds': ROUNDS, 'minimum_class_count': MIN_CLASS}
    if min(counts.values()) < MIN_CLASS:
        _log(f'{name}: disabled, insufficient class counts')
        return None, dict(DISABLED), {**details, 'available': False,
                  'reason': 'Baseline retained: insufficient eligible fitting examples per class'}
    _log(f'{name}: fitting {ROUNDS} multiclass rounds with {len(features)} features')
    model = _fit(fit, features, output / (name + '_model.txt'))
    _log(f'{name}: scoring {sidecar.height:,} full-pool eligible pairs')
    rows = _score(model, sidecar, features)
    rule, calibration = _select(rows, baseline, cal)
    checked = _locked(rows, baseline, check, rule)
    scope = ','.join(sorted(check['co'].unique().to_list())) if 'co' in check.columns else 'unspecified'
    before_f = checked.get('before', {}).get('macro_f05')
    after_f = checked.get('after', {}).get('macro_f05')
    gain = after_f - before_f if before_f is not None and after_f is not None else None
    _log(f'{name}: check countries={scope}, owners={check.height:,}; frozen rule {rule}; '
         f'check passed={checked.get("passed", False)}; macro F0.5 before={before_f}, '
         f'after={after_f}, gain={gain}')
    return model, rule, {**details, 'available': True, 'rule': rule,
                         'calibration': calibration, 'check': checked}


def run(train_frame, refs, targets, fit_frame, cal_refs, check_refs,
        baseline_train, core_features, output_dir, fit_refs=None, s2_count=None):
    'Return runtime models/rules plus a separate JSON-serializable report'
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    core = _features(core_features)
    _validate_fit(train_frame, fit_frame, cal_refs, check_refs, fit_refs)
    _log(f'building train directional support over {train_frame.height:,} full candidate pairs')
    sidecar, directional, support, directions = swaps.build(train_frame, refs, targets, s2_count)
    _log(f'train sidecar: {sidecar.height:,} eligible pairs; {directions.height:,} directions')
    fit = sidecar.join(fit_frame.select(KEYS), on=KEYS, how='semi').filter(pl.col('co').is_in(COUNTRIES))
    cal, check = _country(cal_refs, COUNTRIES), _country(check_refs, COUNTRIES)
    models, rules, feature_sets, heads = {}, {}, {}, {}
    for name, ff in [('control', core), ('treatment', core + directional)]:
        model, rule, evidence = _head(name, fit, sidecar, ff, baseline_train, cal, check, output)
        models[name], rules[name], feature_sets[name], heads[name] = model, rule, ff, evidence
    control_check = heads['control'].get('check', {})
    treatment_check = heads['treatment'].get('check', {})
    promoted = bool(heads['control']['available'] and heads['treatment']['available'] and
                    treatment_check.get('passed', False) and
                    treatment_check['after']['macro_f05'] >= control_check['after']['macro_f05'] - 1e-12)
    transfer_heads = {}

    for src, dst in [('us', 'india'), ('india', 'us')]:
        name = 'transfer_' + src
        model, rule, evidence = _head(name, fit.filter(pl.col('co') == src), sidecar,
                                     core + directional, baseline_train,
                                     _country(cal_refs, [src]), _country(check_refs, [dst]), output)
        models[name], rules[name], feature_sets[name], transfer_heads[src] = model, rule, core + directional, evidence
    transfer_enabled = all(h.get('check', {}).get('passed', False) for h in transfer_heads.values())
    _log(f'promotion: main={promoted}; France reciprocal intersection={transfer_enabled}')
    report = {'support': support, 'heads': heads, 'promoted': promoted,
              'promotion_rule': 'Treatment frozen check guards pass and check macro F0.5 is at least control check macro F0.5.',
              'parameters': {**PARAMS, 'num_threads': int(os.environ.get('ER_THREADS', '48'))},
              'main_countries': COUNTRIES, 'class_order': ['alias', 'decoy', 'other_owned'],
              'historical_upstream_exposure_warning': WARNING,
              'transfer': {'enabled': transfer_enabled, 'heads': transfer_heads,
                           'france_policy': ('Intersection of both frozen reciprocal models only' if transfer_enabled
                                             else 'France retained unchanged: reciprocal source check guards did not both pass'),
                           'limitation': 'No labeled French evidence; reciprocal US/India checks cannot establish French accuracy.'}}
    feature_sets_path = output / 'features.json'
    feature_sets_path.write_text(json.dumps(feature_sets, indent=2))
    (output / 'rules.json').write_text(json.dumps({'rules': rules, 'promoted': promoted,
                                                 'transfer_enabled': transfer_enabled}, indent=2))
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    if directions.width:
        directions.write_parquet(output / 'direction_support.parquet')
    return {'report': report, 'models': models, 'features': feature_sets, 'rules': rules,
            'promoted': promoted, 'transfer_enabled': transfer_enabled, 'output_dir': str(output)}


def _test_directions(rows, directions):
    'bounded label-free global test inspection, including reverse support'
    country = rows.group_by('co').agg(pl.len().alias('eligible_pairs'),
        pl.col('qid').n_unique().alias('references'), pl.col('tid').n_unique().alias('targets'),
        pl.col('ds_address_exact').sum().alias('exact_address_pairs'),
        pl.col('ds_numeric_conflict').sum().alias('numeric_conflict_pairs')).sort('co').to_dicts()
    top = {}
    if directions.width:
        keys = ['_swap_country', '_swap_from', '_swap_to']
        reverse = directions.select(*keys, 'edges', 'qids', 'targets', 'address_edges').rename(
            {'_swap_from': '_swap_to', '_swap_to': '_swap_from',
             **{c: 'reverse_' + c for c in ['edges', 'qids', 'targets', 'address_edges']}})
        table = directions.join(reverse, on=keys, how='left').with_columns(
            pl.col('reverse_edges', 'reverse_qids', 'reverse_targets', 'reverse_address_edges').fill_null(0))
        for part in table.partition_by('_swap_country'):
            name = str(part['_swap_country'][0])
            top[name] = part.sort(['edges', '_swap_from', '_swap_to'], descending=[True, False, False]).head(30).to_dicts()
    return {'countries': country, 'top_directions_by_country': top,
            'inspection_scope': 'Complete test candidate pool; label-free direction totals include all candidate owners.'}


def _prediction_summary(rows):
    aggregates = []
    for name in ['p_alias', 'p_decoy', 'p_other']:
        aggregates.extend([pl.col(name).mean().alias(name + '_mean'),
                           pl.col(name).quantile(.5).alias(name + '_median'),
                           pl.col(name).quantile(.9).alias(name + '_p90'),
                           pl.col(name).quantile(.99).alias(name + '_p99')])
    aggregates.extend([(pl.col('p_decoy') >= .75).sum().alias('decoy_rank_ge_075'),
                       (pl.col('p_decoy') >= .9).sum().alias('decoy_rank_ge_090'),
                       (pl.col('p_alias') >= .99).sum().alias('alias_rank_ge_099')])
    return rows.group_by('co').agg(pl.len().alias('eligible_pairs'), *aggregates).sort('co').to_dicts()


def infer(bundle, test_frame, test_refs, test_targets, baseline_test, s2_count=None):
    'replay frozen rules without labels, fitting, threshold selection, or checking'
    diagnostics = {'promoted': bool(bundle['promoted']), 'transfer_enabled': bool(bundle['transfer_enabled']),
                   'main_removed': 0, 'main_added': 0, 'france_removed': 0, 'france_added': 0}

    _log(f'building test directional inspection over {test_frame.height:,} full candidate pairs')
    rows, _, support, directions = swaps.build(test_frame, test_refs, test_targets, s2_count)
    _log(f'test sidecar: {rows.height:,} eligible pairs; {directions.height:,} directions')
    diagnostics['support'] = support
    diagnostics['direction_inspection'] = _test_directions(rows, directions)
    output = Path(bundle['output_dir']) if bundle.get('output_dir') else None
    if output is not None:
        path = output / 'test_direction_support.parquet'
        if not directions.width:
            directions = pl.DataFrame(schema={'_swap_country': pl.String, '_swap_from': pl.String,
                                              '_swap_to': pl.String, 'edges': pl.UInt32})
        directions.write_parquet(path)
        diagnostics['artifacts'] = {'test_direction_support': str(path),
                                    'inference_diagnostics': str(output / 'inference_diagnostics.json')}
    treatment_scored = None
    if bundle['models'].get('treatment') is not None:
        _log('scoring treatment for test probability inspection, including vetoed policies')
        treatment_scored = _score(bundle['models']['treatment'], rows, bundle['features']['treatment'])
        diagnostics['treatment_predictions_by_country'] = _prediction_summary(treatment_scored)
    res = baseline_test
    if bundle['promoted']:
        scored = treatment_scored
        anchors = _country(test_refs, COUNTRIES)
        drops, adds = policy.propose_actions(scored, baseline_test, anchors, bundle['rules']['treatment'])
        res = policy.apply_actions(res, drops, adds, anchors)
        diagnostics.update(main_removed=drops.height, main_added=adds.height)
    if bundle['transfer_enabled']:
        anchors = _country(test_refs, ['france'])
        actions = []
        for country in COUNTRIES:
            name = 'transfer_' + country
            scored = _score(bundle['models'][name], rows, bundle['features'][name])
            actions.append(policy.propose_actions(scored, baseline_test, anchors, bundle['rules'][name]))
        drops = actions[0][0].join(actions[1][0].select(KEYS), on=KEYS, how='semi')
        adds = actions[0][1].join(actions[1][1].select(KEYS), on=KEYS, how='semi')
        res = policy.apply_actions(res, drops, adds, anchors)
        diagnostics.update(france_removed=drops.height, france_added=adds.height)
    if not bundle['promoted'] and not bundle['transfer_enabled']:
        diagnostics['reason'] = 'All policies disabled; baseline retained unchanged; full test directions inspected'
    if output is not None:
        (output / 'inference_diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
    _log(f'test actions: main +{diagnostics["main_added"]}/-{diagnostics["main_removed"]}; '
         f'France +{diagnostics["france_added"]}/-{diagnostics["france_removed"]}')
    return res, diagnostics

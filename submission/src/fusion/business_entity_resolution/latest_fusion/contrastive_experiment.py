'Fixed-budget matched control/treatment owner-switch experiment'
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from . import contrastive_owners as contrastive
from . import owner_switch as policy
from .structured_protocol import WARNING

KEYS = ['qid', 'tid']
COUNTRIES = ['us', 'india']
MIN_CLASS = 50
ROUNDS = 180
DISABLED = {'enabled': False, 'min_alias': None, 'min_margin': None}
PARAMS = dict(objective='multiclass', num_class=3, metric='multi_logloss',
              learning_rate=.04, num_leaves=15, max_depth=5,
              min_data_in_leaf=50, lambda_l2=40., verbosity=-1,
              seed=2709, deterministic=True, force_col_wise=True)


def _log(message):
    print(time.strftime('%H:%M:%S') + ' contrastive: ' + message, flush=True)


def _anchors(refs):
    return refs.rename({'rid': 'qid'}) if 'rid' in refs else refs


def _country(refs):
    return _anchors(refs).filter(pl.col('co').is_in(COUNTRIES))


def _features(core_features):
    core = list(dict.fromkeys(core_features))
    forbidden = set(KEYS + ['rid', 'eid', 'y', 'own', 'deg', 'fold', 'co', 'nm', 'ad'])
    bad = forbidden.intersection(core) | {c for c in core if c.startswith('ct_')}
    if bad or not core:
        raise ValueError(f'Control requires nonempty label-free core features; forbidden: {sorted(bad)}')
    return core


def _build(frame, refs, targets):
    'null score fill is local to alternative ranking, never model inputs'
    _log(f'building triadic features over {frame.height:,} complete candidate pairs')
    view = frame.with_columns(pl.col('newest').fill_null(1e-4))
    augmented, names, diagnostics = contrastive.build(view, refs, targets)
    augmented = augmented.with_columns(frame['newest'])
    if 'co' not in augmented:
        augmented = augmented.join(refs.select(pl.col('rid').alias('qid'), 'co'),
            on='qid', how='left', validate='m:1', maintain_order='left')
    _log(f'triadic features complete: {diagnostics["alternative_pairs"]:,} pairs with alternatives')
    return augmented, names, diagnostics


def _validate_fit(train, fit, cal_refs, check_refs, fit_refs):
    policy._keys(fit, KEYS, 'Fitting candidates')
    if fit.select(KEYS).join(train.select(KEYS), on=KEYS, how='anti').height:
        raise ValueError('Fitting pairs outside full candidate pool')
    cal, check = _anchors(cal_refs).select('qid'), _anchors(check_refs).select('qid')
    policy._anchors(cal)
    policy._anchors(check)
    if cal.join(check, on='qid').height:
        raise ValueError('Calibration/check reference overlap')
    cal_targets = train.join(cal, on='qid', how='semi').select('tid').unique()
    check_targets = train.join(check, on='qid', how='semi').select('tid').unique()
    if cal_targets.join(check_targets, on='tid').height:
        raise ValueError('Calibration/check target overlap')
    held_targets = pl.concat([cal_targets, check_targets]).unique()
    if fit.select('tid').join(held_targets, on='tid').height:
        raise ValueError('Fitting targets overlap calibration/check targets')
    policy._labels(fit)
    if 'own' not in fit or fit['own'].null_count():
        raise ValueError('Fitting requires truth ownership')
    original = train.select(*KEYS, pl.col('y').alias('_y'), pl.col('own').alias('_own'))
    if fit.select(*KEYS, 'y', 'own').join(original, on=KEYS).filter(
            (pl.col('y') != pl.col('_y')) | (pl.col('own') != pl.col('_own'))).height:
        raise ValueError('Fitting labels differ from full candidate pool')
    if fit_refs is not None:
        allowed = _anchors(fit_refs).select(pl.col('qid').cast(fit.schema['own']).alias('own'))
        if fit.filter(pl.col('own') >= 0).select('own').unique().join(allowed, on='own', how='anti').height:
            raise ValueError('Fitting true owners outside supplied fit references')


def _class_counts(frame):
    return {'alias': frame.filter(pl.col('y') == 1).height,
            'decoy': frame.filter((pl.col('y') == 0) & (pl.col('own') < 0)).height,
            'other_owned': frame.filter((pl.col('y') == 0) & (pl.col('own') >= 0)).height}


def _fit(frame, features, path):
    labels = np.where(frame['y'].to_numpy() == 1, 0,
                      np.where(frame['own'].to_numpy() < 0, 1, 2)).astype(np.int32)
    model = lgb.train({**PARAMS, 'num_threads': int(os.environ.get('ER_THREADS', '48'))},
        lgb.Dataset(frame.select(features).to_numpy(), label=labels, feature_name=features),
        num_boost_round=ROUNDS)
    model.save_model(str(path))
    return model


def _score(model, rows, features, chunk_size=250_000):
    'predict fixed class order over every candidate, in bounded array chunks'
    if model.feature_name() != list(features):
        raise ValueError('Contrastive model feature order differs from inference schema')
    probs = np.empty((rows.height, 3), dtype=np.float32)
    for start in range(0, rows.height, chunk_size):
        chunk = rows.slice(start, chunk_size)
        p = np.asarray(model.predict(chunk.select(features).to_numpy(),
            num_threads=int(os.environ.get('ER_THREADS', '48'))))
        if (p.shape != (chunk.height, 3) or not np.isfinite(p).all() or
                (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1., atol=1e-6)):
            raise ValueError('Invalid three-class contrastive predictions')
        probs[start:start + chunk.height] = p
    return rows.select(*KEYS, *[c for c in ('co', 'y', 'own') if c in rows]).with_columns(
        *[pl.Series(name, probs[:, i]) for i, name in enumerate(['p_alias', 'p_decoy', 'p_other'])])


def _propose(rows, baseline, anchors, rule):
    if rule['enabled']:
        return policy.propose_switches(rows, baseline, anchors, rule['min_alias'], rule['min_margin'])
    return policy.propose_switches(rows, baseline, anchors, 1., 1.).head(0)


def _locked(rows, baseline, anchors, rule):
    if not anchors.height:
        return {'passed': False, 'reason': 'No check anchors', 'after': {'macro_f05': 0.}}
    evidence = policy.evaluate_switches(baseline, _propose(rows, baseline, anchors, rule), anchors)
    checks = {'enabled': bool(rule['enabled']), 'five_corrected': evidence['corrected'] >= 5,
              'zero_broken': evidence['broken'] == 0,
              'positive_macro_gain': evidence['after']['macro_f05'] > evidence['before']['macro_f05'] + 1e-12,
              'precision_nondecreasing': evidence['after']['pair_precision'] >= evidence['before']['pair_precision'] - 1e-12,
              'recall_nondecreasing': evidence['after']['pair_recall'] >= evidence['before']['pair_recall'] - 1e-12}
    for co, metrics in evidence['countries'].items():
        checks[co + '_nondecreasing'] = all(metrics['after'][k] >= metrics['before'][k] - 1e-12
                                           for k in policy.METRICS)
    return {**evidence, 'checks': checks, 'passed': all(checks.values())}


def _head(name, fit, rows, features, baseline, cal, check, output):
    counts = _class_counts(fit)
    _log(f'{name}: {fit.height:,} fit rows; class counts {counts}')
    details = {'features': list(features), 'fit_rows': fit.height, 'class_counts': counts,
               'rounds': ROUNDS, 'minimum_class_count': MIN_CLASS}
    if min(counts.values()) < MIN_CLASS:
        _log(f'{name}: disabled, insufficient fitting examples per class')
        return None, dict(DISABLED), {**details, 'available': False,
            'reason': 'Baseline retained: insufficient fitting examples per class'}
    _log(f'{name}: fitting {ROUNDS} multiclass rounds with {len(features)} features')
    model = _fit(fit, features, output / (name + '_model.txt'))
    _log(f'{name}: scoring {rows.height:,} complete candidate pairs')
    scored = _score(model, rows, features)
    if cal.height:
        rule, calibration = policy.select_policy(scored, baseline, cal)
    else:
        rule, calibration = dict(DISABLED), {'passed': False, 'reason': 'No calibration anchors'}
    check_report = _locked(scored, baseline, check, rule)
    _log(f'{name}: frozen rule {rule}; check passed={check_report.get("passed", False)}; '
         f'check macro F0.5={check_report.get("after", {}).get("macro_f05")}')
    return model, rule, {**details, 'available': True, 'rule': rule,
                        'calibration': calibration, 'check': check_report}


def run(train_frame, refs, targets, fit_frame, cal_refs, check_refs,
        baseline_train, core_features, output_dir, fit_refs=None, s2_count=None, **unused):
    'Return runtime models/features/frozen rules plus a JSONable report'
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    core = _features(core_features)
    _log(f'start: {train_frame.height:,} training candidates, {fit_frame.height:,} supplied fitting pairs')
    _validate_fit(train_frame, fit_frame, cal_refs, check_refs, fit_refs)
    rows, added, support = _build(train_frame, refs, targets)
    if any(c not in rows or not rows.schema[c].is_numeric() for c in core):
        raise ValueError('Core feature schema is missing or nonnumeric')
    fit = rows.join(fit_frame.select(KEYS), on=KEYS, how='semi', maintain_order='left').filter(
        pl.col('co').is_in(COUNTRIES))
    cal, check = _country(cal_refs), _country(check_refs)
    models, rules, ff, heads = {}, {}, {}, {}
    for name, names in [('control', core), ('treatment', core + added)]:
        models[name], rules[name], heads[name] = _head(name, fit, rows, names, baseline_train, cal, check, output)
        ff[name] = list(names)
    control_check = heads['control'].get('check', {})
    treatment_check = heads['treatment'].get('check', {})
    promoted = bool(heads['control']['available'] and heads['treatment']['available'] and
        treatment_check.get('passed', False) and
        treatment_check['after']['macro_f05'] >= control_check['after']['macro_f05'] - 1e-12)
    _log(f'promotion decision: {promoted}')
    report = {'support': support, 'heads': heads, 'promoted': promoted,
              'parameters': {**PARAMS, 'num_threads': int(os.environ.get('ER_THREADS', '48'))},
              'main_countries': COUNTRIES, 'class_order': ['alias', 'decoy', 'other_owned'],
              'identical_fit_keys': True, 'fit_pairs': fit.height,
              'promotion_rule': 'Locked treatment check gain >0, at least five corrections, zero broken links, overall/country P/R/F nonregression, and check macro F0.5 >= matched control.',
              'historical_upstream_exposure_warning': WARNING,
              'interpretation': 'Experimental scoped action rule; no posterior or hidden-test improvement guarantee.'}
    for filename, content in [('features.json', ff), ('rules.json', {'rules': rules, 'promoted': promoted}),
                              ('report.json', report)]:
        (output / filename).write_text(json.dumps(content, indent=2, allow_nan=False))
    return {'report': report, 'models': models, 'features': ff, 'rules': rules,
            'promoted': promoted, 'output_dir': str(output)}


def infer(bundle, test_frame, test_refs, test_targets, baseline_test, s2_count=None, **unused):
    'replay promoted treatment only; otherwise return the exact baseline'
    diagnostics = {'promoted': bool(bundle['promoted']), 'switches': 0,
                   'france_policy': 'Baseline unchanged', 'accepted_targets': baseline_test.height}
    if not bundle['promoted']:
        _log('inference retained exact baseline: treatment not promoted')
        return baseline_test.clone(), diagnostics
    rule = bundle['rules']['treatment']
    model = bundle['models']['treatment']
    if model is None or not rule['enabled']:
        raise ValueError('Promoted contrastive bundle lacks model or enabled frozen rule')
    rows, added, support = _build(test_frame, test_refs, test_targets)
    ff = bundle['features']['treatment']
    if [c for c in ff if c.startswith('ct_')] != added:
        raise ValueError('Contrastive feature schema changed since training')
    scored = _score(model, rows, ff)
    _log(f'inference scored {rows.height:,} complete candidates; replaying frozen treatment rule')
    switches = _propose(scored, baseline_test, _country(test_refs).select('qid'), rule)
    accepted = policy.apply_switches(baseline_test, switches)
    diagnostics.update(switches=switches.height, support=support,
        frozen_rule=rule, accepted_targets=accepted.height)
    _log(f'inference complete: {switches.height:,} owner switches, {accepted.height:,} accepted targets')
    output = Path(bundle['output_dir']) if bundle.get('output_dir') else None
    if output is not None:
        switches.write_parquet(output / 'switches_test.parquet')
    return accepted, diagnostics

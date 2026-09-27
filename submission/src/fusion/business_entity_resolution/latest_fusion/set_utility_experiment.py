'fixed-budget exact set-utility regression with frozen threshold replay'
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from er.stack import decode
from .set_utility import FEATURE_NAMES, build_actions, choose_actions, apply_actions
from .structured_protocol import split_references, WARNING

COUNTRIES = ['india', 'us']
THRESHOLDS = [0., .001, .005, .01, .02]
ROUNDS = 180
MIN_FIT = 100
PARAMS = dict(objective='regression', metric='l2', learning_rate=.04,
              num_leaves=15, min_data_in_leaf=50, lambda_l2=40.,
              verbosity=-1, seed=2709, deterministic=True, force_col_wise=True)
KEYS = ['qid', 'tid']


def _log(message):
    print(time.strftime('%H:%M:%S') + ' set utility: ' + message, flush=True)


def _anchors(refs, dtype=None):
    expression = pl.col('rid') if 'rid' in refs else pl.col('qid')
    if dtype is not None:
        expression = expression.cast(dtype)
    return refs.select(expression.alias('qid'), *[c for c in ['deg', 'co'] if c in refs])


def _known(refs, dtype=None):
    return _anchors(refs, dtype).filter(pl.col('co').is_in(COUNTRIES))


def _feature_sets(core):
    core = list(dict.fromkeys(core))
    forbidden = set(KEYS + ['rid', 'eid', 'y', 'own', 'deg', 'fold', 'co', 'nm', 'ad', 'utility_delta'])
    if forbidden.intersection(core) or set(FEATURE_NAMES).intersection(core):
        raise ValueError('Labels, identifiers, or set-utility fields entered core features')
    return {'control': ['action_sign', 'action_probability'] + core,
            'treatment': list(FEATURE_NAMES) + core}


def _actions(frame, refs, baseline, core):
    if frame.select(KEYS).unique().height != frame.height:
        raise ValueError('Duplicate candidate pairs')
    for column in ['baseline_p'] + (['baseline_raw'] if 'baseline_raw' in frame else []):
        if frame.filter(pl.col(column).is_null() | ~pl.col(column).is_finite() |
                        (pl.col(column) < 0) | (pl.col(column) > 1)).height:
            raise ValueError('Baseline probabilities must be nonnull finite in [0, 1]')
    order = ['tid', 'baseline_p'] + (['baseline_raw'] if 'baseline_raw' in frame else []) + ['qid']
    descending = [False, True] + ([True] if 'baseline_raw' in frame else []) + [False]
    winners = frame.sort(order, descending=descending).unique('tid', keep='first', maintain_order=True)
    winners = winners.select(*KEYS, pl.col('baseline_p').alias('p'), *(['y'] if 'y' in winners else []))
    actions, _ = build_actions(winners, baseline, _anchors(refs, frame.schema['qid']))
    enriched = actions.join(frame.select(*KEYS, *core), on=KEYS, how='left', validate='1:1', maintain_order='left')
    if actions.select(KEYS).join(frame.select(KEYS), on=KEYS, how='anti').height:
        raise ValueError('Baseline action pair missing from full candidate pool')
    if any(not enriched.schema[column].is_numeric() for column in core):
        raise ValueError('Core action features must be numeric')
    return enriched


def _fit(actions, features, path):
    model = lgb.train({**PARAMS, 'num_threads': int(os.environ.get('ER_THREADS', '2'))},
        lgb.Dataset(actions.select(features).to_numpy(), label=actions['utility_delta'].to_numpy(),
                    feature_name=features), num_boost_round=ROUNDS)
    model.save_model(str(path))
    return model


def _predict(model, actions, features):
    parts = []
    for chunk in actions.iter_slices(250_000):
        prediction = np.asarray(model.predict(chunk.select(features).to_numpy(),
            num_threads=int(os.environ.get('ER_THREADS', '2')))).reshape(-1)
        if prediction.size != chunk.height or not np.isfinite(prediction).all():
            raise ValueError('Invalid learned utility predictions')
        parts.append(prediction)
    return np.concatenate(parts) if parts else np.empty(0)


def _metrics(accepted, refs):
    if not refs.height:
        return None
    return {'overall': decode.score(accepted, refs),
            'countries': {country: decode.score(accepted, refs.filter(pl.col('co') == country))
                          for country in refs['co'].unique().sort().to_list()}}


def _evaluate(actions, predictions, baseline, refs, threshold):
    scoped = actions.join(refs.select('qid'), on='qid', how='semi', maintain_order='left')
    if threshold is None:
        chosen = scoped.head(0)
    else:
        scored = actions.with_columns(pl.Series('predicted_delta', predictions)).join(
            refs.select('qid'), on='qid', how='semi', maintain_order='left')
        chosen = choose_actions(scored, 'predicted_delta', threshold)
    after = apply_actions(baseline, chosen)
    before_metrics, after_metrics = _metrics(baseline, refs), _metrics(after, refs)
    passed = False
    guards = {}
    if before_metrics is not None:
        gain = after_metrics['overall']['macro_f05'] - before_metrics['overall']['macro_f05']
        guards['positive_macro_gain'] = gain > 1e-12
        for name in ['overall'] + list(before_metrics['countries']):
            old = before_metrics['overall'] if name == 'overall' else before_metrics['countries'][name]
            new = after_metrics['overall'] if name == 'overall' else after_metrics['countries'][name]
            for metric in ['macro_f05', 'pair_precision', 'pair_recall']:
                guards[f'{name}_{metric}_nondecreasing'] = new[metric] >= old[metric] - 1e-12
        passed = all(guards.values())
    counts = {'TPadded': chosen.filter((pl.col('operation') == 1) & (pl.col('y') == 1)).height,
              'FPadded': chosen.filter((pl.col('operation') == 1) & (pl.col('y') == 0)).height,
              'TPremoved': chosen.filter((pl.col('operation') == -1) & (pl.col('y') == 1)).height,
              'FPremoved': chosen.filter((pl.col('operation') == -1) & (pl.col('y') == 0)).height}
    if 'utility_delta' in chosen:
        counts.update(beneficial_actions=chosen.filter(pl.col('utility_delta') > 0).height,
                      harmful_actions=chosen.filter(pl.col('utility_delta') < 0).height,
                      neutral_actions=chosen.filter(pl.col('utility_delta') == 0).height)
    return {'threshold': threshold, 'before': before_metrics, 'after': after_metrics,
            'chosen_actions': chosen.height, **counts, 'guards': guards, 'passed': passed}


def _select(actions, predictions, baseline, refs):
    evaluated = [_evaluate(actions, predictions, baseline, refs, threshold) for threshold in THRESHOLDS]
    passed = [result for result in evaluated if result['passed']]
    sel = max(passed, key=lambda result: result['after']['overall']['macro_f05']) if passed else _evaluate(
        actions, predictions, baseline, refs, None)
    return sel['threshold'], {**sel, 'evaluated': evaluated}


def _isolated_fit(train, refs, fit_frame, fit_refs, original_cal, check, actions):
    if train.filter(pl.col('y').is_null() | pl.col('own').is_null() | ~pl.col('y').is_in([0, 1]) |
                    (pl.col('y') != (pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.Int8))).height:
        raise ValueError('Training labels disagree with ownership')
    if fit_frame.select(KEYS).unique().height != fit_frame.height or fit_frame.select(KEYS).join(
            train.select(KEYS), on=KEYS, how='anti').height:
        raise ValueError('Invalid isolated fitting pair keys')
    comparison = fit_frame.select(*KEYS, 'y', 'own').join(train.select(*KEYS,
        pl.col('y').alias('_original_y'), pl.col('own').alias('_original_own')), on=KEYS)
    if comparison.filter((pl.col('y') != pl.col('_original_y')) |
                         (pl.col('own') != pl.col('_original_own')) | pl.col('y').is_null() | pl.col('own').is_null()).height:
        raise ValueError('Isolated fitting labels differ from full pool')
    protected = pl.concat([_anchors(original_cal, train.schema['qid']).select('qid'),
                           _anchors(check, train.schema['qid']).select('qid')]).unique()
    held_targets = train.join(protected, on='qid', how='semi').select('tid').unique()
    if fit_frame.join(held_targets, on='tid', how='semi').height:
        raise ValueError('Fitting targets touch protected owners')
    owners = _known(fit_refs, train.schema['qid'])
    if owners.join(protected, on='qid').height:
        raise ValueError('Fit and protected owners overlap')
    fit_owner_ids = owners.select(pl.col('qid').cast(fit_frame.schema['own']).alias('own'))
    if fit_frame.filter(pl.col('own') >= 0).select('own').unique().join(fit_owner_ids, on='own', how='anti').height:

        all_ids = _anchors(fit_refs, train.schema['qid']).select(pl.col('qid').cast(fit_frame.schema['own']).alias('own'))
        if fit_frame.filter(pl.col('own') >= 0).select('own').unique().join(all_ids, on='own', how='anti').height:
            raise ValueError('Fitting true owners outside fit references')
    unsafe = train.join(held_targets, on='tid', how='semi').select('qid').unique()
    safe = owners.join(unsafe, on='qid', how='anti')
    fitted = actions.join(safe.select('qid'), on='qid', how='semi').join(
        fit_frame.select(KEYS), on=KEYS, how='semi', maintain_order='left')
    return fitted, {'assigned_fit_owners': owners.height, 'safe_fit_owners': safe.height,
                    'fit_owners_excluded_for_protected_target_incidence': owners.height-safe.height,
                    'actual_action_fit_owners': fitted['qid'].n_unique(),
                    'safe_fit_owners_without_model_actions': safe.height-fitted['qid'].n_unique(),
                    'fit_action_rows': fitted.height, 'protected_targets': held_targets.height,
                    'fit_actions_touching_protected_targets': fitted.join(held_targets, on='tid', how='semi').height,
                    'scope': 'Whole fit owners touching any original calibration/check target are excluded; each fitted action pair must occur in isolated fitting pairs.'}


def run(train_frame, refs, targets, fit_frame, cal_refs, check_refs, baseline_train,
        core_features, output_dir, fit_refs=None, s2_count=None):
    'fit two regressors, freeze calibration thresholds, then veto on check'
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ff = _feature_sets(core_features)
    core = list(dict.fromkeys(core_features))
    _log(f'building fixed actions over {train_frame.height:,} full candidate pairs')
    actions = _actions(train_frame, refs, baseline_train, core)
    _log(f'built {actions.height:,} actions for {actions["qid"].n_unique():,} owners')
    if 'utility_delta' not in actions:
        raise ValueError('Exact training utility labels are required')
    assigned_fit, original_cal, original_check, _ = split_references(refs)
    if fit_refs is None:
        fit_refs = assigned_fit
    if _anchors(fit_refs).select('qid').join(_anchors(assigned_fit).select('qid'), on='qid', how='anti').height:
        raise ValueError('Fit references differ from frozen name-group fit partition')
    cal, check = _known(cal_refs, train_frame.schema['qid']), _known(check_refs, train_frame.schema['qid'])
    if cal.select('qid').join(check.select('qid'), on='qid').height:
        raise ValueError('Calibration/check owners overlap')
    if cal.select('qid').join(_anchors(original_cal, train_frame.schema['qid']).select('qid'), on='qid', how='anti').height or check.select('qid').join(
            _anchors(original_check, train_frame.schema['qid']).select('qid'), on='qid', how='anti').height:
        raise ValueError('Calibration/check references differ from frozen partitions')
    cal_targets = train_frame.join(cal.select('qid'), on='qid', how='semi').select('tid').unique()
    check_targets = train_frame.join(check.select('qid'), on='qid', how='semi').select('tid').unique()
    if cal_targets.join(check_targets, on='tid').height:
        raise ValueError('Calibration/check targets overlap')
    fitted, isolation = _isolated_fit(train_frame, refs, fit_frame, fit_refs, original_cal,
                                      original_check, actions)
    _log(f'{fitted.height:,} isolated fit actions; {cal.height:,} calibration and {check.height:,} check owners')
    models, cuts, heads = {}, {}, {}
    sufficient = fitted.height >= MIN_FIT and fitted['utility_delta'].n_unique() > 1
    for name in ['control', 'treatment']:
        _log(f'{name}: fitting {ROUNDS} regression rounds, {len(ff[name])} features; sufficient={sufficient}')
        model = _fit(fitted, ff[name], output / f'{name}_model.txt') if sufficient else None
        models[name] = model
        if model is None:
            cuts[name] = None
            heads[name] = {'available': False, 'features': ff[name],
                           'reason': 'Insufficient isolated action rows or constant exact utility labels',
                           'check': _evaluate(actions, np.zeros(actions.height), baseline_train, check, None)}
        else:
            pred = _predict(model, actions, ff[name])
            cut, calibration = _select(actions, pred, baseline_train, cal)
            cuts[name] = cut
            heads[name] = {'available': True, 'features': ff[name], 'threshold': cut,
                           'calibration': calibration,
                           'check': _evaluate(actions, pred, baseline_train, check, cut)}
            _log(f'{name}: frozen calibration threshold={cut}; check passed={heads[name]["check"]["passed"]}; '
                 f'check macro F0.5={heads[name]["check"]["after"]["overall"]["macro_f05"] if heads[name]["check"]["after"] else None}')
    treatment_check, control_check = heads['treatment']['check'], heads['control']['check']
    promoted = bool(sufficient and treatment_check['passed'] and control_check['after'] is not None and
        treatment_check['after']['overall']['macro_f05'] >= control_check['after']['overall']['macro_f05'] - 1e-12)
    _log(f'promotion={promoted}')
    report = {'promoted': promoted, 'heads': heads, 'isolation': isolation,
              'threshold_grid': THRESHOLDS, 'rounds': ROUNDS, 'parameters': PARAMS,
              'minimum_fit_actions': MIN_FIT, 'fit_utility_min': fitted['utility_delta'].min(),
              'fit_utility_max': fitted['utility_delta'].max(), 'fit_utility_mean': fitted['utility_delta'].mean(),
              'fit_utility_counts': {'positive': fitted.filter(pl.col('utility_delta') > 0).height,
                                     'zero': fitted.filter(pl.col('utility_delta') == 0).height,
                                     'negative': fitted.filter(pl.col('utility_delta') < 0).height},
              'training_target': 'Exact per-owner F0.5(after)-F0.5(before), including correctly empty zero-degree owners',
              'promotion_rule': 'Frozen treatment check improves macro F0.5, preserves country macro F0.5 and overall/country precision and recall, and matches or exceeds frozen control check macro F0.5',
              'main_countries': COUNTRIES, 'france_policy': 'Baseline unchanged',
              'prediction_interpretation': 'Learned utility estimates; no guaranteed future gain',
              'historical_upstream_exposure_warning': WARNING}
    (output / 'features.json').write_text(json.dumps(ff, indent=2))
    (output / 'policy.json').write_text(json.dumps({'thresholds': cuts, 'promoted': promoted}, indent=2))
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    return {'report': report, 'promoted': promoted, 'models': models, 'features': ff,
            'core_features': core, 'thresholds': cuts, 'output_dir': str(output)}


def infer(bundle, test_frame, test_refs, test_targets, baseline_test, s2_count=None):
    'Replay the frozen treatment on IN/US only; no labels enter inference'
    if not bundle['promoted']:
        return baseline_test.clone(), {'promoted': False, 'chosen_actions': 0, 'changed': False,
                                      'france_policy': 'Baseline unchanged'}
    unlabeled = test_frame.drop([c for c in ['y', 'own'] if c in test_frame])
    base = baseline_test.drop([c for c in ['y', 'own'] if c in baseline_test])
    actions = _actions(unlabeled, test_refs, base, bundle['core_features'])
    model = bundle['models']['treatment']
    pred = _predict(model, actions, bundle['features']['treatment'])
    known = _known(test_refs, test_frame.schema['qid']).select('qid')
    scored = actions.with_columns(pl.Series('predicted_delta', pred)).join(known, on='qid', how='semi')
    chosen = choose_actions(scored, 'predicted_delta', bundle['thresholds']['treatment'])
    accepted = apply_actions(baseline_test, chosen)
    return accepted, {'promoted': True, 'chosen_actions': chosen.height,
                      'additions': chosen.filter(pl.col('operation') == 1).height,
                      'removals': chosen.filter(pl.col('operation') == -1).height,
                      'changed': bool(chosen.height), 'threshold': bundle['thresholds']['treatment'],
                      'france_policy': 'Baseline unchanged', 'global_unique_targets': accepted['tid'].n_unique() == accepted.height}

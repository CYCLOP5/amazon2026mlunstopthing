'three fitting-only recall verifiers for the frozen-incumbent campaign'
import json
import os
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import polars as pl


METHODS = ('honest_leaf', 'crossview_consensus', 'hard_negative')
SEED = 270927
FORBIDDEN = {'qid', 'tid', 'rid', 'eid', 'y', 'own', 'deg', 'fold', 'co', 'seg',
             'nm', 'ad', 'target', 'label'}
PARAMS = dict(objective='binary', metric='binary_logloss', verbosity=-1,
              learning_rate=.045, num_leaves=31, max_depth=7,
              min_data_in_leaf=100, lambda_l2=15., deterministic=True,
              force_col_wise=True, seed=SEED)


def _threads():
    return max(1, int(os.environ.get('ER_THREADS', '8')))


def _matrix(frame, features):
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing verifier features: {sorted(missing)}')
    if any(not (frame.schema[c].is_numeric() or frame.schema[c] == pl.Boolean)
           for c in features):
        raise ValueError('Verifier features must be numeric')
    x = frame.select(features).cast(pl.Float32).to_numpy()
    if np.isinf(x).any():
        raise ValueError('Verifier features contain infinity')
    return x


def _hash_targets(tids):
    'fixed integer hash: all candidate owners of a target stay together'
    x = np.asarray(tids, dtype=np.uint64) ^ np.uint64(SEED)
    with np.errstate(over='ignore'):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def _validate_fit(frame, features):
    if not features or len(set(features)) != len(features):
        raise ValueError('Verifier requires unique nonempty feature names')
    if FORBIDDEN.intersection(features):
        raise ValueError('Labels, ownership, population metadata and IDs cannot be model features')
    if not {'qid', 'tid', 'y'}.issubset(frame.columns):
        raise ValueError('Fitting rows require qid, tid and y')
    if frame.select('qid', 'tid', 'y').null_count().sum_horizontal().sum():
        raise ValueError('Fitting IDs and labels must be nonnull')
    if frame.select('qid', 'tid').unique().height != frame.height:
        raise ValueError('Duplicate fitting pair')
    if frame.filter(~pl.col('y').is_in([0, 1])).height:
        raise ValueError('Fitting labels must be binary')
    if 'co' in frame and frame.filter(~pl.col('co').is_in(['india', 'us'])).height:
        raise ValueError('Campaign verifiers fit India/US only')

    _matrix(frame.head(1), features)


def _fit_model(frame, features, path, rounds, *, params=None, weights=None):
    print(time.strftime('%H:%M:%S') + f' recall_a: fitting {Path(path).stem}: '
          f'{frame.height:,} rows, {len(features)} features, {rounds} rounds', flush=True)
    model = lgb.train({**PARAMS, 'num_threads': _threads(), **(params or {})},
        lgb.Dataset(_matrix(frame, features), label=frame['y'].to_numpy(),
                    weight=weights, feature_name=list(features)),
        num_boost_round=rounds)
    model.save_model(str(path))
    print(time.strftime('%H:%M:%S') + f' recall_a: saved {Path(path).name}', flush=True)
    return model


def _predict_model(model, frame, features, chunk_size=250_000, *, leaves=False):
    if model.feature_name() != list(features):
        raise ValueError('Stored model feature order differs from inference schema')
    output = np.empty(frame.height, dtype=np.int32 if leaves else np.float32)
    for start in range(0, frame.height, chunk_size):
        part = frame.slice(start, chunk_size)
        values = np.asarray(model.predict(_matrix(part, features),
            num_threads=_threads(), pred_leaf=leaves)).reshape(-1)
        if values.size != part.height or not np.isfinite(values).all():
            raise ValueError('Invalid verifier predictions')
        if not leaves and ((values < 0).any() or (values > 1).any()):
            raise ValueError('Verifier predictions are outside [0,1]')
        output[start:start + part.height] = values
        if frame.height > chunk_size and (start//chunk_size) % 4 == 0:
            print(time.strftime('%H:%M:%S') + f' recall_a: scored '
                  f'{start + part.height:,}/{frame.height:,} rows', flush=True)
    return output


def wilson_lower(positive, total, z=1.6448536269514722):
    'one-sided 95% wilson lower bound; an unsupported leaf always abstains'
    positive, total = np.asarray(positive, float), np.asarray(total, float)
    if ((positive < 0).any() or (total < positive).any() or
            not np.isfinite(positive).all() or not np.isfinite(total).all()):
        raise ValueError('Invalid leaf evidence counts')
    n = np.maximum(total, 1.)
    rate = positive/n
    lower = (rate + z*z/(2*n) - z*np.sqrt(rate*(1-rate)/n + z*z/(4*n*n))) / (1+z*z/n)
    return np.where(total > 0, np.clip(lower, 0., 1.), 0.).astype(np.float32)


def _views(features):
    score = [c for c in features if c in ('baseline_p', 'baseline_raw', 'newest', 'gate')
             or c.endswith(('_logit', '_present', '_owner_margin'))]
    structure = [c for c in features if c not in score]
    if not score or not structure:
        raise ValueError('Cross-view consensus needs both score and structural features')
    return {'scores': score, 'structure': structure}


def fit(method, fit_frame, features, output_dir, **kwargs):
    'fit one predefined method, returning runtime objects and jsonable evidence'
    if method not in METHODS:
        raise ValueError(f'Unknown campaign verifier: {method}')
    features = list(features)
    _validate_fit(fit_frame, features)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    positive = int(fit_frame['y'].sum())
    report = {'method': method, 'fit_rows': fit_frame.height, 'positive_rows': positive,
              'negative_rows': fit_frame.height-positive, 'features': features,
              'seed': SEED, 'available': False, 'labels_used': 'supplied isolated fit only',
              'calibration_or_check_access': False, 'main_countries': ['india', 'us'],
              'score_interpretation': 'Uncalibrated ranking evidence; deployment requires frozen campaign guards'}
    bundle = {'method': method, 'features': features, 'models': {}, 'report': report,
              'output_dir': str(output)}
    if min(positive, fit_frame.height-positive) < 20:
        report['reason'] = 'Fewer than 20 fitting examples in at least one class'
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        return bundle
    test_params = dict(kwargs.get('_model_params') or {})
    rounds = int(kwargs.get('_rounds', 220))
    if rounds <= 0:
        raise ValueError('Training rounds must be positive')
    if method == 'honest_leaf':

        structure_mask = (_hash_targets(fit_frame['tid'].to_numpy()) % 5) < 3
        structure = fit_frame.filter(pl.Series(structure_mask))
        evidence = fit_frame.filter(pl.Series(~structure_mask))
        if (not structure.height or not evidence.height or
                structure['y'].n_unique() != 2):
            report['reason'] = 'Insufficient distinct targets/classes for honest tree split'
        else:
            params = {**test_params, 'learning_rate': 1., 'num_leaves': 127,
                      'max_depth': 9, 'lambda_l2': 2.}
            model = _fit_model(structure, features, output/'honest_structure.txt', 1, params=params)
            leaf = _predict_model(model, evidence, features, leaves=True)
            size = model.dump_model()['tree_info'][0]['num_leaves']
            total = np.bincount(leaf, minlength=size)
            success = np.bincount(leaf, weights=evidence['y'].to_numpy(), minlength=size)
            lower = wilson_lower(success, total)
            bundle['models']['tree'] = model
            bundle['leaf_scores'] = lower
            report.update(available=True, structure_rows=structure.height,
                evidence_rows=evidence.height, structure_targets=structure['tid'].n_unique(),
                evidence_targets=evidence['tid'].n_unique(), target_overlap=0,
                tree_rounds=1, leaf_count=size, score='One-sided 95% Wilson bound on independent leaf evidence',
                leaf_evidence=[{'leaf': i, 'positive': int(success[i]), 'total': int(total[i]),
                    'lower': float(lower[i])} for i in range(size)],
                split='Fixed SplitMix64 target hash modulo5: 0..2 structure, 3..4 evidence')
            np.save(output/'leaf_scores.npy', lower)
    elif method == 'crossview_consensus':
        views = _views(features)
        for name, view in views.items():
            bundle['models'][name] = _fit_model(fit_frame, view, output/(name+'_model.txt'),
                                               rounds, params=test_params)
        bundle['views'] = views
        report.update(available=True, views=views, rounds_per_view=rounds,
            score='Minimum of disjoint score-only and structural-only verifier outputs',
            interpretation='Feature views are disjoint; statistical independence is not assumed')
    else:


        pilot_rounds = min(100, rounds)
        pilot = _fit_model(fit_frame, features, output/'mining_pilot.txt', pilot_rounds,
                           params=test_params)
        pilot_p = _predict_model(pilot, fit_frame, features)
        y = fit_frame['y'].to_numpy()
        hard = (y == 0) & (pilot_p >= .01)
        easy_sample = (_hash_targets(fit_frame['tid'].to_numpy()) % 20) == 0
        retained = (y == 1) | hard | easy_sample
        sel = fit_frame.filter(pl.Series(retained))
        weights = np.where(y[retained] == 1, 1., np.where(hard[retained], 3., 20.)).astype(np.float32)
        model = _fit_model(sel, features, output/'hard_negative_model.txt', rounds,
                           params=test_params, weights=weights)
        bundle['models']['verifier'] = model
        report.update(available=True, pilot_rounds=pilot_rounds, verifier_rounds=rounds,
            mined_rows=sel.height, hard_negative_rows=int(hard.sum()),
            sampled_easy_negative_rows=int(((y == 0) & ~hard & easy_sample).sum()),
            easy_negative_sampling='Fixed whole-target hash modulo20=0 with inverse weight20',
            hard_negative_definition='Fit-negative pilot score >= .01; weight3',
            positive_weight=1., score='Binary verifier after one fixed hard-negative mining pass',
            mean_fit_weight=float(weights.mean()))
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return bundle


def predict(bundle, frame, **kwargs):
    'return input-order alias evidence without reading labels or retuning'
    if not bundle['report']['available']:
        return np.zeros(frame.height, dtype=np.float32)
    method = bundle['method']
    if method == 'honest_leaf':
        leaf = _predict_model(bundle['models']['tree'], frame, bundle['features'], leaves=True)
        probs = bundle['leaf_scores'][leaf]
    elif method == 'crossview_consensus':
        probs = np.ones(frame.height, dtype=np.float32)
        for name, ff in bundle['views'].items():
            probs = np.minimum(probs, _predict_model(bundle['models'][name], frame, ff))
    elif method == 'hard_negative':
        probs = _predict_model(bundle['models']['verifier'], frame, bundle['features'])
    else:
        raise ValueError(f'Unknown campaign verifier: {method}')
    return np.asarray(probs, dtype=np.float32)

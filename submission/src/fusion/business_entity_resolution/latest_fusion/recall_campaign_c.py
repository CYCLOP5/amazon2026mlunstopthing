'monotone evidence and cross-country consensus verifiers, with fixed budgets'
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from .pipeline import log

METHODS = ('monotone_verifier', 'country_consensus')
PARAMS = dict(objective='binary', metric='binary_logloss', learning_rate=.05,
              num_leaves=31, max_depth=6, min_data_in_leaf=75, lambda_l2=25.,
              verbosity=-1, deterministic=True, force_col_wise=True, seed=27109)
ROUNDS = 240
INCREASING = {'baseline_p', 'newest_logit', 'gate_logit', 'neural_logit',
              'graph_logit', 'hybrid_logit', 'friend_logit', 'name_core_exact',
              'name_jaccard', 'name_ref_coverage', 'name_target_coverage',
              'house_equal', 'newest_owner_margin', 'gate_owner_margin'}
DECREASING = {'house_conflict'}
FORBIDDEN = {'qid', 'tid', 'rid', 'y', 'own', 'co', 'fold', 'deg', 'seg'}


def _matrix(frame, features):
    if not features or FORBIDDEN.intersection(features):
        raise ValueError('Verifier features must be nonempty and label-free')
    x = frame.select(features).cast(pl.Float32).to_numpy()
    if np.isinf(x).any():
        raise ValueError('Infinite verifier features')
    return x


def fit(method, fit_frame, features, output_dir, **kwargs):
    if method not in METHODS:
        raise ValueError('Unknown C verifier: '+str(method))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = fit_frame.filter(pl.col('co').is_in(['us', 'india']))
    names = list(features)
    _matrix(frame.head(1), names)
    report = {'method': method, 'features': names, 'fit_rows': frame.height,
              'rounds': ROUNDS, 'parameters': PARAMS, 'heads': {}, 'available': True,
              'selection': 'Fixed model recipe; no parameter search',
              'inference_uses_labels': False}
    models = {}
    groups = [('monotone', frame)] if method == 'monotone_verifier' else [
        (co, frame.filter(pl.col('co') == co)) for co in ('us', 'india')]
    for name, rows in groups:
        positives = int(rows['y'].sum() or 0)
        report['heads'][name] = {'rows': rows.height, 'positives': positives}
        if min(positives, rows.height-positives) < 25:
            report.update(available=False, reason='Insufficient class support for '+name)
            models = {}
            break
        params = {**PARAMS, 'num_threads': int(os.environ.get('ER_THREADS', '48'))}
        if method == 'monotone_verifier':
            constraints = [1 if c in INCREASING else -1 if c in DECREASING else 0 for c in names]
            params.update(monotone_constraints=constraints, monotone_constraints_method='advanced')
            report['monotone_constraints'] = dict(zip(names, constraints))
        log(f'recall {method}: fitting {name} on {rows.height:,} rows, {ROUNDS} rounds')
        model = lgb.train(params, lgb.Dataset(_matrix(rows, names), label=rows['y'].to_numpy(),
                          feature_name=names), num_boost_round=ROUNDS)
        model.save_model(str(output/(name+'_model.txt')))
        log(f'recall {method}: saved {name} model')
        models[name] = model
    report['interpretation'] = ('Monotone evidence constrains stronger individual evidence from lowering alias score.'
        if method == 'monotone_verifier' else
        'Minimum of independent US-only and India-only expert scores tests cross-country transfer agreement.')
    (output/'model_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return {'method': method, 'models': models, 'features': names, 'report': report}


def predict(bundle, frame, **kwargs):
    if not bundle['report']['available']:
        return np.zeros(frame.height, dtype=np.float32)
    res = np.ones(frame.height, dtype=np.float32)
    for model in bundle['models'].values():
        if model.feature_name() != bundle['features']:
            raise ValueError('Stored verifier feature order changed')
        for start in range(0, frame.height, 250_000):
            chunk = frame.slice(start, 250_000)
            prob = np.asarray(model.predict(_matrix(chunk, bundle['features']),
                num_threads=int(os.environ.get('ER_THREADS', '48'))), dtype=np.float32)
            res[start:start+chunk.height] = np.minimum(res[start:start+chunk.height], prob)
            if start % 1_000_000 == 0 and frame.height > 250_000:
                log(f'recall {bundle["method"]}: scored {start+chunk.height:,}/{frame.height:,} rows')
    return res

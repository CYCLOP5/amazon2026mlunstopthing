'full-pool rerankers and an owner-cross-fitted, contextual booster mixture'
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit, softmax

from .pipeline import log

BACKENDS = ('lightgbm', 'xgboost', 'catboost')
FORBIDDEN = {'qid', 'tid', 'rid', 'eid', 'y', 'own', 'co', 'fold', 'deg', 'seg'}
CONTEXT = ['target_address_empty', 'house_conflict', 'name_jaccard',
           'ref_name_twins_log', 'newest_owner_margin', 'model_spread']
ROUNDS = {'lightgbm': 320, 'xgboost': 280, 'catboost': 240}


def matrix(frame, features):
    if not features or len(features) != len(set(features)) or FORBIDDEN.intersection(features):
        raise ValueError('Model features must be unique and exclude labels, IDs and countries')
    x = frame.select(features).cast(pl.Float32).to_numpy()
    if np.isinf(x).any():
        raise ValueError('Infinite model features')
    return x


def owner_folds(frame, count=3):
    'keep all candidates of owned targets and their true-owner group together'
    owned, tids = frame['own'].to_numpy(), frame['tid'].to_numpy()
    keys = np.where(owned >= 0, owned, tids.astype(np.int64)+(1 << 32)).astype(np.uint64)
    with np.errstate(over='ignore'):
        keys = (keys ^ np.uint64(27943)) + np.uint64(0x9E3779B97F4A7C15)
        keys = (keys ^ (keys >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        keys = (keys ^ (keys >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return ((keys ^ (keys >> np.uint64(31))) % count).astype(np.int8)


def _weights(frame):
    n = frame.select(pl.len().over('tid').alias('n'))['n'].to_numpy()
    values = 1./np.sqrt(n)
    return (values/values.mean()).astype(np.float32)


def train_backend(backend, frame, features, path, rounds=None):
    x, y, weight = matrix(frame, features), frame['y'].to_numpy(), _weights(frame)
    threads = int(os.environ.get('ER_THREADS', '48'))
    rounds = rounds or ROUNDS[backend]
    log(f'innovation: fit {backend} {frame.height:,} rows x {len(features)} features, {rounds} rounds')
    if backend == 'lightgbm':
        model = lgb.train(dict(objective='binary', learning_rate=.04, num_leaves=63,
            max_depth=8, min_data_in_leaf=80, lambda_l2=25., verbosity=-1,
            feature_fraction=.9, deterministic=True, force_col_wise=True,
            num_threads=threads, seed=27943),
            lgb.Dataset(x, label=y, weight=weight, feature_name=features), num_boost_round=rounds)
        model.save_model(str(path))
    elif backend == 'xgboost':
        import xgboost as xgb
        model = xgb.train(dict(objective='binary:logistic', tree_method='hist', device='cpu',
            max_depth=6, eta=.045, reg_lambda=25., min_child_weight=30., max_bin=128,
            colsample_bytree=.85, seed=27943, nthread=threads, base_score=.5),
            xgb.DMatrix(x, label=y, weight=weight, feature_names=features), num_boost_round=rounds)
        model.save_model(str(path))
    elif backend == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=rounds, learning_rate=.05, depth=6,
            l2_leaf_reg=25., border_count=128, loss_function='Logloss',
            random_seed=27943, thread_count=threads, task_type='CPU',
            allow_writing_files=False, verbose=False)
        model.fit(x, y, sample_weight=weight)
        model.save_model(str(path))
    else:
        raise ValueError('Unknown backend: '+backend)
    return model


def predict_raw(backend, model, frame, features, chunk=250_000):
    res = np.empty(frame.height, dtype=np.float32)
    for start in range(0, frame.height, chunk):
        part = frame.slice(start, chunk)
        x = matrix(part, features)
        if backend == 'lightgbm':
            raw = model.predict(x, raw_score=True, num_threads=int(os.environ.get('ER_THREADS', '48')))
        elif backend == 'xgboost':
            import xgboost as xgb
            raw = model.predict(xgb.DMatrix(x, feature_names=features), output_margin=True)
        else:
            raw = model.predict(x, prediction_type='RawFormulaVal')
        res[start:start+part.height] = np.asarray(raw).reshape(-1)
        if frame.height > chunk and start % 2_000_000 == 0:
            log(f'innovation {backend}: scored {start+part.height:,}/{frame.height:,}')
    if not np.isfinite(res).all():
        raise ValueError('Nonfinite expert scores')
    return res


def context_matrix(frame, logits):
    'small observable context with fixed scales; no fitted split information'
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[0] != frame.height or not np.isfinite(logits).all():
        raise ValueError('Gate context requires complete finite expert scores')
    columns = [np.ones(frame.height)]
    for name in CONTEXT:
        value = frame[name].fill_null(0.).to_numpy() if name in frame else np.zeros(frame.height)
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite gate context feature: '+name)
        divisor = 5. if name == 'ref_name_twins_log' else 10. if name == 'newest_owner_margin' else 1.
        columns.append(np.clip(value/divisor, -1., 1.))
    columns.append(np.clip(np.std(logits, axis=1)/5., 0., 1.))
    return np.column_stack(columns).astype(np.float64)


def gate_loss_gradient(theta, context, logits, labels, weights, penalty=.01):
    params = np.asarray(theta).reshape(context.shape[1], logits.shape[1])
    alpha = softmax(context @ params, axis=1)
    raw = np.sum(alpha*logits, axis=1)
    residual = (expit(raw)-labels)*weights/weights.sum()
    loss = np.sum(weights*(np.logaddexp(0., raw)-labels*raw))/weights.sum()
    loss += penalty*np.square(params).sum()/2
    gradient = context.T @ (residual[:, None]*alpha*(logits-raw[:, None])) + penalty*params
    return float(loss), gradient.ravel()


def fit_gate(frame, logits, labels, weights=None):
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if logits.shape != (frame.height, len(BACKENDS)) or labels.shape != (frame.height,) or \
            not frame.height or not np.isfinite(logits).all() or not np.isin(labels, [0., 1.]).all():
        raise ValueError('Gate requires complete finite OOF expert scores and binary fit labels')
    weights = np.ones(frame.height) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (frame.height,) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Gate weights must be finite, positive and aligned with fitting rows')
    logits = np.clip(logits, -12., 12.)
    context = context_matrix(frame, logits)
    zero = np.zeros(context.shape[1]*logits.shape[1])
    res = minimize(gate_loss_gradient, zero, args=(context, logits, labels, weights),
                      jac=True, method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-10})
    initial = gate_loss_gradient(zero, context, logits, labels, weights)[0]
    acceptable = np.isfinite(res.fun) and res.fun <= initial and np.isfinite(res.x).all()
    theta = (res.x if acceptable else zero).reshape(context.shape[1], len(BACKENDS))
    report = {'optimizer_success': bool(res.success), 'message': str(res.message),
        'iterations': int(res.nit), 'equal_mixture_loss': initial,
        'gated_regularized_loss': float(res.fun) if acceptable else initial,
        'used_learned_gate': bool(acceptable), 'theta': theta.tolist(),
        'context': ['intercept']+CONTEXT+['expert_logit_std'],
        'fit': 'Only out-of-fold predictions on isolated fitting labels', 'l2_penalty': .01}
    return theta, report


def masked_training_rows(frame, features):
    'teach retrieval reranker to use raw evidence when upstream scores are absent'
    sel = frame.filter((pl.col('tid').hash(seed=27943) % 5) == 0)
    expressions = []
    for name in features:
        if name.endswith('_logit') or name == 'baseline_raw':

            expressions.append(pl.lit(1e-4 if name == 'baseline_raw' else -9.21024, pl.Float32).alias(name))
        elif name.endswith('_present') or name in ('model_spread', 'old_models_mean',
                'new_old_gap', 'gate_neural_gap', 'models_above_90', 'member_spread'):
            expressions.append(pl.lit(0., pl.Float32).alias(name))
        elif name == 'baseline_p':
            expressions.append(pl.lit(1e-4, pl.Float32).alias(name))
        elif name.endswith('_behind_best'):
            maximum = name.removesuffix('_behind_best')+'_target_max'
            if maximum in sel:
                expressions.append((-pl.col(maximum)).alias(name))
    return pl.concat([frame, sel.with_columns(expressions)], how='vertical_relaxed')


def fit(mode, frame, features, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features = list(dict.fromkeys(features))
    matrix(frame.head(1), features)
    if frame['y'].n_unique() != 2:
        raise ValueError('Innovation fitting requires both match and nonmatch examples')
    report = {'mode': mode, 'features': features, 'fit_rows': frame.height,
              'rounds': ROUNDS, 'device': 'cpu', 'full_candidate_reranking': True}
    bundle = {'mode': mode, 'features': features, 'models': {}, 'report': report}
    if mode != 'gated_ensemble':
        tr = masked_training_rows(frame, features) if mode == 'reciprocal_retrieval' else frame
        bundle['models']['lightgbm'] = [train_backend('lightgbm', tr, features, output/'reranker.txt')]
        report['augmented_training_rows'] = tr.height
    else:
        folds = owner_folds(frame)
        oof = np.full((frame.height, len(BACKENDS)), np.nan, dtype=np.float32)
        cv = []
        for k in range(3):
            train, valid = frame.filter(pl.Series(folds != k)), frame.filter(pl.Series(folds == k))
            if train['y'].n_unique() != 2 or not valid.height:
                raise ValueError('Insufficient fit-only support for owner-group cross-fitting')
            if train.select('tid').unique().join(valid.select('tid').unique(), on='tid').height:
                raise ValueError('Cross-fitting target overlap')
            train_owners = train.filter(pl.col('own') >= 0).select('own').unique()
            valid_owners = valid.filter(pl.col('own') >= 0).select('own').unique()
            if train_owners.join(valid_owners, on='own').height:
                raise ValueError('Cross-fitting true-owner overlap')
            cv.append({'fold': k, 'train_rows': train.height, 'valid_rows': valid.height,
                       'target_overlap': 0, 'true_owner_overlap': 0})
            for i, backend in enumerate(BACKENDS):
                suffix = {'lightgbm': 'txt', 'xgboost': 'ubj', 'catboost': 'cbm'}[backend]
                model = train_backend(backend, train, features, output/f'{backend}_fold{k}.{suffix}')
                bundle['models'].setdefault(backend, []).append(model)
                oof[folds == k, i] = predict_raw(backend, model, valid, features)
        theta, gate_report = fit_gate(frame, oof, frame['y'].to_numpy(), _weights(frame))
        bundle['gate'] = theta
        report.update(cross_fitting=cv, gate=gate_report,
            expert_families=list(BACKENDS), gate_training_rows=frame.height)
        frame.select('qid', 'tid', 'y').with_columns(pl.Series('cv_fold', folds),
            *[pl.Series(name+'_oof_logit', oof[:, i]) for i, name in enumerate(BACKENDS)]).write_parquet(output/'oof_predictions.parquet')
    (output/'model_report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return bundle


def predict(bundle, frame):
    if not frame.height:
        return np.empty(0, dtype=np.float32)
    columns = []
    for backend, models in bundle['models'].items():
        raw = np.zeros(frame.height, dtype=np.float64)
        for model in models:
            raw += predict_raw(backend, model, frame, bundle['features'])/len(models)
        columns.append(np.clip(raw, -12., 12.))
    logits = np.column_stack(columns)
    if bundle['mode'] == 'gated_ensemble':
        context = context_matrix(frame, logits)
        alpha = softmax(context @ bundle['gate'], axis=1)
        raw = np.sum(alpha*logits, axis=1)
    else:
        raw = logits[:, 0]
    res = expit(raw).astype(np.float32)
    if not np.isfinite(res).all():
        raise ValueError('Nonfinite model innovation probabilities')
    return res

'weighted numeric boosters with explicit absolute-logit prediction contracts'
import json
from pathlib import Path

import numpy as np
import polars as pl

from er.safe import THREADS, guard
from final_hybrid.fit import weighted_partitions
from innovation.common import log

BACKENDS = ('xgboost', 'catboost')
MODES = ('independent', 'residual')
CHUNK_ROWS = 250_000
_FORBIDDEN = {'qid', 'tid', 'rid', 'y', 'own', 'fold', 'co', 'deg'}


def _validate(backend, mode, device, columns=None):
    if backend not in BACKENDS:
        raise ValueError(f'Unknown booster backend: {backend}')
    if mode not in MODES:
        raise ValueError(f'Unknown booster mode: {mode}')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if columns is not None:
        if not columns or len(set(columns)) != len(columns):
            raise ValueError('Expected nonempty, unique feature columns')
        if _FORBIDDEN.intersection(columns):
            raise ValueError('IDs, labels, countries and split metadata cannot be model features')
        if mode == 'independent' and any(
                c in ('base_lg', 'has_parent', 'from_prepared') or c.startswith('parent')
                for c in columns):
            raise ValueError('Independent head cannot include parent features')


def clean_X(values):
    'return a contiguous float32 matrix; represent infinity as missing values'
    out = np.asarray(values, dtype=np.float32, order='C')
    if out.ndim != 2:
        raise ValueError('Feature matrix must have two dimensions')
    if np.isinf(out).any():
        out = out.copy()
        out[np.isinf(out)] = np.nan
    return out


def matrix(frame, columns):
    return clean_X(frame.select(columns).cast(pl.Float32).to_numpy())


def _offset(frame, mode):
    if mode == 'independent':
        return np.zeros(frame.height, dtype=np.float32)
    values = frame['base_lg'].to_numpy().astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Residual baseline logits must be finite')
    return values


def _indices(values, n, name):
    values = np.asarray(values)
    if values.dtype == bool:
        if values.shape != (n,):
            raise ValueError(f'{name} mask has wrong shape')
        return np.flatnonzero(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f'{name} must be a boolean mask or integer indices')
    if ((values < 0) | (values >= n)).any() or np.unique(values).size != values.size:
        raise ValueError(f'Invalid {name} indices')
    return values


def _params(backend, device, rounds):
    if backend == 'xgboost':
        return {'objective': 'binary:logistic', 'eval_metric': 'logloss',
                'tree_method': 'hist', 'device': 'cuda:0' if device == 'cuda' else 'cpu',
                'max_depth': 5, 'eta': .04, 'reg_lambda': 20.,
                'min_child_weight': 20., 'max_bin': 256, 'colsample_bytree': .9,
                'seed': 942, 'base_score': .5, 'nthread': THREADS}
    params = {'loss_function': 'Logloss', 'eval_metric': 'Logloss',
              'iterations': int(rounds), 'learning_rate': .04, 'depth': 5,
              'l2_leaf_reg': 20., 'border_count': 256, 'random_seed': 942,
              'boost_from_average': False, 'thread_count': THREADS,
              'allow_writing_files': False, 'verbose': False,
              'task_type': 'GPU' if device == 'cuda' else 'CPU'}
    if device == 'cuda':

        params['devices'] = '0'
    else:

        params['rsm'] = .9
    return params


def fit_one_head(X, y, weights, offset, tr, va, backend='xgboost',
                 mode='independent', device='cpu', rounds=700, path=None,
                 columns=None):
    'fit one split and return (model, metadata), without scoring other rows'
    X = clean_X(X)
    columns = list(columns) if columns is not None else [f'f{i}' for i in range(X.shape[1])]
    _validate(backend, mode, device, columns)
    if len(columns) != X.shape[1] or int(rounds) < 1:
        raise ValueError('Feature count mismatch or nonpositive rounds')
    n = len(X)
    tr, va = _indices(tr, n, 'training'), _indices(va, n, 'validation')
    if not len(tr) or not len(va) or np.intersect1d(tr, va).size:
        raise ValueError('Training and validation must be nonempty and disjoint')
    y, weights = np.asarray(y, dtype=np.float32), np.asarray(weights, dtype=np.float32)
    if y.shape != (n,) or weights.shape != (n,):
        raise ValueError('Labels and weights must match feature rows')
    supervised = np.concatenate((tr, va))
    if not np.isin(y[supervised], (0., 1.)).all():
        raise ValueError('Expected binary training/validation labels')
    if not np.isfinite(weights[supervised]).all() or (weights[supervised] <= 0).any():
        raise ValueError('Training/validation weights must be finite and positive')
    if np.unique(y[tr]).size != 2 or np.unique(y[va]).size != 2:
        raise ValueError('Training and validation each require both classes')
    if mode == 'residual':
        offset = np.asarray(offset, dtype=np.float32)
        if offset.shape != (n,) or not np.isfinite(offset[supervised]).all():
            raise ValueError('Residual baseline logits must be finite and match rows')
    params = _params(backend, device, rounds)
    if backend == 'xgboost':
        import xgboost as xgb
        def dataset(ix):
            return xgb.DMatrix(X[ix], label=y[ix], weight=weights[ix],
                base_margin=offset[ix] if mode == 'residual' else None,
                feature_names=columns, nthread=THREADS)
        train, valid = dataset(tr), dataset(va)
        model = xgb.train(params, train, num_boost_round=int(rounds),
            evals=[(valid, 'valid')], early_stopping_rounds=70, verbose_eval=False)
        iteration, score = int(model.best_iteration), float(model.best_score)
        version = xgb.__version__
    else:
        import catboost as cb
        def dataset(ix):
            return cb.Pool(X[ix], label=y[ix], weight=weights[ix],
                baseline=offset[ix] if mode == 'residual' else None,
                feature_names=columns, thread_count=THREADS)
        train, valid = dataset(tr), dataset(va)
        model = cb.CatBoostClassifier(**params)
        model.fit(train, eval_set=valid, early_stopping_rounds=70, use_best_model=True)
        iteration = int(model.get_best_iteration())
        score = float(model.get_best_score()['validation']['Logloss'])
        version = cb.__version__
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(path))
    info = {'backend': backend, 'version': version, 'mode': mode, 'device': device,
            'iteration': iteration, 'trees_used': iteration + 1,
            'weighted_logloss': score, 'train_rows': int(len(tr)),
            'valid_rows': int(len(va)), 'parameters': params,
            'model_file': str(path) if path is not None else None}
    return model, info


def _backend(model):
    module = type(model).__module__.split('.')[0]
    if module not in BACKENDS:
        raise ValueError(f'Unsupported booster type: {type(model)}')
    return module


def load_models(paths, device='cpu'):
    'Reload native JSON/CBM models; JSON retains the best-iteration attribute'
    models = []
    for path in paths:
        path = Path(path)
        if path.suffix == '.json':
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(path))
            model.set_param({'device': 'cuda:0' if device == 'cuda' else 'cpu', 'nthread': THREADS})
        elif path.suffix == '.cbm':
            import catboost as cb
            model = cb.CatBoostClassifier()
            model.load_model(str(path))
        else:
            raise ValueError('Expected native .json or .cbm model file')
        _validate(_backend(model), 'independent', device)
        models.append(model)
    return models


def _predict_matrix(model, X, offset, columns, mode, device):
    backend = _backend(model)
    if backend == 'xgboost':
        import xgboost as xgb
        data = xgb.DMatrix(X, feature_names=columns, nthread=THREADS,
                          base_margin=offset if mode == 'residual' else None)
        best = model.attr('best_iteration')
        iterations = (0, int(best) + 1) if best is not None else (0, 0)

        values = model.predict(data, output_margin=True, iteration_range=iterations)
    else:

        values = model.predict(X, prediction_type='RawFormulaVal',
            thread_count=THREADS, task_type='GPU' if device == 'cuda' else 'CPU')
        if mode == 'residual':
            values = values + offset
    return np.asarray(values, dtype=np.float32)


def predict_raw(models, frame, columns, mode='independent', device='cpu',
                chunk_rows=CHUNK_ROWS):
    'predict absolute logits, averaging cv models and adding offsets once'
    models = list(models)
    if not models or int(chunk_rows) < 1:
        raise ValueError('Prediction requires models and a positive chunk size')
    if isinstance(models[0], (str, Path)):
        models = load_models(models, device)
    columns = list(columns)
    backend = _backend(models[0])
    _validate(backend, mode, device, columns)
    if any(_backend(m) != backend for m in models):
        raise ValueError('Cannot ensemble different booster backends')
    for model in models:
        if backend == 'xgboost':
            model.set_param({'device': 'cuda:0' if device == 'cuda' else 'cpu', 'nthread': THREADS})
    out = np.empty(frame.height, dtype=np.float32)
    for start in range(0, frame.height, int(chunk_rows)):
        chunk = frame.slice(start, int(chunk_rows))
        X, offset = matrix(chunk, columns), _offset(chunk, mode)
        total = np.zeros(chunk.height, dtype=np.float64)
        for model in models:
            total += _predict_matrix(model, X, offset, columns, mode, device)
        values = total / len(models)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite booster prediction')
        stop = start+chunk.height
        out[start:stop] = values
        if start == 0 or stop == frame.height or stop//1_000_000 > start//1_000_000:
            log(f'{backend}_{mode}: predicted {stop:,}/{frame.height:,} rows '
                f'with {len(models)} model(s), device={device}')
        guard('boosted hybrid prediction')
    return out


def fit_head(frame, refs, output, backend='xgboost', mode='independent',
             device='cpu', rounds=700):
    'Fit three owner-group CV models and return absolute logits/models/info'
    from boosted_hybrid.features import model_columns
    columns = list(model_columns(frame, mode))
    _validate(backend, mode, device, columns)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fit, groups, weights, proportions = weighted_partitions(frame, refs, cv=3)
    fit_ids = np.flatnonzero(fit)
    fitted_frame = frame.filter(pl.Series(fit))
    X = matrix(fitted_frame, columns)
    y = fitted_frame['y'].to_numpy().astype(np.float32)
    offset = _offset(fitted_frame, mode)
    weights, groups = weights[fit], groups[fit]
    if not len(y) or np.unique(y).size != 2:
        raise ValueError('Booster fit partition requires both classes')
    models, history, oof = [], [], np.empty(len(y), dtype=np.float32)
    stem = f'{backend}_{mode}'
    extension = 'json' if backend == 'xgboost' else 'cbm'
    log(f'{stem}: fit rows={len(y):,}, features={len(columns)}, device={device}')
    for fold in range(3):
        tr, va = groups != fold, groups == fold
        model, info = fit_one_head(X, y, weights, offset, tr, va,
            backend=backend, mode=mode, device=device, rounds=rounds,
            path=output/f'{stem}_{fold}.{extension}', columns=columns)
        oof[va] = _predict_matrix(model, X[va], offset[va], columns, mode, device)
        info['fold'] = fold
        models.append(model)
        history.append(info)
        log(f'{stem}: owner CV {fold+1}/3, iteration={info["iteration"]}, '
            f'weighted_logloss={info["weighted_logloss"]:.8f}, '
            f'train={info["train_rows"]:,}, valid={info["valid_rows"]:,}')
        guard('boosted hybrid fitting')
    del fitted_frame, X
    raw = predict_raw(models, frame, columns, mode, device)
    raw[fit_ids] = oof
    info = {'backend': backend, 'mode': mode, 'device': device, 'columns': columns,
            'cv': history, 'fit_rows': int(len(y)),
            'fit_positive_rate': float(y.mean()),
            'weighted_fit_positive_rate': float(np.average(y, weights=weights)),
            'prediction': 'absolute logit; fit rows are true-owner OOF, other rows CV ensemble',
            'baseline': 'base_lg training offset added exactly once' if mode == 'residual'
                        else 'zero initial logit; intercept learned from supervised fit only',
            'chunk_rows': CHUNK_ROWS,
            'sampling': {'country_candidate_fit_fractions': proportions,
                'decoy_weight': 12.5, 'positive_weight': 1.,
                'owned_negative_weight': 'inverse country fit fraction',
                'holdouts': 'candidate and true owner; three-fold true-owner CV'},
            'model_files': [entry['model_file'] for entry in history]}
    (output/f'{stem}_info.json').write_text(json.dumps(info, indent=2)+'\n')
    return raw, models, info

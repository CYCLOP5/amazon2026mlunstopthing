'Bounded unlabeled source/test density-ratio weights for candidate fitting'
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

FEATURES = ['target_address_empty', 'ref_address_empty', 'house_equal', 'house_conflict',
    'name_core_exact', 'name_jaccard', 'name_ref_coverage', 'name_target_coverage',
    'ref_name_twins_log', 'target_name_twins_log', 'competitor_count_log', 'newest_present']
COUNTRIES = ('india', 'us')
SEED = 27107
BOUNDS = (.25, 4.)
PARAMS = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
    'num_leaves': 15, 'max_depth': 5, 'min_data_in_leaf': 200, 'learning_rate': .05,
    'lambda_l2': 10., 'deterministic': True, 'force_col_wise': True, 'seed': SEED}


def _schema(schema):
    missing = set(FEATURES + ['co', 'qid', 'tid']) - set(schema)
    if missing:
        raise ValueError('Missing domain covariates/metadata: ' + str(sorted(missing)))
    bad = [c for c in FEATURES if not schema[c].is_numeric()]
    if bad:
        raise ValueError('Domain features must be numeric: ' + str(bad))


def _sample(frame, country, size, domain):
    'lazy top-k limits collected test covariates to the fixed sample cap'
    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    projected = lazy.select('co', 'qid', 'tid', *FEATURES).filter(pl.col('co') == country)
    sampled = projected.with_columns(
        pl.struct('qid', 'tid').hash(seed=SEED+domain).alias('_sample_hash'))
    return sampled.top_k(size, by='_sample_hash').sort('_sample_hash', 'qid', 'tid').collect(engine='streaming')


def _validation_mask(frame, domain):
    return (frame['tid'].hash(seed=SEED+100+domain).to_numpy() % 5) == 0


def _matrix(frame):

    x = frame.select(FEATURES).cast(pl.Float32).to_numpy()
    if np.isinf(x).any():
        raise ValueError('Domain covariates contain infinity')
    return x


def bounded_mean_one(ratios):
    'scale and clip simultaneously so final bounds and mean one both hold'
    ratios = np.asarray(ratios, dtype=float)
    if not ratios.size:
        return ratios.copy()
    if not np.isfinite(ratios).all() or (ratios <= 0).any():
        raise ValueError('Density ratios must be finite and positive')
    ratios = np.clip(ratios, *BOUNDS)
    left, right = 0., BOUNDS[1] / float(ratios.min()) + 1.
    for _ in range(80):
        scale = (left+right)/2
        if np.clip(scale*ratios, *BOUNDS).mean() > 1.:
            right = scale
        else:
            left = scale
    return np.clip(((left+right)/2)*ratios, *BOUNDS)


def _ess(weights):
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum()**2 / np.square(weights).sum()) if weights.size else 0.


def fit_weights(source_frame, test_path, output=None, *, sample_size=100_000,
                _rounds=120, _model_params=None, _minimum_fit=500,
                _minimum_validation=100, _auc_window=(.55, .95)):
    'return source-order float32 weights and jsonable diagnostics'
    if sample_size <= 0 or _rounds <= 0:
        raise ValueError('Sample budget and rounds must be positive')
    _schema(source_frame.schema)
    test_lazy = pl.scan_parquet(test_path)
    _schema(test_lazy.collect_schema())
    if source_frame['co'].null_count() or source_frame.select('qid', 'tid').null_count().sum_horizontal().sum():
        raise ValueError('Domain metadata contains null IDs/countries')
    source_counts = dict(source_frame.group_by('co').len().rows())
    test_counts = dict(test_lazy.select('co').group_by('co').len().collect(engine='streaming').rows())
    weights = np.ones(source_frame.height, dtype=np.float32)
    report = {'features': list(FEATURES), 'domain_label': 'source=0,unlabeled_test=1',
              'truth_labels_read': False, 'feature_matrix_excludes_ids_country_labels': True,
              'sample_size_per_country_domain': sample_size, 'seed': SEED, 'rounds': _rounds,
              'bounds': list(BOUNDS), 'auc_window': list(_auc_window), 'minimum_ess_fraction': .25,
              'validation': 'Whole target groups: domain-specific seeded tid hash modulo5=0',
              'test_access': 'Lazy feature/ID projection with country-filtered bounded top-k collection',
              'countries': {}, 'unknown_country_policy': 'Unit weights'}
    dst = Path(output) if output is not None else None
    if dst is not None:
        dst.mkdir(parents=True, exist_ok=True)
    for country in COUNTRIES:
        nsource, ntest = int(source_counts.get(country, 0)), int(test_counts.get(country, 0))
        size = min(sample_size, nsource, ntest)
        diagnostic = {'source_rows': nsource, 'test_rows': ntest,
                      'sample_rows_per_domain': size, 'used_weights': False,
                      'fallback_reason': None, 'auc': None,
                      'ess': float(nsource), 'ess_fraction': 1.,
                      'weight_mean': 1., 'weight_min': 1., 'weight_max': 1.,
                      'clipped_ratio_fraction': 0., 'final_bound_fraction': 0.}
        report['countries'][country] = diagnostic
        if size < _minimum_fit + _minimum_validation:
            diagnostic['fallback_reason'] = 'insufficient_domain_support'
            continue
        source_sample = _sample(source_frame, country, size, 0)
        test_sample = _sample(test_lazy, country, size, 1)
        source_valid, test_valid = _validation_mask(source_sample, 0), _validation_mask(test_sample, 1)
        fit_counts = [int((~source_valid).sum()), int((~test_valid).sum())]
        valid_counts = [int(source_valid.sum()), int(test_valid.sum())]
        diagnostic.update(fit_rows_by_domain=fit_counts, validation_rows_by_domain=valid_counts,
                          fit_target_groups_by_domain=[source_sample.filter(pl.Series(~source_valid))['tid'].n_unique(),
                                                       test_sample.filter(pl.Series(~test_valid))['tid'].n_unique()],
                          validation_target_groups_by_domain=[source_sample.filter(pl.Series(source_valid))['tid'].n_unique(),
                                                              test_sample.filter(pl.Series(test_valid))['tid'].n_unique()])
        if min(fit_counts) < _minimum_fit or min(valid_counts) < _minimum_validation:
            diagnostic['fallback_reason'] = 'insufficient_grouped_fit_or_validation'
            continue
        xs, xt = _matrix(source_sample), _matrix(test_sample)
        diagnostic['feature_support'] = {}
        for i, feature in enumerate(FEATURES):
            sf, tf = xs[:, i][np.isfinite(xs[:, i])], xt[:, i][np.isfinite(xt[:, i])]
            diagnostic['feature_support'][feature] = {
                'source_range': [float(sf.min()), float(sf.max())] if sf.size else None,
                'test_range': [float(tf.min()), float(tf.max())] if tf.size else None,
                'source_missing_fraction': float(1.-sf.size/len(xs)),
                'test_missing_fraction': float(1.-tf.size/len(xt)),
                'test_outside_source_range_fraction': float(((tf < sf.min()) | (tf > sf.max())).mean())
                    if sf.size and tf.size else None}
        xfit = np.concatenate([xs[~source_valid], xt[~test_valid]])
        yfit = np.r_[np.zeros(fit_counts[0]), np.ones(fit_counts[1])]
        params = {**PARAMS, 'num_threads': int(os.environ.get('ER_THREADS', '48')), **(_model_params or {})}
        model = lgb.train(params, lgb.Dataset(xfit, label=yfit, feature_name=FEATURES), num_boost_round=_rounds)
        if dst is not None:
            path = dst/(country+'_domain_model.txt')
            model.save_model(str(path))
            diagnostic['model_file'] = str(path)
        valid_x = np.concatenate([xs[source_valid], xt[test_valid]])
        valid_y = np.r_[np.zeros(valid_counts[0]), np.ones(valid_counts[1])]
        pvalid = np.asarray(model.predict(valid_x, num_threads=params['num_threads']))
        auc = float(roc_auc_score(valid_y, pvalid))
        diagnostic['auc'] = auc
        diagnostic['prior_odds_correction'] = fit_counts[0]/fit_counts[1]
        if auc <= _auc_window[0] or auc >= _auc_window[1]:
            diagnostic['fallback_reason'] = 'no_useful_shift' if auc <= _auc_window[0] else 'poor_domain_overlap'
            diagnostic.update(ess=float(nsource), ess_fraction=1., weight_mean=1., weight_min=1., weight_max=1.)
            continue
        ids = np.flatnonzero(source_frame['co'].to_numpy() == country)
        probs = np.empty(len(ids), dtype=float)
        for start in range(0, len(ids), 250_000):
            part = source_frame[ids[start:start+250_000]]
            probs[start:start+len(part)] = model.predict(_matrix(part), num_threads=params['num_threads'])
        probs = np.clip(probs, 1e-6, 1.-1e-6)
        ratios = probs/(1.-probs) * diagnostic['prior_odds_correction']
        adjusted = bounded_mean_one(ratios)
        ess = _ess(adjusted)
        diagnostic.update(clipped_ratio_fraction=float(((ratios < BOUNDS[0]) | (ratios > BOUNDS[1])).mean()),
            raw_ratio_min=float(ratios.min()), raw_ratio_max=float(ratios.max()),
            ess=ess, ess_fraction=ess/nsource, weight_mean=float(adjusted.mean()),
            weight_min=float(adjusted.min()), weight_max=float(adjusted.max()),
            final_bound_fraction=float(((adjusted <= BOUNDS[0]+1e-8) | (adjusted >= BOUNDS[1]-1e-8)).mean()))
        if ess/nsource < .25:
            diagnostic['fallback_reason'] = 'insufficient_effective_source_support'
            continue
        weights[ids] = adjusted.astype(np.float32)
        diagnostic['used_weights'] = True
    if not np.isfinite(weights).all() or (weights < BOUNDS[0]-1e-6).any() or (weights > BOUNDS[1]+1e-6).any():
        raise AssertionError('Invalid final domain weights')
    report['weighted_source_rows'] = sum(d['source_rows'] for d in report['countries'].values() if d['used_weights'])
    report['overall_ess'] = _ess(weights)
    return weights, report

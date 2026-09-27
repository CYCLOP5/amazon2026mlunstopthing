'cpu-only booster contracts, using synthetic rows and no challenge data'
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boosted_hybrid.models import clean_X, fit_head, fit_one_head, predict_raw
from final_hybrid.fit import weighted_partitions


def _small_matrix():
    n = 900
    rng = np.random.default_rng(17)
    y = (np.arange(n) % 2).astype(np.float32)
    X = (y + rng.normal(0., .25, n)).astype(np.float32).reshape(-1, 1)
    tr = np.arange(n) < 600
    return X, y, np.where(y == 1, 1., 2.).astype(np.float32), tr, ~tr


@pytest.mark.parametrize('backend', ('xgboost', 'catboost'))
@pytest.mark.parametrize('mode', ('independent', 'residual'))
def test_saved_models_replay_absolute_logits_and_offset_once(tmp_path, backend, mode):
    pytest.importorskip(backend)
    X, y, weights, tr, va = _small_matrix()
    offset = np.full(len(y), -2., dtype=np.float32)
    extension = 'json' if backend == 'xgboost' else 'cbm'
    path = tmp_path/f'model.{extension}'
    model, info = fit_one_head(X, y, weights, offset, tr, va,
        backend, mode, 'cpu', 9, path, ['signal'])
    frame = pl.DataFrame({'signal': X[:, 0], 'base_lg': offset})
    raw = predict_raw([model], frame, ['signal'], mode, chunk_rows=113)
    reloaded = predict_raw([path], frame, ['signal'], mode, chunk_rows=79)
    np.testing.assert_allclose(raw, reloaded, atol=1e-6)
    shifted = frame.with_columns((pl.col('base_lg') + .75).alias('base_lg'))
    raw_shifted = predict_raw([path], shifted, ['signal'], mode)
    np.testing.assert_allclose(raw_shifted - raw,
        np.full(len(raw), .75 if mode == 'residual' else 0.), atol=1e-6)
    assert raw.dtype == np.float32
    assert np.isfinite(raw).all()
    assert raw[y == 1].mean() > raw[y == 0].mean()
    assert info['train_rows'] == 600 and info['valid_rows'] == 300
    assert np.isfinite(info['weighted_logloss'])
    assert 0 <= info['iteration'] < 9


def _owner_fixture():
    refs = pl.DataFrame({'rid': np.arange(300, dtype=np.uint32),
        'fold': [0]*150+[1]*100+[2]*50, 'co': ['us']*300})


    rows = []
    for own in range(300):
        for cand in (own, (own+1) % 300, (own+7) % 300):
            rows.append({'qid': cand, 'tid': 2*own, 'own': own,
                'fold': 0 if cand < 150 else 1 if cand < 250 else 2,
                'co': 'us', 'y': int(cand == own),
                'signal': float(cand == own)+.01*(own % 5),
                'base_lg': -1.2, 'has_parent': 1, 'parent_best': .8})
    frame = pl.DataFrame(rows).with_columns(pl.col('qid', 'tid').cast(pl.UInt32))
    return frame, refs


@pytest.mark.parametrize('backend', ('xgboost', 'catboost'))
def test_fit_uses_both_holdouts_and_true_owner_oof(tmp_path, backend):
    pytest.importorskip(backend)
    frame, refs = _owner_fixture()
    fit, groups, _, _ = weighted_partitions(frame, refs)
    changed = frame.with_columns(pl.when(pl.Series(fit)).then(pl.col('y'))
        .otherwise(1-pl.col('y')).alias('y'))
    raw, models, info = fit_head(frame, refs, tmp_path/'original', backend, rounds=5)
    altered, _, changed_info = fit_head(changed, refs, tmp_path/'mutated', backend, rounds=5)


    np.testing.assert_array_equal(raw, altered)
    assert info['fit_rows'] == int(fit.sum())
    assert changed_info['fit_positive_rate'] == info['fit_positive_rate']
    assert info['columns'] == ['signal']
    np.testing.assert_allclose(raw[~fit], predict_raw(models, frame, ['signal'])[~fit], atol=1e-6)
    for fold, model in enumerate(models):
        valid = fit & (groups == fold)
        np.testing.assert_allclose(raw[valid],
            predict_raw([model], frame.filter(pl.Series(valid)), ['signal']), atol=1e-6)
        assert info['cv'][fold]['valid_rows'] == int(valid.sum())
    assert (tmp_path/'original'/f'{backend}_independent_info.json').is_file()


def test_xgboost_replay_uses_best_iteration_after_early_stop(tmp_path):
    xgb = pytest.importorskip('xgboost')
    X, y, weights, tr, va = _small_matrix()
    y[va] = 1-y[va]
    path = tmp_path/'early.json'
    model, info = fit_one_head(X, y, weights, None, tr, va, rounds=90,
        path=path, columns=['signal'])
    assert model.num_boosted_rounds() > info['trees_used']
    frame = pl.DataFrame({'signal': X[:, 0]})
    raw = predict_raw([path], frame, ['signal'])
    data = xgb.DMatrix(X, feature_names=['signal'])
    exp = model.predict(data, output_margin=True,
        iteration_range=(0, info['trees_used']))
    np.testing.assert_allclose(raw, exp, atol=1e-6)
    assert not np.allclose(raw, model.predict(data, output_margin=True))


def test_float32_matrix_cleans_infinities_and_guards_feature_leakage():
    X = clean_X(np.array([[1., np.inf], [np.nan, -np.inf]], dtype=np.float64))
    assert X.dtype == np.float32 and X.flags.c_contiguous
    assert np.isnan(X[0, 1]) and np.isnan(X[1]).all()
    data, y, weights, tr, va = _small_matrix()
    for column in ('y', 'co', 'qid', 'own', 'fold', 'base_lg', 'parent_best', 'has_parent', 'from_prepared'):
        with pytest.raises(ValueError, match='features'):
            fit_one_head(data, y, weights, None, tr, va, columns=[column])
    with pytest.raises(ValueError, match='disjoint'):
        fit_one_head(data, y, weights, None, tr, tr, columns=['signal'])

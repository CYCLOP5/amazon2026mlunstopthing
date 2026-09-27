'gate numerics, true-owner cross-fitting, and all three real cpu backends'
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest
from scipy.special import expit, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from latest_fusion import innovation_models as models


def fitting_frame(owners=150):
    rng = np.random.default_rng(27943)
    tids = np.repeat(np.arange(owners * 2), 2)
    own = tids // 2
    decoy = tids % 13 == 0
    own = np.where(decoy, -1, own)
    y = ((np.arange(len(tids)) % 2 == 0) & ~decoy).astype(np.uint8)
    return pl.DataFrame({
        'qid': np.where(np.arange(len(tids)) % 2 == 0, tids // 2, tids // 2 + owners),
        'tid': tids, 'own': own, 'y': y,
        'co': np.where(tids % 2 == 0, 'us', 'india'),
        'signal': y + rng.normal(0., .07, len(tids)),
        'name_jaccard': np.clip(y + rng.normal(0., .1, len(tids)), 0., 1.),
        'target_address_empty': (tids % 3 == 0).astype(np.float32),
    })


def test_gate_gradient_matches_independent_finite_difference():
    rng = np.random.default_rng(934)
    context = rng.normal(size=(19, 4))
    logits = rng.normal(size=(19, 3)) * 4
    labels = rng.integers(0, 2, size=19)
    weights = rng.uniform(.2, 3., size=19)
    theta = rng.normal(size=12) * .3
    loss, analytic = models.gate_loss_gradient(theta, context, logits, labels, weights)
    finite_difference = np.empty_like(theta)
    for index in range(theta.size):
        offset = np.zeros_like(theta)
        offset[index] = 1e-6
        hi = models.gate_loss_gradient(theta + offset, context, logits, labels, weights)[0]
        lo = models.gate_loss_gradient(theta - offset, context, logits, labels, weights)[0]
        finite_difference[index] = (hi - lo) / 2e-6
    assert np.isfinite(loss)
    np.testing.assert_allclose(analytic, finite_difference, atol=3e-8, rtol=3e-6)


def test_gate_learns_contextual_expert_switch_without_outcome_features():
    n = 400
    context = np.repeat([0., 1.], n // 2)
    labels = (np.arange(n) % 2).astype(float)
    correct = (2 * labels - 1) * 5
    first = np.where(context == 0, correct, -correct)
    logits = np.column_stack([first, -first, np.zeros(n)])
    frame = pl.DataFrame({'target_address_empty': context})
    theta, report = models.fit_gate(frame, logits, labels)
    alpha = softmax(models.context_matrix(frame, logits) @ theta, axis=1)
    probs = expit((alpha * logits).sum(axis=1))
    assert report['used_learned_gate']
    assert report['gated_regularized_loss'] < report['equal_mixture_loss'] - .3
    assert np.mean((probs >= .5) == labels) == 1.
    assert alpha[:n // 2, 0].mean() > alpha[:n // 2, 1].mean()
    assert alpha[n // 2:, 1].mean() > alpha[n // 2:, 0].mean()


def test_owner_folds_keep_siblings_candidates_and_decoys_together():
    frame = fitting_frame()
    folds = models.owner_folds(frame)
    assigned = frame.with_columns(pl.Series('assigned_fold', folds))
    assert assigned.group_by('tid').agg(pl.col('assigned_fold').n_unique())['assigned_fold'].max() == 1
    owned = assigned.filter(pl.col('own') >= 0)
    assert owned.group_by('own').agg(pl.col('assigned_fold').n_unique())['assigned_fold'].max() == 1
    assert set(folds) == {0, 1, 2}
    reverse = frame.reverse()
    np.testing.assert_array_equal(models.owner_folds(reverse)[::-1], folds)


def test_real_cross_fitted_backends_never_predict_training_targets(tmp_path, monkeypatch):
    monkeypatch.setenv('ER_THREADS', '1')
    monkeypatch.setattr(models, 'ROUNDS', {backend: 4 for backend in models.BACKENDS})
    frame = fitting_frame()
    ff = ['signal', 'name_jaccard', 'target_address_empty']
    real_train, real_predict = models.train_backend, models.predict_raw
    training_members = {}
    oof_calls = []
    checking_oof = True

    def training(backend, rows, selected, path, **kwargs):
        model = real_train(backend, rows, selected, path, **kwargs)
        training_members[id(model)] = (
            set(rows['tid'].to_list()), set(rows.filter(pl.col('own') >= 0)['own'].to_list()))
        return model

    def prediction(backend, model, rows, selected, **kwargs):
        if checking_oof:
            tids, owners = training_members[id(model)]
            assert tids.isdisjoint(rows['tid'].to_list())
            assert owners.isdisjoint(rows.filter(pl.col('own') >= 0)['own'].to_list())
            oof_calls.append((backend, rows.height))
        return real_predict(backend, model, rows, selected, **kwargs)

    monkeypatch.setattr(models, 'train_backend', training)
    monkeypatch.setattr(models, 'predict_raw', prediction)
    bundle = models.fit('gated_ensemble', frame, ff, tmp_path)
    checking_oof = False
    assert len(oof_calls) == 9
    assert set(bundle['models']) == set(models.BACKENDS)
    assert all(len(experts) == 3 for experts in bundle['models'].values())
    oof = pl.read_parquet(tmp_path / 'oof_predictions.parquet')
    assert oof.height == frame.height
    assert np.isfinite(oof.select([backend + '_oof_logit' for backend in models.BACKENDS]).to_numpy()).all()
    for backend in models.BACKENDS:
        assert sum(count for family, count in oof_calls if family == backend) == frame.height
        assert len(list(tmp_path.glob(backend + '_fold*'))) == 3
        logits = oof[backend + '_oof_logit'].to_numpy()
        assert logits[frame['y'].to_numpy() == 1].mean() > logits[frame['y'].to_numpy() == 0].mean()
    a = models.predict(bundle, frame)
    poisoned = frame.with_columns((1 - pl.col('y')).alias('y'),
                                 pl.lit(-123).alias('own'), pl.lit('france').alias('co'))
    np.testing.assert_array_equal(a, models.predict(bundle, poisoned))
    assert a.dtype == np.float32 and np.isfinite(a).all()
    assert ((a > 0) & (a < 1)).all()
    assert models.predict(bundle, frame.head(0)).shape == (0,)


@pytest.mark.parametrize('bad', [np.zeros((4,)), np.zeros((3, 3)),
                               np.full((4, 3), np.inf), np.full((4, 3), np.nan)])
def test_gate_rejects_incomplete_or_nonfinite_oof_predictions(bad):
    with pytest.raises(ValueError, match='OOF'):
        models.fit_gate(pl.DataFrame({'x': np.arange(4)}), bad, [0, 1, 0, 1])


@pytest.mark.parametrize('weights', [[1, 1, 1], [1, 0, 1, 1], [1, -1, 1, 1],
                                   [1, np.nan, 1, 1], [1, np.inf, 1, 1]])
def test_gate_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match='weights'):
        models.fit_gate(pl.DataFrame({'x': np.arange(4)}), np.zeros((4, 3)), [0, 1, 0, 1], weights)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_gate_rejects_nonfinite_observable_context(value):
    frame = pl.DataFrame({'name_jaccard': [value, .5]})
    with pytest.raises(ValueError, match='Nonfinite gate context feature: name_jaccard'):
        models.fit_gate(frame, np.zeros((2, 3)), [0, 1])


@pytest.mark.parametrize('feature', ['qid', 'tid', 'own', 'y', 'co', 'fold', 'deg'])
def test_models_reject_label_identity_and_partition_features(feature):
    with pytest.raises(ValueError, match='exclude labels'):
        models.matrix(fitting_frame(), ['signal', feature])

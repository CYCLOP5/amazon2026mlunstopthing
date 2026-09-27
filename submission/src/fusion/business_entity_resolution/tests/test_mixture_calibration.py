from pathlib import Path
import json
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from latest_fusion import mixture_calibration as mixture


def source_frame():
    sizes = [1000, 600, 400]
    classes = np.repeat(np.arange(3), sizes)
    qid = np.arange(sum(sizes), dtype=np.int64)
    own = np.where(classes == 0, qid, np.where(classes == 1, qid+sum(sizes), -1))
    return pl.DataFrame({'qid': qid, 'tid': qid, 'own': own, 'y': (classes == 0).astype(np.int8),
        'co': ['us']*sum(sizes), 'seg': [2]*sum(sizes), 'source': ['s3']*sum(sizes),
        'target_address_empty': [1.]*sum(sizes), 'raw': np.choose(classes, [.9, .6, .1])})


def target_frame(scores):
    n = len(scores)
    return pl.DataFrame({'tid': np.arange(n), 'co': ['us']*n, 'seg': [2]*n,
                         'source': ['s3']*n, 'target_address_empty': [1.]*n, 'raw': scores})


def test_three_class_em_recovers_known_shift_and_normalized_priors():
    x = np.arange(50)
    density = np.stack([np.exp(-.5*((x-centre)/3.)**2) for centre in [39, 25, 10]])
    density /= density.sum(axis=1, keepdims=True)
    truth = np.array([.35, .25, .4])
    counts = 200_000 * (truth @ density)
    res = mixture.estimate_prior(density, counts, [.6, .2, .2])
    assert res['passed'], res
    np.testing.assert_allclose(res['prior'], truth, atol=2e-6)
    assert res['mixture_tv'] < 2e-6
    assert np.isfinite(res['prior']).all() and sum(res['prior']) == pytest.approx(1.)


def test_source_target_identity_exactly_preserves_matched_control():
    src = source_frame()
    bundle = mixture.fit_source(src)
    target = src.drop('y', 'own', 'qid')
    recipe = mixture.adapt(bundle, target)
    assert bundle['cells']['country:us']['class_order'] == ['alias', 'other_owned', 'decoy']
    assert all(cell['reason'] == 'source_target_identity' for cell in recipe['diagnostics'].values())
    assert recipe['adapted_cells'] == 0
    assert np.array_equal(mixture.predict(bundle, target), mixture.predict(bundle, target, recipe))
    json.dumps(bundle, allow_nan=False)
    json.dumps(recipe, allow_nan=False)


def test_unlabeled_shift_adapts_with_shrinkage_and_label_poison_has_no_effect():
    bundle = mixture.fit_source(source_frame())
    target = target_frame(np.r_[np.full(850, .9), np.full(600, .6), np.full(550, .1)])
    recipe = mixture.adapt(bundle, target)
    assert recipe['adapted_cells'] > 0
    for key, cell in recipe['cells'].items():
        prior = np.array(cell['prior'])
        src = np.array(bundle['cells'][key]['source_prior'])
        assert prior.sum() == pytest.approx(1.) and np.isfinite(prior).all()
        assert ((prior/src >= .5-1e-10) & (prior/src <= 2.+1e-10)).all()
        assert .425 < prior[0] < .5
    poison = target.with_columns(pl.lit('not a label').alias('y'),
                                 pl.lit(None).alias('own'), pl.lit(99).alias('fold'))
    assert mixture.adapt(bundle, poison) == recipe
    assert np.array_equal(mixture.predict(bundle, poison, recipe), mixture.predict(bundle, target, recipe))
    assert np.isfinite(mixture.predict(bundle, target, recipe)).all()


def test_identical_densities_abstain_and_unmodelled_mass_falls_back():
    uniform = np.full((3, 50), 1/50)
    res = mixture.estimate_prior(uniform, np.full(50, 100.), [.5, .3, .2])
    assert not res['passed'] and res['reason'] == 'unidentifiable_densities'
    no_support = mixture.estimate_prior([[1., 0., 0.], [0., 1., 0.]], [0., 0., 1000.], [.5, .5])
    assert not no_support['passed'] and no_support['reason'] == 'bad_mixture_fit'
    assert no_support['unsupported_target_mass'] == 1.
    bundle = mixture.fit_source(source_frame())
    target = target_frame(np.zeros(2000))
    recipe = mixture.adapt(bundle, target)
    assert not recipe['adapted_cells']
    assert all(not diagnostic['passed'] for diagnostic in recipe['diagnostics'].values())
    assert all(diagnostic['reason'] == 'bad_mixture_fit' for diagnostic in recipe['diagnostics'].values())
    assert np.array_equal(mixture.predict(bundle, target, recipe), mixture.predict(bundle, target))


def test_sparse_children_back_off_empty_and_unknown_countries_remain_finite():
    src = source_frame().with_columns(pl.when(pl.col('tid') < 10).then(pl.lit('s2')).otherwise(pl.lit('s3')).alias('source'))
    bundle = mixture.fit_source(src)
    assert 'child:us|2|s2|1.0' not in bundle['cells']
    sparse = target_frame(np.full(10, .9)).with_columns(pl.lit('s2').alias('source'))
    recipe = mixture.adapt(bundle, sparse)
    assert all(v['reason'] == 'sparse_target' for v in recipe['diagnostics'].values())
    assert np.array_equal(mixture.predict(bundle, sparse, recipe), mixture.predict(bundle, sparse))
    assert mixture.adapt(bundle, sparse.head(0))['cells'] == {}
    assert mixture.predict(bundle, sparse.head(0)).shape == (0,)
    france = sparse.with_columns(pl.lit('france').alias('co'))
    np.testing.assert_allclose(mixture.predict(bundle, france), france['raw'].to_numpy())
    assert mixture.adapt(bundle, france)['cells'] == {}


def test_binary_fallback_and_source_isolation_checks():
    src = source_frame().with_columns(pl.when(pl.col('y') == 1).then(.9).otherwise(.2).alias('raw'))
    bundle = mixture.fit_source(src)
    assert all(cell['class_order'] == ['alias', 'nonalias'] for cell in bundle['cells'].values())
    with pytest.raises(ValueError, match='fold0'):
        mixture.fit_source(src.with_columns(pl.lit(1).alias('fold')))
    with pytest.raises(ValueError, match='partition1'):
        mixture.fit_source(src.with_columns(pl.lit(0).alias('fold')))
    with pytest.raises(ValueError, match='India/US'):
        mixture.fit_source(src.with_columns(pl.lit('france').alias('co')))
    with pytest.raises(ValueError, match='ownership'):
        mixture.fit_source(src.with_columns(pl.lit(1).alias('y')))
    empty = mixture.fit_source(src.head(0))
    assert not empty['cells']
    np.testing.assert_allclose(mixture.predict(empty, src), src['raw'].to_numpy())


def test_bounds_do_not_break_normalization_for_extreme_priors():
    prior = mixture._bounded_prior([.999999999, .0000000005, .0000000005], [.01, .39, .6])
    assert prior.sum() == pytest.approx(1.)
    assert np.all(prior / [.01, .39, .6] >= .5-1e-12)
    assert np.all(prior / [.01, .39, .6] <= 2.+1e-12)

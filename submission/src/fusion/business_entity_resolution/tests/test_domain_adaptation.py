from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from latest_fusion import domain_adaptation as domain


def population(n=2000, shift=.16, perfect=False):
    def frame(seed, target):
        rng = np.random.default_rng(seed)
        values = {c: rng.random(n) for c in domain.FEATURES}
        values['name_jaccard'] = np.clip(rng.normal(.4+(shift if target else 0.), .15, n), 0., 1.)
        if perfect:
            values['name_jaccard'] = np.full(n, float(target))
        return pl.DataFrame({'qid': np.arange(n), 'tid': np.arange(n)//2, 'co': ['us']*n,
                            **values, 'y': rng.integers(0, 2, n), 'own': rng.integers(-1, n, n)})
    return frame(100, False), frame(101, True)


def fit(source, path, output=None):
    return domain.fit_weights(source, path, output, sample_size=1500,
        _rounds=25, _model_params={'min_data_in_leaf': 25, 'num_threads': 1},
        _minimum_fit=100, _minimum_validation=50)


def test_genuine_domain_fit_weights_bounds_means_and_truth_label_invariance(tmp_path):
    src, target = population()
    path = tmp_path/'test.parquet'
    target.write_parquet(path)
    weights, report = fit(src, path, tmp_path/'models')
    assert report['countries']['us']['used_weights'], report
    assert .55 < report['countries']['us']['auc'] < .95
    assert weights.dtype == np.float32 and len(weights) == src.height
    assert np.isfinite(weights).all() and weights.min() >= .25 and weights.max() <= 4.
    assert weights.mean() == pytest.approx(1., abs=1e-6)
    assert not np.allclose(weights, 1.)
    assert report['countries']['us']['ess_fraction'] >= .25
    assert (tmp_path/'models/us_domain_model.txt').is_file()
    poison_source = src.with_columns(pl.lit('poison').alias('y'), pl.lit(None).alias('own'),
                                        pl.lit(999).alias('fold'), pl.lit(-1).alias('deg'))
    target.with_columns(pl.lit('wrong').alias('y'), pl.lit(None).alias('own')).write_parquet(path)
    poisoned, poisoned_report = fit(poison_source, path, tmp_path/'poisoned')
    np.testing.assert_array_equal(poisoned, weights)
    assert poisoned_report['countries']['us']['auc'] == report['countries']['us']['auc']
    assert poisoned_report['features'] == domain.FEATURES
    assert not set(['qid', 'tid', 'y', 'own', 'fold', 'deg']).intersection(report['features'])


def test_sampling_deterministic_bounded_and_validation_target_disjoint(tmp_path):
    src, target = population()
    path = tmp_path/'test.parquet'
    target.write_parquet(path)
    a = domain._sample(pl.scan_parquet(path), 'us', 200, 1)
    b = domain._sample(pl.scan_parquet(path), 'us', 200, 1)
    assert a.equals(b) and a.height == 200
    assert 'y' not in a and 'own' not in a
    mask = domain._validation_mask(a, 1)
    assert not set(a.filter(pl.Series(mask))['tid']).intersection(a.filter(pl.Series(~mask))['tid'])


def test_poor_overlap_and_missing_country_support_retain_unit_weights(tmp_path):
    src, target = population(perfect=True)
    india = src.head(20).with_columns(pl.lit('india').alias('co'))
    src = pl.concat([src, india])
    path = tmp_path/'test.parquet'
    target.write_parquet(path)
    weights, report = fit(src, path)
    np.testing.assert_array_equal(weights, np.ones(src.height, dtype=np.float32))
    assert report['countries']['us']['fallback_reason'] == 'poor_domain_overlap'
    assert report['countries']['india']['fallback_reason'] == 'insufficient_domain_support'
    assert report['countries']['us']['feature_support']['name_jaccard']['test_outside_source_range_fraction'] == 1.


def test_no_detectable_shift_retains_units(tmp_path):
    src, target = population(shift=0.)
    path = tmp_path/'test.parquet'
    target.write_parquet(path)
    weights, report = fit(src, path)
    assert report['countries']['us']['auc'] <= .55
    assert report['countries']['us']['fallback_reason'] == 'no_useful_shift'
    np.testing.assert_array_equal(weights, np.ones(src.height, dtype=np.float32))


def test_clipped_scaling_is_country_normalized_and_not_clip_then_divide():
    ratios = np.r_[np.full(999, .00001), 1000.]
    weights = domain.bounded_mean_one(ratios)
    assert weights.mean() == pytest.approx(1., abs=1e-12)
    assert weights.min() >= .25 and weights.max() <= 4.
    assert domain._ess(weights)/len(weights) >= .25
    for invalid in ([0., 1.], [np.nan, 1.], [-1., 2.]):
        with pytest.raises(ValueError):
            domain.bounded_mean_one(invalid)


def test_non_numeric_features_fail_before_domain_fitting(tmp_path):
    src, target = population()
    path = tmp_path/'test.parquet'
    target.write_parquet(path)
    with pytest.raises(ValueError, match='numeric'):
        fit(src.with_columns(pl.lit('text').alias('name_jaccard')), path)

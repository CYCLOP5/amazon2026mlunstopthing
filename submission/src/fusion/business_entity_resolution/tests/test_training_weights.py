from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import polars as pl
import pytest
from latest_fusion.training_weights import sample_weights


def fixture():
    return pl.DataFrame({'qid': [1, 2, 2, 2], 'target_address_empty': [0., 0., 1., 1.],
                         'y': [1, 0, 0, 1]})


def test_each_business_gets_equal_total_weight():
    d = fixture()
    w = sample_weights(d, 'owner_balanced')
    assert w.mean() == pytest.approx(1.)
    assert w[0] == pytest.approx(w[1:].sum())
    assert sample_weights(d) is None


def test_missing_address_emphasis_does_not_depend_on_labels():
    d = fixture()
    w = sample_weights(d, 'owner_missing_address', 4.)
    np.testing.assert_array_equal(w, sample_weights(d.with_columns(1-pl.col('y')), 'owner_missing_address', 4.))
    assert w[2] == pytest.approx(4*w[1])
    assert w[3] == pytest.approx(w[2])
    assert w.mean() == pytest.approx(1.)
    np.testing.assert_allclose(sample_weights(d.reverse(), 'owner_missing_address', 4.)[::-1], w)


def test_bad_weight_configuration_rejected():
    with pytest.raises(ValueError, match='Unknown'):
        sample_weights(fixture(), 'unknown')
    with pytest.raises(ValueError, match='Empty'):
        sample_weights(fixture().head(0), 'owner_balanced')
    with pytest.raises(ValueError, match='multiplier'):
        sample_weights(fixture(), 'owner_missing_address', float('nan'))

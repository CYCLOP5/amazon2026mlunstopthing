'compare the batch decoder against the original expected-f implementation'
from pathlib import Path
import sys
import numpy as np
import polars as pl
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from er.stack import decode
from final_repair.fast_decode import expected_f, apply


def fixture(seed=16, owners=250):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 45, owners)
    q = np.repeat(np.arange(owners), sizes)
    p = rng.uniform(0, 1, len(q))
    d = pl.DataFrame({'qid': q, 'tid': np.arange(len(q)), 'p': p, 'y': rng.integers(0, 2, len(q))})

    competitors = d.sample(fraction=.2, seed=seed).with_columns((pl.col('qid')+owners).alias('qid'))
    return pl.concat([d, competitors]).sample(fraction=1, shuffle=True, seed=seed)


@pytest.mark.parametrize('floor', [0., .1, .3, .5, 1.01])
def test_random_equivalence(floor):
    pairs = fixture()
    assert expected_f(pairs, floor).equals(decode.expected_f(pairs, floor))


def test_singletons_extremes_ties_and_truncation():
    groups = [[.5], [np.nextafter(.5, 1)], [0.], [1.], [.5]*3, [0.]*4, [1.]*30, [.6]*30, [.5, .5, 0., 1.]]
    q = [i for i, group in enumerate(groups) for _ in group]
    p = [v for group in groups for v in group]
    pairs = pl.DataFrame({'qid': q, 'tid': np.arange(len(q)), 'p': p})
    for floor in (0., .3, .5):
        res = expected_f(pairs, floor)
        assert res.equals(decode.expected_f(pairs, floor))
        assert res.filter(pl.col('qid') == 6).height == 25
    assert expected_f(pairs.head(0), .3).equals(pairs.head(0))


def test_other_rules_delegate():
    pairs = fixture(owners=8)
    for rule in ('plain_threshold', 'top1_threshold'):
        assert apply(rule, pairs, .4).equals(decode.apply(rule, pairs, .4))


@pytest.mark.parametrize('dtype', [pl.Float32, pl.Float64])
def test_rounded_probability_dtype_and_group_order(dtype):
    pairs = fixture(seed=37, owners=200).with_columns(pl.col('p').round(2).cast(dtype))
    assert expected_f(pairs, .05).equals(decode.expected_f(pairs, .05))

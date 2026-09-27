from pathlib import Path
import sys
import polars as pl
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from latest_fusion.rescue_policy import apply_additions, select_additions, promotion_additions


def population():
    anchors = pl.DataFrame({'qid': range(1, 21), 'deg': [2]*20, 'co': ['us']*20})
    base = pl.DataFrame({'qid': range(1, 21), 'tid': range(101, 121), 'p': [.99]*20, 'y': [1]*20})
    actions = pl.DataFrame({'qid': range(1, 21), 'tid': range(201, 221), 'score': [.999]*20, 'y': [1]*20, 'co': ['us']*20})
    return anchors, base, actions


def test_preserves_baseline_and_rejects_swapped_target():
    anchors, base, actions = population()
    swapped = actions.head(1).with_columns(pl.lit(101).cast(pl.Int64).alias('tid'), pl.lit(2).cast(pl.Int64).alias('qid'))
    out = apply_additions(base, swapped, 0.)
    assert out.equals(base)
    assert not promotion_additions(swapped, base, anchors, min_actions=1)['passed']


def test_preserves_owners_outside_search():
    anchors, base, actions = population()
    out = apply_additions(base, actions, .9)
    assert out.head(base.height).equals(base)
    policy, evidence = select_additions(actions, base, anchors.head(10))
    assert policy['enabled'] and evidence['added_tp'] == 10


def test_zero_fp_positive_gain_and_uncertainty():
    anchors, base, actions = population()
    policy, evidence = select_additions(actions, base, anchors)
    assert policy['enabled'] and evidence['added_fp'] == 0
    report = promotion_additions(actions, base, anchors)
    assert report['passed']
    assert report['after']['pair_precision'] == report['before']['pair_precision']
    assert report['after']['pair_recall'] > report['before']['pair_recall']
    assert report['action_precision_ci95'][0] < .999


def test_any_fp_rejected():
    anchors, base, actions = population()
    actions = actions.with_columns(pl.when(pl.col('qid') == 1).then(0).otherwise(1).alias('y'))
    assert not select_additions(actions, base, anchors)[0]['enabled']
    report = promotion_additions(actions, base, anchors)
    assert not report['passed'] and report['added_fp'] == 1


def test_no_actions_fallback():
    anchors, base, actions = population()
    assert not select_additions(actions.head(0), base, anchors)[0]['enabled']


def test_check_labels_cannot_change_search_policy():
    anchors, base, actions = population()
    search = anchors.head(10)
    original = select_additions(actions, base, search)[0]
    modified = actions.with_columns(pl.when(pl.col('qid') > 10).then(0).otherwise(pl.col('y')).alias('y'))
    assert select_additions(modified, base, search)[0] == original


def test_unique_global_ownership_required():
    anchors, base, actions = population()
    with pytest.raises(ValueError):
        select_additions(pl.concat([actions, actions.head(1)]), base, anchors)


def test_false_positive_score_tie_discards_true_tie_too():
    anchors, base, actions = population()
    actions = actions.with_columns(
        pl.when(pl.col('qid') <= 2).then(.98).otherwise(.981).alias('score'),
        pl.when(pl.col('qid') == 1).then(0).otherwise(1).alias('y'))
    policy, evidence = select_additions(actions, base, anchors)
    assert policy['enabled'] and policy['threshold'] > .98
    assert evidence['added_tp'] == 18 and evidence['added_fp'] == 0
    assert len(evidence['evaluated']) <= 10


def test_float32_cutoff_does_not_round_back_to_false_score():
    import numpy as np
    anchors, base, actions = population()
    false_score = np.float32(.98)
    true_score = np.nextafter(false_score, np.float32(np.inf))
    scores = np.full(actions.height, true_score, dtype=np.float32)
    scores[0] = false_score
    actions = actions.with_columns(pl.Series('score', scores),
        pl.when(pl.col('qid') == 1).then(0).otherwise(1).alias('y'))
    policy, evidence = select_additions(actions, base, anchors)
    assert policy['enabled'] and policy['threshold'] == float(true_score)
    assert evidence['added_tp'] == 19 and evidence['added_fp'] == 0
    assert apply_additions(base, actions, policy['threshold']).height == 39

from pathlib import Path
import sys

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from latest_fusion.directional_policy import (apply_actions, evaluate_actions,
                                              propose_actions, select_policy)


def population(n=30):
    anchors = pl.DataFrame({'qid': list(range(n)), 'deg': [2]*n, 'co': ['us']*n})
    base = pl.DataFrame({'qid': list(range(n)), 'tid': list(range(100, 100+n)),
                         'y': [1]*n, 'p': [.99]*n})
    rows = pl.DataFrame({'qid': list(range(n)), 'tid': list(range(200, 200+n)),
                         'y': [1]*n, 'p': [.999]*n, 'co': ['us']*n,
                         'own': list(range(n)), 'fold': [0]*n,
                         'p_alias': [.9999]*n, 'p_decoy': [.00005]*n, 'p_other': [.00005]*n})
    return anchors, base, rows


def rule(drop=None, add=None):
    return {'enabled': True, 'drop_threshold': drop, 'add_threshold': add}


def test_true_additions_fixed_grid_and_frozen_replay():
    anchors, base, rows = population()
    frozen, report = select_policy(rows, base, anchors)
    assert frozen == rule(add=.99)
    assert report['passed'] and report['TPadded'] == 30 and report['FPadded'] == 0
    assert report['after']['macro_f05'] == 1
    assert 0 < report['action_benefit_wilson98_lower'] < 1
    drops, adds = propose_actions(rows.drop('y', 'own'), base, anchors.select('qid'), frozen)
    assert drops.height == 0 and adds.height == 30


def test_ambiguous_global_owners_removed_before_scope():
    anchors, base, rows = population()
    competitor = rows.head(1).with_columns(pl.lit(999).alias('qid'), pl.lit(999).alias('own'))
    cands = pl.concat([rows, competitor], how='vertical_relaxed')
    _, adds = propose_actions(cands, base, anchors, rule(add=.99))
    assert adds.height == 29 and 200 not in adds['tid']


def test_occupied_targets_cannot_reassign_even_if_dropped():
    anchors, base, rows = population()
    attempted = rows.head(1).with_columns(pl.lit(101).alias('tid'))
    drops = base.filter(pl.col('tid') == 101).select('qid', 'tid')
    with pytest.raises(ValueError, match='disjoint'):
        apply_actions(base, drops, attempted, anchors)
    res = apply_actions(base, drops.head(0), attempted, anchors)
    assert res.equals(base)


def test_scope_preserves_owners_and_ignores_outside_labels():
    anchors, base, rows = population(60)
    cal = anchors.head(30)
    first = select_policy(rows, base, cal)[0]
    changed_rows = rows.with_columns(pl.when(pl.col('qid') >= 30).then(0).otherwise(pl.col('y')).alias('y'))
    changed_base = base.with_columns(pl.when(pl.col('qid') >= 30).then(0).otherwise(pl.col('y')).alias('y'))
    assert select_policy(changed_rows, changed_base, cal)[0] == first
    drops, adds = propose_actions(rows, base, cal, first)
    out = apply_actions(base, drops, adds, cal)
    assert out.filter(pl.col('qid') >= 30).equals(base.filter(pl.col('qid') >= 30))
    requested = base.tail(1).select('qid', 'tid')
    assert apply_actions(base, requested, rows.head(0), cal).equals(base)


def test_zero_degree_false_links_deletion_exact_macro_and_decoys():
    anchors, base, rows = population()
    anchors = anchors.with_columns(pl.lit(0).alias('deg'))
    base = base.with_columns(pl.lit(0).alias('y'), pl.lit(-1).alias('own'))
    rows = rows.with_columns(pl.Series('tid', base['tid']), pl.lit(0).alias('y'),
                             pl.lit(-1).alias('own'), pl.lit(.9999).alias('p_decoy'),
                             pl.lit(.00005).alias('p_alias'))
    frozen, report = select_policy(rows, base, anchors)
    assert frozen == rule(drop=.75)
    assert report['after']['macro_f05'] == 1 and report['before']['macro_f05'] == 0
    assert report['TPremoved'] == 0 and report['FPremoved'] == report['removed_decoy'] == 30


def test_removed_true_link_fails_despite_net_precision_improvement():
    anchors, base, rows = population()
    base = base.with_columns(pl.when(pl.col('qid') == 0).then(1).otherwise(0).alias('y'))
    rows = rows.with_columns(pl.Series('tid', base['tid']), pl.Series('y', base['y']),
                             pl.lit(.9999).alias('p_decoy'), pl.lit(.00005).alias('p_alias'),
                             pl.when(pl.col('qid') == 0).then(pl.col('qid')).otherwise(-1).alias('own'))
    frozen, report = select_policy(rows, base, anchors)
    assert not frozen['enabled']
    assert all(not trial['checks']['zero_removed_tp'] for trial in report['evaluated'][:4])


def test_added_false_link_rejects_all_thresholds_and_bad_owner_label():
    anchors, base, rows = population()
    rows = rows.with_columns(pl.when(pl.col('qid') == 0).then(0).otherwise(1).alias('y'),
                             pl.when(pl.col('qid') == 0).then(-1).otherwise(pl.col('own')).alias('own'))
    frozen, report = select_policy(rows, base, anchors)
    assert not frozen['enabled']
    assert any(not trial['checks']['zero_added_fp'] for trial in report['evaluated'])
    _, adds = propose_actions(rows, base, anchors, rule(add=.99))

    with pytest.raises(ValueError, match='owner'):
        evaluate_actions(base.with_columns(pl.col('qid').alias('own')), rows.select('qid', 'tid').head(0),
                         adds.with_columns(pl.col('qid').alias('own')), anchors)


def test_combination_requires_both_independent_guards():
    anchors, base, rows = population(60)
    anchors = anchors.with_columns(pl.when(pl.col('qid') >= 30).then(0).otherwise(pl.col('deg')).alias('deg'))
    base = base.with_columns(pl.when(pl.col('qid') >= 30).then(0).otherwise(1).alias('y'))
    drops = rows.tail(30).with_columns((pl.col('tid')-100).alias('tid'), pl.lit(0).alias('y'),
                                      pl.lit(-1).alias('own'), pl.lit(.9999).alias('p_decoy'),
                                      pl.lit(.00005).alias('p_alias'))
    frozen, report = select_policy(pl.concat([rows.head(30), drops], how='vertical_relaxed'), base, anchors)
    assert frozen == rule(drop=.75, add=.99)
    assert report['TPadded'] == report['FPremoved'] == 30 and report['after']['macro_f05'] == 1


def test_added_wrong_owner_and_decoy_counts_without_baseline_owner_column():
    anchors, base, rows = population()
    rows = rows.head(2).with_columns(pl.lit(0).alias('y'), pl.Series('own', [-1, 1000]))
    report = evaluate_actions(base, rows.select('qid', 'tid').head(0), rows, anchors)
    assert report['added_decoy'] == report['added_other'] == 1
    assert report['FPadded'] == 2


def test_removed_owner_provenance_can_come_from_sidecar():
    anchors, base, rows = population()
    base = base.with_columns(pl.lit(0).alias('y'))
    rows = rows.with_columns(pl.Series('tid', base['tid']), pl.lit(0).alias('y'),
                             pl.lit(-1).alias('own'), pl.lit(.9999).alias('p_decoy'))
    drops, adds = propose_actions(rows, base, anchors, rule(drop=.99))
    report = evaluate_actions(base, drops, adds, anchors)
    assert report['removed_decoy'] == report['FPremoved'] == 30


def test_empty_minimum_duplicates_and_invalid_scores():
    anchors, base, rows = population()
    frozen, report = select_policy(rows.head(0), base, anchors)
    assert not frozen['enabled'] and not report['changed'] and len(report['evaluated']) == 7
    assert not select_policy(rows.head(24), base, anchors)[0]['enabled']
    with pytest.raises(ValueError, match='duplicate'):
        propose_actions(pl.concat([rows, rows.head(1)]), base, anchors, rule(add=.99))
    with pytest.raises(ValueError, match='duplicate'):
        apply_actions(pl.concat([base, base.head(1)]), rows.select('qid', 'tid').head(0), rows.head(0), anchors)
    with pytest.raises(ValueError, match='finite'):
        propose_actions(rows.with_columns(pl.lit(float('nan')).alias('p_alias')), base, anchors, rule(add=.99))


def test_no_actions_and_zero_degree_owners_in_exact_metric():
    anchors, base, rows = population(2)
    anchors = anchors.with_columns(pl.Series('deg', [0, 2]))
    base = base.tail(1)
    report = evaluate_actions(base, rows.select('qid', 'tid').head(0), rows.head(0), anchors)
    assert report['before']['macro_f05'] == pytest.approx((1 + 1.25/1.5)/2)
    assert report['before'] == report['after'] and not report['changed']

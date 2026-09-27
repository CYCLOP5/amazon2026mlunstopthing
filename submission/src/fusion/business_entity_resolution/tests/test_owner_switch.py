from pathlib import Path
import sys

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from latest_fusion.owner_switch import (apply_switches, evaluate_switches,
                                        propose_switches, select_policy, GRID)


def population(n=6):
    anchors = pl.DataFrame({'qid': list(range(2*n)), 'deg': [0]*n + [1]*n,
                            'co': ['us']*(2*n)})
    base = pl.DataFrame({'qid': list(range(n)), 'tid': list(range(n)),
                         'p': [.95]*n, 'y': [0]*n, 'own': list(range(n, 2*n))})
    rows = pl.DataFrame({'qid': list(range(n)) + list(range(n, 2*n)),
                         'tid': list(range(n))*2, 'p_alias': [.001]*n + [.9999]*n,
                         'p_decoy': [.001]*(2*n), 'p_other': [.001]*(2*n),
                         'y': [0]*n + [1]*n, 'own': list(range(n, 2*n))*2})
    return rows, base, anchors


def test_true_ownership_switch_and_exact_zero_degree_macro():
    rows, base, anchors = population()
    switches = propose_switches(rows, base, anchors, .999, .99)
    assert switches['old_qid'].to_list() == list(range(6))
    assert switches['qid'].to_list() == list(range(6, 12))
    assert switches['old_p'].to_list() == [.001]*6
    res = apply_switches(base, switches)
    assert res['p'].to_list() == [.9999]*6
    assert res['y'].to_list() == [1]*6
    assert res['own'].equals(base['own'])
    assert res['tid'].equals(base['tid']) and res.columns == base.columns
    report = evaluate_switches(base, switches, anchors)
    assert report['corrected'] == 6 and report['broken'] == 0
    assert report['before']['macro_f05'] == 0 and report['after']['macro_f05'] == 1
    assert report['countries']['us']['after']['macro_f05'] == 1
    frozen, evidence = select_policy(rows, base, anchors)
    assert frozen == {'enabled': True, 'min_alias': .999, 'min_margin': .99}
    assert evidence['passed'] and len(evidence['evaluated']) == len(GRID) == 4
    replay = propose_switches(rows.drop('y', 'own'), base.drop('y', 'own'),
                              anchors.select('qid'), frozen['min_alias'], frozen['min_margin'])
    assert replay.select('old_qid', 'qid', 'tid').equals(switches.select('old_qid', 'qid', 'tid'))


def test_broken_true_link_rejects_rule_even_with_net_improvement():
    rows, base, anchors = population(7)

    base = base.with_columns(pl.when(pl.col('tid') == 6).then(1).otherwise(pl.col('y')).alias('y'),
                             pl.when(pl.col('tid') == 6).then(6).otherwise(pl.col('own')).alias('own'))
    rows = rows.with_columns(pl.when(pl.col('tid') == 6).then(6).otherwise(pl.col('own')).alias('own'))
    rows = rows.with_columns((pl.col('qid') == pl.col('own')).cast(pl.Int8).alias('y'))
    anchors = anchors.with_columns(pl.when(pl.col('qid') == 6).then(1)
        .when(pl.col('qid') == 13).then(0).otherwise(pl.col('deg')).alias('deg'))
    frozen, report = select_policy(rows, base, anchors)
    assert not frozen['enabled']
    assert all(trial['corrected'] == 6 and trial['broken'] == 1 for trial in report['evaluated'])
    assert all(not trial['checks']['zero_broken'] for trial in report['evaluated'])


def test_whole_transaction_scope_and_full_pool_competitor():
    rows, base, anchors = population()
    only_new = anchors.filter(pl.col('qid') >= 6)
    assert propose_switches(rows, base, only_new, .999, .99).height == 0
    only_old = anchors.filter(pl.col('qid') < 6)
    assert propose_switches(rows, base, only_old, .999, .99).height == 0
    scope = anchors.filter(pl.col('qid').is_in([0, 6]))
    switches = propose_switches(rows, base, scope, .999, .99)
    assert switches.height == 1
    res = apply_switches(base, switches)
    assert res.filter(pl.col('tid') != 0).equals(base.filter(pl.col('tid') != 0))
    with pytest.raises(ValueError, match='Both switch owners'):
        evaluate_switches(base, switches, only_new)
    outside = rows.filter(pl.col('qid') == 6).with_columns(pl.lit(99).alias('qid'),
                  pl.lit(1.).alias('p_alias'))
    assert propose_switches(pl.concat([rows, outside], how='vertical_relaxed'), base,
                            scope, .999, .99).height == 0


def test_calibration_never_reads_outside_scope_labels():
    rows, base, anchors = population(7)
    scope = anchors.filter(~pl.col('qid').is_in([6, 13]))
    initial = select_policy(rows, base, scope)
    poisoned_rows = rows.with_columns(pl.when(pl.col('qid').is_in([6, 13])).then(None)
                                      .otherwise(pl.col('y')).alias('y'))
    poisoned_base = base.with_columns(pl.when(pl.col('qid') == 6).then(None)
                                      .otherwise(pl.col('y')).alias('y'))
    changed = select_policy(poisoned_rows, poisoned_base, scope)
    assert changed == initial


def test_country_decline_rejects_net_global_gain_without_broken_true_links():
    rows, base, anchors = population(5)


    extra_anchors = pl.DataFrame({'qid': [10, 11], 'deg': [1, 0], 'co': ['india', 'india']})
    extra_base = pl.DataFrame({'qid': [10], 'tid': [99], 'p': [.95], 'y': [0], 'own': [99]})
    extra_rows = pl.DataFrame({'qid': [10, 11], 'tid': [99, 99],
        'p_alias': [.001, .9999], 'p_decoy': [.001, .001], 'p_other': [.001, .001],
        'y': [0, 0], 'own': [99, 99]})
    rule, report = select_policy(pl.concat([rows, extra_rows]), pl.concat([base, extra_base]),
                                pl.concat([anchors, extra_anchors]))
    assert not rule['enabled']
    for trial in report['evaluated']:
        assert trial['corrected'] == 5 and trial['broken'] == 0 and trial['wrong_to_wrong'] == 1
        assert trial['checks']['overall_nondecreasing']
        assert not trial['checks']['india_nondecreasing']


def test_ties_margin_current_best_and_unaccepted_target():
    rows, base, anchors = population(1)

    tied = rows.with_columns(pl.lit(.9999).alias('p_alias'))
    assert propose_switches(tied.reverse(), base, anchors, .98, 0).height == 0
    assert propose_switches(rows, base, anchors, .999, .999).height == 0

    extra = rows.tail(1).with_columns(pl.lit(999).alias('tid'))
    switches = propose_switches(pl.concat([rows, extra], how='vertical_relaxed'), base, anchors, .999, .99)
    assert switches['tid'].to_list() == [0]
    malformed = switches.with_columns(pl.lit(999).alias('tid'))
    with pytest.raises(ValueError, match='current owner pair'):
        apply_switches(base, malformed)


def test_empty_exact_preservation_and_minimum_corrected_guard():
    rows, base, anchors = population(4)
    frozen, report = select_policy(rows, base, anchors)
    assert not frozen['enabled'] and all(not t['checks']['five_corrected'] for t in report['evaluated'])
    switches = propose_switches(rows, base, anchors, 1., 1.)
    assert switches.height == 0 and apply_switches(base, switches).equals(base)
    evidence = evaluate_switches(base, switches, anchors)
    assert evidence['before'] == evidence['after'] and evidence['switches'] == 0
    emptybase = base.head(0)
    emptyswitch = propose_switches(rows, emptybase, anchors, .999, .99)
    assert emptyswitch.height == 0 and apply_switches(emptybase, emptyswitch).equals(emptybase)
    zero = anchors.with_columns(pl.lit(0).alias('deg'))
    assert evaluate_switches(emptybase, emptyswitch, zero)['after']['macro_f05'] == 1


def test_invalid_pairs_scores_and_global_duplicates_rejected():
    rows, base, anchors = population()
    for bad, message in [(pl.concat([rows, rows.head(1)]), 'duplicate'),
                         (rows.filter(~((pl.col('qid') == 0) & (pl.col('tid') == 0))), 'Missing current'),
                         (rows.with_columns(pl.lit(float('nan')).alias('p_alias')), 'finite numeric'),
                         (rows.with_columns(pl.lit(-1.).alias('p_other')), 'finite numeric')]:
        with pytest.raises(ValueError, match=message):
            propose_switches(bad, base, anchors, .999, .99)
    with pytest.raises(ValueError, match='duplicate'):
        propose_switches(rows, pl.concat([base, base.head(1)]), anchors, .999, .99)
    switches = propose_switches(rows, base, anchors, .999, .99)
    with pytest.raises(ValueError, match='duplicate'):
        apply_switches(base, pl.concat([switches, switches.head(1)]))

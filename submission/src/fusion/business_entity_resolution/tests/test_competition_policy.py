from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from latest_fusion.competition_policy import evidence_actions, lock_family_policy, apply_family_policy
from latest_fusion.rescue_policy import apply_additions


def evidence_fixture():
    refs = pl.DataFrame({'rid': [1, 2, 3, 4], 'co': ['us']*4,
        'nm': ['Distinct Owner LLC', 'Rare Canonical Inc', 'Repeated Name LLC', 'Other Owner'],
        'ad': ['12 Main Street', '8 Side Road', '3 West Street', '4 East Street']})
    targets = pl.DataFrame({'rid': [10, 20, 30], 'co': ['us']*3,
        'nm': ['Distinct Owner LLC', 'Canonical Rare Ltd', 'Repeated Name LLC'],
        'ad': ['12 Main Street', '', '']})
    actions = pl.DataFrame({'qid': [1, 2, 3], 'tid': [10, 20, 30], 'co': ['us']*3,
        'score': [.97, .99, .99999], 'min_head_margin': [.1]*3, 'y': [1, 1, 0]})
    pool = actions.select('qid', 'tid').with_columns(
        pl.lit(.1).alias('rescue_rescue_shared_idf_sum_margin'),
        pl.lit(.9).alias('rescue_member_vote_fraction'),
        pl.lit(.8).alias('rescue_family_vote_fraction'))
    return actions, pool, refs, targets


def test_full_country_signature_veto_and_label_independence():
    actions, pool, refs, targets = evidence_fixture()

    twin = refs.head(1).with_columns(pl.lit(99).alias('rid'),
        pl.lit('Name Repeated Inc').alias('nm'), pl.lit('99 South Road').alias('ad'))
    refs = pl.concat([refs, twin], how='vertical_relaxed')
    res = evidence_actions(actions, pool, refs, targets)
    assert res['tid'].to_list() == [10, 20]
    assert res['evidence_family'].to_list() == ['addressed', 'unique_name']
    flipped = evidence_actions(actions.with_columns((1-pl.col('y')).alias('y')), pool, refs, targets)
    assert flipped.drop('y').equals(res.drop('y'))

    other = twin.with_columns(pl.lit('india').alias('co'))
    assert evidence_actions(actions, pool, pl.concat([refs.filter(pl.col('rid') != 99), other], how='vertical_relaxed'), targets).height == 3


def test_ties_small_margins_and_conflicting_addresses_abstain():
    actions, pool, refs, targets = evidence_fixture()
    assert evidence_actions(actions.with_columns(pl.lit(0.).alias('min_head_margin')), pool, refs, targets).is_empty()
    assert evidence_actions(actions.with_columns(pl.lit(.02).alias('min_head_margin')), pool, refs, targets).is_empty()
    changed = targets.with_columns(pl.when(pl.col('rid') == 10).then(pl.lit('13 Main Street')).otherwise(pl.col('ad')).alias('ad'))
    assert 10 not in evidence_actions(actions, pool, refs, changed)['tid'].to_list()
    weak = pool.with_columns(pl.lit(.79).alias('rescue_member_vote_fraction'))
    assert evidence_actions(actions, weak, refs, targets)['tid'].to_list() == [10]


def calibration_fixture():
    anchors = pl.DataFrame({'qid': range(1, 81), 'deg': [2]*80, 'co': ['us']*80})
    base = pl.DataFrame({'qid': range(1, 81), 'tid': range(101, 181), 'p': [.99]*80, 'y': [1]*80})
    addressed = pl.DataFrame({'qid': list(range(1, 21))+list(range(41, 61)),
        'tid': range(201, 241), 'score': [.97]*40, 'y': [1]*40, 'co': ['us']*40,
        'evidence_family': ['addressed']*40})
    ambiguous = addressed.head(1).with_columns(pl.lit(25).alias('qid'), pl.lit(999).alias('tid'),
        pl.lit(.99999).alias('score'), pl.lit(0).alias('y'), pl.lit('unique_name').alias('evidence_family'))
    return anchors.head(40), anchors.tail(40), base, pl.concat([addressed, ambiguous], how='vertical_relaxed')


def test_bad_family_does_not_poison_supported_address_family():
    cal, search, base, actions = calibration_fixture()
    policy, report = lock_family_policy(actions, base, cal, search)
    assert policy['enabled']
    assert policy['families']['addressed']['enabled']
    assert policy['families']['addressed']['calibration_floor'] == 0
    assert policy['families']['unique_name']['calibration_floor'] > .99999
    assert not policy['families']['unique_name']['enabled']
    sel = apply_family_policy(actions, policy)
    assert sel.height == 40 and sel['y'].sum() == 40
    assert report['combined_search']['passed']
    final = apply_additions(base, sel, 0.)
    assert base.join(final, on=['qid', 'tid'], how='anti').is_empty()


def test_frozen_policy_does_not_use_evaluation_labels_or_unsupported_family():
    cal, search, base, actions = calibration_fixture()
    policy, _ = lock_family_policy(actions, base, cal, search)
    flipped = actions.with_columns((1-pl.col('y')).alias('y'))
    assert apply_family_policy(actions, policy).drop('y').equals(apply_family_policy(flipped, policy).drop('y'))
    unsupported = actions.filter(pl.col('qid') > 40)
    rejected, _ = lock_family_policy(unsupported, base, cal, search)
    assert not rejected['enabled']
    assert apply_family_policy(actions, rejected).is_empty()


def test_raw_calibration_support_below_locked_threshold_does_not_enable_family():
    cal, search, base, actions = calibration_fixture()


    actions = actions.with_columns(pl.when(pl.col('qid') <= 20).then(.85)
                                   .otherwise(pl.col('score')).alias('score'))
    policy, report = lock_family_policy(actions, base, cal, search)
    evidence = report['families']['addressed']['calibration']
    assert evidence['actions'] == 20
    assert evidence['selected_actions'] == 0
    assert not policy['families']['addressed']['enabled']
    assert not policy['enabled']
    assert apply_family_policy(actions, policy).is_empty()

from copy import deepcopy
import json
from pathlib import Path
import sys

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from latest_fusion.drop_review import (INCUMBENT_SHA, STRENGTH_SHA, combined_guard,
    evaluate_arms, fixed_arms, intersection, parse_sources)


def population():
    base = pl.DataFrame({'qid': [1, 2], 'tid': [10, 20], 'p': [.9, .95],
                         'y': [0, 1], 'own': [-1, 2]})
    strength = pl.DataFrame({'qid': [5, 2, 7], 'tid': [10, 20, 70],
                             'p': [.99, .99, .99], 'y': [0, 1, 1], 'own': [-1, 2, 7]})
    adds = pl.DataFrame({'qid': [3, 2], 'tid': [30, 21], 'p': [.999, .999],
                         'y': [1, 1], 'own': [3, 2]})
    anchors = pl.DataFrame({'qid': [1, 2, 3, 4], 'deg': [0, 2, 1, 0],
                            'co': ['us', 'india', 'us', 'india']})
    return base, strength, adds, anchors


def test_exact_pair_intersection_preserves_scores_and_cannot_introduce_strength_additions():
    base, strength, adds, anchors = population()
    dropped = intersection(base, strength)
    assert dropped.equals(base.filter(pl.col('qid') == 2))
    assert dropped['p'].to_list() == [.95]

    assert 10 not in dropped['tid']
    assert 70 not in dropped['tid']
    assert base.height == 2 and strength.height == 3


def test_identical_additions_use_original_occupancy_and_no_resurrection():
    base, strength, adds, anchors = population()
    arms = fixed_arms(base, strength, adds)
    assert arms['A'].height == 4 and arms['DA'].height == 3
    assert arms['A'].join(base.select('tid'), on='tid', how='anti').equals(
        arms['DA'].join(arms['D'].select('tid'), on='tid', how='anti'))
    resurrect = adds.head(1).with_columns(pl.lit(10, pl.Int64).alias('tid'))
    with pytest.raises(ValueError, match='original baseline occupancy'):
        fixed_arms(base, strength, resurrect)


def test_country_effects_and_correct_empty_entities_use_actual_macro():
    base, strength, adds, anchors = population()
    res = evaluate_arms(fixed_arms(base, strength, adds), {'structured_check': anchors})['structured_check']
    assert res['D']['effects_vs_B']['FPremoved'] == 1
    assert res['D']['effects_vs_B']['TPremoved'] == 0
    assert res['DA']['effects_vs_D']['TPadded'] == 2
    assert res['DA']['effects_vs_D']['FPadded'] == 0
    assert res['DA']['effects_vs_D']['country']['india']['TPadded'] == 1
    assert res['DA']['metrics']['overall']['macro_f05'] == 1.
    assert res['D']['metrics']['country']['us']['macro_f05'] == .5
    assert combined_guard(res)['passed']

    damaged = deepcopy(res)
    damaged['DA']['metrics']['country']['india']['pair_precision'] = .9
    guard = combined_guard(damaged)
    assert not guard['passed']
    assert 'india_pair_precision_nondecreasing' in guard['failure_reasons']


def sources():
    sel = {'name': 'competition63_1.0', 'model': 'competition63', 'strength': 1.0,
                'policy': {'rule': 'expected_f', 'threshold': .8}}
    inc = {'matching_sha256': INCUMBENT_SHA, 'selected': 'residual_0.5'}
    strength = {'matching_sha256': STRENGTH_SHA, 'locked_candidate': deepcopy(sel)}
    additions = {'promoted': False, 'selected': {'family': 'contrastive', 'head': 'treatment',
        'rule': {'enabled': True, 'min_alias': .99, 'min_margin': .05}}}
    return inc, strength, sel, additions


@pytest.mark.parametrize('fault', ['incumbent_hash', 'strength_hash', 'strength', 'floor', 'addition_cut'])
def test_saved_provenance_and_thresholds_are_required(fault):
    inc, strength, policy, additions = sources()
    if fault == 'incumbent_hash':
        inc['matching_sha256'] = 'wrong'
    elif fault == 'strength_hash':
        strength['matching_sha256'] = 'wrong'
    elif fault == 'strength':
        policy['strength'] = 1.25
        strength['locked_candidate'] = deepcopy(policy)
    elif fault == 'floor':
        policy['policy']['threshold'] = .65
        strength['locked_candidate'] = deepcopy(policy)
    else:
        additions['selected']['rule']['min_alias'] = .995
    with pytest.raises(ValueError):
        parse_sources(inc, strength, policy, additions)


def test_actual_completed_source_reports_parse_without_mutation():
    workspace = ROOT.parent
    paths = [workspace/'artifacts/latest_fusion_results/report.json',
             workspace/'artifacts/competition_runs/strength/report.json',
             workspace/'artifacts/competition_runs/strength/selected_policy.json',
             workspace/'artifacts/structured_runs/additions/experiment/report.json']
    if not all(path.is_file() for path in paths):
        pytest.skip('Completed local source reports not present')
    reports = [json.loads(path.read_text()) for path in paths]
    original = deepcopy(reports)
    policy, sel = parse_sources(*reports)
    assert policy['strength'] == 1. and policy['policy']['threshold'] == .8
    assert sel['rule']['min_alias'] == .99
    assert reports == original

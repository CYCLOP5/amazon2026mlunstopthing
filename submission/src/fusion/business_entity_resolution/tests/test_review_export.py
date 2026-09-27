from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from latest_fusion.review_export import (INCUMBENT_SHA, require_uncertainty_only,
    assert_metric_replay, frozen_test_assignments)
from er.stack.pipeline import export


def source_report():
    guards = {key: True for key in ('gain', 'precision', 'recall', 'india_score', 'us_score',
        'overall_absolute_precision', 'india_absolute_precision', 'us_absolute_precision')}
    guards['positive_lower_bound'] = False
    comparison = {'incumbent': {'macro_f05': .99, 'pair_precision': .999, 'pair_recall': .97},
                  'candidate': {'macro_f05': .991, 'pair_precision': .9991, 'pair_recall': .971}}
    return {'promoted': False, 'submission_changed': False, 'matching_sha256': INCUMBENT_SHA,
        'check': {'passed': False, 'checks': guards, 'paired_gain': .0004,
                  'paired_standard_error': .0002, 'paired_family_interval': [-.0001, .0009],
                  **deepcopy(comparison), 'countries': {country: deepcopy(comparison) for country in ('india', 'us')}},
        'density_replay_check': {'passed': True, 'checks': {region+'_'+metric: True
            for region in ('overall', 'india', 'us') for metric in ('macro_f05', 'precision', 'recall')}},
        'locked_candidate': {'model': 'competition63', 'strength': .75,
            'policy': {'rule': 'top1_threshold', 'threshold': .5}, 'eligible': True,
            'search_checks': {'overall_gain': True, **{region+'_'+metric: True
                for region in ('overall', 'india', 'us') for metric in ('macro_f05', 'precision', 'recall')}}}}


def test_only_uncertainty_failure_can_be_exported():
    report = source_report()
    assert require_uncertainty_only(report) == report['locked_candidate']
    for guard in ('precision', 'recall', 'india_score'):
        bad = deepcopy(report); bad['check']['checks'][guard] = False
        with pytest.raises(ValueError, match='other than uncertainty'):
            require_uncertainty_only(bad)
    bad = deepcopy(report); bad['density_replay_check']['checks']['us_precision'] = False
    with pytest.raises(ValueError, match='density/country'):
        require_uncertainty_only(bad)
    bad = deepcopy(report); bad['locked_candidate']['eligible'] = False
    with pytest.raises(ValueError, match='eligible locked'):
        require_uncertainty_only(bad)
    for metric in ('pair_precision', 'pair_recall'):
        bad = deepcopy(report)
        bad['check']['countries']['us']['candidate'][metric] = .9
        with pytest.raises(ValueError, match='regressed'):
            require_uncertainty_only(bad)
    bad = deepcopy(report); del bad['locked_candidate']['search_checks']['india_precision']
    with pytest.raises(ValueError, match='eligible locked'):
        require_uncertainty_only(bad)
    bad = deepcopy(report); del bad['check']['countries']['india']
    with pytest.raises(ValueError, match='Incomplete'):
        require_uncertainty_only(bad)


def test_saved_metric_replay_rejects_different_policy_outcomes():
    saved = {'macro_f05': .991, 'pair_precision': .999, 'pair_recall': .98, 'pairs': 100}
    assert_metric_replay(saved, saved, 'locked candidate')
    for key in saved:
        changed = {**saved, key: saved[key]+.001}
        with pytest.raises(ValueError, match='replay mismatch'):
            assert_metric_replay(changed, saved, 'locked candidate')
    for got, exp in (({**saved, 'macro_f05': float('nan')}, saved),
                             (saved, {**saved, 'pair_precision': float('nan')})):
        with pytest.raises(ValueError, match='replay mismatch'):
            assert_metric_replay(got, exp, 'nonfinite candidate')


def test_real_tsv_export_keeps_incumbent_france_owner(tmp_path):
    refs = pl.DataFrame({'rid': [0, 1, 2, 3], 'eid': ['r0', 'r1', 'r2', 'r3'],
        'co': ['us', 'india', 'france', 'france']})
    targets = pl.DataFrame({'rid': [0, 1, 2], 'eid': ['t0', 't1', 't2']})
    frame = pl.DataFrame({'qid': [0, 1, 2, 3], 'tid': [0, 1, 2, 2],
        'co': ['us', 'india', 'france', 'france']})
    old = pl.DataFrame({'qid': [0, 2], 'tid': [0, 2], 'p': [.9, .8]})
    raw = np.array([.99, .99, .001, .999], dtype=np.float32)
    recipe = {'curves': {f'{country}|0': {'centres': [-12, 0, 12], 'posterior': [0, .5, 1]}
                        for country in ('us', 'india')}}
    frame = frame.with_columns(pl.lit(0).alias('seg'))
    policy = {'rule': 'top1_threshold', 'threshold': .5}
    accepted, _, delta = frozen_test_assignments(frame, raw, recipe, refs, old, policy, ['us', 'india'])
    assert accepted.filter(pl.col('tid') == 2)['qid'].to_list() == [2]
    assert delta['added_pairs'] == 1 and delta['removed_pairs'] == 0
    assert delta['added_by_country'].get('france', 0) == 0
    res = export(frame.with_columns(pl.Series('p', raw)), 'p', policy['rule'], .5,
                    refs, targets, str(tmp_path/'output'), accepted=accepted)
    assert res['matches'] == 3 and res['candidate_pairs'] == 4
    rows = pl.read_csv(tmp_path/'output/matching_results.tsv', separator='\t', infer_schema=False)
    assert rows.filter(pl.col('source1_entity_id') == 'r2')['matched_entity_ids'].item() == 't2'
    assert rows.filter(pl.col('source1_entity_id') == 'r3')['matched_entity_ids'].fill_null('').item() == ''

'saved-trial provenance guards and exact threaded decoder parity'
from copy import deepcopy
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]


def trial_report():
    sel = {'name': 'competition63_1.0', 'model': 'competition63', 'strength': 1.0,
                'policy': {'rule': 'expected_f', 'threshold': .8}, 'eligible': True}
    alternate = {'name': 'competition63_1.25', 'model': 'competition63', 'strength': 1.25,
                 'policy': {'rule': 'expected_f', 'threshold': .65}, 'eligible': False,
                 'search_checks': {'overall_precision': False},
                 'search': {'overall': {'macro_f05': .993, 'pair_precision': .9989,
                                       'pair_recall': .98, 'pairs': 110}}}
    return {'promoted': True, 'submission_changed': True,
            'locked_candidate': sel, 'trials': [deepcopy(sel), alternate]}


def test_exact_saved_tradeoff_trial_can_keep_original_precision_failure():
    from latest_fusion.tradeoff_review import locked_trial
    report = trial_report()
    alternate = locked_trial(report, 'competition63_1.25')
    assert alternate == report['trials'][1]
    assert alternate['eligible'] is False
    assert alternate['policy']['threshold'] == .65
    assert alternate['strength'] == 1.25
    assert report == trial_report()


@pytest.mark.parametrize('fault', ['unpromoted', 'unchanged', 'missing', 'duplicate', 'different_checkpoint', 'selected'])
def test_tradeoff_trial_provenance_failures_are_rejected(fault):
    from latest_fusion.tradeoff_review import locked_trial
    report = trial_report()
    name = 'competition63_1.25'
    if fault == 'unpromoted':
        report['promoted'] = False
    elif fault == 'unchanged':
        report['submission_changed'] = False
    elif fault == 'missing':
        name = 'competition63_1.250'
    elif fault == 'duplicate':
        report['trials'].append(deepcopy(report['trials'][1]))
    elif fault == 'different_checkpoint':
        report['trials'][1]['model'] = 'competition31'
    elif fault == 'selected':
        name = 'competition63_1.0'
    with pytest.raises(ValueError):
        locked_trial(report, name)


def test_expected_f_decoder_threads_preserve_ties_and_large_groups(monkeypatch):
    from latest_fusion import tuning, decoder
    lengths = np.array([1, 2, 4, 8, 16, 32, 64, 70])
    qid = np.repeat(np.arange(len(lengths)), lengths)
    tid = np.arange(len(qid))
    prob = np.linspace(.55, .999, len(qid))

    frame = pl.DataFrame({'qid': np.r_[qid, 1], 'tid': np.r_[tid, 0],
        'p': np.r_[prob, prob[0]], 'raw': np.r_[prob, .9],
        'y': np.ones(len(qid)+1, dtype=np.uint8)}).sample(fraction=1, shuffle=True, seed=13)
    calls = []
    original = decoder.choose

    def capture(*args, **kwargs):
        calls.append(kwargs['threads'])
        return original(*args, **kwargs)

    monkeypatch.setattr(decoder, 'choose', capture)
    monkeypatch.setattr(tuning.os, 'cpu_count', lambda: 64)
    monkeypatch.setenv('ER_DECODER_THREADS', '1')
    one = tuning.apply_policy(frame, {'rule': 'expected_f', 'threshold': .05})
    monkeypatch.setenv('ER_DECODER_THREADS', '4')
    four = tuning.apply_policy(frame, {'rule': 'expected_f', 'threshold': .05})
    assert calls == [1, 4]
    assert one.equals(four)
    assert one.filter(pl.col('tid') == 0)['qid'].to_list() == [1]
    assert one.filter(pl.col('qid') == 7).height > 64


def test_real_saved_model_calibration_replay_and_frozen_france_export(tmp_path, monkeypatch):
    from test_latest_fusion_pipeline import test_complete_frozen_fusion
    from latest_fusion import tune_pipeline, tradeoff_review
    from latest_fusion.pipeline import sha256
    test_complete_frozen_fusion(tmp_path, monkeypatch)
    incumbent = tmp_path/'result'
    prev = json.loads((incumbent/'report.json').read_text())
    prev['selected'] = 'residual_0.5'
    (incumbent/'report.json').write_text(json.dumps(prev))
    shutil.copyfile(tmp_path/'recipe.json', tmp_path/'latest_fusion_calibration.json')
    cfg = {'incumbent_method': 'residual_0.5', 'incumbent_matching_sha256': prev['matching_sha256'],
        'incumbent_job': 'fixture', 'incumbent_leaderboard_user_reported': .98805,
        'strengths': [.5, .65], 'incumbent_strengths': [.5, .65], 'heads': [],
        'expected_f_floors': [.05, .8], 'minimum_gain': -1,
        'max_precision_loss': .00015, 'max_recall_loss': .0001}
    config = tmp_path/'tuning.json'; config.write_text(json.dumps(cfg))
    original_promotion = tune_pipeline.promotion

    def export_gate(*args, **kwargs):


        res = original_promotion(*args, **kwargs)
        res['passed'] = True
        return res

    monkeypatch.setattr(tune_pipeline, 'promotion', export_gate)
    completed = tmp_path/'selected'
    src = tune_pipeline.run(tmp_path/'data', tmp_path/'prepared/prepared-train.parquet',
        tmp_path/'prepared/prepared-test.parquet', incumbent, tmp_path/'metadata.json', config, completed)
    assert src['promoted'] and src['locked_candidate']['name'] == 'incumbent_0.5'


    src['submission_changed'] = True
    (completed/'report.json').write_text(json.dumps(src))
    experimental = tmp_path/'experimental'
    res = tradeoff_review.run(tmp_path/'data', tmp_path/'prepared/prepared-train.parquet',
        tmp_path/'prepared/prepared-test.parquet', completed, tmp_path/'metadata.json',
        experimental, 'incumbent_0.65', src['matching_sha256'])
    assert res['experimental'] and not res['promoted']
    assert not res['training_performed'] and not res['new_threshold_search']
    assert res['reviewed_trial']['strength'] == .65
    assert res['validation']['source1'] == 240
    assert res['changes_vs_current']['added_by_country'].get('france', 0) == 0
    assert res['changes_vs_current']['removed_by_country'].get('france', 0) == 0
    refs = pl.read_parquet(tmp_path/'data/test/ref.parquet')
    france = refs.filter(pl.col('co') == 'france')['rid']
    before = pl.read_parquet(completed/'accepted_test.parquet').filter(pl.col('qid').is_in(france.implode())).sort('qid', 'tid')
    after = pl.read_parquet(experimental/'accepted_test.parquet').filter(pl.col('qid').is_in(france.implode())).sort('qid', 'tid')
    assert before.equals(after)
    assert sha256(completed/'output/matching_results.tsv') == src['matching_sha256']
    with pytest.raises(ValueError, match='comparison submission'):
        tradeoff_review.run(tmp_path/'data', tmp_path/'prepared/prepared-train.parquet',
            tmp_path/'prepared/prepared-test.parquet', completed, tmp_path/'metadata.json',
            tmp_path/'rejected', 'incumbent_0.65', 'wrong')

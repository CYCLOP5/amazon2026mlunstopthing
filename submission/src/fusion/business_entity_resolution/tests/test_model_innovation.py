'end-to-end fresh-model experiments, full owner decoding and promotion guards'
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from latest_fusion import innovation_models as models
from latest_fusion import model_innovation as pipeline
from latest_fusion.tuning import top_pairs, apply_policy
from test_structured_pipeline import make_fixture


@pytest.mark.parametrize('mode', pipeline.MODES)
def test_real_models_full_pipeline_validate_and_preserve_benchmark(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(models, 'ROUNDS', dict.fromkeys(models.BACKENDS, 4))
    data, prepared, inc, checksum = make_fixture(tmp_path)
    original = (inc/'output'/'matching_results.tsv').read_bytes()
    report = pipeline.run(data, prepared/'prepared-train.parquet', prepared/'prepared-test.parquet',
                          inc, tmp_path/'result', mode, expected_sha=checksum)
    assert report['incumbent_sha256'] == checksum
    assert (inc/'output'/'matching_results.tsv').read_bytes() == original
    assert report['validation']['source1'] == 1200
    assert report['validation']['candidate_pairs'] >= 3600
    assert report['protocol']['target_overlap'] == 0
    assert report['protocol']['calibration_check_target_overlap'] == 0
    assert report['selection']['locked_before_check'] is True
    assert report['submission_status'] == 'experimental_candidate_not_promoted_do_not_submit'
    assert (tmp_path/'result'/'models'/'model_report.json').exists()
    assert (tmp_path/'result'/'output'/'export.json').exists()
    assert (tmp_path/'result'/'test_predictions.parquet').exists()
    if mode == 'gated_ensemble':
        assert len(report['model']['cross_fitting']) == 3
        assert report['model']['expert_families'] == ['lightgbm', 'xgboost', 'catboost']
        assert (tmp_path/'result'/'models'/'oof_predictions.parquet').exists()
    old = pl.read_parquet(inc/'accepted_test.parquet').select('qid', 'tid', 'p')
    new = pl.read_parquet(tmp_path/'result'/'accepted_test.parquet')
    france = pl.read_parquet(data/'test'/'ref.parquet').filter(pl.col('co') == 'france').select(pl.col('rid').alias('qid'))
    assert old.join(france, on='qid', how='semi').sort('qid').equals(
        new.join(france, on='qid', how='semi').sort('qid'))


def test_full_model_can_reverse_owner_and_reject_existing_match():
    frame = pl.DataFrame({'qid': [0, 1, 2], 'tid': [7, 7, 8], 'co': ['us']*3,
        'baseline_p': [.999, .01, .99]})
    scored = pipeline.score_frame(frame, np.array([.01, .999, .001]), 1.)
    accepted = apply_policy(top_pairs(scored), {'rule': 'top1_threshold', 'threshold': .5})
    assert accepted.select('qid', 'tid').rows() == [(1, 7)]


def test_novel_edges_do_not_inherit_low_incumbent_offset():
    frame = pl.DataFrame({'qid': [0, 1], 'tid': [7, 8], 'co': ['us']*2,
        'baseline_p': [1e-4, 1e-4], '_innovation_new': [True, False]})
    scored = pipeline.score_frame(frame, np.array([.99, .99]), .5)
    assert scored['p'][0] == pytest.approx(.99)
    assert scored['p'][1] < .5


def test_precision_regression_rejected_despite_recall_gain():
    owners = pl.DataFrame({'qid': [0, 1], 'co': ['us', 'india'], 'deg': [2, 1]})
    base = pl.DataFrame({'qid': [0, 1], 'tid': [7, 8], 'y': [1, 1], 'p': [.999, .999]})
    cand = pl.concat([base, pl.DataFrame({'qid': [0, 1], 'tid': [9, 10], 'y': [1, 0], 'p': [.99, .99]})])
    evidence = pipeline.preservation(cand, base, owners)
    assert evidence['after']['overall']['pair_recall'] > evidence['before']['overall']['pair_recall']
    assert not evidence['passed']
    assert not evidence['checks']['overall_pair_precision']
    assert not evidence['checks']['india_pair_precision']


def test_score_input_validation():
    frame = pl.DataFrame({'qid': [0], 'tid': [7], 'co': ['us'], 'baseline_p': [.5]})
    for invalid in (np.array([np.nan]), np.array([-1.]), np.array([1.1]), np.array([])):
        with pytest.raises(ValueError, match='Invalid innovation'):
            pipeline.score_frame(frame, invalid, 1.)

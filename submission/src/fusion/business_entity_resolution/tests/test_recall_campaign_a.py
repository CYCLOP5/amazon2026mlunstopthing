from pathlib import Path
import json
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from latest_fusion import recall_campaign_a as campaign


FEATURES = ['baseline_p', 'newest_logit', 'name_jaccard', 'house_equal']


def fixture(n=600):
    rng = np.random.default_rng(27)
    truth = (np.arange(n) % 2 == 0).astype(np.int8)
    signal = truth + rng.normal(0, .15, n)
    return pl.DataFrame({'qid': np.arange(n), 'tid': np.arange(n)//2,
        'y': truth, 'own': np.where(truth, np.arange(n), -1), 'co': ['us']*n,
        'baseline_p': np.clip(signal, .001, .999), 'newest_logit': signal*3,
        'name_jaccard': np.clip(truth + rng.normal(0, .2, n), 0, 1),
        'house_equal': truth.astype(float)})


def test_honest_leaf_evidence_is_disjoint_and_cannot_change_tree(tmp_path, monkeypatch):
    monkeypatch.setenv('ER_THREADS', '1')
    frame = fixture()
    evidence = campaign._hash_targets(frame['tid'].to_numpy()) % 5 >= 3

    assert np.all(evidence[::2] == evidence[1::2])
    first = campaign.fit('honest_leaf', frame, FEATURES, tmp_path/'first',
                         _model_params={'min_data_in_leaf': 5})
    changed = frame.with_columns(pl.Series('y', np.where(evidence, 1-frame['y'].to_numpy(), frame['y'].to_numpy())))
    second = campaign.fit('honest_leaf', changed, FEATURES, tmp_path/'second',
                          _model_params={'min_data_in_leaf': 5})
    assert first['models']['tree'].model_to_string() == second['models']['tree'].model_to_string()
    assert not np.array_equal(first['leaf_scores'], second['leaf_scores'])
    assert first['report']['structure_rows'] + first['report']['evidence_rows'] == frame.height
    assert first['report']['target_overlap'] == 0
    assert all(e['lower'] < 1 for e in first['report']['leaf_evidence'])


@pytest.mark.parametrize('method', campaign.METHODS)
def test_real_models_fit_then_predict_without_reading_outcome_labels(method, tmp_path, monkeypatch):
    monkeypatch.setenv('ER_THREADS', '1')
    frame = fixture()
    bundle = campaign.fit(method, frame, FEATURES, tmp_path/method,
                           _rounds=5, _model_params={'min_data_in_leaf': 5})
    assert bundle['report']['available']
    a = campaign.predict(bundle, frame)
    b = campaign.predict(bundle, frame.drop('y', 'own', 'co'))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (frame.height,) and a.dtype == np.float32
    assert np.isfinite(a).all() and ((0 <= a) & (a <= 1)).all()
    assert a[frame['y'].to_numpy() == 1].mean() > a[frame['y'].to_numpy() == 0].mean()
    report = json.loads((tmp_path/method/'report.json').read_text())
    assert not report['calibration_or_check_access']


def test_wilson_abstains_without_support_and_keeps_finite_sample_uncertainty():
    lower = campaign.wilson_lower([0, 10, 1000, 900], [0, 10, 1000, 1000])
    assert lower[0] == 0
    assert 0 < lower[1] < lower[2] < 1
    assert lower[3] < .9
    with pytest.raises(ValueError, match='Invalid leaf'):
        campaign.wilson_lower([2], [1])


def test_crossview_conservative_envelope_and_disjoint_features(monkeypatch):
    views = campaign._views(FEATURES)
    assert set(views['scores']).isdisjoint(views['structure'])
    assert set(views['scores'] + views['structure']) == set(FEATURES)
    bundle = {'method': 'crossview_consensus', 'report': {'available': True},
              'views': views, 'models': {'scores': 'a', 'structure': 'b'}}
    monkeypatch.setattr(campaign, '_predict_model', lambda model, *args:
                       np.array([.9, .7]) if model == 'a' else np.array([.2, .99]))
    np.testing.assert_allclose(campaign.predict(bundle, pl.DataFrame({'x': [0, 1]})), [.2, .7])


def test_negative_mining_preserves_positive_and_sampling_weights(tmp_path, monkeypatch):
    frame = fixture(200)
    pilot = np.where(np.arange(frame.height) % 4 == 1, .02, .001)
    calls = []
    def train(rows, features, path, rounds, **kwargs):
        calls.append((rows, kwargs.get('weights')))
        return object()
    monkeypatch.setattr(campaign, '_fit_model', train)
    monkeypatch.setattr(campaign, '_predict_model', lambda *args, **kwargs: pilot)
    res = campaign.fit('hard_negative', frame, FEATURES, tmp_path)
    sel, weights = calls[1]
    assert sel.filter(pl.col('y') == 1).height == frame.filter(pl.col('y') == 1).height
    index = sel['qid'].to_numpy()
    y = sel['y'].to_numpy()
    np.testing.assert_array_equal(weights[y == 1], np.ones((y == 1).sum()))
    np.testing.assert_array_equal(weights[(y == 0) & (pilot[index] >= .01)],
                                  np.full(((y == 0) & (pilot[index] >= .01)).sum(), 3.))
    np.testing.assert_array_equal(weights[(y == 0) & (pilot[index] < .01)],
                                  np.full(((y == 0) & (pilot[index] < .01)).sum(), 20.))
    assert res['report']['hard_negative_rows'] == 50


def test_forbidden_features_and_sparse_fit_fall_back(tmp_path):
    frame = fixture()
    with pytest.raises(ValueError, match='cannot be model features'):
        campaign.fit('hard_negative', frame, FEATURES+['own'], tmp_path)
    with pytest.raises(ValueError, match='India/US only'):
        campaign.fit('hard_negative', frame.with_columns(pl.lit('france').alias('co')), FEATURES, tmp_path)
    unavailable = campaign.fit('hard_negative', frame.filter(pl.col('y') == 1), FEATURES, tmp_path)
    assert not unavailable['report']['available']
    np.testing.assert_array_equal(campaign.predict(unavailable, frame), np.zeros(frame.height))


@pytest.mark.parametrize('method', campaign.METHODS)
def test_full_campaign_replay_fallback_validates(tmp_path, monkeypatch, method):
    from latest_fusion.structured_pipeline import run
    from test_structured_pipeline import make_fixture

    monkeypatch.setenv('ER_THREADS', '2')
    data, prepared, incumbent, checksum = make_fixture(tmp_path)
    report = run(data, prepared/'prepared-train.parquet', prepared/'prepared-test.parquet',
                 incumbent, tmp_path/'result', method, expected_sha=checksum)
    assert report['matching_sha256'] == checksum
    assert report['changed'] is False
    assert report['validation'] == {'source1': 1200, 'matches': 1200, 'candidate_pairs': 3600}
    assert report['protocol']['target_overlap'] == 0
    assert report['protocol']['calibration_check_target_overlap'] == 0
    assert report['experiment']['campaign_size'] == 8
    assert (tmp_path/'result'/'experiment'/'locked_rule.json').exists()

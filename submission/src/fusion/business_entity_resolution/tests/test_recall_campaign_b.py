from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from latest_fusion import recall_campaign_b as campaign


FEATURES = ['baseline_p', 'name_jaccard', 'house_equal', 'gate_logit']


def population(n=900):
    rng = np.random.default_rng(521)
    qid = np.arange(n) % 30
    state = np.arange(n) % 3
    y = (state == 0).astype(np.int8)
    own = np.where(y == 1, qid, np.where(state == 1, -1, (qid + 1) % 30))
    strength = np.clip(.15 + .65*y + rng.normal(0, .18, n), 0, 1)
    frame = pl.DataFrame({'qid': qid, 'tid': np.arange(n), 'co': ['us', 'india']*(n//2),
        'y': y, 'own': own, 'baseline_p': strength, 'name_jaccard': strength,
        'house_equal': (strength > .6).astype(np.int8), 'gate_logit': strength * 3 - 1.5})
    targets = pl.DataFrame({'rid': np.arange(n), 'co': frame['co'],
        'nm': [f'shop group {x}' for x in qid], 'ad': [f'{x} main road' for x in qid]})
    base = frame.filter(pl.col('tid') % 4 == 0).select('qid', 'tid').with_columns(pl.lit(.999).alias('p'))
    return frame, targets, base


def fit(method, frame, targets, baseline, path):
    return campaign.fit(method, frame, FEATURES, path, baseline=baseline, targets=targets,
        s2_count=450, _rounds=15, _minimum_class=5,
        _model_params={'min_data_in_leaf': 5, 'num_threads': 1, 'lambda_l2': 1.})


@pytest.mark.parametrize('method', campaign.METHODS)
def test_real_fit_finite_predictions_and_poisoned_inference_labels(method, tmp_path):
    frame, targets, base = population()
    bundle = fit(method, frame, targets, base, tmp_path/method)
    assert bundle['available'], bundle['report']
    probs = campaign.predict(bundle, frame, baseline=base, targets=targets, s2_count=450)
    assert probs.shape == (frame.height,)
    assert probs.dtype == np.float32
    assert np.isfinite(probs).all()
    assert probs.min() >= 0 and probs.max() <= 1
    assert np.ptp(probs) > .1
    poison = frame.with_columns(pl.lit('poison').alias('y'), pl.lit(None).alias('own'),
                                pl.lit(999).alias('deg'), pl.lit(-1).alias('fold'))
    poisoned_baseline = base.with_columns(pl.lit('poison').alias('y'), pl.lit(None).alias('own'))
    poisoned_targets = targets.with_columns(pl.lit('poison').alias('own'), pl.lit(-1).alias('deg'))
    replay = campaign.predict(bundle, poison, baseline=poisoned_baseline,
                              targets=poisoned_targets, s2_count=450)
    np.testing.assert_array_equal(probs, replay)
    assert not campaign.FORBIDDEN.intersection(bundle['features'])
    assert len(list((tmp_path/method).glob('*model.txt'))) >= 1


def test_sibling_excludes_self_before_capping_and_preserves_input_order():
    targets = pl.DataFrame({'rid': [0, 1, 2, 3, 4, 5], 'co': ['us']*6,
        'nm': ['alpha', 'beta', 'gamma', 'delta', 'alpha', 'orphan'],
        'ad': ['1 first', '2 second', '3 third', '4 fourth', '1 first', '5 fifth']})
    base = pl.DataFrame({'qid': [7, 7, 7, 7], 'tid': [0, 1, 2, 3], 'p': [.999, .998, .997, .996]})
    frame = pl.DataFrame({'qid': [7, 99, 7, 7], 'tid': [4, 5, 0, 3]})
    out = campaign.sibling_features(frame, base, targets, 4, chunk_size=2)
    assert out.select('qid', 'tid').equals(frame)

    assert out['sb_name_exact'].to_list() == [1., 0., 0., 0.]
    assert out['sb_joint_exact'].to_list() == [1., 0., 0., 0.]
    np.testing.assert_allclose(out['sb_count_log'], [np.log(4), 0., np.log(4), np.log(4)])
    assert out['sb_cross_source_name_max'][0] == 1.
    assert out['sb_cross_source_address'][0] == 1.

    assert campaign.sibling_features(frame, base, targets, 4, chunk_size=50).equals(out)


def test_sibling_missing_scores_or_duplicate_occupancy_fail_closed():
    frame, targets, base = population()
    with pytest.raises(ValueError, match='duplicate'):
        campaign.sibling_features(frame, pl.concat([base, base.head(1)]), targets, 450)
    with pytest.raises(ValueError, match='score'):
        campaign.sibling_features(frame, base.with_columns(pl.lit(float('nan')).alias('p')), targets, 450)


def test_source_experts_fit_disjoint_sources_and_exclude_french_labels(tmp_path):
    frame, targets, base = population()
    a = fit('source_corroboration', frame, targets, base, tmp_path/'a')

    changed = frame.with_columns(
        pl.when(pl.col('tid') >= 450).then(1-pl.col('y')).otherwise(pl.col('y')).alias('y'))
    changed = changed.with_columns(pl.when(pl.col('tid') >= 450).then(
        pl.when(pl.col('y') == 1).then(pl.col('qid')).otherwise(-1)).otherwise(pl.col('own')).alias('own'))
    french = changed.head(15).with_columns(pl.lit('france').alias('co'), pl.lit(None).alias('own'))
    b = fit('source_corroboration', pl.concat([changed, french], how='vertical_relaxed'),
            targets, base, tmp_path/'b')
    assert a['models']['source2'].model_to_string() == b['models']['source2'].model_to_string()
    assert a['models']['source3'].model_to_string() != b['models']['source3'].model_to_string()
    assert b['report']['excluded_other_country_rows'] == 15
    assert a['report']['heads']['source2']['rows'] == 450
    assert a['report']['heads']['source3']['rows'] == 450


def test_insufficient_expert_support_abstains_without_using_other_source(tmp_path):
    frame, targets, base = population()
    bundle = fit('source_corroboration', frame.filter(pl.col('tid') < 450), targets, base, tmp_path)
    assert not bundle['available']
    assert not bundle['report']['heads']['source3']['available']
    np.testing.assert_array_equal(campaign.predict(bundle, frame), np.zeros(frame.height, np.float32))


def test_feature_label_and_missing_boundary_rejected(tmp_path):
    frame, targets, base = population()
    with pytest.raises(ValueError, match='label/ID free'):
        campaign.fit('orphan_factor', frame, FEATURES+['own'], tmp_path)
    with pytest.raises(ValueError, match='s2_count'):
        campaign.fit('source_corroboration', frame, FEATURES, tmp_path)


@pytest.mark.parametrize('method', campaign.METHODS)
def test_full_structured_campaign_replay_fallback_validates(tmp_path, monkeypatch, method):
    from test_structured_pipeline import make_fixture
    from latest_fusion.structured_pipeline import run

    monkeypatch.setattr(campaign, 'ROUNDS', 10)
    data, prepared, incumbent, checksum = make_fixture(tmp_path)
    output = tmp_path/'result'
    report = run(data, prepared/'prepared-train.parquet', prepared/'prepared-test.parquet',
                 incumbent, output, method, expected_sha=checksum)
    assert report['matching_sha256'] == checksum
    assert report['changed'] is False
    assert report['validation'] == {'source1': 1200, 'matches': 1200, 'candidate_pairs': 3600}
    assert report['protocol']['target_overlap'] == 0
    assert report['protocol']['calibration_check_target_overlap'] == 0
    assert report['experiment']['campaign_size'] == 8
    assert report['experiment']['model']['method'] == method
    assert (output/'experiment'/'locked_rule.json').is_file()
    assert (output/'output'/'matching_results.tsv').read_bytes() == (
        incumbent/'output'/'matching_results.tsv').read_bytes()

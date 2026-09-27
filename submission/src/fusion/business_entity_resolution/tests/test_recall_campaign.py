'recall campaign fitting, global competition, frozen checks and full export'
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from latest_fusion import recall_campaign_c as models
from latest_fusion import recall_campaign_experiment as campaign
from latest_fusion.structured_pipeline import run
from test_structured_pipeline import make_fixture


def fitting_rows():
    n = 1000
    evidence = np.tile(np.linspace(0., 1., 250), 4)
    return pl.DataFrame({'qid': np.arange(n), 'tid': np.arange(n),
        'co': ['us']*500+['india']*500, 'y': (evidence > .5).astype(np.uint8),
        'baseline_p': evidence, 'house_conflict': np.zeros(n), 'unconstrained': np.zeros(n)})


@pytest.mark.parametrize('method', models.METHODS)
def test_real_heads_label_free_and_country_complete(tmp_path, monkeypatch, method):
    monkeypatch.setattr(models, 'ROUNDS', 30)
    frame = fitting_rows()
    ff = ['baseline_p', 'house_conflict', 'unconstrained']
    bundle = models.fit(method, frame, ff, tmp_path)
    first = models.predict(bundle, frame)
    poisoned = frame.with_columns((1-pl.col('y')).alias('y'), pl.lit(-1).alias('own'),
                                  pl.lit('france').alias('co'))
    np.testing.assert_array_equal(first, models.predict(bundle, poisoned))
    assert bundle['report']['available']
    assert np.isfinite(first).all() and (first >= 0).all() and (first <= 1).all()
    if method == 'monotone_verifier':
        assert (np.diff(first[:250]) >= -1e-7).all()
    else:
        assert set(bundle['models']) == {'us', 'india'}
        assert all(head['rows'] == 500 for head in bundle['report']['heads'].values())


def test_forbidden_truth_features_and_missing_country_support(tmp_path):
    frame = fitting_rows()
    with pytest.raises(ValueError, match='label-free'):
        models.fit('monotone_verifier', frame, ['baseline_p', 'y'], tmp_path)
    bundle = models.fit('country_consensus', frame.filter(pl.col('co') == 'us'), ['baseline_p'], tmp_path)
    assert not bundle['report']['available']
    assert not models.predict(bundle, frame).any()


@pytest.mark.parametrize('method', models.METHODS)
def test_full_campaign_replay_fallback_validates(tmp_path, monkeypatch, method):
    monkeypatch.setattr(models, 'ROUNDS', 10)
    data, prepared, incumbent, checksum = make_fixture(tmp_path)
    report = run(data, prepared/'prepared-train.parquet', prepared/'prepared-test.parquet',
                 incumbent, tmp_path/'result', method, expected_sha=checksum)
    assert report['matching_sha256'] == checksum
    assert report['changed'] is False
    assert report['validation'] == {'source1': 1200, 'matches': 1200, 'candidate_pairs': 3600}
    assert report['experiment']['campaign_size'] == 8
    assert (tmp_path/'result'/'experiment'/'locked_rule.json').exists()


def action_fixture():
    n = 120
    refs = pl.DataFrame({'rid': np.arange(n), 'co': ['us', 'india']*(n//2), 'deg': [2]*n})
    qids = np.tile(np.arange(n), 3)
    frame = pl.DataFrame({'qid': qids, 'tid': np.arange(3*n),
        'co': [refs['co'][int(i)] for i in qids],
        'own': np.r_[np.arange(n), np.arange(n), np.full(n, -1)],
        'y': np.r_[np.ones(2*n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)],
        'fixture_signal': np.r_[np.full(2*n, .9999), np.full(n, .01)],
        'baseline_p': np.r_[np.full(n, .9999), np.full(2*n, .1)],
        'baseline_raw': np.r_[np.full(n, .9999), np.full(2*n, .1)]})
    baseline = frame.head(n).select('qid', 'tid', 'co', 'y', 'own',
        pl.col('baseline_raw').alias('raw'), pl.col('baseline_p').alias('p'))
    return refs, frame, baseline


@pytest.mark.parametrize('false_check', [False, True])
def test_calibration_lock_and_no_check_fallback(tmp_path, monkeypatch, false_check):
    refs, frame, baseline = action_fixture()
    if false_check:


        frame = frame.with_columns(pl.when(pl.col('tid') == 330).then(.99999)
                                    .otherwise(pl.col('fixture_signal')).alias('fixture_signal'))
    fake = SimpleNamespace(fit=lambda *a, **k: {'report': {'available': True}},
                           predict=lambda model, rows, **k: rows['fixture_signal'].to_numpy())
    monkeypatch.setattr(campaign, '_module', lambda method: fake)
    fitrefs, cal, check = refs[:40], refs[40:80], refs[80:]
    fit = frame.filter(pl.col('qid') < 40)
    bundle = campaign.run(frame, refs, None, fit, cal, check, baseline,
                          ['fixture_signal'], tmp_path, fit_refs=fitrefs, method='monotone_verifier')
    assert bundle['rule']['enabled']
    assert bundle['promoted'] is (not false_check)
    lock = json.loads((tmp_path/'locked_rule.json').read_text())
    assert lock['rule'] == bundle['rule']
    assert 'check' not in lock
    assert bundle['report']['check']['FPadded'] == int(false_check)

    testrefs = refs.with_columns(pl.when(pl.col('rid') < 10).then(pl.lit('france'))
                                 .otherwise(pl.col('co')).alias('co'))
    base = baseline.select('qid', 'tid', 'p')
    accepted, info = campaign.infer(bundle, frame.drop('y', 'own'), testrefs, None, base)
    assert base.join(accepted, on=['qid', 'tid'], how='anti').height == 0
    assert accepted['tid'].n_unique() == accepted.height
    assert info['additions'] == (0 if false_check else 110)


def test_scoring_retains_global_competitors_before_scope():
    frame = pl.DataFrame({'qid': [1, 2, 3], 'tid': [7, 8, 8], 'baseline_p': [.9, .5, .5],
                          'signal': [.9999, .99, .9999]})
    base = pl.DataFrame({'qid': [1], 'tid': [7], 'p': [.9]})
    fake = SimpleNamespace(predict=lambda model, rows, **kw: rows['signal'].to_numpy())
    scored = campaign.score_candidates(fake, {}, frame, base)
    assert scored['qid'].to_list() == [2, 3]
    anchors = pl.DataFrame({'qid': [2]})
    res = campaign.actions.propose_additions(scored, base, anchors,
        {'enabled': True, 'min_alias': .98, 'min_margin': .001})
    assert res.height == 0

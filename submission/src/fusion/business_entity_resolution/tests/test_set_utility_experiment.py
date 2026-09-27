from pathlib import Path
import json
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from latest_fusion import set_utility_experiment as experiment
from latest_fusion.structured_protocol import prepare_protocol, split_references


class FakeModel:
    def predict(self, values, **kwargs):

        return np.where((values[:, 0] == -1) & (values[:, 1] < .5), .1, -.1)


def fixture():
    count = 200
    refs = pl.DataFrame({'rid': range(count), 'nm': [f'Business {i}' for i in range(count)],
        'ad': ['road'] * count, 'co': ['india' if i % 3 else 'us' for i in range(count)],
        'fold': [1] * count, 'deg': [i % 2 for i in range(count)]})
    frame = pl.DataFrame({'qid': range(count), 'tid': range(count),
        'own': [i if i % 2 else -1 for i in range(count)], 'y': [i % 2 for i in range(count)],
        'baseline_p': [.9 if i % 2 else .2 for i in range(count)],
        'baseline_raw': [.9 if i % 2 else .2 for i in range(count)]})
    assigned_fit, _, assigned_check, _ = split_references(refs)
    unsafe = assigned_fit.filter(pl.col('deg') == 0)['rid'][0]
    held = assigned_check['rid'][0]
    extra = frame.filter(pl.col('qid') == held).with_columns(pl.lit(unsafe, dtype=pl.Int64).alias('qid'),
        pl.lit(0, dtype=pl.Int64).alias('y'), pl.lit(.01).alias('baseline_p'), pl.lit(.01).alias('baseline_raw'))
    frame = pl.concat([frame, extra])
    fit, cal, check, fitframe, _ = prepare_protocol(refs, frame)
    base = frame.filter(pl.col('qid') == pl.col('tid')).select('qid', 'tid', pl.col('baseline_p').alias('p'), 'y')
    return refs, frame, fit, cal, check, fitframe, base, unsafe


def install_fake(monkeypatch):
    monkeypatch.setattr(experiment, 'MIN_FIT', 1)
    def fake_fit(actions, features, path):
        path.write_text('fixed fake model')
        return FakeModel()
    monkeypatch.setattr(experiment, '_fit', fake_fit)


def test_run_fits_exact_utility_labels_excludes_whole_unsafe_owners_and_persists(monkeypatch, tmp_path):
    refs, frame, fit, cal, check, fitframe, base, unsafe = fixture()
    seen = []
    monkeypatch.setattr(experiment, 'MIN_FIT', 1)
    def inspect_fit(actions, features, path):
        assert unsafe not in actions['qid'].to_list()
        assert set(actions['utility_delta'].to_list()) == {-1., 1.}
        assert not set(features) & {'qid', 'tid', 'deg', 'y', 'own', 'co', 'utility_delta'}
        seen.append(features)
        path.write_text('persisted model')
        return FakeModel()
    monkeypatch.setattr(experiment, '_fit', inspect_fit)
    bundle = experiment.run(frame, refs, pl.DataFrame(), fitframe, cal, check, base,
                            ['baseline_p'], tmp_path, fit_refs=fit)
    assert bundle['promoted'] and len(seen) == 2
    assert len(seen[1]) == len(seen[0])+12
    assert bundle['report']['isolation']['fit_owners_excluded_for_protected_target_incidence'] >= 1
    for head in bundle['report']['heads'].values():
        assert [entry['threshold'] for entry in head['calibration']['evaluated']] == experiment.THRESHOLDS
        assert head['check']['passed']
        assert head['check']['guards']['india_macro_f05_nondecreasing']
        assert head['check']['guards']['us_macro_f05_nondecreasing']
    assert (tmp_path/'control_model.txt').exists() and (tmp_path/'treatment_model.txt').exists()
    assert json.loads((tmp_path/'report.json').read_text())['promoted']


def test_inference_keeps_france_and_ignores_test_labels_and_degrees(monkeypatch, tmp_path):
    refs, frame, fit, cal, check, fitframe, base, _ = fixture()
    install_fake(monkeypatch)
    bundle = experiment.run(frame, refs, pl.DataFrame(), fitframe, cal, check, base,
                            ['baseline_p'], tmp_path, fit_refs=fit)
    testrefs = refs.with_columns(pl.when(pl.col('rid') == 0).then(pl.lit('france')).otherwise(pl.col('co')).alias('co'),
                                 pl.lit(-999).alias('deg'))
    accepted, report = experiment.infer(bundle, frame, testrefs, pl.DataFrame(), base.drop('y'))
    assert accepted.filter(pl.col('qid') == 0).equals(base.drop('y').filter(pl.col('qid') == 0))
    assert report['chosen_actions'] > 0 and report['global_unique_targets']
    unlabeled, report2 = experiment.infer(bundle, frame.drop('y', 'own'), testrefs.drop('deg'), pl.DataFrame(), base.drop('y'))
    assert accepted.equals(unlabeled) and report == report2


def test_no_promotion_replays_exact_baseline_and_missing_training_labels_fail(monkeypatch, tmp_path):
    refs, frame, fit, cal, check, fitframe, base, _ = fixture()
    monkeypatch.setattr(experiment, 'MIN_FIT', 10_000)
    bundle = experiment.run(frame, refs, pl.DataFrame(), fitframe, cal, check, base,
                            ['baseline_p'], tmp_path, fit_refs=fit)
    assert not bundle['promoted']
    accepted, report = experiment.infer(bundle, frame, refs, pl.DataFrame(), base)
    assert accepted.equals(base) and report['chosen_actions'] == 0
    disabled = {'promoted': False}
    assert experiment.infer(disabled, frame, refs, pl.DataFrame(), base)[0].equals(base)
    with pytest.raises(ValueError, match='labels'):
        experiment.run(frame.drop('y'), refs, pl.DataFrame(), fitframe, cal, check, base,
                       ['baseline_p'], tmp_path/'missing', fit_refs=fit)


def test_global_winner_uses_raw_probability_before_qid_tiebreak():
    frame = pl.DataFrame({'qid': [0, 1], 'tid': [5, 5], 'baseline_p': [.9, .9],
                          'baseline_raw': [.7, .8]})
    refs = pl.DataFrame({'rid': [0, 1], 'co': ['india', 'india']})
    base = pl.DataFrame(schema={'qid': pl.Int64, 'tid': pl.Int64, 'p': pl.Float64})
    actions = experiment._actions(frame, refs, base, ['baseline_p'])
    assert actions['qid'].to_list() == [1]
    tied = experiment._actions(frame.with_columns(pl.lit(.7).alias('baseline_raw')), refs, base, ['baseline_p'])
    assert tied['qid'].to_list() == [0]


def test_check_never_reselects_calibration_threshold(monkeypatch, tmp_path):
    refs, frame, fit, cal, check, fitframe, base, _ = fixture()
    install_fake(monkeypatch)
    first = experiment.run(frame, refs, pl.DataFrame(), fitframe, cal, check, base,
                           ['baseline_p'], tmp_path/'first', fit_refs=fit)


    checkids = check['rid'].implode()
    changedbase = base.filter(~(pl.col('qid').is_in(checkids) & (pl.col('y') == 1)))
    second = experiment.run(frame, refs, pl.DataFrame(), fitframe, cal, check, changedbase,
                            ['baseline_p'], tmp_path/'second', fit_refs=fit)
    assert first['thresholds'] == second['thresholds']


def test_precision_or_recall_regression_vetoes_policy():
    refs = pl.DataFrame({'qid': [0], 'co': ['india'], 'deg': [2]})
    base = pl.DataFrame({'qid': [0, 0, 0], 'tid': [1, 2, 3], 'p': [.9, .8, .1], 'y': [1, 1, 0]})
    actions = base.head(1).with_columns(pl.lit(-1).alias('operation'))
    evidence = experiment._evaluate(actions, np.array([.1]), base, refs, 0)
    assert not evidence['passed'] and not evidence['guards']['overall_pair_recall_nondecreasing']


def test_feature_label_leakage_is_rejected():
    for column in ['qid', 'tid', 'co', 'deg', 'y', 'own', 'utility_delta', 'accepted_count']:
        with pytest.raises(ValueError, match='features'):
            experiment._feature_sets(['baseline_p', column])


def test_real_regressor_learns_signed_utility_and_reloads_identically(tmp_path):
    import lightgbm as lgb
    actions = pl.DataFrame({'action_sign': [-1.] * 180,
        'action_probability': [.2] * 90 + [.9] * 90, 'utility_delta': [1.] * 90 + [-1.] * 90})
    ff = ['action_sign', 'action_probability']
    path = tmp_path/'regression.txt'
    model = experiment._fit(actions, ff, path)
    pred = experiment._predict(model, actions, ff)
    assert pred[0] > 0 and pred[-1] < 0
    reloaded = lgb.Booster(model_file=str(path))
    np.testing.assert_allclose(pred, experiment._predict(reloaded, actions, ff))
    assert reloaded.params['objective'] == 'regression'

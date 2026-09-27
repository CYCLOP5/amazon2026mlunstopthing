from pathlib import Path
import json
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from latest_fusion import additions_experiment as experiment


def baseline():
    return pl.DataFrame({'qid': [9], 'tid': [9], 'p': [.95], 'y': [1], 'own': [9]})


def rows():
    return pl.DataFrame({'qid': [1, 2, 1, 2, 1, 2, 1], 'tid': [0, 0, 1, 1, 2, 2, 9],
        'p_alias': [.999, .999, .995, .999, .999, .1, 1.],
        'p_decoy': [0.] * 7, 'p_other': [0.] * 7,
        'y': [1, 0, 1, 0, 1, 0, 0], 'own': [1, 1, 1, 1, 1, 1, 9]})


def test_global_ties_occupancy_and_outside_competitor():
    anchors = pl.DataFrame({'qid': [1], 'deg': [3], 'co': ['us']})
    rule = {'enabled': True, 'min_alias': .98, 'min_margin': .05}
    adds = experiment.propose_additions(rows(), baseline(), anchors, rule)
    assert adds['tid'].to_list() == [2]

    res = experiment.apply_additions(baseline(), adds)
    assert res.head(1).equals(baseline())
    assert res['tid'].to_list() == [9, 2]
    with pytest.raises(ValueError, match='globally unoccupied'):
        experiment.apply_additions(baseline(), rows().filter(pl.col('tid') == 9).with_columns(pl.col('p_alias').alias('p')))


def test_exact_empty_zero_degree_and_country_guards():
    anchors = pl.DataFrame({'qid': [1, 2], 'deg': [1, 0], 'co': ['us', 'india']})
    base = baseline().head(0)
    add = rows().filter(pl.col('tid') == 2).head(1).with_columns(pl.col('p_alias').alias('p'))
    evidence = experiment.evaluate_additions(base, add, anchors)
    assert evidence['before']['macro_f05'] == .5
    assert evidence['after']['macro_f05'] == 1.
    assert evidence['countries']['india']['after']['macro_f05'] == 1.


def test_frozen_inference_and_france(monkeypatch):
    bundle = {'promoted': True, 'family': 'contrastive', 'head': 'treatment',
              'model': object(), 'features': ['x'],
              'rule': {'enabled': True, 'min_alias': .98, 'min_margin': .05}}
    france = rows().filter(pl.col('tid') == 2).head(1).with_columns(pl.lit(3).alias('qid'), pl.lit(3).alias('tid'))
    scored = pl.concat([rows(), france], how='vertical_relaxed').with_columns(
        pl.when((pl.col('qid') == 1) & (pl.col('tid') == 1)).then(.1).otherwise(pl.col('p_alias')).alias('p_alias'))
    monkeypatch.setattr(experiment, '_build_score', lambda *args: ({'treatment': scored}, {}))
    monkeypatch.setattr(experiment, 'select_policy', lambda *args: pytest.fail('Replay must not retune'))
    refs = pl.DataFrame({'rid': [1, 2, 3], 'co': ['us', 'us', 'france']})
    out, diagnostics = experiment.infer(bundle, rows(), refs, refs, baseline())
    assert out['tid'].to_list() == [9, 1, 2]
    assert out.head(1).equals(baseline())
    assert diagnostics['additions'] == 2

    bundle['promoted'] = False
    monkeypatch.setattr(experiment, '_build_score', lambda *args: pytest.fail('Frozen baseline'))
    out, _ = experiment.infer(bundle, rows(), refs, refs, baseline())
    assert out.equals(baseline())


def test_actual_stored_lightgbm_model_reused(tmp_path, monkeypatch):
    rng = np.random.default_rng(2709)
    x = rng.normal(size=(90, 1))
    labels = np.arange(90) % 3
    model = lgb.train({'objective': 'multiclass', 'num_class': 3, 'verbosity': -1,
                       'num_threads': 1, 'min_data_in_leaf': 2},
                      lgb.Dataset(x, label=labels, feature_name=['x']), num_boost_round=3)
    path = tmp_path / 'model.txt'
    model.save_model(str(path))
    original_hash = experiment._hash(path)
    restored = lgb.Booster(model_file=str(path))
    frame = pl.DataFrame({'qid': [1, 2], 'tid': [1, 2], 'x': [.2, -.1], 'newest': [.7, .8], 'baseline_raw': [.6, .9],
                          'co': ['us', 'india'], 'y': [1, 1], 'own': [1, 2]})
    monkeypatch.setattr(experiment.contrastive, '_build', lambda *args: (frame, [], {}))
    monkeypatch.setattr(lgb, 'train', lambda *args, **kwargs: pytest.fail('No training allowed'))
    pred, _ = experiment._build_score('contrastive', frame, frame, frame,
                                             {'control': restored}, {'control': ['x']}, None)
    scores = pred['control']
    np.testing.assert_allclose(scores.select('p_alias', 'p_decoy', 'p_other').to_numpy(),
                               model.predict(frame.select('x').to_numpy()), rtol=1e-6)
    assert scores['baseline_p'].to_list() == [.7, .8]
    assert scores['raw'].to_list() == [.6, .9]
    assert experiment._hash(path) == original_hash


def test_single_winner_before_check_and_no_failed_check_fallback(tmp_path, monkeypatch):
    ncal, ncheck = 6, 26
    owners = list(range(1, ncal + ncheck + 1))
    train = pl.DataFrame({'qid': [0] + owners, 'tid': [0] + owners,
                         'y': [1] * (len(owners) + 1), 'own': [0] + owners, 'x': [1.] * (len(owners) + 1)})
    refs = pl.DataFrame({'rid': [0] + owners, 'deg': [1] * (len(owners) + 1), 'co': ['us'] * (len(owners) + 1)})
    cal = refs.filter(pl.col('rid').is_between(1, ncal))
    check = refs.filter(pl.col('rid') > ncal)
    fit = train.filter(pl.col('qid') == 0)
    base = baseline().head(0)
    sources = {}
    class FakeModel:
        def __init__(self, model_file):
            self.head = Path(model_file).name.split('_')[0]
        def feature_name(self):
            return ['x']
    for family in ('contrastive', 'directional'):
        directory = tmp_path / family / 'experiment'
        directory.mkdir(parents=True)
        (directory / 'features.json').write_text(json.dumps({'control': ['x'], 'treatment': ['x']}))
        for head in ('control', 'treatment'):
            (directory / (head + '_model.txt')).write_text(head)
        sources[family] = directory.parent
    monkeypatch.setattr(lgb, 'Booster', FakeModel)
    monkeypatch.setattr(lgb, 'train', lambda *args, **kwargs: pytest.fail('No training allowed'))
    def build(family, frame, refs, targets, models, features, s2):
        res = {}
        for head in models:
            score = .9999 if head == 'treatment' else .1
            res[head] = frame.with_columns(pl.lit(score).alias('p_alias'), pl.lit(0.).alias('p_decoy'),
                pl.lit(0.).alias('p_other'), pl.lit(.8).alias('baseline_p'), pl.lit(score).alias('p'), pl.lit('us').alias('co'))

            if family == 'contrastive' and head == 'treatment':
                res[head] = res[head].with_columns(pl.when(pl.col('qid') == 7).then(0).otherwise(pl.col('y')).alias('y'),
                    pl.when(pl.col('qid') == 7).then(-1).otherwise(pl.col('own')).alias('own'))
        return res, {}
    monkeypatch.setattr(experiment, '_build_score', build)
    checked_scores = []
    evaluate = experiment.evaluate_additions
    def observe(base, adds, anchors):
        if anchors['qid'].min() == 7:
            checked_scores.append(adds.height)
        return evaluate(base, adds, anchors)
    monkeypatch.setattr(experiment, 'evaluate_additions', observe)
    bundle = experiment.run(train, refs, refs, fit, cal, check, base, ['x'], tmp_path / 'output',
        fit_refs=refs.head(1), pretrained_heads=sources)
    assert not bundle['promoted']
    assert bundle['family'] == 'contrastive' and bundle['head'] == 'treatment'
    assert checked_scores == [26, 0]
    assert bundle['report']['check']['FPadded'] == 1
    assert all(len(head['calibration']['trials']) == 12 for head in bundle['report']['heads'].values())
    assert (tmp_path / 'output' / 'selected_model.txt').read_text() == 'treatment'
    checked_scores.clear()
    def insufficient_calibration(*args):
        scored, support = build(*args)
        return {head: frame.with_columns(pl.when(pl.col('qid').is_in([5, 6])).then(.1)
            .otherwise(pl.col('p_alias')).alias('p_alias')) for head, frame in scored.items()}, support
    monkeypatch.setattr(experiment, '_build_score', insufficient_calibration)
    frozen = experiment.run(train, refs, refs, fit, cal, check, base, ['x'], tmp_path / 'frozen',
        fit_refs=refs.head(1), pretrained_heads=sources)
    assert not frozen['promoted'] and frozen['report']['selected'] is None
    assert checked_scores == []

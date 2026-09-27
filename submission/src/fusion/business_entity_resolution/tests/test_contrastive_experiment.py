from pathlib import Path
import json
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from latest_fusion import contrastive_experiment as experiment


def population(nfit=50, nheld=5):
    refs = pl.DataFrame({'rid': list(range(8)), 'nm': ['Lake Club']*8,
        'ad': ['24 Lake Road', '26 Lake Road']*4,
        'co': ['us']*6 + ['france']*2, 'deg': [0, nfit, 0, nheld, 0, nheld, 0, 1]})
    targets, rows, base = [], [], []
    for qold, qnew, start, n in [(0, 1, 0, nfit), (2, 3, 1000, nheld), (4, 5, 2000, nheld)]:
        for tid in range(start, start+n):
            targets.append((tid, 'Lake Club', '26 Lake Road', 'us'))
            for qid in (qold, qnew):
                rows.append((qid, tid, .001 if qid == qold else .9, .95, qid == qnew, qnew,
                             'us', float(qid == qnew)))
            if qold > 0:
                base.append((qold, tid, .95, 0))
    for tid in range(3000, 3000+nfit):
        targets.append((tid, 'Orphan', '', 'us'))
        rows.append((0, tid, .01, .01, False, -1, 'us', 0.))
    train = pl.DataFrame(rows, schema=['qid', 'tid', 'newest', 'baseline_p', 'y', 'own', 'co', 'signal'],
                         orient='row').with_columns(pl.col('y').cast(pl.Int8))
    train = train.with_columns(pl.when((pl.col('qid') == 0) & (pl.col('tid') == 0))
                               .then(None).otherwise(pl.col('newest')).alias('newest'))
    targets = pl.DataFrame(targets, schema=['rid', 'nm', 'ad', 'co'], orient='row')
    fit = train.filter((pl.col('tid') < 1000) | (pl.col('tid') >= 3000))
    cal = refs.filter(pl.col('rid').is_in([2, 3]))
    check = refs.filter(pl.col('rid').is_in([4, 5]))
    baseline = pl.DataFrame(base, schema=['qid', 'tid', 'p', 'y'], orient='row')
    return train, refs, targets, fit, cal, check, baseline


class ScriptModel:
    def __init__(self, features):
        self.features = list(features)

    def feature_name(self):
        return self.features

    def predict(self, matrix, **kwargs):
        if 'ct_numeric_q_support' in self.features:
            alias = np.where(matrix[:, self.features.index('ct_numeric_q_support')] > 0, .9999, .001)
        else:
            alias = np.full(len(matrix), .4)
        return np.column_stack([alias, (1-alias)/2, (1-alias)/2])


def test_matched_fit_frozen_promotion_and_label_free_inference(monkeypatch, tmp_path):
    train, refs, targets, fit, cal, check, baseline = population()
    calls = []

    def fit_model(frame, features, path):
        calls.append((frame.select('qid', 'tid'), frame['newest'], list(features)))
        path.write_text('synthetic model')
        return ScriptModel(features)

    monkeypatch.setattr(experiment, '_fit', fit_model)
    core = ['signal', 'newest', 'baseline_p']
    bundle = experiment.run(train, refs, targets, fit, cal, check, baseline, core,
                            tmp_path, fit_refs=refs.head(2), s2_count=42)
    assert bundle['promoted']
    assert calls[0][0].equals(calls[1][0]) and calls[0][0].equals(fit.select('qid', 'tid'))
    assert calls[0][1].null_count() == calls[1][1].null_count() == 1
    assert train['newest'].null_count() == 1
    assert calls[0][2] == core and len(calls[1][2]) == len(core) + 24
    assert bundle['report']['heads']['treatment']['check']['corrected'] == 5
    assert bundle['report']['heads']['control']['check']['switches'] == 0
    assert json.loads((tmp_path/'report.json').read_text()) == bundle['report']
    assert (tmp_path/'control_model.txt').exists() and (tmp_path/'treatment_model.txt').exists()
    test = train.filter(pl.col('tid').is_between(2000, 2004)).drop('y', 'own')
    testbase = baseline.filter(pl.col('tid').is_between(2000, 2004)).drop('y')

    french_target = pl.DataFrame({'rid': [9000], 'nm': ['Lake Club'], 'ad': ['26 Lake Road'], 'co': ['france']})
    french = test.head(2).with_columns(pl.Series('qid', [6, 7]), pl.lit(9000).cast(pl.Int64).alias('tid'),
                                    pl.lit('france').alias('co'))
    test = pl.concat([test, french], how='vertical_relaxed')
    testbase = pl.concat([testbase, pl.DataFrame({'qid': [6], 'tid': [9000], 'p': [.95]})])
    testtargets = pl.concat([targets, french_target])
    out, diag = experiment.infer(bundle, test, refs, testtargets, testbase, s2_count=42)
    assert out.height == testbase.height and out['tid'].equals(testbase['tid'])
    assert out.filter(pl.col('tid') == 9000).equals(testbase.filter(pl.col('tid') == 9000))
    assert out.filter(pl.col('tid') < 9000)['qid'].to_list() == [5]*5
    assert diag['switches'] == 5 and (tmp_path/'switches_test.parquet').exists()


def test_small_classes_fail_closed_and_disabled_infer_is_exact(tmp_path):
    train, refs, targets, fit, cal, check, baseline = population(nfit=2)
    bundle = experiment.run(train, refs, targets, fit, cal, check, baseline, ['signal'], tmp_path)
    assert not bundle['promoted'] and all(model is None for model in bundle['models'].values())
    assert not bundle['report']['heads']['treatment']['available']

    out, diag = experiment.infer(bundle, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), baseline)
    assert out.equals(baseline) and diag['switches'] == 0


def test_check_breakage_vetoes_calibrated_rule(monkeypatch, tmp_path):
    train, refs, targets, fit, cal, check, baseline = population()
    monkeypatch.setattr(experiment, '_fit', lambda frame, features, path: ScriptModel(features))

    train = train.with_columns(pl.when(pl.col('tid').is_between(2000, 2004)).then(4)
                               .otherwise(pl.col('own')).alias('own'))
    train = train.with_columns((pl.col('qid') == pl.col('own')).cast(pl.Int8).alias('y'))
    baseline = baseline.with_columns(pl.when(pl.col('tid').is_between(2000, 2004)).then(1)
                                     .otherwise(pl.col('y')).alias('y'))
    check = check.with_columns(pl.when(pl.col('rid') == 4).then(5).otherwise(0).alias('deg'))
    bundle = experiment.run(train, refs, targets, fit, cal, check, baseline, ['signal'], tmp_path)
    treatment = bundle['report']['heads']['treatment']
    assert treatment['rule']['enabled'] and treatment['calibration']['corrected'] == 5
    assert treatment['check']['broken'] == 5 and not bundle['promoted']


def test_real_fixed_multiclass_model_schema_and_probability_order(tmp_path):
    train, refs, targets, fit, cal, check, baseline = population()
    augmented, added, _ = experiment._build(fit, refs, targets)
    ff = ['signal'] + added
    model = experiment._fit(augmented, ff, tmp_path/'real_model.txt')
    scored = experiment._score(model, augmented, ff, chunk_size=7)
    assert model.feature_name() == ff and (tmp_path/'real_model.txt').exists()
    assert scored.select('qid', 'tid').equals(fit.select('qid', 'tid'))
    probs = scored.select('p_alias', 'p_decoy', 'p_other').to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.) and np.isfinite(probs).all()
    assert experiment._class_counts(augmented) == {'alias': 50, 'decoy': 50, 'other_owned': 50}
    with pytest.raises(ValueError, match='feature order'):
        experiment._score(model, augmented, ff[::-1])


def test_invalid_prediction_and_isolation_are_rejected(tmp_path):
    train, refs, targets, fit, cal, check, baseline = population(nfit=2)
    with pytest.raises(ValueError, match='Fitting targets overlap'):
        experiment.run(train, refs, targets, train, cal, check, baseline, ['signal'], tmp_path)
    with pytest.raises(ValueError, match='forbidden'):
        experiment.run(train, refs, targets, fit, cal, check, baseline, ['signal', 'y'], tmp_path)

    class BadModel(ScriptModel):
        def predict(self, matrix, **kwargs):
            return np.full((len(matrix), 3), .9)

    with pytest.raises(ValueError, match='Invalid three-class'):
        experiment._score(BadModel(['signal']), fit, ['signal'])

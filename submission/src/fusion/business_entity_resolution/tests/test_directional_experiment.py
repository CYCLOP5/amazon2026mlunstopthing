from pathlib import Path
import json
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from latest_fusion import directional_experiment as experiment


def fixture():
    refs = pl.DataFrame({'rid': [0, 1, 2], 'co': ['us'] * 3,
                         'nm': ['common original'] * 3, 'ad': ['1 main'] * 3,
                         'deg': [1] * 3, 'fold': [1] * 3})
    targets = pl.DataFrame({'rid': [10, 11, 12], 'co': ['us'] * 3,
                            'nm': ['common changed'] * 3, 'ad': ['1 main'] * 3})
    rows = pl.DataFrame({'qid': [0, 1, 2], 'tid': [10, 11, 12], 'co': ['us'] * 3,
                         'y': [1] * 3, 'own': [0, 1, 2], 'score': [.9] * 3,
                         'baseline_p': [.9] * 3})
    base = rows.select('qid', 'tid', 'y', pl.col('baseline_p').alias('p'))
    return rows, refs, targets, base


def test_insufficient_classes_persists_disabled_and_replays_without_labels(tmp_path):
    rows, refs, targets, base = fixture()
    bundle = experiment.run(rows, refs, targets, rows.head(1), refs.slice(1, 1), refs.slice(2, 1),
                            base, ['score'], tmp_path, fit_refs=refs.head(1))
    assert not bundle['promoted'] and not bundle['transfer_enabled']
    assert bundle['models']['treatment'] is None
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['heads']['control']['class_counts'] == {'alias': 1, 'decoy': 0, 'other_owned': 0}
    assert 'unchanged' in report['transfer']['france_policy']

    accepted, diagnostics = experiment.infer(bundle, rows.drop('y', 'own'), refs, targets, base.drop('y'))
    assert accepted.equals(base.drop('y')) and not diagnostics['promoted']
    assert diagnostics['support']['candidate_pairs'] == 3
    assert diagnostics['direction_inspection']['countries'][0]['eligible_pairs'] == 3
    assert diagnostics['direction_inspection']['top_directions_by_country']['us'][0]['reverse_edges'] == 0
    assert (tmp_path / 'test_direction_support.parquet').exists()
    assert (tmp_path / 'inference_diagnostics.json').exists()


def test_target_isolation_and_forbidden_features_fail_closed(tmp_path):
    rows, refs, targets, base = fixture()
    with pytest.raises(ValueError, match='overlap'):
        experiment.run(rows, refs, targets, rows, refs.slice(1, 1), refs.slice(2, 1),
                       base, ['score'], tmp_path)
    for field in ['qid', 'tid', 'own', 'y', 'deg', 'fold', 'ds_shared']:
        with pytest.raises(ValueError, match='Forbidden'):
            experiment.run(rows, refs, targets, rows.head(1), refs.slice(1, 1), refs.slice(2, 1),
                           base, ['score', field], tmp_path)


def test_real_multiclass_fit_predict_and_serialization(tmp_path, monkeypatch):
    monkeypatch.setenv('ER_THREADS', '2')
    n = 90
    rows = pl.DataFrame({'qid': list(range(3*n)), 'tid': list(range(3*n)),
                         'score': [-4.] * n + [0.] * n + [4.] * n,
                         'y': [1] * n + [0] * (2*n),
                         'own': list(range(n)) + [-1] * n + [9999] * n,
                         'baseline_p': [.5] * (3*n)})
    path = tmp_path / 'model.txt'
    model = experiment._fit(rows, ['score'], path)
    scored = experiment._score(model, rows, ['score'])
    probs = scored.select('p_alias', 'p_decoy', 'p_other').to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.)
    assert (probs.argmax(axis=1) == np.repeat([0, 1, 2], n)).all()
    assert path.exists()
    reloaded = experiment.lgb.Booster(model_file=str(path))
    assert np.allclose(reloaded.predict(rows.select('score').to_numpy(), num_threads=2), probs)
    test_rows, refs, targets, base = fixture()
    bundle = {'promoted': False, 'transfer_enabled': False, 'output_dir': str(tmp_path),
              'models': {'treatment': reloaded}, 'features': {'treatment': ['score']},
              'rules': {'treatment': {'enabled': True, 'drop_threshold': 0., 'add_threshold': 0.}}}
    accepted, diagnostics = experiment.infer(bundle, test_rows.drop('y', 'own'), refs,
                                              targets, base.drop('y'))
    assert accepted.equals(base.drop('y'))
    assert diagnostics['main_removed'] == diagnostics['main_added'] == 0
    assert diagnostics['treatment_predictions_by_country'][0]['eligible_pairs'] == 3
    assert 'p_decoy_p99' in diagnostics['treatment_predictions_by_country'][0]

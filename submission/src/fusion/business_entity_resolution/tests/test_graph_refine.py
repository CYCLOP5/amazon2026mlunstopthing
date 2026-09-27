import json
from pathlib import Path
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_resolution.refine import reuse_predictions, run
from graph_resolution.pipeline import run as run_parent
from test_stack import make_fixture


def test_saved_pool_is_exact_and_labels_do_not_enter_features(tmp_path):
    raw = pl.DataFrame({'qid': [0, 2], 'tid': [0, 2], 'prob': [.9, .999],
                        'gate_prob': [.9, .999], 'neural_prob': [.9, .999], 'y': [1, 1]})
    saved = pl.DataFrame({'qid': [0, 1], 'tid': [0, 0], 'binary': [.8, .1], 'rank': [.7, .2], 'y': [0, 1]})
    path = tmp_path/'pred.parquet'
    saved.write_parquet(path)
    frozen = pl.DataFrame({'qid': [0, 2], 'sid': [0, 2], 'confidence': [.999, .999]})
    pairs, seeds, _ = reuse_predictions(raw, frozen, path)
    assert pairs.select('qid', 'tid').equals(saved.select('qid', 'tid'))
    assert 'y' not in pairs.columns
    assert seeds.filter(pl.col('sid') == 0)['confidence'].to_list() == [.8]
    assert seeds.filter(pl.col('sid') == 2).height == 1
    assert pairs.filter(pl.col('qid') == 1)['has_upstream_score'][0] == 0


def test_refinement_bundle_and_submission(tmp_path):
    data, roots = make_fixture(tmp_path, 160)
    sc = {'data': str(data), 'train_roots': roots['train'], 'test_roots': roots['test'],
          'lexical_train': [], 'lexical_test': [], 'rounds': 8}
    run_parent(sc, tmp_path/'parent', tmp_path/'parent_work')
    sc['graph_predictions'] = str(tmp_path/'parent')
    run(sc, tmp_path/'refined', tmp_path/'work')
    report = json.loads((tmp_path/'refined/report.json').read_text())
    assert report['audit_entities'] == 80
    assert (tmp_path/'refined/bundle/fold_2.txt').exists()
    assert 'paired_bootstrap_95_ci' in report['comparison']
    res = pl.read_csv(tmp_path/'refined/output/matching_results.tsv', separator='\t')
    assert res.height == 32
    for split in ('train', 'test'):
        filename = 'validation_predictions.parquet' if split == 'train' else 'test_predictions.parquet'
        before = pl.read_parquet(tmp_path/'parent'/filename).select('qid', 'tid').sort('tid', 'qid')
        after = pl.read_parquet(tmp_path/'refined'/filename).select('qid', 'tid').sort('tid', 'qid')
        assert before.equals(after)

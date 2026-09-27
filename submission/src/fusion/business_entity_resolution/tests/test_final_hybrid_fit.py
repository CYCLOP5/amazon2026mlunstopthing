'synthetic correctness checks; real-data retrieval and fitting run in azure'
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final_hybrid.fit import (
    attach_parent, weighted_partitions, update_probabilities, union_predictions,
    feature_columns, fit_residual, load_scored, run,
)
from innovation.common import read_parent
from test_edit_channel import fixture


def test_weights_restore_negative_sampling_and_keep_both_holdouts():
    refs = pl.DataFrame({'rid': np.arange(300, dtype=np.uint32),
        'fold': [0]*100+[1]*100+[2]*100, 'co': ['us']*300})
    eligible = refs.filter((pl.col('fold') == 0) & ((pl.col('rid').hash(1033)%5) != 0))['rid'].to_list()
    tune = refs.filter((pl.col('fold') == 0) & ((pl.col('rid').hash(1033)%5) == 0))['rid'][0]
    a, b = eligible[:2]
    decoy = pl.DataFrame({'tid': np.arange(100, dtype=np.uint32)}).filter(
        (pl.col('tid').hash(713)%100) < 8)['tid'][0]
    frame = pl.DataFrame({'qid': [a, a, a, a, tune, 220],
        'tid': [0, 1, decoy, 3, 4, 5], 'own': [a, b, -1, 120, a, a],
        'fold': [0, 0, 0, 0, 0, 2], 'co': ['us']*6,
        'y': [1, 0, 0, 0, 0, 0]}).with_columns(pl.col('qid', 'tid').cast(pl.UInt32))
    fit, groups, weights, rates = weighted_partitions(frame, refs)
    assert fit.tolist() == [True, True, True, False, False, False]
    assert weights[0] == 1
    assert weights[1] == pytest.approx(300/len(eligible))
    assert weights[2] == 12.5
    assert groups[0] == groups[4] == groups[5]


def test_residual_adds_offset_once_and_union_preserves_full_pool():
    parent = pl.DataFrame({'qid': [0, 1, 2], 'tid': [0, 0, 1], 'p': [.8, .1, .7], 'y': [1, 0, 1]})
    sel = pl.DataFrame({'qid': [0, 3], 'tid': [0, 0], 'y': [1, 0]})
    frame = attach_parent(sel, parent)
    update = update_probabilities(frame, np.zeros(2), 1.)
    np.testing.assert_allclose(update['p'], [.8, .01], rtol=1e-6)
    combined = union_predictions(parent, update, frame)
    assert combined.height == 4
    assert combined.select('qid', 'tid').n_unique() == 4
    assert combined.filter(pl.col('qid') == 2)['p'][0] == .7
    assert combined.filter(pl.col('qid') == 3)['y'][0] == 0
    with pytest.raises(ValueError, match='Duplicate'):
        attach_parent(pl.concat([sel, sel]), parent)


def test_saved_booster_keeps_tree_residual_separate_from_offset(tmp_path):
    import lightgbm as lgb
    n = 1800
    rng = np.random.default_rng(1)
    y = np.arange(n)%2
    signal = y+rng.normal(0, .3, n)
    frame = pl.DataFrame({'y': y, 'base_lg': np.full(n, -2.), 'has_parent': np.ones(n), 'signal': signal})
    fit = np.ones(n, dtype=bool)
    groups = (np.arange(n)//2)%3
    residual, models, info = fit_residual(frame, ['signal'], fit, groups,
        np.ones(n), 12, tmp_path, 'fixture')
    valid = groups == 0
    reloaded = lgb.Booster(model_file=str(tmp_path/'fixture_0.txt'))
    np.testing.assert_allclose(residual[valid], reloaded.predict(
        frame.select('signal').to_numpy()[valid], raw_score=True), atol=1e-6)
    p = 1/(1+np.exp(-(residual+frame['base_lg'].to_numpy())))
    initial = 1/(1+np.exp(2))
    assert np.mean(p[y == 1]) > initial
    assert info['fit_rows'] == n


def test_synthetic_pipeline_covers_new_pairs_and_exports_valid_baseline(tmp_path):
    data, parent, prepared, refs = fixture(tmp_path, n=300)
    refs = refs.with_columns(pl.Series('fold', [0]*100+[1]*100+[2]*100))
    refs.write_parquet(data/'train/ref.parquet')
    hybrid = tmp_path/'hybrid'
    for split in ('train', 'test'):
        old, _ = read_parent(parent, split)
        count = 300 if split == 'train' else 32
        keys = pl.concat([old.select('qid', 'tid'),
            pl.DataFrame({'qid': [4], 'tid': [0]}, schema=old.select('qid', 'tid').schema)]).unique()
        part = keys.with_columns(
            pl.when(pl.col('qid') == pl.col('tid')%count).then(4.).otherwise(-4.).alias('ce_base_lg'),
            pl.when(pl.col('qid') == pl.col('tid')%count).then(5.).otherwise(-5.).alias('expert_lg'),
            pl.lit(.5).alias('dense_name'), pl.lit(.4).alias('dense_address'),
            pl.lit(.2).alias('lex_name'), pl.lit(.3).alias('lex_address'))
        (hybrid/split/'parts').mkdir(parents=True)
        part.write_parquet(hybrid/split/'parts/part_00000.parquet')
        keys.write_parquet(hybrid/split/'candidates.parquet')
    (hybrid/'_SUCCESS').write_text('complete\n')
    output = tmp_path/'result'
    report = run(data, parent, prepared, hybrid, output, rounds=3)
    assert report['selected'] == 'parent'
    assert report['comparison']['paired_delta'] == 0
    assert report['retrieval']['genuinely_new_pairs'] == 1
    frame = pl.read_parquet(output/'train_features.parquet')
    surface = feature_columns(frame, 'surface')
    assert not {'qid', 'tid', 'y', 'own', 'co', 'fold', 'parent_p'} & set(surface)
    assert 'expert_lg' not in feature_columns(frame, 'semantic')
    assert 'expert_lg' in feature_columns(frame, 'hybrid')
    assert not any(c.startswith(('dense_', 'ce_')) for c in surface)
    from scripts.validate_outputs import validate
    assert validate(data, output/'output')['source1'] == 32
    assert report['export']['candidate_pairs'] == 32*2*3

    path = hybrid/'test/parts/part_00000.parquet'
    pl.read_parquet(path).head(1).write_parquet(path)
    with pytest.raises(ValueError, match='exact candidate pool'):
        load_scored(hybrid, 'test')

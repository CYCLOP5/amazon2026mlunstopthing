'synthetic protocol, isolation, keyed-fusion and decoder checks'
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from large_reranker.prepare import (load_predictions, attach_probabilities, choose_targets,
                                    neural_partitions, isolate_neural_decoys, prepare)
from large_reranker.fit import read_scored, patch, feature_columns, fit_head, run, base_predictions, labeled_pool
from final_hybrid.fit import weighted_partitions
from test_stack import make_fixture


def test_country_balanced_atomic_selection_ignores_truth():
    d = pl.DataFrame({'qid': [0, 1, 2, 3, 4, 5], 'tid': [0, 0, 1, 1, 1, 2],
        'hybrid_p': [.6]*6, 'friend_p': [.6]*6, 'graph_p': [.4]*6, 'friend_graph_p': [.5]*6,
        'y': [1, 0, 0, 1, 0, 1], 'own': [0, 0, 3, 3, 3, 5]})
    targets = pl.DataFrame({'rid': [0, 1, 2], 'co': ['us', 'india', 'france'], 'ad': ['', '', '']})
    chosen, report = choose_targets(d, targets, 3, 4)
    changed, _ = choose_targets(d.with_columns(1-pl.col('y'), -pl.col('own')), targets, 3, 4)
    assert chosen.equals(changed)
    assert report['selected_pairs'] <= 4
    sel = d.join(chosen, on='tid')
    assert sel.height == report['selected_pairs']
    assert sel['tid'].n_unique() == report['selected_targets']
    assert report['countries']['france']['selected'] == 1


def test_fold2_training_excludes_all_labeled_holdouts_and_groups_decoys():
    refs = pl.DataFrame({'rid': np.arange(400, dtype=np.uint32), 'fold': [0]*100+[1]*100+[2]*200})
    rows = pl.DataFrame({'qid': [220]*6+[30, 140], 'tid': [5, 6, 7, 7, 8, 9, 10, 11],
        'own': [230, 40, 120, -1, -1, 230, 230, 230], 'fold': [2]*6+[0, 1],
        'y': [0]*8, 'co': ['us']*8}).with_columns(pl.col('qid', 'tid').cast(pl.UInt32))
    fit, valid = neural_partitions(rows, refs)
    assert not (fit & valid).any()
    assert not (fit | valid)[[1, 2, 6, 7]].any()
    assert valid[0] == valid[5]

    other = rows.with_columns(pl.when(pl.col('tid') == 7).then(221).otherwise(pl.col('qid')).cast(pl.UInt32).alias('qid'))
    _, valid_other = neural_partitions(other, refs)
    assert valid[3] == valid_other[3]

    grid = refs.filter(pl.col('fold') == 2).select(pl.col('rid').alias('qid')).join(
        refs.filter(pl.col('fold') == 2).select(pl.col('rid').cast(pl.Int64).alias('own')), how='cross')
    grid = grid.with_columns(pl.col('own').cast(pl.UInt32).alias('tid'), pl.lit(2).alias('fold'),
        pl.lit('us').alias('co'), (pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.UInt8).alias('y'))
    gf, gv = neural_partitions(grid, refs)
    heldowners = set(grid.filter(pl.Series(gv))['own'].to_list())
    fitting = grid.filter(pl.Series(gf))
    assert heldowners.isdisjoint(fitting['own'].to_list())
    assert heldowners.isdisjoint(fitting['qid'].to_list())


def test_neural_decoys_never_share_full_union_tune_or_audit_target_text():
    refs = pl.DataFrame({'rid': np.arange(300, dtype=np.uint32),
        'fold': [0]*100+[1]*100+[2]*100, 'deg': [1]*300})
    tune_qid = refs.filter((pl.col('fold') == 0) & ((pl.col('rid').hash(1033)%5) == 0))['rid'][0]
    cache = pl.DataFrame({'qid': [220]*4, 'tid': [0, 1, 2, 3], 'own': [-1, -1, -1, 230],
        'fold': [2]*4, 'y': [0]*4, 'co': ['us']*4}).with_columns(pl.col('qid', 'tid').cast(pl.UInt32))

    full = pl.concat([cache.select('qid', 'tid'), pl.DataFrame({'qid': [100, tune_qid, 100],
        'tid': [0, 1, 3]}).with_columns(pl.col('qid', 'tid').cast(pl.UInt32))])
    safe, report = isolate_neural_decoys(cache, full, refs)
    assert safe['tid'].to_list() == [2, 3]
    assert report['excluded_decoy_targets_touching_tune_audit'] == 2
    assert report['remaining_cached_decoy_targets'] == 1

    assert safe.filter(pl.col('own') == 230).height == 1


def inputs(tmp_path, n=600):
    data, roots = make_fixture(tmp_path, n)
    refs = pl.read_parquet(data/'train/ref.parquet').with_columns(pl.Series('fold', [i%3 for i in range(n)]))
    refs.write_parquet(data/'train/ref.parquet')
    hybrid, friend, graph, cache = [tmp_path/name for name in ('hybrid', 'friend', 'graph', 'cache')]
    for p in (hybrid, friend, graph, cache):
        p.mkdir()
    for split in ('train', 'test'):
        base = pl.read_parquet(Path(roots[split][0])/'parts/part_0.parquet')
        name = 'validation_predictions.parquet' if split == 'train' else 'test_predictions.parquet'
        base.select('qid', 'tid', pl.col('prob').alias('p')).write_parquet(hybrid/name)
        base.select('qid', 'tid', pl.col('prob').alias('head')).write_parquet(graph/name)

        base.slice(1).select('qid', 'tid', pl.col('prob').alias('p2')).write_parquet(
            friend/('val_pred.parquet' if split == 'train' else 'test_pred.parquet'))
        feature = base.select('qid', 'tid', pl.col('neural_prob').alias('expert_lg'))
        feature = feature.join(pl.read_parquet(data/split/'ref.parquet').select(
            pl.col('rid').alias('qid'), 'co', *(['fold'] if split == 'train' else [])), on='qid')
        if split == 'train':
            own = pl.concat([pl.read_parquet(data/split/f's{s}.parquet').select(pl.col('rid').alias('tid'), 'own')
                             for s in (2, 3)])
            feature = feature.join(own, on='tid').with_columns(
                (pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.UInt8).alias('y'))
        feature.write_parquet(cache/f'{split}_features.parquet')
    (friend/'report.json').write_text(json.dumps({'methods': {'round2': {'score': 'p2',
        'tune': {'rule': 'top1_threshold', 'threshold': .5}}}}))
    (hybrid/'report.json').write_text(json.dumps({'decision': {'rule': 'top1_threshold', 'threshold': .5}}))
    return data, hybrid, friend, graph, cache


def test_fusion_preserves_missing_side_scores_and_unselected_exactly(tmp_path):
    data, hybrid, friend, graph, cache = inputs(tmp_path)
    full = load_predictions(hybrid, friend, graph, 'test')
    missing = full.filter(pl.col('has_friend_p') == 0)
    np.testing.assert_array_equal(missing['friend_graph_p'], missing['graph_p'])
    missing_pure = base_predictions(full, 'friend').join(missing.select('qid', 'tid'), on=['qid', 'tid'])
    assert missing_pure['p'].to_list() == [0.]
    base = full.select('qid', 'tid', pl.col('friend_graph_p').alias('p'))
    frame = attach_probabilities(pl.read_parquet(cache/'test_features.parquet').head(2), full)
    updated = patch(base, frame, np.full(2, 9.), 1.)
    untouched = base.join(frame.select('qid', 'tid'), on=['qid', 'tid'], how='anti')
    assert untouched.equals(updated.join(frame.select('qid', 'tid'), on=['qid', 'tid'], how='anti'))
    assert updated.head(2)['p'].max() > .99
    assert not {'qid', 'tid', 'own', 'fold', 'y', 'co'} & set(feature_columns(frame))


def test_prepare_score_fit_export_and_exact_scoring_contract(tmp_path):
    data, hybrid, friend, graph, cache = inputs(tmp_path, n=900)
    prepared, scored, output = [tmp_path/n for n in ('prepared', 'scored', 'result')]
    selection = prepare(data, hybrid, cache, cache, friend, graph, prepared, max_targets=10000)
    assert selection['neural_training']['fit']['rows'] > 0
    assert selection['neural_training']['valid']['rows'] > 0
    for split in ('train', 'test'):
        (scored/split).mkdir(parents=True)
        frame = pl.read_parquet(prepared/split/'features.parquet')
        frame.select('qid', 'tid').with_columns(pl.lit(0.).alias('large_lg')).write_parquet(scored/split/'part_00000.parquet')
    (scored/'_SUCCESS').write_text('complete\n')
    report = run(data, hybrid, cache, cache, friend, graph, prepared, scored, output, rounds=3)
    assert report['promotion_policy']['audit_used_for_selection'] is False
    assert report['france_policy']['weight'] == 0
    assert (output/'bundle/head_0.txt').exists()

    train_pool = labeled_pool(data, hybrid, friend, graph, 'train')
    train_frame = read_scored(prepared, scored, 'train', train_pool)
    refs = pl.read_parquet(data/'train/ref.parquet')
    fitmask, groups, _, _ = weighted_partitions(train_frame, refs)
    cols = feature_columns(train_frame)
    for tag in ('control', 'mutated'):
        (tmp_path/tag).mkdir()
    control, models, _, _ = fit_head(train_frame, refs, cols, 3, tmp_path/'control')
    altered = train_frame.with_columns(pl.when(pl.Series(~fitmask)).then(1-pl.col('y')).otherwise(pl.col('y')).alias('y'))
    replay, _, _, _ = fit_head(altered, refs, cols, 3, tmp_path/'mutated')
    np.testing.assert_array_equal(control, replay)
    import lightgbm as lgb
    saved = lgb.Booster(model_file=str(tmp_path/'control/head_0.txt'))
    cv0 = fitmask & (groups == 0)
    np.testing.assert_allclose(control[cv0], saved.predict(
        train_frame.select(cols).cast(pl.Float32).to_numpy()[cv0], raw_score=True), atol=1e-6)
    from scripts.validate_outputs import validate
    assert validate(data, output/'output')['source1'] == 32
    full = load_predictions(hybrid, friend, graph, 'test')
    final = pl.read_parquet(output/'test_predictions.parquet')
    assert final.height == full.height
    assert final.select('qid', 'tid').n_unique() == full.height

    part = scored/'test/part_00000.parquet'
    pl.read_parquet(part).head(1).write_parquet(scored/'test/part_extra.parquet')
    with pytest.raises(ValueError, match='Duplicate'):
        read_scored(prepared, scored, 'test', full)

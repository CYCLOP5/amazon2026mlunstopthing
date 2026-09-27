from pathlib import Path
import json
import sys

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from latest_fusion import features
from latest_fusion import innovation_retrieval as retrieval


def fixture():
    refs = pl.DataFrame({'rid': [0, 1, 2], 'co': ['us', 'us', 'india'],
        'nm': ['Aurelia Trading', 'Borealis Manufacturing', 'Zenith Services'],
        'ad': ['1 Oldlane Road', '99 Elsewhere Street', '2 Southbank Road'],
        'fold': [1, 1, 1], 'deg': [2, 0, 1]})
    targets = pl.DataFrame({'rid': [10, 11, 21], 'co': ['us', 'us', 'india'],
        'nm': ['Aurelia Nova Incorporated', 'Aurelia Nova Corp', 'Aurelia Nova Corp'],
        'ad': ['71 Cedarbank Grove', '71 Cedarbank Grove', '71 Cedarbank Grove'],
        'own': [0, 0, 2]})
    frame = pl.DataFrame({'qid': [1, 0, 2], 'tid': [10, 11, 21], 'co': ['us', 'us', 'india'],
        'y': [0, 1, 1], 'own': [0, 0, 2], 'fold': [1, 1, 1], 'seg': [2, 2, 2],
        **{score: [.4, .9999, .3] for score in features.SCORES}, 'np_m0': [.4, .9999, .3]})
    frame, _ = features.prepare_features(frame, refs, targets)
    frame = frame.with_columns(pl.Series('baseline_p', [.4, .9999, .3]),
                               pl.Series('baseline_raw', [.4, .9999, .3]))
    base = pl.DataFrame({'qid': [0], 'tid': [11], 'p': [.9999]})
    return frame, refs, targets, base


def test_new_alias_edge_expands_universe_with_measured_features_and_truth_attached_last():
    frame, refs, targets, base = fixture()
    expanded, names, report = retrieval.expand(frame, refs, targets, base, s2_count=20)
    assert report['genuinely_new_pairs'] == 1
    assert report['expanded_pairs'] == frame.height+1
    assert report['selection_uses_labels'] is False
    fresh = expanded.filter(pl.col('ir_new_edge') == 1)
    assert fresh.select('qid', 'tid').rows() == [(0, 10)]
    assert fresh['y'].to_list() == [1]
    assert fresh['own'].to_list() == [0]
    assert fresh['name_jaccard'][0] > 0
    assert fresh['ir_name_witnesses'][0] > 0 and fresh['ir_address_witnesses'][0] > 0
    assert fresh['ir_s2_path'][0] == 1
    assert fresh['newest'].null_count() == 1
    assert fresh['np_m0'].null_count() == 1
    assert fresh['newest_present'][0] == fresh['np_m0_present'][0] == 0
    assert fresh['newest_logit'][0] == pytest.approx(np.log(1e-4/(1-1e-4)), rel=1e-6)
    assert fresh['baseline_p'][0] == pytest.approx(1e-4)
    assert fresh['competitor_count_log'][0] == pytest.approx(np.log(3))

    old = expanded.filter(pl.col('ir_new_edge') == 0)
    assert old.select('qid', 'tid', 'baseline_p', 'y', 'newest').equals(
        frame.select('qid', 'tid', 'baseline_p', 'y', 'newest'))
    assert not {'qid', 'tid', 'y', 'own', 'fold', 'deg', 'co'}.intersection(names)


def test_proposals_and_evidence_ignore_truth_fold_and_degree():
    frame, refs, targets, base = fixture()
    keys, evidence, _ = retrieval.retrieve_pairs(frame, refs, targets, base, 20)
    poisoned = frame.with_columns((1-pl.col('y')).alias('y'), pl.lit(-1).alias('own'), pl.lit(99).alias('fold'))
    other_keys, other_evidence, _ = retrieval.retrieve_pairs(poisoned,
        refs.with_columns(pl.lit(9000).alias('deg'), pl.lit(0).alias('fold')),
        targets.with_columns(pl.lit(-1).alias('own')), base, 20)
    assert keys.sort('qid', 'tid').equals(other_keys.sort('qid', 'tid'))
    assert evidence.sort('qid', 'tid').equals(other_evidence.sort('qid', 'tid'))


def test_self_alias_cannot_generate_own_evidence():
    frame, refs, targets, base = fixture()

    second = frame.head(1).with_columns(pl.lit(11, pl.Int64).alias('tid'))
    frame = pl.concat([frame, second]).with_columns(
        pl.when(pl.col('tid') == 11).then(.996).otherwise(pl.col('baseline_p')).alias('baseline_p'))
    base = base.with_columns(pl.lit(.996).alias('p'))
    _, evidence, _ = retrieval.retrieve_pairs(frame, refs, targets, base, 20)
    own = evidence.filter((pl.col('qid') == 0) & (pl.col('tid') == 11))

    assert not own.height or (own['ir_name_alias_paths'][0] == 0 and own['ir_address_alias_paths'][0] == 0)


def test_reciprocal_collision_caps_and_country_blocking():
    frame, refs, targets, base = fixture()
    _, evidence, _ = retrieval.retrieve_pairs(frame, refs, targets, base, 20)
    assert not evidence.filter((pl.col('qid') == 0) & (pl.col('tid') == 21)).height
    duplicate = targets.filter(pl.col('rid') == 10).with_columns(pl.lit(12, pl.Int64).alias('rid'))
    targets = pl.concat([targets, duplicate])
    new, _, report = retrieval.retrieve_pairs(frame, refs, targets, base, 20, max_query_df=1)
    assert new.height == 0
    assert report['queried_targets'] == 3


def test_pair_cap_and_label_free_schema():
    frame, refs, targets, base = fixture()
    test_frame = frame.drop('y', 'own', 'fold')
    test_targets = targets.drop('own')
    expanded, _, report = retrieval.expand(test_frame, refs.drop('fold', 'deg'), test_targets,
                                          base, 20, max_queries=1, owners_per_target=1)
    assert report['queried_targets'] == 1
    assert report['genuinely_new_pairs'] == 1
    assert 'y' not in expanded and 'own' not in expanded
    with pytest.raises(ValueError, match='already been expanded'):
        retrieval.expand(expanded, refs, targets, base, 20)


def test_global_collision_feature_is_not_computed_on_new_edge_subset():
    frame, refs, targets, base = fixture()
    duplicate = refs.filter(pl.col('rid') == 0).with_columns(pl.lit(3, pl.Int64).alias('rid'))
    refs = pl.concat([refs, duplicate])
    rows, _, _ = retrieval.expand(frame, refs, targets, base, 20)
    fresh = rows.filter((pl.col('qid') == 0) & (pl.col('tid') == 10))
    assert fresh['ref_name_twins_log'][0] == pytest.approx(np.log(3))


def test_full_model_pipeline_retrieval_runs_isolation_reranking_and_export(tmp_path, monkeypatch):
    import lightgbm as lgb
    from er.stack import decode
    from er.stack.inputs import load_refs
    from latest_fusion import innovation_models as models
    from latest_fusion.model_innovation import run
    from latest_fusion.structured_pipeline import replay, METHOD
    from latest_fusion.tuning import fit_population
    from test_structured_pipeline import make_fixture

    monkeypatch.setenv('ER_THREADS', '2')
    monkeypatch.setattr(models, 'ROUNDS', {backend: 5 for backend in models.BACKENDS})
    data, prepared, incumbent, checksum = make_fixture(tmp_path)



    for split in ('train', 'test'):
        refpath = data/split/'ref.parquet'
        sourcepath = data/split/'s3.parquet'
        refs = pl.read_parquet(refpath).with_columns(
            pl.when(pl.col('rid') == 500).then(pl.lit('Aurelia Novastream')).otherwise(pl.col('nm')).alias('nm'),
            pl.when(pl.col('rid') == 500).then(pl.lit('71 Cedarbank Grove')).otherwise(pl.col('ad')).alias('ad'))
        if split == 'train':
            refs = refs.with_columns(pl.when(pl.col('rid') == 500).then(2).otherwise(pl.col('deg')).alias('deg'))
        refs.write_parquet(refpath)
        target = pl.read_parquet(sourcepath).with_columns(
            pl.when(pl.col('rid') == 1700).then(pl.lit('Aurelia Novastream')).otherwise(pl.col('nm')).alias('nm'),
            pl.when(pl.col('rid') == 1700).then(pl.lit('71 Cedarbank Grove')).otherwise(pl.col('ad')).alias('ad'),
            pl.when(pl.col('rid') == 1700).then(500).otherwise(pl.col('own')).alias('own'))
        target.write_parquet(sourcepath)
        path = prepared/f'prepared-{split}.parquet'
        cands = pl.read_parquet(path).with_columns(
            pl.when(pl.col('tid') == 1700).then(501).otherwise(pl.col('qid')).cast(pl.UInt32).alias('qid'))
        if split == 'train':
            cands = cands.with_columns(pl.when(pl.col('tid') == 1700).then(500).otherwise(pl.col('own')).alias('own'))
        cands.write_parquet(path)
        artifact = 'validation_predictions.parquet' if split == 'train' else 'test_predictions.parquet'
        cands.select('qid', 'tid', *(['y'] if split == 'train' else [])).write_parquet(incumbent/artifact)
    prev = json.loads((incumbent/'report.json').read_text())
    residual = lgb.Booster(model_file=str(incumbent/'residual_model.txt'))
    original = pl.read_parquet(prepared/'prepared-train.parquet')
    _, _, accepted = replay(original, residual, ['newest_logit'], prev, 'train')
    trainrefs = load_refs(str(data), 'train')
    _, tuneids = fit_population(trainrefs)
    tune = trainrefs.filter(pl.col('rid').is_in(pl.Series(tuneids).implode())).select(pl.col('rid').alias('qid'), 'deg')
    prev['methods'][METHOD]['tune'].update(decode.score(accepted, tune))
    (incumbent/'report.json').write_text(json.dumps(prev))
    report = run(data, prepared/'prepared-train.parquet', prepared/'prepared-test.parquet',
                 incumbent, tmp_path/'result', 'reciprocal_retrieval', expected_sha=checksum)
    assert report['validation']['source1'] == 1200
    assert report['protocol']['target_overlap'] == 0
    assert report['protocol']['calibration_check_target_overlap'] == 0
    assert report['train_preparation']['selection_uses_labels'] is False
    assert report['train_preparation']['genuinely_new_pairs'] >= 1
    assert report['train_preparation']['new_true_pairs'] >= 1
    assert report['train_preparation']['new_covered_true_pairs'] > report['train_preparation']['old_covered_true_pairs']
    assert report['model']['full_candidate_reranking'] is True
    assert (tmp_path/'result'/'output'/'matching_results.tsv').exists()
    assert (tmp_path/'result'/'locked_policy.json').exists()

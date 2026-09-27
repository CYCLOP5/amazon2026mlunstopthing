'small transfer and immutable-cache checks; full data processing is cloud only'
import json
from pathlib import Path
import sys

import polars as pl
import pytest
from polars.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boosted_hybrid.features import normalize_records, pair_features, model_columns, run


def records():
    return pl.DataFrame({'rid': [0, 1, 2, 3, 4],
        'nm': ['Café Lumière S.A.R.L.', 'Cafe Lumiere SAS', 'Mimosa', 'Other LLC', 'Other LLC'],
        'ad': ['12 Rue Sainte Anne, LA', '12 Rue Saint Anne, LA', '', '10 Main St, LA', '10 Main St, MH'],
        'co': ['france', 'france', 'france', 'us', 'india']}).with_columns(pl.col('rid').cast(pl.UInt32))


def test_country_scoping_preserves_french_named_components():
    d = normalize_records(records())
    assert d['address_scoped'][0] == '12 rue sainte anne la'
    assert d['address_scoped'][1] == '12 rue saint anne la'
    assert d['state'].to_list() == ['', '', '', 'la', 'mh']
    assert d['sainte'].to_list() == [1, 0, 0, 0, 0]
    assert d['address_scoped'][3] == '10 main st'
    assert d['name_core'][0] == 'cafe lumiere'
    assert d['legal'][0] == 'sarl'
    assert d['legal'][2] == ''


def test_raw_and_legal_comparators_keep_ambiguous_evidence_and_mask_empty():
    d = normalize_records(records())
    keys = pl.DataFrame({'qid': [0, 2, 3], 'tid': [1, 2, 4]}, schema={'qid': pl.UInt32, 'tid': pl.UInt32})
    p = pair_features(keys, d, d)
    assert p['raw_name_core_tset'][0] == 100
    assert p['raw_name_fold_ratio'][0] < 100
    assert p['raw_legal_eq'][0] == 0
    assert p['geo_sainte_conflict'][0] == 1
    assert p['raw_address_fold_ratio'][1] == 0
    assert p['geo_address_missing1'][1] == 1
    assert p['geo_state_conflict'][2] == 1
    assert p['raw_first_number_eq'][0] == 1
    assert all(t.is_numeric() for t in p.schema.values())


def test_model_columns_remove_labels_parent_and_sampler_features():
    d = pl.DataFrame({k: [1.] for k in ('qid', 'tid', 'co', 'fold', 'own', 'y', 'deg',
        '_prior', 'base_lg', 'has_parent', 'parent_p', 'parent_best', 'from_prepared',
        'hit_dense_name', 'dense_name_rank', 'rrf', 's1_occupancy', 's1_name_freq',
        'dense_name', 'lex_address', 'ce_base_lg_margin', 'raw_name_fold_ratio')})
    assert set(model_columns(d, 'independent')) == {'s1_name_freq', 'dense_name',
        'lex_address', 'ce_base_lg_margin', 'raw_name_fold_ratio'}
    residual = model_columns(d, 'residual')
    assert {'base_lg', 'has_parent', 'parent_p'} <= set(residual)
    assert not {'qid', 'tid', 'co', 'y', 'own', 'deg', 'from_prepared', 's1_occupancy'} & set(residual)
    with pytest.raises(ValueError, match='Unknown'):
        model_columns(d, 'other')


def write_fixture(tmp_path):
    data, cache = tmp_path/'data', tmp_path/'cache'
    cache.mkdir()
    raw = records().with_columns(pl.col('rid').cast(pl.String).alias('eid'),
        pl.lit(0).alias('fold'), pl.lit(1).alias('deg'))
    for split in ('train', 'test'):
        (data/split).mkdir(parents=True)
        raw.write_parquet(data/split/'ref.parquet')
        raw.head(3).write_parquet(data/split/'s2.parquet')
        raw.tail(2).write_parquet(data/split/'s3.parquet')
        base = pl.DataFrame({'qid': [0, 2, 3], 'tid': [1, 2, 4], 'n_tset': [90., 100., 100.],
            'a_tset': [95., 0., 100.], 'parent_p': [.2, .8, .3],
            'y': [0, 1, 0], 'own': [1, 2, 4], 'co': ['france', 'france', 'us']})
        base = base.with_columns(pl.col('qid', 'tid').cast(pl.UInt32), pl.col('n_tset').cast(pl.Float32))
        base.write_parquet(cache/f'{split}_features.parquet')
    return data, cache


def test_run_preserves_cached_values_order_and_ignores_labels(tmp_path):
    data, cache = write_fixture(tmp_path)
    base = pl.read_parquet(cache/'train_features.parquet')
    report = run(data, cache, tmp_path/'output', 'train')
    res = pl.read_parquet(tmp_path/'output/train_features.parquet')
    assert_frame_equal(res.select(base.columns), base)
    assert report['labels_used'] is False
    assert (tmp_path/'output/_train_SUCCESS').exists()
    assert json.loads((tmp_path/'output/features_train.json').read_text())['pairs'] == 3
    modified = base.with_columns(pl.lit(99).alias('own'), pl.lit(1).alias('y'))
    modified.write_parquet(cache/'train_features.parquet')
    run(data, cache, tmp_path/'output2', 'train')
    result2 = pl.read_parquet(tmp_path/'output2/train_features.parquet')
    assert_frame_equal(res.select(report['added_columns']), result2.select(report['added_columns']))
    run(data, cache, tmp_path/'test_output', 'test')
    assert (tmp_path/'test_output/_test_SUCCESS').exists()


def test_missing_raw_record_fails_instead_of_silent_imputation():
    d = normalize_records(records())
    with pytest.raises(ValueError, match='missing observable'):
        pair_features(pl.DataFrame({'qid': [99], 'tid': [0]},
            schema={'qid': pl.UInt32, 'tid': pl.UInt32}), d, d)

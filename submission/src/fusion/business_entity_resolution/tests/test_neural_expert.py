'small tests of supervised isolation and neural text preparation; no gpu needed'
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from innovation.neural_expert import attach_text, owner_partitions, pair_texts, prepare
from test_edit_channel import fixture


def test_masking_keeps_distinctive_names_numbers_and_addresses():
    row = {'nm1': 'ACME 12 Holdings North', 'nm2': 'Acme 12 Holdings South',
           'ad1': '1958 River Road', 'ad2': 'Door No #834 House No 1958 River Rd', 'y': 0}
    first, second = pair_texts(row, mask_shared=True)
    assert first == 'address: 1958 River Road\nname: business 12 business North'
    assert second == 'address: Door No #834 House No 1958 River Rd\nname: business 12 business South'
    assert pair_texts(dict(row, y=1), True) == (first, second)
    assert pair_texts(row)[0].endswith('ACME 12 Holdings North')


def test_missing_addresses_do_not_trigger_name_masking():
    row = {'nm1': 'Acme Ltd', 'nm2': 'Acme Limited', 'ad1': None, 'ad2': '12 Road'}
    assert pair_texts(row, True) == pair_texts(row, False)
    assert pair_texts({'nm1': None, 'nm2': None, 'ad1': None, 'ad2': None}) == (
        'address: \nname: ', 'address: \nname: ')


def test_owner_and_candidate_partitions_exclude_outer_and_internal_holdouts():
    refs = pl.DataFrame({'rid': np.arange(500, dtype=np.uint32),
                         'fold': [0]*100+[1]*100+[2]*300})
    holdout = refs.filter((pl.col('fold') == 2) &
                          ((pl.col('rid').cast(pl.Int64).hash(3719)%10) == 0))['rid'][0]
    train = refs.filter((pl.col('fold') == 2) &
                        ((pl.col('rid').cast(pl.Int64).hash(3719)%10) != 0))['rid'].to_list()
    cand, owner = train[:2]
    rows = pl.DataFrame({'qid': [cand]*5+[0, 100, holdout, cand],
        'own': [owner, 0, 100, holdout, -1, owner, owner, owner, owner],
        'tid': list(range(9))}).with_columns(pl.col('qid', 'tid').cast(pl.UInt32))
    sel = owner_partitions(rows, refs)
    assert sel['_fit'].to_list() == [True, False, False, False, False, False, False, False, True]
    assert sel['_valid'].to_list() == [False, False, False, True, False, False, False, False, False]


def test_text_join_preserves_pair_order_allows_null_text_rejects_unknown_ids():
    refs = pl.DataFrame({'rid': [2, 1], 'nm': ['B', None], 'ad': ['B Road', ''], 'co': ['us', 'us']})
    targets = pl.DataFrame({'rid': [8, 9], 'nm': ['A', 'B'], 'ad': ['', None], 'co': ['us', 'us']})
    pairs = pl.DataFrame({'qid': [1, 2], 'tid': [9, 8], 'y': [0, 0]})
    res = attach_text(pairs, refs, targets)
    assert res.select('qid', 'tid').equals(pairs.select('qid', 'tid'))
    assert res['nm1'].to_list() == [None, 'B']
    with pytest.raises(ValueError, match='unknown IDs'):
        attach_text(pl.DataFrame({'qid': [55], 'tid': [8]}), refs, targets)


def test_preparation_respects_budget_and_isolates_all_evaluation_labels(tmp_path):
    data, parent, prepared, refs = fixture(tmp_path, n=600)
    output = tmp_path/'training'
    prepare(data, prepared, output, max_pairs=600)
    train = pl.read_parquet(output/'fit.parquet')
    valid = pl.read_parquet(output/'valid.parquet')
    owners = pl.concat([pl.read_parquet(data/'train'/f's{i}.parquet', columns=['rid', 'own'])
                       for i in (2, 3)]).rename({'rid': 'tid'})
    for pairs, name in ((train, '_fit'), (valid, '_valid')):
        sel = owner_partitions(pairs.select('qid', 'tid').join(owners, on='tid'), refs)
        assert sel[name].all()
    assert train.height <= 600
    assert train.select('qid', 'tid').n_unique() == train.height
    assert train.join(valid.select('qid', 'tid'), on=['qid', 'tid']).height == 0

    path = prepared/'train/features.parquet'
    frame = pl.read_parquet(path)
    partition = owner_partitions(frame, refs)
    changed = frame.with_columns(pl.when(pl.Series(~(partition['_fit'] | partition['_valid'])))
        .then(1-pl.col('y')).otherwise(pl.col('y')).alias('y'))
    changed.write_parquet(path)
    prepare(data, prepared, tmp_path/'training2', max_pairs=600)
    assert train.equals(pl.read_parquet(tmp_path/'training2/fit.parquet'))
    assert valid.equals(pl.read_parquet(tmp_path/'training2/valid.parquet'))
    assert (output/'_SUCCESS').exists()
    assert json.loads((output/'preparation.json').read_text())['fit_pairs'] == train.height

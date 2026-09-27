import json
from pathlib import Path
import sys

import polars as pl
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from large_reranker.protocol import encode_rows, PREFIX, SUFFIX, worker_tasks
from large_reranker.gpu import gather


class Tokenizer:
    def encode(self,text,add_special_tokens=False):

        return text.split()


def test_both_records_and_suffix_survive_long_fields():
    row = {'nm1':'FIRST '+('a '*1000),'ad1':'11 RUE '+('b '*1000),'co1':'france',
           'nm2':'SECOND '+('c '*1000),'ad2':'12 RUE '+('d '*1000),'co2':'france','y':1}
    ids = encode_rows(Tokenizer(),[row],256)[0]
    assert len(ids) <= 256
    assert 'FIRST' in ids and 'SECOND' in ids and '11' in ids and '12' in ids
    assert ids[-len(SUFFIX.split()):] == SUFFIX.split()
    assert encode_rows(Tokenizer(),[{**row,'y':0}],256)[0] == ids


def test_replica_tasks_cover_every_shard_exactly_once():
    tasks = list(range(103))
    parts = [worker_tasks(tasks,r,4,s,2) for s in range(2) for r in range(4)]
    assert sorted(x for part in parts for x in part) == tasks
    assert max(map(len,parts))-min(map(len,parts)) <= 1


def test_pair_coverage_rejects_duplicate_or_missing_score(tmp_path):
    prepared, a, b, out = [tmp_path/n for n in ('prepared','a','b','out')]
    for split in ('train','test'):
        (prepared/split).mkdir(parents=True)
        pl.DataFrame({'qid':[1,2],'tid':[10,20]}).write_parquet(prepared/split/'features.parquet')
        for i,root in enumerate((a,b)):
            (root/split).mkdir(parents=True)
            pl.DataFrame({'qid':[i+1],'tid':[(i+1)*10],'large_lg':[1.]}).write_parquet(root/split/'part.parquet')
    for i,root in enumerate((a,b)):
        (root/'_SUCCESS').write_text('done')
        (root/'shard.json').write_text(json.dumps({'shard_index':i,'shards':2}))
    assert gather(prepared,[a,b],out) == out
    pl.DataFrame({'qid':[1],'tid':[10],'large_lg':[1.]}).write_parquet(b/'test/part.parquet')
    with pytest.raises(ValueError,match='coverage'):
        gather(prepared,[a,b],tmp_path/'bad')

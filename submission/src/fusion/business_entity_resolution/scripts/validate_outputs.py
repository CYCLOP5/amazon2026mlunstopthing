'check tsv structure, ids, duplicates, s1 coverage, and matches ⊆ candidates'
import argparse
from pathlib import Path
import polars as pl


def read_lists(path, column, refs):
    d = pl.read_csv(path, separator='\t', quote_char=None, infer_schema_length=0)
    if d.columns != ['source1_entity_id',column]:
        raise ValueError(f'{path}: invalid header')
    if d.height != refs.height or d['source1_entity_id'].n_unique() != d.height:
        raise ValueError(f'{path}: missing or duplicate Source-1 rows')
    if d.join(refs,on='source1_entity_id',how='anti').height:
        raise ValueError(f'{path}: unknown Source-1 ID')
    d = d.with_columns(pl.col(column).fill_null(''))
    if d.filter(pl.col(column).str.contains(r'(^,|,,|,$)')).height:
        raise ValueError(f'{path}: empty ID inside nonempty list')
    d = d.with_columns(pl.col(column).str.split(',').list.eval(pl.element().filter(pl.element() != '')))
    if d.filter(pl.col(column).list.len() != pl.col(column).list.n_unique()).height:
        raise ValueError(f'{path}: repeated target ID in a row')
    return d


def validate(data, output):
    refs = pl.read_parquet(data/'test/ref.parquet',columns=['eid']).rename({'eid':'source1_entity_id'})
    targets = pl.concat([pl.read_parquet(data/f'test/s{sr}.parquet',columns=['eid']) for sr in (2,3)])
    cands = read_lists(output/'candidate_pairs.tsv','candidate_entity_ids',refs)
    matches = read_lists(output/'matching_results.tsv','matched_entity_ids',refs)
    both = cands.join(matches,on='source1_entity_id')
    if both.filter(pl.col('matched_entity_ids').list.set_difference('candidate_entity_ids').list.len() > 0).height:
        raise ValueError('Matches outside the candidate pool')

    ids = cands.select(pl.col('candidate_entity_ids').explode().alias('eid')).drop_nulls().unique()
    if ids.join(targets,on='eid',how='anti').height:
        raise ValueError('Unknown Source-2/3 ID in candidates')
    res = {'source1':refs.height,'matches':int(matches['matched_entity_ids'].list.len().sum()),
              'candidate_pairs':int(cands['candidate_entity_ids'].list.len().sum())}
    print(f'PASS: {res}')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a = ap.parse_args()
    validate(a.data,a.output)

'data and text protocol for a small, independently trained neural specialist'
import json
from pathlib import Path
import re

import polars as pl

from innovation.common import log

BASE_MODEL = 'intfloat/multilingual-e5-large'

BASE_REVISION = 'ab10c1a7f42e74530fe7ae5be82e6d4f11a719eb'


def pair_texts(row, mask_shared=False):
    'address first protects it from truncation; no label controls augmentation'
    n1, n2 = row['nm1'] or '', row['nm2'] or ''
    a1, a2 = row['ad1'] or '', row['ad2'] or ''
    if mask_shared and a1.strip() and a2.strip():
        common = ({x.casefold() for x in re.findall(r'\w+', n1) if x.isalpha() and len(x) >= 2} &
                  {x.casefold() for x in re.findall(r'\w+', n2) if x.isalpha() and len(x) >= 2})
        def replace(text):
            return re.sub(r'\w+', lambda m: 'business' if m.group().casefold() in common else m.group(), text)
        n1, n2 = replace(n1), replace(n2)
    return (f'address: {a1}\nname: {n1}', f'address: {a2}\nname: {n2}')


def owner_partitions(rows, refs):
    'both candidate and owner must be fold 2, even for competing negatives'
    d = rows.join(refs.select(pl.col('rid').cast(pl.Int64).alias('own'),
        pl.col('fold').alias('_owner_fold')), on='own', how='left', maintain_order='left').join(
        refs.select(pl.col('rid').alias('qid'), pl.col('fold').alias('_candidate_fold')),
        on='qid', how='left', maintain_order='left')
    permitted = ((pl.col('_owner_fold') == 2) & (pl.col('_candidate_fold') == 2) &
                 (pl.col('own') >= 0)).fill_null(False)
    own_valid = (pl.col('own').cast(pl.Int64).hash(3719)%10) == 0
    candidate_valid = (pl.col('qid').cast(pl.Int64).hash(3719)%10) == 0
    return d.with_columns((permitted & ~own_valid & ~candidate_valid).alias('_fit'),
                          (permitted & own_valid).alias('_valid'))


def attach_text(pairs, refs, targets):
    left = refs.select(pl.col('rid').alias('qid'), pl.col('nm').alias('nm1'),
                       pl.col('ad').alias('ad1'), pl.col('co').alias('co1'),
                       pl.lit(True).alias('_ref_found'))
    right = targets.select(pl.col('rid').alias('tid'), pl.col('nm').alias('nm2'),
                           pl.col('ad').alias('ad2'), pl.col('co').alias('co2'),
                           pl.lit(True).alias('_target_found'))
    text = pairs.join(left, on='qid', how='left', validate='m:1', maintain_order='left').join(
        right, on='tid', how='left', validate='m:1', maintain_order='left')
    if (text.height != pairs.height or text['_ref_found'].null_count() or
        text['_target_found'].null_count()):
        raise ValueError('Neural text pairs reference unknown IDs')
    return text.drop('_ref_found', '_target_found')


def prepare(data, prepared, output, max_pairs=600_000):
    from er.stack.inputs import load_refs, load_targets
    if max_pairs < 20:
        raise ValueError('max_pairs must be at least 20')
    if not (Path(prepared)/'_SUCCESS').exists():
        raise RuntimeError('Prepared pair selection is incomplete')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    refs, targets = load_refs(data, 'train'), load_targets(data, 'train')
    f = pl.read_parquet(Path(prepared)/'train/features.parquet', columns=['qid', 'tid', 'own', 'y'])
    hard = owner_partitions(f, refs)

    valid = hard.filter(pl.col('_valid')).sort(pl.struct('qid', 'tid').hash(481)).head(30_000)
    train = hard.filter(pl.col('_fit')).sort(pl.struct('qid', 'tid').hash(479)).head(max_pairs*3//4)
    train = train.select('qid', 'tid', 'y').with_columns(pl.lit('hard').alias('kind'))
    owned = pl.concat([pl.read_parquet(Path(data)/'train'/f's{s}.parquet', columns=['rid', 'own']) for s in (2, 3)])
    positive = owned.filter(pl.col('own') >= 0).select(pl.col('own').cast(pl.UInt32).alias('qid'),
        pl.col('rid').cast(pl.UInt32).alias('tid'), pl.col('own').cast(pl.Int64), pl.lit(1, pl.UInt8).alias('y'))
    positive = owner_partitions(positive, refs).filter(pl.col('_fit')).sort(
        pl.struct('qid', 'tid').hash(487)).head(max_pairs//4).select('qid', 'tid', 'y').with_columns(
            pl.lit('positive_anchor').alias('kind'))
    train = pl.concat([train, positive], how='vertical_relaxed').unique(['qid', 'tid'], keep='first')
    train = train.sort(pl.struct('qid', 'tid').hash(491))
    valid = valid.select('qid', 'tid', 'y').with_columns(pl.lit('hard_holdout').alias('kind'))
    if train.height < 10 or min(train['y'].sum(), train.height-train['y'].sum()) < 2:
        raise ValueError('Neural training selection has insufficient classes')
    if valid.height < 2:
        raise ValueError('Neural internal owner holdout is empty')
    for name, pairs in [('fit', train), ('valid', valid)]:
        if pairs.select('qid', 'tid').n_unique() != pairs.height:
            raise ValueError('Duplicate neural training pair')
        text = attach_text(pairs, refs, targets)
        text.write_parquet(output/f'{name}.parquet')
    report = {'fit_pairs': train.height, 'valid_pairs': valid.height,
        'fit_positive_rate': float(train['y'].mean()), 'valid_positive_rate': float(valid['y'].mean()),
        'fit_kinds': train.group_by('kind').len().to_dicts(), 'model': BASE_MODEL,
        'revision': BASE_REVISION, 'source': 'supplied challenge labels only',
        'partition': 'candidate and true owner fold 2; internal owner AND candidate hash holdout',
        'augmentation': '30% shared-name masking, only for two addressed records; labels unchanged',
        'max_pairs': max_pairs}
    (output/'preparation.json').write_text(json.dumps(report, indent=2))
    (output/'_SUCCESS').write_text('complete\n')
    log(f'Neural expert preparation: {report}')

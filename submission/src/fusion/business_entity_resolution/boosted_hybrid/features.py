'add country-aware surface evidence to immutable cached hybrid features'
import json
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from er.features.name import cp
from er.normalize.normalizer import ascii_fold
from er.normalize.rules import ADDR_NULL_COMPONENTS, IN_STATES, US_STATES
from er.safe import guard
from er.stack.inputs import load_refs, load_targets
from innovation.common import log

CHUNK_ROWS = 200_000
LEGAL_SUFFIX = r"(?:sarl|sas|sasu|eurl|sci|sa|ste|societe|cie|ltd|limited|llc|llp|inc|incorporated|corp|corporation|pvt|private|plc)"


def _clean(expr):
    return expr.fill_null('').str.normalize('NFKC').str.to_lowercase().str.replace_all(
        r"[^\p{L}\p{N}]+", ' ').str.strip_chars()


def normalize_records(records):
    'return temporary observable fields; preserve sainte and named components'
    required = {'rid', 'nm', 'ad', 'co'}
    if not required.issubset(records.columns):
        raise ValueError('Raw records lack required observable fields')
    d = records.select('rid', 'co', _clean(pl.col('nm')).alias('name_unicode'),
        _clean(pl.col('ad')).alias('address_unicode'),
        pl.col('ad').fill_null('').str.strip_chars().alias('_ad'))

    d = d.with_columns(ascii_fold(d['name_unicode']).alias('name_fold'),
        ascii_fold(d['address_unicode']).alias('address_fold'),
        ascii_fold(d['_ad']).alias('_ad_fold'))


    states = {}
    for country, vocabulary in (('us', US_STATES), ('india', IN_STATES)):
        states[country] = {**vocabulary, **{v: v for v in vocabulary.values()}}
    components = d.select('rid', 'co', pl.col('_ad_fold').str.split(',').alias('_comp')).explode(
        '_comp', empty_as_null=True)
    components = components.with_columns(pl.col('_comp').str.strip_chars().alias('_comp'))
    state = pl.lit(None, dtype=pl.String)
    for country, vocabulary in states.items():
        state = pl.when(pl.col('co') == country).then(
            pl.col('_comp').replace_strict(vocabulary, default=None)).otherwise(state)
    components = components.with_columns(state.alias('_state'))
    scoped = components.group_by('rid', maintain_order=True).agg(
        pl.col('_state').drop_nulls().first().fill_null('').alias('state'),
        pl.col('_comp').filter(pl.col('_state').is_null()).str.join(' ').alias('_address_scoped'))
    d = d.join(scoped, on='rid', how='left', maintain_order='left', validate='1:1')


    d = d.with_columns(pl.col('name_fold').alias('_name'))

    d = d.with_columns(pl.col('_name').str.replace_all(r'\bs a r l\b', 'sarl')
        .str.replace_all(r'\bs a s u\b', 'sasu').str.replace_all(r'\bs a s\b', 'sas').alias('_name'))
    d = d.with_columns(
        pl.col('_name').str.replace(rf'(?:\s+{LEGAL_SUFFIX})+$', '').alias('name_core'),
        pl.col('_name').str.extract(rf'\b((?:{LEGAL_SUFFIX})(?:\s+{LEGAL_SUFFIX})*)$', 1)
            .fill_null('').alias('legal'),
        _clean(pl.col('_address_scoped')).alias('address_scoped'),
        pl.col('_ad_fold').is_in(ADDR_NULL_COMPONENTS).cast(pl.UInt8).alias('address_missing'),
        pl.col('address_fold').str.extract_all(r'\b\d+\b').list.join(' ').alias('numbers'),
        pl.col('address_fold').str.contains(r'\bsainte\b').cast(pl.UInt8).alias('sainte'),
        pl.col('address_fold').str.contains(r'\bsaint\b').cast(pl.UInt8).alias('saint'),
        pl.col('address_fold').str.contains(r'\b(?:suite|ste)\b').cast(pl.UInt8).alias('suite'))
    return d.drop('_ad', '_ad_fold', '_address_scoped', '_name')


def pair_features(keys, refs, targets):
    'new numeric columns only, in exactly the supplied pair order'
    attrs = [c for c in refs.columns if c not in {'rid', 'co'}]
    p = keys.join(refs.select(pl.col('rid').alias('qid'),
        *[pl.col(c).alias(c+'_1') for c in attrs]), on='qid', how='left',
        maintain_order='left', validate='m:1').join(
        targets.select(pl.col('rid').alias('tid'), *[pl.col(c).alias(c+'_2') for c in attrs]),
        on='tid', how='left', maintain_order='left', validate='m:1')
    if p['name_unicode_1'].null_count() or p['name_unicode_2'].null_count():
        raise ValueError('Cached pair refers to a missing observable record')
    values = {}
    for kind in ('name_unicode', 'name_fold', 'name_core', 'address_unicode', 'address_fold', 'address_scoped'):
        a, b = p[kind+'_1'].to_list(), p[kind+'_2'].to_list()
        both = p.select(((pl.col(kind+'_1') != '') & (pl.col(kind+'_2') != ''))
                        .cast(pl.Float32)).to_series().to_numpy()
        for suffix, scorer in (('ratio', fuzz.ratio), ('tset', fuzz.token_set_ratio)):
            values['raw_'+kind+'_'+suffix] = cp(a, b, scorer)*both
        if kind in ('name_fold', 'address_fold'):
            values['raw_'+kind+'_jw'] = cp(a, b, JaroWinkler.normalized_similarity)*both
    def nonempty_eq(a, b):
        return ((pl.col(a) != '') & (pl.col(b) != '') & (pl.col(a) == pl.col(b))).cast(pl.UInt8)
    out = p.select(
        *[pl.col('address_missing_'+s).alias('geo_address_missing'+s) for s in ('1', '2')],
        nonempty_eq('state_1', 'state_2').alias('geo_state_eq'),
        ((pl.col('state_1') != '') & (pl.col('state_2') != '') &
            (pl.col('state_1') != pl.col('state_2'))).cast(pl.UInt8).alias('geo_state_conflict'),
        nonempty_eq('numbers_1', 'numbers_2').alias('raw_numbers_eq'),
        ((pl.col('numbers_1') != '') & (pl.col('numbers_2') != '') &
            (pl.col('numbers_1').str.split(' ').list.first() ==
             pl.col('numbers_2').str.split(' ').list.first())).cast(pl.UInt8).alias('raw_first_number_eq'),
        ((pl.col('numbers_1') != '') & (pl.col('numbers_2') != '') &
            (pl.col('numbers_1').str.split(' ').list.first() !=
             pl.col('numbers_2').str.split(' ').list.first())).cast(pl.UInt8).alias('raw_first_number_conflict'),
        ((pl.col('numbers_1') != '') & (pl.col('numbers_2') != '')).cast(pl.UInt8).alias('raw_numbers_present'),
        nonempty_eq('legal_1', 'legal_2').alias('raw_legal_eq'),
        *[(pl.col('legal_'+s) == '').cast(pl.UInt8).alias('raw_legal_missing'+s) for s in ('1', '2')],
        *[(pl.col(k+'_1') != pl.col(k+'_2')).cast(pl.UInt8).alias('geo_'+k+'_conflict')
          for k in ('saint', 'sainte', 'suite')],
        *[pl.col(k+'_'+s).str.len_chars().cast(pl.Float32).alias('raw_'+k+'_len'+s)
          for k in ('name_fold', 'address_fold') for s in ('1', '2')])
    return pl.DataFrame(values).hstack(out)


def model_columns(frame, mode):
    'explicit numeric model inputs; labels and identifiers are never features'
    if mode not in ('independent', 'residual'):
        raise ValueError('Unknown boosted head mode')
    exclude = {'qid', 'tid', 'rid', 'eid', 'y', 'own', 'owner', 'co', 'country', 'fold', 'deg',
               'from_prepared'}
    res = []
    for name, dtype in frame.schema.items():
        if name in exclude or name.startswith('_') or not dtype.is_numeric():
            continue
        if name.startswith('s1_') and name != 's1_name_freq':
            continue
        if mode == 'independent' and (name in {'base_lg', 'has_parent', 'rrf', 'p', 'p1', 'p2'} or
                name.startswith(('parent_', 'hit_')) or
                (name.startswith(('lex_', 'dense_')) and name.endswith('_rank'))):
            continue
        res.append(name)
    return res


def run(data, hybrid_result, output, split):
    if split not in ('train', 'test'):
        raise ValueError('Unknown split')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(Path(hybrid_result)/f'{split}_features.parquet')
    if frame.select('qid', 'tid').n_unique() != frame.height:
        raise ValueError('Cached hybrid feature pairs are duplicated')
    refs = normalize_records(load_refs(data, split).select('rid', 'nm', 'ad', 'co'))
    log(f'BOOSTED {split}: normalized {refs.height:,} references with country-scoped surface fields')
    sel = frame.select(pl.col('tid').alias('rid')).unique()
    targets = load_targets(data, split).join(sel, on='rid', how='semi')
    targets = normalize_records(targets.select('rid', 'nm', 'ad', 'co'))
    log(f'BOOSTED {split}: normalized {targets.height:,} selected targets; adding evidence to {frame.height:,} cached pairs')
    parts = []
    for off in range(0, frame.height, CHUNK_ROWS):
        keys = frame.slice(off, CHUNK_ROWS).select('qid', 'tid')
        parts.append(pair_features(keys, refs, targets))
        if off == 0 or (off//CHUNK_ROWS+1)%5 == 0 or off+keys.height == frame.height:
            log(f'BOOSTED {split}: raw surface evidence complete for {off+keys.height:,}/{frame.height:,} pairs')
        guard('boosted hybrid raw pair features')
    if not parts:
        raise ValueError('Empty cached hybrid feature table')
    extra = pl.concat(parts)
    if set(extra.columns) & set(frame.columns):
        raise ValueError('New features collide with cached legacy columns')

    frame = frame.hstack(extra)
    changes = []
    for raw, old in (('raw_name_core_tset', 'n_tset'), ('raw_address_fold_tset', 'a_tset')):
        if old in frame.columns:
            changes.append((pl.col(raw)-pl.col(old)).cast(pl.Float32).alias(raw+'_legacy_delta'))
    frame = frame.with_columns(changes)
    frame.write_parquet(output/f'{split}_features.parquet')
    report = {'split': split, 'pairs': frame.height, 'references': refs.height,
        'selected_targets': targets.height, 'chunk_rows': CHUNK_ROWS,
        'added_columns': extra.columns+[e.meta.output_name() for e in changes],
        'model_columns': {mode: model_columns(frame, mode) for mode in ('independent', 'residual')},
        'normalization': 'Unicode NFKC/raw accent fold; suffix legal forms; country-scoped US/India states; preserve French address components',
        'labels_used': False}
    (output/f'features_{split}.json').write_text(json.dumps(report, indent=2))
    (output/f'_{split}_SUCCESS').write_text('complete\n')
    log(f'BOOSTED {split}: augmented feature cache complete; {frame.height:,} pairs, {len(report["added_columns"])} added columns')
    return report

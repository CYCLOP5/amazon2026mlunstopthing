'reciprocal rare-token witnesses expand the candidate universe before fitting'
import hashlib
import math
import re
import time

import numpy as np
import polars as pl

from . import features as original


KEYS = ['qid', 'tid']
EXTRA = ['ir_new_edge', 'ir_name_witnesses', 'ir_address_witnesses',
         'ir_unique_name_words', 'ir_reciprocal_name', 'ir_reciprocal_address',
         'ir_name_alias_paths', 'ir_address_alias_paths', 'ir_seed_confidence',
         'ir_direct_path', 'ir_s2_path', 'ir_s3_path', 'ir_witness_score']
HIGH_BIT = 1 << 63
STOP = {'the', 'and', 'of', 'for', 'company', 'private', 'limited', 'ltd', 'inc',
        'corp', 'corporation', 'llc', 'pvt', 'road', 'street', 'avenue'}


def _log(message):
    print(time.strftime('%H:%M:%S') + ' reciprocal retrieval: ' + message, flush=True)


def _token(value, char=False):

    digest = int.from_bytes(hashlib.blake2b(value.encode('utf-8'), digest_size=8).digest(), 'little')
    return (digest & (HIGH_BIT-1)) | (HIGH_BIT if char else 0)


def _name_tokens(value):
    words = sorted({w for w in value.split() if len(w) >= 3 and w not in STOP},
                   key=lambda word: (-len(word), word))[:4]
    exact = [_token('word:'+word) for word in words]

    grams = {_token('gram:'+word[i:i+4], True)
             for word in words for i in range(max(0, len(word)-3))}
    return exact + sorted(grams)[:6]


def _address_tokens(value):
    words = sorted({w for w in value.split() if len(w) >= 3 and not w.isdigit() and w not in STOP},
                   key=lambda word: (-len(word), word))[:4]
    numbers = re.findall(r'\d+', value)
    tokens = {_token('address:'+word) for word in words}
    if numbers:
        house = str(int(numbers[0])) if len(numbers[0]) < 20 else numbers[0]
        tokens.update(_token('house:'+house+':'+word) for word in words[:3])
    return sorted(tokens)


def _normalize(records):
    return records.with_columns(
        original.normalized('nm').str.replace_all(original.LEGAL, '').str.replace_all(r'\s+', ' ')
            .str.strip_chars().alias('_name'),
        original.normalized('ad').alias('_address'))


def _postings(records, field, columns, function):

    dictionary = records.select(field).unique().with_columns(
        pl.col(field).map_elements(function, return_dtype=pl.List(pl.UInt64)).alias('_tokens'))
    return (records.select(*columns, field).join(dictionary, on=field, how='left')
            .drop(field).explode('_tokens').rename({'_tokens': '_token'})
            .filter(pl.col('_token').is_not_null()))


def _view_hits(anchors, queries, view, owner_cap, query_cap):
    field, function = ('_name', _name_tokens) if view == 'name' else ('_address', _address_tokens)
    query = _postings(queries, field, ['tid'], function)
    query_df = query.group_by('_token').agg(pl.col('tid').n_unique().alias('_query_df')).filter(
        pl.col('_query_df') <= query_cap)
    query = query.join(query_df, on='_token', how='inner')
    index = _postings(anchors, field, ['qid', '_sid', '_source', '_confidence'], function)


    index = index.join(query_df.select('_token'), on='_token', how='semi')
    owner_df = index.group_by('_token').agg(pl.col('qid').n_unique().alias('_owner_df')).filter(
        pl.col('_owner_df') <= owner_cap)
    index = index.join(owner_df, on='_token', how='inner')
    hit = index.join(query, on='_token', how='inner').filter(pl.col('_sid') != pl.col('tid').cast(pl.Int64))


    tokens = hit.group_by(*KEYS, '_token').agg(
        pl.col('_owner_df').first(), pl.col('_query_df').first())
    summary = tokens.group_by(KEYS).agg(
        pl.len().cast(pl.Float32).alias('ir_'+view+'_witnesses'),
        (1./(pl.col('_owner_df')*pl.col('_query_df')).cast(pl.Float64).sqrt()).sum()
            .cast(pl.Float32).alias('ir_reciprocal_'+view))
    paths = hit.group_by(KEYS).agg(
        pl.col('_sid').filter(pl.col('_sid') >= 0).n_unique().cast(pl.Float32).alias('ir_'+view+'_alias_paths'),
        pl.col('_confidence').max().alias('_confidence_'+view),
        (pl.col('_source') == 1).any().alias('_direct_'+view),
        (pl.col('_source') == 2).any().alias('_s2_'+view),
        (pl.col('_source') == 3).any().alias('_s3_'+view))
    res = summary.join(paths, on=KEYS, how='left')
    if view == 'name':
        unique_words = tokens.filter((pl.col('_token') < HIGH_BIT) & (pl.col('_owner_df') == 1) &
                                    (pl.col('_query_df') <= 4)).group_by(KEYS).len().rename(
                                        {'len': 'ir_unique_name_words'})
        res = res.join(unique_words, on=KEYS, how='left').with_columns(
            pl.col('ir_unique_name_words').fill_null(0).cast(pl.Float32))
    return res


def _queries(frame, targets, baseline, maximum):
    score_column = next((c for c in ('baseline_p', 'newest', 'p') if c in frame), None)
    if score_column is None:
        raise ValueError('Retrieval queries require baseline_p/newest/p')
    summary = frame.group_by('tid').agg(pl.col(score_column).fill_null(0).max().alias('_best'),
                                      pl.len().alias('_competitors'))
    occupied = baseline.select('tid').with_columns(pl.lit(True).alias('_occupied'))
    query = (targets.select(pl.col('rid').alias('tid'), 'co', 'nm', 'ad')
        .join(summary, on='tid', how='left').join(occupied, on='tid', how='left')
        .with_columns(pl.col('_best').fill_null(0), pl.col('_competitors').fill_null(0),
                      pl.col('_occupied').fill_null(False))
        .filter(~pl.col('_occupied') | ((pl.col('_best') < .999) & (pl.col('_competitors') > 1))))
    eligible = query.height

    query = query.with_columns((.5-(pl.col('_best')-.5).abs()).alias('_uncertainty')).sort(
        ['_occupied', '_uncertainty', '_best', 'tid'], descending=[False, True, True, False]).head(maximum)
    return _normalize(query.select('tid', 'co', 'nm', 'ad')), eligible


def _anchors(refs, targets, baseline, s2_count):
    direct = refs.select(pl.col('rid').alias('qid'), 'co', 'nm', 'ad').with_columns(
        pl.lit(-1, pl.Int64).alias('_sid'), pl.lit(1, pl.UInt8).alias('_source'),
        pl.lit(1., pl.Float32).alias('_confidence'))


    aliases = (baseline.select('qid', 'tid', 'p').filter(pl.col('p') >= .995)
        .with_columns(pl.when(pl.col('tid') < s2_count).then(2).otherwise(3).cast(pl.UInt8).alias('_source'))
        .sort(['qid', '_source', 'p', 'tid'], descending=[False, False, True, False])
        .unique(['qid', '_source'], keep='first', maintain_order=True))
    aliases = (aliases.join(targets.select(pl.col('rid').alias('tid'), 'co', 'nm', 'ad'), on='tid', how='inner')
        .join(refs.select(pl.col('rid').alias('qid'), pl.col('co').alias('_owner_country')), on='qid', how='inner')
        .filter(pl.col('co') == pl.col('_owner_country'))
        .select('qid', 'co', 'nm', 'ad', pl.col('tid').cast(pl.Int64).alias('_sid'), '_source',
                pl.col('p').cast(pl.Float32).alias('_confidence')))
    return _normalize(pl.concat([direct, aliases], how='vertical_relaxed')), aliases.height


def retrieve_pairs(frame, refs, targets, baseline, s2_count, *, max_queries=300_000,
                   owners_per_target=8, max_owner_df=8, max_query_df=64):
    'return proposed keys and measured evidence for old and new query edges'
    if min(max_queries, owners_per_target, max_owner_df, max_query_df) <= 0:
        raise ValueError('Retrieval budgets must be positive')
    if frame.select(KEYS).unique().height != frame.height or refs['rid'].n_unique() != refs.height or targets['rid'].n_unique() != targets.height:
        raise ValueError('Duplicate candidate or record keys')
    if baseline['tid'].n_unique() != baseline.height:
        raise ValueError('Baseline must have one owner per target')
    queries, eligible = _queries(frame, targets, baseline, max_queries)
    _log(f'{queries.height:,}/{eligible:,} uncertain targets; constructing frozen owner anchors')
    anchors, aliases = _anchors(refs, targets, baseline, s2_count)
    parts = []
    for country in sorted(queries['co'].unique().to_list()):
        query = queries.filter(pl.col('co') == country)
        index = anchors.filter(pl.col('co') == country)
        _log(f'{country}: {index.height:,} anchors, {query.height:,} queries; independent name/address witnesses')
        name = _view_hits(index, query, 'name', max_owner_df, max_query_df)
        address = _view_hits(index, query, 'address', max_owner_df, max_query_df)
        both = name.join(address, on=KEYS, how='full', coalesce=True)
        both = both.with_columns(
            *[pl.col(c).fill_null(0) for c in both.columns if c not in KEYS],
        ).with_columns(
            pl.max_horizontal('_confidence_name', '_confidence_address').alias('ir_seed_confidence'),
            (pl.col('_direct_name') | pl.col('_direct_address')).cast(pl.Float32).alias('ir_direct_path'),
            (pl.col('_s2_name') | pl.col('_s2_address')).cast(pl.Float32).alias('ir_s2_path'),
            (pl.col('_s3_name') | pl.col('_s3_address')).cast(pl.Float32).alias('ir_s3_path'))
        both = both.with_columns((pl.col('ir_reciprocal_name')*pl.col('ir_reciprocal_address')).sqrt()
            .add(.25*pl.col('ir_unique_name_words')).cast(pl.Float32).alias('ir_witness_score'))


        parts.append(both.select(*KEYS, *EXTRA[1:]))
    schema = {**{c: frame.schema[c] for c in KEYS}, **{c: pl.Float32 for c in EXTRA[1:]}}
    evidence = pl.concat(parts, how='vertical_relaxed') if parts else pl.DataFrame(schema=schema)
    evidence = evidence.with_columns(pl.col('qid').cast(frame.schema['qid']), pl.col('tid').cast(frame.schema['tid']),
                                    *[pl.col(c).cast(pl.Float32) for c in EXTRA[1:]])
    eligible_edges = evidence.filter(((pl.col('ir_name_witnesses') > 0) & (pl.col('ir_address_witnesses') > 0)) |
                                     (pl.col('ir_unique_name_words') >= 2))
    absent = eligible_edges.join(frame.select(KEYS), on=KEYS, how='anti')
    absent = (absent.sort(['tid', 'ir_witness_score', 'qid'], descending=[False, True, False])
        .group_by('tid', maintain_order=True).head(owners_per_target))
    report = {'technique': 'Reciprocal rare-token witness graph with independent name/address paths',
        'selection_uses_labels': False, 'eligible_uncertain_targets': eligible, 'queried_targets': queries.height,
        'max_queries': max_queries, 'owners_per_target': owners_per_target, 'max_owner_token_df': max_owner_df,
        'max_query_token_df': max_query_df, 'anchors': anchors.height, 'accepted_alias_anchors': aliases,
        'witnessed_edges': evidence.height, 'old_witnessed_edges': evidence.join(frame.select(KEYS), on=KEYS, how='semi').height,
        'genuinely_new_pairs': absent.height, 'new_targets': absent['tid'].n_unique(),
        'countries': queries.group_by('co').len().sort('co').to_dicts(),
        'name_tokens': 'Up to four complete nontrivial words plus six deterministic character-fourgram minhash witnesses',
        'reciprocity': 'Every token rare in distinct anchor owners and uncertain query targets; support=1/sqrt(owner_df*query_df)',
        'proposal_rule': 'Independent name+address witness, or two distinct owner-unique full name words',
        'acceptance': 'None: downstream full-union learned reranking required'}
    return absent.select(KEYS), evidence, report


def _record_subset(full, ids, prefix, idname):
    subset = full.join(ids.rename({idname: 'rid'}), on='rid', how='semi')
    records = original._records(subset, prefix, idname)


    cores = full.select('rid', 'co', original.normalized('nm').str.replace_all(original.LEGAL, ' legal ')
        .str.replace_all(r'\blegal\b', '').str.replace_all(r'\s+', ' ').str.strip_chars().alias('_core'))
    counts = cores.group_by('co', '_core').len().rename({'len': '_twins'})
    twins = cores.join(counts, on=['co', '_core'], how='left').select(pl.col('rid').alias(idname),
        pl.when(pl.col('_core') == '').then(0).otherwise(pl.col('_twins')).alias(prefix+'_duplicates'))
    return records.drop(prefix+'_duplicates').join(twins, on=idname, how='left', validate='1:1')


def _new_rows(frame, pairs, refs, targets):
    if not pairs.height:
        return frame.head(0)
    rows = pairs.join(refs.select(pl.col('rid').alias('qid'), 'co',
        *(['fold'] if 'fold' in frame else [])), on='qid', how='left', validate='m:1', maintain_order='left')
    if 'y' in frame or 'own' in frame:
        if 'own' not in targets:
            raise ValueError('New supervised edges require canonical targets.own, after retrieval')
        rows = rows.join(targets.select(pl.col('rid').alias('tid'), 'own'), on='tid', how='left',
                         validate='m:1', maintain_order='left')
        rows = rows.with_columns((pl.col('qid').cast(pl.Int64) == pl.col('own')).cast(pl.UInt8).alias('y'))
    rr = _record_subset(refs, pairs.select('qid').unique(), 'r', 'qid')
    tt = _record_subset(targets, pairs.select('tid').unique(), 't', 'tid')
    lexical = original._lexical(rows, rr, tt)
    rows = pl.concat([rows, lexical], how='horizontal')
    if 'seg' in frame:
        relation = (pairs.join(refs.select(pl.col('rid').alias('qid'), pl.col('ad').fill_null('').str.extract(r'(\d+)', 1)
            .str.strip_chars_start('0').replace('', '0').fill_null('').alias('_house')), on='qid')
            .join(targets.select(pl.col('rid').alias('tid'), pl.col('ad').fill_null('').str.extract_all(r'\d+')
                .list.eval(pl.element().str.strip_chars_start('0').replace('', '0')).alias('_numbers')), on='tid'))
        segment = relation.select(*KEYS, pl.when((pl.col('_house') != '') & pl.col('_numbers').list.contains(pl.col('_house')))
            .then(0).when((pl.col('_house') != '') & (pl.col('_numbers').list.len() > 0)).then(1)
            .otherwise(2).alias('seg'))
        rows = rows.join(segment, on=KEYS, how='left', validate='1:1', maintain_order='left')
    unknown = []
    for column, dtype in frame.schema.items():
        if column in rows:
            continue
        if column in original.SCORES + original.MEMBERS:
            value = None
        elif column in ('baseline_p', 'baseline_raw'):
            value = 1e-4
        elif column.endswith('_logit'):
            value = math.log(1e-4/(1-1e-4))
        else:
            value = 0 if dtype.is_numeric() or dtype == pl.Boolean else None
        unknown.append(pl.lit(value, dtype=dtype).alias(column))
    return rows.with_columns(unknown).select([pl.col(c).cast(t) for c, t in frame.schema.items()])


def _competition(frame):
    scores = [c for c in ('newest', 'gate', 'hybrid', 'graph') if c in frame]
    aggregated = [pl.len().cast(pl.Float32).log1p().alias('competitor_count_log')]
    for score in scores:
        aggregated.extend([pl.col(score).fill_null(0).max().alias(score+'_target_max'),
            pl.when(pl.len() > 1).then(pl.col(score).fill_null(0).top_k(2).min()).otherwise(0.).alias(score+'_target_second')])
    summary = frame.group_by('tid').agg(aggregated)
    replaced = [c for c in summary.columns if c != 'tid' and c in frame]
    res = frame.drop(replaced).join(summary, on='tid', how='left', maintain_order='left', validate='m:1')
    for score in scores:
        res = res.with_columns((pl.col(score).fill_null(0)-pl.col(score+'_target_max')).cast(pl.Float32).alias(score+'_behind_best'),
            (pl.col(score+'_target_max')-pl.col(score+'_target_second')).cast(pl.Float32).alias(score+'_owner_margin'))
    return res


def expand(frame, refs, targets, baseline, s2_count, **kwargs):
    'expand features and attach truth only after proposal keys have frozen'
    started = time.monotonic()
    if set(EXTRA).intersection(frame.columns):
        raise ValueError('Candidate frame has already been expanded')
    proposals, evidence, report = retrieve_pairs(frame, refs, targets, baseline, s2_count, **kwargs)
    _log(f'{proposals.height:,} genuinely new edges; computing fresh pair features')
    fresh = _new_rows(frame, proposals, refs, targets)
    original_rows = frame.with_columns(pl.lit(0., pl.Float32).alias('ir_new_edge'))
    fresh = fresh.with_columns(pl.lit(1., pl.Float32).alias('ir_new_edge'))
    combined = pl.concat([original_rows, fresh], how='vertical_relaxed')
    combined = combined.join(evidence, on=KEYS, how='left', maintain_order='left', validate='1:1').with_columns(
        *[pl.col(c).fill_null(0).cast(pl.Float32) for c in EXTRA[1:]])
    combined = _competition(combined)
    report.update(before_pairs=frame.height, expanded_pairs=combined.height,
                  measured_features=list(EXTRA), elapsed_seconds=time.monotonic()-started,
                  label_attachment='Canonical targets.own joined only after proposal keys frozen',
                  upstream_missing_policy='Scores null/present0/logit(logit1e-4); baseline_p/raw1e-4; other unavailable numeric derivatives0')
    if 'y' in fresh:
        report['new_positive_pairs_after_label_attachment'] = int(fresh['y'].sum() or 0)
        report['new_negative_pairs_after_label_attachment'] = fresh.height-report['new_positive_pairs_after_label_attachment']
    _log(f'candidate union {frame.height:,} -> {combined.height:,}; all edges require model reranking')
    return combined, list(EXTRA), report

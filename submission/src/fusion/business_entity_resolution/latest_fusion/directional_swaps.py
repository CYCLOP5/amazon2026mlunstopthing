'label-free directed name edits on the complete supplied candidate population'
import polars as pl

LEGAL = r'\b(?:incorporated|inc|corporation|corp|limited|ltd|llc|llp|pvt|plc|sarl|sas|sasu|eurl|sa|sci|snc|gmbh)\b'
DOTTED_LEGAL = r'(?i)\b(?:s\.\s*a\.\s*r\.\s*l\.?|s\.\s*a\.\s*s\.\s*u\.?|s\.\s*a\.\s*s\.?|e\.\s*u\.\s*r\.\s*l\.?|l\.\s*l\.\s*c\.?)'
SUPPORT = ['edges', 'qids', 'targets', 'address_edges', 'address_targets', 's2_edges', 's3_edges']
FEATURES = ['ds_missing', 'ds_extra', 'ds_shared', 'ds_ref_coverage', 'ds_target_coverage',
            'ds_two_word', 'ds_address_exact', 'ds_ref_address_present',
            'ds_target_address_present', 'ds_numeric_conflict', 'ds_exact_alternative_refs']
FEATURES += [f'ds_{side}_{name}' for side in ('forward', 'reverse') for name in SUPPORT]
FEATURES += ['ds_direction_ratio', 'ds_address_direction_ratio', 'ds_target_direction_ratio']
FEATURES += [f'ds_context_{side}_{name}' for side in ('forward', 'reverse') for name in SUPPORT]
FEATURES += ['ds_context_direction_ratio', 'ds_context_address_direction_ratio', 'ds_context_target_direction_ratio']


def _normal(column):
    return (pl.col(column).fill_null('').str.to_lowercase().str.normalize('NFKD')
            .str.replace_all(r'\p{M}', '').str.replace_all(r'[^\p{L}\p{N}]+', ' ')
            .str.replace_all(r'\s+', ' ').str.strip_chars())


def _address_normal(column):


    return (pl.col(column).fill_null('').str.to_lowercase().str.normalize('NFKD')
            .str.replace_all(r'\p{M}', '').str.replace_all(r'[^\p{L}\p{N}/-]+', ' ')
            .str.replace_all(r'\s*([/-])\s*', '${1}')
            .str.replace_all(r'\s+', ' ').str.strip_chars())


def _records(records, key):
    if records['rid'].n_unique() != records.height:
        raise ValueError('Duplicate record IDs')
    d = records.select(pl.col('rid').alias(key), 'co', 'nm', 'ad').with_columns(
        pl.col('nm').fill_null('').str.replace_all(DOTTED_LEGAL, ' ').alias('_legal_nm'))
    return (d.with_columns(_normal('_legal_nm').str.replace_all(LEGAL, ' ')
                          .str.replace_all(r'\s+', ' ').str.strip_chars().alias('_name'),
                          _address_normal('ad').alias('_address'))
            .with_columns(pl.col('_name').str.split(' ').list.eval(
                pl.element().filter(pl.element() != '')).alias('_all_tokens'))
            .with_columns(pl.col('_all_tokens').list.unique().list.sort().alias('_tokens'))
            .with_columns((pl.col('_all_tokens').list.len() != pl.col('_tokens').list.len()).alias('_repeated'))
            .select(key, 'co', '_name', '_address', '_tokens', '_repeated'))


def _counts(d, keys):
    return d.group_by(keys).agg(
        pl.len().alias('edges'), pl.col('qid').n_unique().alias('qids'),
        pl.col('tid').n_unique().alias('targets'),
        pl.col('ds_address_exact').sum().alias('address_edges'),
        pl.col('tid').filter(pl.col('ds_address_exact') == 1).n_unique().alias('address_targets'),
        pl.col('_s2').sum().alias('s2_edges'), pl.col('_s3').sum().alias('s3_edges'))


def _exclusive_targets(d, keys, address=False):
    if address:
        d = d.filter(pl.col('ds_address_exact') == 1)



    return (d.group_by(keys + ['tid']).agg(pl.col('qid').n_unique().alias('_owners'),
                                               pl.col('qid').first())
            .filter(pl.col('_owners') == 1).group_by(keys + ['qid']).len(
                name='_exclusive_address_targets' if address else '_exclusive_targets'))


def _support(d, keys, prefix):
    table = _counts(d, keys)
    own = _counts(d, keys + ['qid']).join(_exclusive_targets(d, keys), on=keys+['qid'], how='left').join(
        _exclusive_targets(d, keys, address=True), on=keys+['qid'], how='left')
    own = own.with_columns(pl.col('_exclusive_targets', '_exclusive_address_targets').fill_null(0))
    own = own.select(*keys, 'qid', pl.col('edges', 'qids', 'address_edges', 's2_edges', 's3_edges'),
                     pl.col('_exclusive_targets').alias('targets'),
                     pl.col('_exclusive_address_targets').alias('address_targets'))
    for side in ('forward', 'reverse'):
        mapping = {'_swap_from': '_swap_to', '_swap_to': '_swap_from'} if side == 'reverse' else {}
        totals = table.rename(mapping).rename({c: '_total_'+c for c in SUPPORT})
        local = own.rename(mapping).rename({c: '_local_'+c for c in SUPPORT})
        d = d.join(totals, on=keys, how='left', validate='m:1', maintain_order='left').join(
            local, on=keys+['qid'], how='left', validate='m:1', maintain_order='left')
        d = d.with_columns([(pl.col('_total_'+c).fill_null(0).cast(pl.Float64) -
                             pl.col('_local_'+c).fill_null(0).cast(pl.Float64)).alias(f'{prefix}_{side}_{c}')
                            for c in SUPPORT]).drop(*['_total_'+c for c in SUPPORT], *['_local_'+c for c in SUPPORT])
    d = d.with_columns(*[((pl.col(prefix+'_forward_'+c)+1) /
        (pl.col(prefix+'_forward_'+c)+pl.col(prefix+'_reverse_'+c)+2)).alias(prefix+'_'+name)
        for c, name in [('edges', 'direction_ratio'), ('address_edges', 'address_direction_ratio'),
                         ('targets', 'target_direction_ratio')]])
    return d, table


def build(frame, refs, targets, s2_count=None):
    'return eligible rows, feature names, bounded summary, and direction table'
    if frame.select('qid', 'tid').n_unique() != frame.height:
        raise ValueError('Duplicate candidate pairs')
    if s2_count is not None and s2_count < 0:
        raise ValueError('s2_count must be nonnegative')
    reserved = set(FEATURES) | {'_swap_from', '_swap_to', '_swap_context'}
    if reserved.intersection(frame.columns):
        raise ValueError('Directional feature columns already exist')
    r, t = _records(refs, 'qid'), _records(targets, 'tid')

    alternatives = (r.filter((pl.col('_name') != '') & (pl.col('_address') != ''))
                    .group_by('co', '_tokens', '_address').len(name='ds_exact_alternative_refs'))
    rr = r.rename({c: '_r'+c for c in ('co', '_name', '_address', '_tokens', '_repeated')})
    tt = t.rename({c: '_t'+c for c in ('co', '_name', '_address', '_tokens', '_repeated')})
    parts, samples, repeated_excluded = [], [], 0
    for chunk in frame.select('qid', 'tid', *(['co'] if 'co' in frame.columns else [])).iter_slices(500_000):
        d = chunk.join(rr, on='qid', how='left', validate='m:1', maintain_order='left').join(
            tt, on='tid', how='left', validate='m:1', maintain_order='left')
        if d['_r_name'].null_count() or d['_t_name'].null_count():
            raise ValueError('Candidate IDs missing from full record tables')
        if d.filter(pl.col('_rco') != pl.col('_tco')).height:
            raise ValueError('Cross-country candidate pair')
        if 'co' in chunk.columns and d.filter(pl.col('co') != pl.col('_rco')).height:
            raise ValueError('Candidate country mismatch')
        repeated_excluded += d.filter(pl.col('_r_repeated') | pl.col('_t_repeated')).height
        d = d.filter(~pl.col('_r_repeated') & ~pl.col('_t_repeated'))
        d = d.with_columns(
            pl.col('_r_tokens').list.set_difference('_t_tokens').list.sort().alias('_missing'),
            pl.col('_t_tokens').list.set_difference('_r_tokens').list.sort().alias('_extra'),
            pl.col('_r_tokens').list.set_intersection('_t_tokens').list.len().alias('ds_shared'),
            pl.col('_r_tokens').list.set_intersection('_t_tokens').list.sort().list.join(' ').alias('_swap_context'))
        d = d.with_columns(pl.col('_missing').list.len().alias('ds_missing'),
                           pl.col('_extra').list.len().alias('ds_extra'))
        d = d.filter(pl.col('ds_missing').is_between(1, 2) &
                     pl.col('ds_extra').is_between(1, 2) & (pl.col('ds_shared') >= 1))
        d = d.with_columns(
            pl.col('_rco').alias('_swap_country'),
            pl.col('_missing').list.join(' ').alias('_swap_from'),
            pl.col('_extra').list.join(' ').alias('_swap_to'),
            (pl.col('ds_shared') / pl.col('_r_tokens').list.len()).alias('ds_ref_coverage'),
            (pl.col('ds_shared') / pl.col('_t_tokens').list.len()).alias('ds_target_coverage'),
            ((pl.col('ds_missing') == 2) | (pl.col('ds_extra') == 2)).alias('ds_two_word'),
            ((pl.col('_r_address') != '') & (pl.col('_r_address') == pl.col('_t_address'))).alias('ds_address_exact'),
            (pl.col('_r_address') != '').alias('ds_ref_address_present'),
            (pl.col('_t_address') != '').alias('ds_target_address_present'),
            pl.col('_r_address').str.extract(r'(\d+)', 1).str.strip_chars_start('0').replace('', '0').alias('_house'),
            pl.col('_t_address').str.extract_all(r'\d+').list.eval(
                pl.element().str.strip_chars_start('0').replace('', '0')).alias('_nums'),
            (pl.col('tid') < s2_count if s2_count is not None else pl.lit(False)).cast(pl.UInt32).alias('_s2'),
            (pl.col('tid') >= s2_count if s2_count is not None else pl.lit(False)).cast(pl.UInt32).alias('_s3'))
        d = d.with_columns(((pl.col('_house').is_not_null()) & (pl.col('_nums').list.len() > 0) &
            ~pl.col('_nums').list.contains(pl.col('_house'))).fill_null(False).alias('ds_numeric_conflict'))
        d = d.join(alternatives.rename({'co': '_swap_country', '_tokens': '_t_tokens', '_address': '_t_address'}),
                   on=['_swap_country', '_t_tokens', '_t_address'], how='left', validate='m:1', maintain_order='left')
        d = d.with_columns(pl.col('ds_exact_alternative_refs').fill_null(0))
        if len(samples) < 200:
            samples.extend(d.head(200-len(samples)).select('qid', 'tid', '_swap_country', '_swap_from', '_swap_to',
                            '_r_name', '_t_name', '_r_address', '_t_address').to_dicts())
        parts.append(d.select('qid', 'tid', '_swap_country', '_swap_from', '_swap_to', '_swap_context',
                              *FEATURES[:11], '_s2', '_s3').with_columns(
                                  pl.col(FEATURES[:11]).cast(pl.Float64)))
    keys = ['_swap_country', '_swap_from', '_swap_to']
    if not parts:
        empty = frame.head(0).with_columns(pl.lit('', pl.String).alias('_swap_from'),
            pl.lit('', pl.String).alias('_swap_to'), *[pl.lit(0., pl.Float64).alias(c) for c in FEATURES])
        return empty, list(FEATURES), {'candidate_pairs': 0, 'swap_pairs': 0, 'samples': [],
                                     's2_count': s2_count}, pl.DataFrame()
    d = pl.concat(parts)
    d, table = _support(d, keys, 'ds')
    d, context_table = _support(d, keys + ['_swap_context'], 'ds_context')
    added = d.select('qid', 'tid', '_swap_from', '_swap_to', *FEATURES)
    output = frame.join(added, on=['qid', 'tid'], how='inner', validate='1:1', maintain_order='left')
    summary = {'candidate_pairs': frame.height, 'swap_pairs': output.height, 'directions': table.height,
               'shared_name_context_directions': context_table.height,
               'repeated_material_token_pairs_excluded': repeated_excluded,
               'exact_address_swap_pairs': int(d['ds_address_exact'].sum()), 's2_count': s2_count,
               'support_scope': 'All eligible candidate edges, separately by country and by shared name context; entire queried qid excluded in both directions.',
               'interpretation': 'Smoothed direction ratios are corpus evidence, not ownership probabilities.',
               'samples': samples}
    return output, list(FEATURES), summary, table.sort(keys)

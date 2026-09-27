'label-free, country-scoped record signatures for candidate proposals'
import time
import polars as pl
from rapidfuzz import fuzz
from er.normalize.normalizer import ascii_fold
from er.features.name import cp

LEGAL = r'(private limited|corporation|limited|sarl|sasu|sas|eurl|sa|llc|llp|inc|ltd|pvt|corp)'


def _sorted(c):
    return pl.col(c).str.split(' ').list.sort().list.join(' ')


def normalize_records(raw):
    'raw rid/nm/ad/co -> rid/co and explicit name/address signature fields'
    if raw['rid'].n_unique() != raw.height:
        raise ValueError('Duplicate record IDs')
    d = raw.select('rid', pl.col('co').fill_null('').str.to_lowercase().str.strip_chars(),
                   pl.col('ad').fill_null('').str.strip_chars().eq('').alias('missing_address'))
    d = d.with_columns(ascii_fold(raw['nm']).alias('_name'), ascii_fold(raw['ad'].fill_null('').str.replace_all(r'(?i)\b(?:n[º°]|no\.?)\s*(\d)', '${1}')).alias('_address'))

    for pattern, value in ((r'\be\s*\.\s*u\s*\.\s*r\s*\.\s*l\s*\.?', 'eurl'),
                           (r'\bs\s*\.\s*a\s*\.\s*s\s*\.\s*u\s*\.?', 'sasu'),
                           (r'\bs\s*\.\s*a\s*\.\s*r\s*\.\s*l\s*\.?', 'sarl'),
                           (r'\bs\s*\.\s*a\s*\.\s*s\s*\.?', 'sas'),
                           (r'\bl\s*\.\s*l\s*\.\s*c\s*\.?', 'llc')):
        d = d.with_columns(pl.col('_name').str.replace_all(pattern, ' '+value+' '))
    d = d.with_columns(pl.col('_name').str.replace_all(r'[^a-z0-9]+', ' ').str.strip_chars(),
                      pl.col('_address').str.replace_all(r'[^a-z0-9/]+', ' ').str.strip_chars())

    fr = pl.col('co').is_in(['france', 'fr'])
    for old, new in (('bd','boulevard'),('boul','boulevard'),('av','avenue'),('r','rue'),('st','saint'),('ste','sainte')):
        d = d.with_columns(pl.when(fr).then(pl.col('_address').str.replace_all(r'\b'+old+r'\b',new))
                           .otherwise(pl.col('_address')).alias('_address'))
    legal_pattern = r'\b'+LEGAL+r'\b'
    d = d.with_columns(pl.col('_name').str.extract_all(legal_pattern).list.eval(
        pl.element().replace({'limited':'ltd','corporation':'corp','private limited':'pvt'}))
        .list.unique().list.sort().list.join(' ').alias('legal_form'),
        pl.col('_name').str.replace_all(legal_pattern, ' ').str.replace_all(r'\s+', ' ')
        .str.strip_chars().alias('_core'))
    d = d.with_columns(_sorted('_core').alias('name_core'),
                      _sorted('_address').alias('address_key'),
                      pl.col('_core').str.replace_all(r'[^a-z]', '').str.len_chars().alias('name_alpha_length'),
                      pl.col('_address').str.replace_all(r'[^a-z]', '').str.len_chars().alias('address_alpha_length'),
                      pl.col('_address').str.contains(r'\d').alias('address_numeric'),
                      pl.col('_address').str.extract_all(r'\b[a-z]+\b').list.len().alias('address_alpha_tokens'))
    d = d.with_columns(pl.concat_str('name_core','legal_form',separator=' ').str.strip_chars().alias('name_full'),
                      ((pl.col('name_alpha_length') >= 4) & (pl.col('name_core') != '')).alias('valid_name'),
                      ((pl.col('address_alpha_length') >= 8) & (pl.col('address_alpha_tokens') >= 2) &
                       pl.col('address_numeric') & ~pl.col('missing_address')).alias('valid_address'))
    alias = pl.col('name_core')
    for countries, noise in ((['france','fr'],'cie|compagnie|groupe|fils|freres|associes|developpement|france'),
                             (['us','usa','united states'],'group|company|sons|associates|holdings'),
                             (['india','in'],'group|enterprises|sons|associates|company')):
        alias = pl.when(pl.col('co').is_in(countries)).then(
            alias.str.replace_all(r'\b(?:'+noise+r')\b',' ').str.replace_all(r'\s+',' ').str.strip_chars()
        ).otherwise(alias)
    d = d.with_columns(alias.alias('name_alias'))
    d = d.with_columns((pl.col('name_alias').str.replace_all(r'[^a-z]','').str.len_chars() >= 4)
                       .alias('valid_alias'))
    return d.drop('_name','_address','_core')


def _unique(frame, keys):
    return frame.filter(pl.len().over(keys) == 1)


def propose(refs, targets):
    'Return (qid/tid/rule/rank/cell_id/similarity proposals, diagnostics)'
    started = time.monotonic()
    if 'name_core' not in refs:
        refs = normalize_records(refs)
    if 'name_core' not in targets:
        targets = normalize_records(targets)
    r = refs.rename({'rid':'qid'}).filter(pl.col('valid_name'))
    t = targets.rename({'rid':'tid'}).filter(pl.col('valid_name'))
    output = []
    rules = [('full_identity',['co','name_full','address_key'],False,False),
             ('core_identity',['co','name_core','address_key'],True,False),
             ('alias_identity',['co','name_alias','address_key'],True,False),
             ('unique_full_name_missing_address',['co','name_full'],False,True),
             ('unique_core_name_missing_address',['co','name_core'],True,True),
             ('unique_address_close_name',['co','address_key'],True,False)]
    for rank, (rule, keys, compatible, missing) in enumerate(rules,1):

        rr = r if missing else r.filter(pl.col('valid_address'))
        if rule == 'alias_identity':
            rr = rr.filter(pl.col('valid_alias'))
        rr = _unique(rr,keys)
        tt = t.filter(pl.col('missing_address') if missing else pl.col('valid_address'))
        if rule == 'alias_identity':
            tt = tt.filter(pl.col('valid_alias'))
        cols = list(dict.fromkeys(['qid',*keys,'legal_form','name_core']))
        d = tt.join(rr.select(cols),on=keys,how='inner',suffix='_ref',validate='m:1')
        if compatible:
            d = d.filter((pl.col('legal_form') == pl.col('legal_form_ref')) |
                         (pl.col('legal_form') == '') | (pl.col('legal_form_ref') == ''))
        if rule == 'unique_address_close_name':
            scores = cp(d['name_core'].to_list(), d['name_core_ref'].to_list(), fuzz.token_sort_ratio)
            d = d.with_columns(pl.Series('similarity', scores).cast(pl.Float64)).filter(pl.col('similarity') >= 95.)
        else:
            d = d.with_columns(pl.lit(100.).alias('similarity'))
        d = d.with_columns(pl.struct(keys).hash(9217).alias('cell_id'),pl.lit(rule).alias('rule'),
                           pl.lit(rank,pl.UInt8).alias('rank')).select('qid','tid','rule','rank','cell_id','similarity')
        output.append(d)
        print(f'IDENTITY {rule}: {d.height:,} proposals ({time.monotonic()-started:.1f}s)',flush=True)
    proposals = pl.concat(output,how='vertical_relaxed').unique(['qid','tid','rule']).with_columns(
        pl.col('qid','tid').cast(pl.UInt32))
    counts = proposals.group_by('tid').agg(pl.col('qid').n_unique().alias('owners'))
    diagnostics = {'label_free': True, 'reference_records': refs.height, 'target_records': targets.height,
        'proposals': proposals.height, 'proposed_targets': counts.height,
        'ambiguous_targets': counts.filter(pl.col('owners') > 1).height,
        'by_rule': dict(proposals.group_by('rule').len().rows()),
        'seconds': time.monotonic()-started}
    return proposals, diagnostics

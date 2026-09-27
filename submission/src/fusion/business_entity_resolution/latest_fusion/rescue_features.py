'label-free ownership consensus for a selected target competition pool'
import polars as pl
from .features import normalized, LEGAL


def _tokens(raw, key):
    return raw.select(pl.col('rid').alias(key), 'co', normalized('nm').str.replace_all(LEGAL, ' ').str.split(' ').list.eval(pl.element().filter(pl.element() != '')).list.unique().alias('_tokens'))


def build_rescue_features(pool, refs, targets, existing_features):
    if pool.select('qid','tid').n_unique() != pool.height:
        raise ValueError('Duplicate candidate pairs')
    if not pool.height:
        raise ValueError('Empty rescue pool')
    if refs['rid'].n_unique() != refs.height or targets['rid'].n_unique() != targets.height:
        raise ValueError('Duplicate raw record IDs')
    if pool.select('qid').unique().join(refs.select(pl.col('rid').alias('qid')),on='qid',how='anti').height or pool.select('tid').unique().join(targets.select(pl.col('rid').alias('tid')),on='tid',how='anti').height:
        raise ValueError('Candidate IDs missing from raw records')
    countries = pool.select('qid','tid','co').join(refs.select(pl.col('rid').alias('qid'),pl.col('co').alias('_ref_co')),on='qid').join(targets.select(pl.col('rid').alias('tid'),pl.col('co').alias('_target_co')),on='tid')
    if countries.filter((pl.col('co')!=pl.col('_ref_co')) | (pl.col('co')!=pl.col('_target_co')) | pl.col('co').is_null() | pl.col('_ref_co').is_null() | pl.col('_target_co').is_null()).height:
        raise ValueError('Candidate country mismatch')
    d = pool.with_row_index('_order')
    names = list(existing_features)
    added = []
    def add(name, expr):
        nonlocal d
        d = d.with_columns(expr.fill_null(0).cast(pl.Float32).alias(name))
        added.append(name)
    def ownership(c):
        nonlocal d

        agg = d.group_by('tid').agg(pl.col(c).max().alias('_max'),pl.col(c).drop_nulls().top_k(2).sort(descending=True).alias('_top'))
        d = d.join(agg,on='tid',how='left',maintain_order='left')
        other = pl.when((pl.col(c)==pl.col('_max')) & (pl.col('_top').list.len()>1)).then(pl.col('_top').list.get(1,null_on_oob=True)).otherwise(pl.when(pl.col(c)==pl.col('_max')).then(0.).otherwise(pl.col('_max')))
        margin = (pl.col(c)-other)
        add('rescue_'+c+'_margin',margin)
        add('rescue_'+c+'_vote',(margin>0)&pl.col(c).is_not_null())
        d = d.drop('_max','_top')
    members = [f'np_m{i}' for i in range(15) if f'np_m{i}' in d]
    families = [c for c in ['newest','gate','neural','hybrid','graph','friend'] if c in d]
    for c in members+families+(['inc_p'] if 'inc_p' in d else []):
        ownership(c)
    if members:
        add('rescue_member_present_count',pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Float32) for c in members]))
        valid = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Float32) for c in members]).clip(1,None)
        margins = [pl.when(pl.col(c).is_not_null()).then(pl.col('rescue_'+c+'_margin')) for c in members]
        add('rescue_member_vote_count',pl.sum_horizontal([pl.col('rescue_'+c+'_vote') for c in members]))
        add('rescue_member_vote_fraction',pl.col('rescue_member_vote_count')/valid)
        add('rescue_member_margin_min',pl.min_horizontal(margins))
        add('rescue_member_margin_mean',pl.sum_horizontal(margins)/valid)
        add('rescue_member_margin_std',(pl.sum_horizontal([m*m for m in margins])/valid-pl.col('rescue_member_margin_mean')**2).clip(0,None).sqrt())
        for cut in [.01,.1]:
            add('rescue_member_margin_above_'+str(cut).replace('.','p'),pl.sum_horizontal([(m>cut).fill_null(False).cast(pl.Float32) for m in margins])/valid)
    if families:
        add('rescue_family_vote_count',pl.sum_horizontal([pl.col('rescue_'+c+'_vote') for c in families]))
        add('rescue_family_vote_fraction',pl.col('rescue_family_vote_count')/pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Float32) for c in families]).clip(1,None))

    rr = _tokens(refs,'qid')
    counts = rr.group_by('co').len().rename({'len':'_documents'})
    df = rr.explode('_tokens', empty_as_null=True).filter(pl.col('_tokens').is_not_null()).group_by('co','_tokens').len().rename({'len':'_df'}).join(counts,on='co').with_columns(((pl.col('_documents')+1)/(pl.col('_df')+1)).log().add(1).alias('_idf')).select('co','_tokens','_idf')
    rr = rr.join(d.select('qid').unique(),on='qid',how='semi')
    tt = _tokens(targets,'tid').join(d.select('tid').unique(),on='tid',how='semi')
    rt = rr.explode('_tokens', empty_as_null=True).join(df,on=['co','_tokens'],how='left').with_columns(pl.when(pl.col('_tokens').is_null()).then(0.).otherwise(pl.col('_idf').fill_null(1.)))
    tx = tt.explode('_tokens', empty_as_null=True).join(df,on=['co','_tokens'],how='left').join(counts,on='co',how='left').with_columns(pl.when(pl.col('_tokens').is_null()).then(0.).otherwise(pl.col('_idf').fill_null((pl.col('_documents').fill_null(0)+1).log()+1)).alias('_idf'))

    a = d.select('_order','qid','tid').join(rt.select('qid','_tokens','_idf'),on='qid').join(tt.select('tid',pl.col('_tokens').alias('_target_tokens')),on='tid')
    a = a.with_columns(pl.col('_target_tokens').list.contains(pl.col('_tokens')).fill_null(False).alias('_shared'))
    b = d.select('_order','qid','tid').join(tx.select('tid','_tokens','_idf'),on='tid').join(rr.select('qid',pl.col('_tokens').alias('_ref_tokens')),on='qid').with_columns(pl.col('_ref_tokens').list.contains(pl.col('_tokens')).fill_null(False).alias('_shared'))
    ag = a.group_by('_order').agg(pl.col('_idf').filter(pl.col('_shared')).sum().alias('rescue_shared_idf_sum'),pl.col('_idf').filter(pl.col('_shared')).max().alias('rescue_shared_idf_max'),pl.col('_idf').filter(~pl.col('_shared')).sum().alias('rescue_missing_idf_sum'),pl.col('_idf').filter(~pl.col('_shared')).max().alias('rescue_missing_idf_max'))
    bg = b.group_by('_order').agg(pl.col('_idf').filter(~pl.col('_shared')).sum().alias('rescue_extra_idf_sum'),pl.col('_idf').filter(~pl.col('_shared')).max().alias('rescue_extra_idf_max'))
    d = d.join(ag,on='_order',how='left',maintain_order='left').join(bg,on='_order',how='left',maintain_order='left')
    rarity = ag.columns[1:]+bg.columns[1:]
    d = d.with_columns(pl.col(rarity).fill_null(0).cast(pl.Float32)); added.extend(rarity)
    s=pl.col('rescue_shared_idf_sum'); m=pl.col('rescue_missing_idf_sum'); e=pl.col('rescue_extra_idf_sum')
    add('rescue_idf_jaccard',s/(s+m+e).clip(1e-9,None))
    add('rescue_idf_ref_coverage',s/(s+m).clip(1e-9,None))
    add('rescue_idf_target_coverage',s/(s+e).clip(1e-9,None))
    ownership('rescue_shared_idf_sum')
    add('rescue_shared_rarity_advantage',pl.col('rescue_rescue_shared_idf_sum_margin')/(pl.col('rescue_shared_idf_sum')-pl.col('rescue_rescue_shared_idf_sum_margin')).clip(1.,None))
    d=d.sort('_order').drop('_order')
    names.extend(added)
    if d.select(pl.any_horizontal([~pl.col(c).is_finite() for c in names]).any()).item():
        raise ValueError('Nonfinite rescue features')
    return d,names

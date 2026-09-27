'label-free ambiguity and leave-one-record-out consensus evidence'
import polars as pl
from er.stack.features import _margin


def ambiguity_features(pairs, refs, targets):
    out = pairs.select('qid', 'tid')
    for key, label in [('addr_can', 'address'), ('name_core', 'name')]:
        freq = refs.filter(pl.col(key) != '').group_by('co', key).len(name='_freq')
        rr = refs.join(freq, on=['co', key], how='left').select(
            pl.col('rid').alias('qid'), pl.col('_freq').fill_null(0).alias(f'ref_{label}_freq'))
        tt = targets.join(freq, on=['co', key], how='left').select(
            pl.col('rid').alias('tid'), pl.col('_freq').fill_null(0).alias(f'target_{label}_freq'))
        out = out.join(rr, on='qid', how='left').join(tt, on='tid', how='left')
    return out


def occupancy_features(df):
    'soft evidence about other records, rather than a one-record-per-source assumption'
    best = pl.col('p1') == pl.col('p1').max().over('tid')
    d = df.with_columns(best.alias('_winner'))
    own = pl.when(pl.col('_winner')).then(pl.col('p1')).otherwise(0.0)
    expressions = []
    for keys, suffix in [(['qid'], 'all'), (['qid', 'is_s3'], 'source')]:
        expressions.append((own.sum().over(keys) - own).alias(f'sib_soft_mass_{suffix}'))
        for cut in (0.5, 0.9, 0.99):
            flag = pl.col('_winner') & (pl.col('p1') >= cut)
            expressions.append((flag.sum().over(keys) - flag.cast(pl.UInt32))
                               .alias(f'sib_others_p{int(cut*100)}_{suffix}'))
    d = d.with_columns(expressions).drop('_winner')
    return _margin(d, 'sib_soft_mass_source', 'tid', 'sib_soft_mass_source_gap')


def consensus_features(pairs, conf, targets):
    'compare only other confident records; missing addresses never count as agreement'
    h = (pl.col('hn1').str.slice(-12).cast(pl.Int64, strict=False))
    attrs = targets.select('rid', 'name_core', 'addr_can', h.alias('hn'))
    s = (pairs.join(conf, on='qid', how='inner').filter(pl.col('tid') != pl.col('sid'))
         .join(attrs.rename({'rid':'tid', 'name_core':'tn', 'addr_can':'ta', 'hn':'th'}), on='tid')
         .join(attrs.rename({'rid':'sid', 'name_core':'sn', 'addr_can':'sa', 'hn':'sh'}), on='sid'))
    fills = {'sib_addr_known':0, 'sib_addr_exact':0, 'sib_name_exact':0,
             'sib_numeric_known':0, 'sib_numeric_agree':0, 'sib_numeric_conflict':0,
             'sib_numeric_agree_cross':0, 'sib_numeric_min_logdist':-1.0,
             'sib_numeric_agree_weight':0.0, 'sib_numeric_conflict_weight':0.0}
    out = pairs.select('qid', 'tid')
    if s.height:
        known = pl.col('th').is_not_null() & pl.col('sh').is_not_null()
        eq = (known & (pl.col('th') == pl.col('sh'))).fill_null(False)
        conflict = (known & (pl.col('th') != pl.col('sh'))).fill_null(False)
        weight = 1 / (1 + (-pl.col('s_lg')).exp())
        g = s.group_by('qid', 'tid').agg(
            (pl.col('sa') != '').sum().alias('sib_addr_known'),
            ((pl.col('ta') != '') & (pl.col('ta') == pl.col('sa'))).sum().alias('sib_addr_exact'),
            ((pl.col('tn') != '') & (pl.col('tn') == pl.col('sn'))).sum().alias('sib_name_exact'),
            known.sum().alias('sib_numeric_known'), eq.sum().alias('sib_numeric_agree'),
            conflict.sum().alias('sib_numeric_conflict'),
            (eq & (pl.col('is_s3') != pl.col('s_is_s3'))).sum().alias('sib_numeric_agree_cross'),
            (pl.col('th')-pl.col('sh')).abs().cast(pl.Float64).log1p().min().alias('sib_numeric_min_logdist'),
            (eq.cast(pl.Float64)*weight).sum().alias('sib_numeric_agree_weight'),
            (conflict.cast(pl.Float64)*weight).sum().alias('sib_numeric_conflict_weight'))
        out = out.join(g, on=['qid','tid'], how='left')
    else:
        out = out.with_columns([pl.lit(None).alias(c) for c in fills])
    out = out.with_columns([pl.col(c).fill_null(v) for c,v in fills.items()])
    out = out.with_columns((pl.col('sib_numeric_agree') / pl.max_horizontal('sib_numeric_known', pl.lit(1)))
                          .alias('sib_numeric_agree_fraction'))
    return out.with_columns(pl.col(pl.Float64).cast(pl.Float32))

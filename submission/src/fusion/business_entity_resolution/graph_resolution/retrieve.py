import numpy as np
import polars as pl

from er.blocking.skeleton_tfidf import block_country
from er.stack.decode import top1


def seed_owners(pairs, minimum=.995):
    'frozen neural evidence only; labels and folds never enter the graph'
    top = pairs.group_by('tid').agg(pl.col('prob').top_k(2).alias('_p'))
    top = top.select('tid',pl.col('_p').list.get(1,null_on_oob=True).fill_null(0).alias('_second'))
    best = top1(pairs,'prob').join(top,on='tid')
    seeds = best.filter((pl.col('prob') >= minimum) & (pl.col('prob')-pl.col('_second') >= .2)
        & (pl.col('gate_prob') >= .9) & (pl.col('neural_prob') >= .9))
    return seeds.select('qid',pl.col('tid').alias('sid'),pl.col('prob').alias('confidence'))


def bridge_candidates(pairs, refs, targets, seeds, log, max_df=256):
    'Retrieve through S1 records AND confidently owned S2/S3 aliases'
    fields = ['co','name_skel','addr_skel','name_core','addr_can']
    direct = refs.select(pl.col('rid').alias('qid'),*fields,pl.lit(-1,pl.Int64).alias('sid'),
                         pl.lit(1.0,pl.Float32).alias('confidence'))
    aliases = seeds.join(targets.select(pl.col('rid').alias('sid'),*fields),on='sid').select(
        'qid',*fields,pl.col('sid').cast(pl.Int64),pl.col('confidence').cast(pl.Float32))
    index = pl.concat([direct,aliases],how='vertical_relaxed')
    best = pairs.group_by('tid').agg(pl.col('prob').max().alias('_best'),pl.col('gate_prob').max().alias('_gate'))
    query = targets.join(best,left_on='rid',right_on='tid',how='left').filter(
        ((pl.col('_best').fill_null(0) < .999) & (pl.col('_gate').fill_null(0) >= .05))
        | (pl.col('addr_can') == ''))
    log(f'graph: {seeds.height:,} confident aliases; {query.height:,} retrieval queries')
    parts = []
    prm = {'tokens':{'name_unigrams':True,'name_bigrams':True,'compact_name':True,
        'address_unigrams':True,'address_bigrams':True},'chunk_rows':100_000,
        'max_df1':max_df,'retrieve_k':32,'top_k':16,'addr_view_k':0,'name_view_k':0}
    for country in sorted(query['co'].unique().to_list()):
        ref = index.filter(pl.col('co') == country).with_row_index('rid')
        qry = query.filter(pl.col('co') == country).rename({'rid':'tid'}).with_row_index('rid')
        if not ref.height or not qry.height:
            continue
        log(f'graph country {country}: index {ref.height:,}, queries {qry.height:,}')
        for chunk in block_country(ref,qry,prm):
            hit = (chunk.join(ref.select(pl.col('rid').alias('s1_rid'),'qid','sid','confidence'),on='s1_rid')
                .join(qry.select(pl.col('rid').alias('o_rid'),'tid'),on='o_rid')
                .filter(pl.col('tid').cast(pl.Int64) != pl.col('sid')))
            parts.append(hit.group_by('qid','tid').agg(
                pl.col('cos_name').max().alias('bridge_name_cos'),
                pl.col('cos_addr').max().alias('bridge_addr_cos'),
                pl.col('bscore').max().alias('bridge_score'),
                (pl.col('sid') >= 0).sum().cast(pl.Int16).alias('bridge_alias_hits'),
                (pl.col('sid') < 0).any().cast(pl.Int8).alias('bridge_direct_hit'),
                pl.col('confidence').max().alias('bridge_seed_confidence')))
    schema = {'qid':pl.UInt32,'tid':pl.UInt32,'bridge_name_cos':pl.Float32,'bridge_addr_cos':pl.Float32,
              'bridge_score':pl.Float32,'bridge_alias_hits':pl.Int16,'bridge_direct_hit':pl.Int8,
              'bridge_seed_confidence':pl.Float32}
    evidence = pl.concat(parts,how='vertical_relaxed') if parts else pl.DataFrame(schema=schema)
    evidence = evidence.with_columns(pl.col('qid','tid').cast(pl.UInt32))


    keys = []
    for field,cap,name in [('name_core',128,'graph_exact_name'),('addr_can',64,'graph_exact_addr')]:
        idx = refs.filter(pl.col(field) != '').select('co',field,pl.col('rid').alias('qid'))
        idx = idx.with_columns(pl.len().over('co',field).alias('_n')).filter(pl.col('_n') <= cap)
        q = query.filter(pl.col(field) != '')
        if field == 'name_core':
            q = q.filter(pl.col('addr_can') == '')
        keys.append(q.select('co',field,pl.col('rid').alias('tid')).join(idx,on=['co',field])
                    .select('qid','tid',pl.lit(1,pl.Int8).alias(name)))
    extra = keys[0].join(keys[1],on=['qid','tid'],how='full',coalesce=True)
    evidence = evidence.join(extra,on=['qid','tid'],how='full',coalesce=True)
    if 'in_neural' not in pairs.columns:
        pairs = pairs.with_columns(pl.lit(1,pl.Int8).alias('in_neural'))
    res = pairs.join(evidence,on=['qid','tid'],how='full',coalesce=True).with_columns(
        pl.col('prob','neural_prob','gate_prob').fill_null(1e-6),pl.col('in_neural').fill_null(0),
        *[pl.col(c).fill_null(0) for c in evidence.columns if c not in ('qid','tid')])
    return res.sort('tid','qid'), {'queries':query.height,'seeds':seeds.height,
        'bridge_proposals':evidence.height,'new_pairs':res.height-pairs.height}


def sibling_seeds(seeds, refs, targets, n_s2):
    def first_number():
        return pl.col('addr_nums').str.split(' ').list.first().str.slice(-12).cast(pl.Int64,strict=False)
    r = refs.select(pl.col('rid').alias('qid'),first_number().alias('_rh'))
    t = targets.select(pl.col('rid').alias('sid'),first_number().alias('_th'))
    d = seeds.join(r,on='qid').join(t,on='sid')
    p = np.clip(d['confidence'].to_numpy(),1e-6,1-1e-6)
    return d.select('qid','sid',(pl.col('sid') >= n_s2).cast(pl.Int8).alias('s_is_s3'),
        (pl.col('_rh') == pl.col('_th')).fill_null(False).cast(pl.Int8).alias('s_hn_exact')).with_columns(
        pl.Series('s_lg',np.log(p/(1-p)).astype(np.float32)))

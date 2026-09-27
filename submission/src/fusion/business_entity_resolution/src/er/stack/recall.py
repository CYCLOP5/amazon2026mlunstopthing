'bounded exact-key retrieval from supplied records only; never a match rule'
import polars as pl


def add_exact_candidates(pairs, refs, targets, max_owners=8):
    if max_owners < 1:
        raise ValueError('exact_max_owners must be positive')
    proposals = []
    for key, label in [('addr_can','exact_address_candidate'), ('name_core','exact_name_candidate')]:


        idx = refs.filter(pl.col(key) != '').select('co',key,pl.col('rid').alias('qid'))
        idx = idx.with_columns(pl.len().over('co',key).alias('_owners')).filter(pl.col('_owners') <= max_owners)
        query = targets.filter(pl.col(key) != '')
        if key == 'name_core':
            query = query.filter(pl.col('addr_can') == '')
        proposals.append(query.select('co',key,pl.col('rid').alias('tid'))
            .join(idx,on=['co',key],how='inner').select('qid','tid',pl.lit(1,pl.Int8).alias(label)))
    extra = proposals[0].join(proposals[1],on=['qid','tid'],how='full',coalesce=True)
    before = pairs.height
    if 'in_neural' not in pairs.columns:
        pairs = pairs.with_columns(pl.lit(1,pl.Int8).alias('in_neural'))
    out = pairs.join(extra,on=['qid','tid'],how='full',coalesce=True).with_columns(
        pl.col('in_neural','exact_address_candidate','exact_name_candidate').fill_null(0),
        pl.col('prob','gate_prob','neural_prob').fill_null(1e-6))
    return out.sort('tid','qid'), {'exact_proposals':extra.height,'exact_new_pairs':out.height-before}

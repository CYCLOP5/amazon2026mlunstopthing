import gc
import json
from pathlib import Path
import time
import numpy as np
import polars as pl

from er.safe import THREADS, guard
from er.stack import decode
from er.stack.inputs import load_pairs,load_refs,load_targets,add_lexical
from er.stack.features import normalize,score_features,text_features,name_frequency,chunks_by_tid,group_features
from er.stack.enhanced import ambiguity_features,consensus_features
from er.stack.pipeline import export,tuning_anchors
from graph_resolution.retrieve import seed_owners,bridge_candidates,sibling_seeds
from graph_resolution.ranker import add_null,training_groups,fit_models,predict_models

START = time.time()


def log(message):
    print(f'[{(time.time()-START)/60:7.1f} min] {message}',flush=True)


def oracle(pairs,refs):
    anchors = refs.filter(pl.col('fold').is_in([0,1])).select(pl.col('rid').alias('qid'),'deg')
    correct = pairs.filter(pl.col('y') == 1).select('qid','y')
    return decode.score(correct,anchors)


def prepare(sc,split,work,output):
    refs,tg = load_refs(sc['data'],split),load_targets(sc['data'],split)
    n_s2 = int((tg['sr'] == 2).sum())
    pairs,info = load_pairs(sc[f'{split}_roots'],tg.height,threads=min(THREADS,24))
    seeds = seed_owners(pairs)

    seeds = seeds.sort(['qid','confidence','sid'],descending=[False,True,False]).with_columns(
        pl.int_range(0,pl.len()).over('qid').alias('_r')).filter(pl.col('_r') < 32).drop('_r')
    if sc.get(f'lexical_{split}'):
        pairs,li = add_lexical(pairs,sc[f'lexical_{split}'],refs,tg,5,.6)
        info.update(li)
    own = None
    if split == 'train':
        own = pl.concat([pl.read_parquet(Path(sc['data'])/split/f's{sr}.parquet',columns=['rid','own']) for sr in (2,3)])
        own = own.select(pl.col('rid').cast(pl.UInt32).alias('tid'),pl.col('own').cast(pl.Int64))
        pairs = pairs.join(own,on='tid').with_columns((pl.col('qid').cast(pl.Int64)==pl.col('own')).cast(pl.UInt8).alias('y')).drop('own')
        info['oracle_before_graph'] = oracle(pairs,refs)
    log(f'{split}: normalize full reference and target pools for graph context')
    ra,ta = normalize(refs.select('rid','nm','ad','co')),normalize(tg.select('rid','nm','ad','co'))
    if sc.get('graph_predictions'):
        from graph_resolution.refine import reuse_predictions
        filename = 'validation_predictions.parquet' if split == 'train' else 'test_predictions.parquet'
        pairs,seeds,extra = reuse_predictions(pairs,seeds,Path(sc['graph_predictions'])/filename)
    else:
        pairs,extra = bridge_candidates(pairs,ra,ta,seeds,log)
    info.update(extra)
    if own is not None:
        pairs = pairs.join(own,on='tid').with_columns((pl.col('qid').cast(pl.Int64)==pl.col('own')).cast(pl.UInt8).alias('y')).drop('own')
        info['oracle_after_graph'] = oracle(pairs,refs)
    log(f'{split} retrieval: {info}')
    (output/f'inputs_{split}.json').write_text(json.dumps(info,indent=2))
    sf = score_features(pairs,n_s2)
    del pairs

    conf = sibling_seeds(seeds,ra,ta,n_s2)
    if split == 'train':
        sf = sf.join(refs.select(pl.col('rid').alias('qid'),'fold'),on='qid')
        relevant = sf.filter(pl.col('fold').is_in([0,1]))['tid'].unique()
        sf = sf.filter(pl.col('tid').is_in(relevant.implode()))
    sf = sf.sort('tid','qid')
    small = ta.select('rid','name_core','addr_can',
        pl.col('addr_nums').str.split(' ').list.first().fill_null('').alias('hn1'))
    name_ref,name_target = name_frequency(ra,ta)
    parts = []
    for chunk in chunks_by_tid(sf.select('qid','tid','is_s3'),100_000):
        text = text_features(chunk,ra,ta,['name','address','numbers'])
        siblings = group_features(chunk,conf,small).join(consensus_features(chunk,conf,small),on=['qid','tid'])
        parts.append(text.join(siblings,on=['qid','tid']))
        if len(parts)%20 == 0:
            log(f'{split}: feature chunks {len(parts)}; approximately {len(parts)*100_000:,}/{sf.height:,} pairs')
        guard('graph feature construction')
    ff = sf.join(pl.concat(parts),on=['qid','tid']).join(name_ref,on='qid').join(name_target,on='tid',how='left')
    ff = ff.join(ambiguity_features(sf.select('qid','tid'),ra,ta),on=['qid','tid'])
    if own is not None:
        groups = training_groups(own.rename({'tid':'rid'}),refs)
        ff = ff.join(groups,on='tid',how='left')
    ff = ff.sort('tid','qid')
    ff.write_parquet(work/f'{split}_features.parquet')
    log(f'{split}: saved {ff.height:,} pairs x {ff.width} features')
    return info


def subset_for_tuning(pairs,anchors):

    tids = pairs.join(anchors.select('qid'),on='qid')['tid'].unique()
    return pairs.filter(pl.col('tid').is_in(tids.implode()))


def blend(pred,weight):
    return pred.select('qid','tid',((1-weight)*pl.col('binary')+weight*pl.col('rank')).alias('p'),
                       *(['y'] if 'y' in pred.columns else []))


def entity_scores(accepted,anchors):
    count = accepted.group_by('qid').agg(pl.len().alias('_n'),pl.col('y').sum().alias('_tp'))
    d = anchors.sort('qid').join(count,on='qid',how='left',maintain_order='left').fill_null(0)
    n,tp,degree = [d[c].to_numpy().astype(float) for c in ('_n','_tp','deg')]
    return np.where(degree==0,(n==0).astype(float),1.25*tp/np.maximum(n+.25*degree,1e-9))


def compare_audit(winner,baseline,anchors):
    delta = entity_scores(winner,anchors)-entity_scores(baseline,anchors)
    rng = np.random.default_rng(913)
    means = [delta[rng.integers(0,len(delta),size=len(delta))].mean() for _ in range(500)]
    return {'paired_delta':float(delta.mean()),'paired_bootstrap_95_ci':np.quantile(means,[.025,.975]).tolist(),
            'replicates':500,'reference':'binary classifier on identical graph candidates',
            'historical_audit_reference':0.988947737846756,
            'historical_comparison_is_not_a_controlled_ablation':True}


def run(sc,output,work):
    output,work = Path(output),Path(work)
    output.mkdir(parents=True,exist_ok=True)
    work.mkdir(parents=True,exist_ok=True)
    (output/'config.json').write_text(json.dumps(sc,indent=2))
    info_train = prepare(sc,'train',work,output)
    gc.collect()
    tr = add_null(pl.read_parquet(work/'train_features.parquet'))
    pred,model_info = fit_models(tr,output/'bundle',log,rounds=sc.get('rounds',1800))
    del tr
    gc.collect()
    pred.write_parquet(output/'validation_predictions.parquet')
    anchors = load_refs(sc['data'],'train').select(pl.col('rid').alias('qid'),'fold','deg')
    tune = tuning_anchors(anchors,{'fit_fold':0,'seed':42,'tune_holdout_buckets':5})
    audit = anchors.filter(pl.col('fold') == 1).select('qid','deg')
    small = subset_for_tuning(pred,tune)
    experiments = {}
    for weight in (0.0,.25,.5,.75,1.0):
        res = decode.tune(blend(small,weight),tune,rules=('top1_threshold','expected_f'),floors=(.1,.3,.5))
        experiments[str(weight)] = res
        log(f'tuning rank weight {weight}: F0.5 {res["macro_f05"]:.7f}')
    winner = max(experiments,key=lambda key:experiments[key]['macro_f05'])
    chosen = experiments[winner]
    accepted = decode.apply(chosen['rule'],blend(pred,float(winner)),chosen['threshold'])
    base = experiments['0.0']
    base_accepted = decode.apply(base['rule'],blend(pred,0),base['threshold'])
    audit_metrics = decode.score(accepted,audit)
    report = {'inputs_train':info_train,'training':model_info,'tuning':experiments,'selected_rank_weight':float(winner),
        'selected_decoder':chosen,'audit':audit_metrics,'binary_audit':decode.score(base_accepted,audit),
        'comparison':compare_audit(accepted,base_accepted,audit),'tune_entities':tune.height,'audit_entities':audit.height}
    (output/'report.json').write_text(json.dumps(report,indent=2))
    (output/'bundle/decision.json').write_text(json.dumps({'weight':float(winner),'decoder':chosen},indent=2))
    log(f'HELDOUT AUDIT: {audit_metrics}; paired comparison {report["comparison"]}')
    del pred,small,accepted,base_accepted
    gc.collect()

    report['inputs_test'] = prepare(sc,'test',work,output)
    if info_train['config_sha256'] != report['inputs_test']['config_sha256']:
        raise ValueError('Train/test upstream model configurations differ')
    gc.collect()
    te = add_null(pl.read_parquet(work/'test_features.parquet'))
    scored = predict_models(te,output/'bundle',log).filter(pl.col('is_null') == 0).drop('is_null')
    del te
    gc.collect()
    scored.write_parquet(output/'test_predictions.parquet')
    out = blend(scored,float(winner))
    refs,tg = load_refs(sc['data'],'test'),load_targets(sc['data'],'test')
    report['export'] = export(out,'p',chosen['rule'],chosen['threshold'],refs,tg,str(output/'output'))
    (output/'report.json').write_text(json.dumps(report,indent=2))
    log('Graph experiment complete; inspect report.json before comparing with the original suite')

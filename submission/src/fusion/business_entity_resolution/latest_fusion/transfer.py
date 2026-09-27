'country-transfer proxy for a gate-offset decision head, without french labels'
import json
import os
from pathlib import Path
import numpy as np
import polars as pl
from latest_fusion.calibration import histogram,curves,adjust,partition,decide
from er.stack import decode

def transfer_features(features):
    banned=('newest','neural','np_m','member_spread','model_spread','new_old_gap','models_above_90','gate_neural_gap')
    return [name for name in features if not any(term in name for term in banned)]

def gate_offset(frame):
    p=np.clip(frame['gate'].fill_null(1e-4).to_numpy().astype(float),1e-4,1-1e-4)
    return np.log(p/(1-p))

def predict(model,frame,features):
    out=np.empty(frame.height,dtype=np.float32)
    for start in range(0,frame.height,1_000_000):
        out[start:start+1_000_000]=model.predict(frame.slice(start,1_000_000).select(features).to_numpy(),raw_score=True)
    return out

def probability(frame,residual=None):
    if residual is None:return frame['gate'].fill_null(1e-4).to_numpy().astype(float)
    return 1/(1+np.exp(-np.clip(gate_offset(frame)+.25*residual,-40,40)))

def transferred_recipe(source_frame,target_frame,source_refs,target_refs,raw_source,raw_target,edges):
    'only source partition1 labels enter the estimator; target contributes density'
    cal=(source_frame['fold'].to_numpy()==0)&(partition(source_frame['qid'])==1)
    source_ids=source_refs.filter((pl.col('fold')==0)&pl.Series(partition(source_refs['rid'])==1))


    sc=np.full(source_frame.height,'transfer',dtype=object)
    tc=np.full(target_frame.height,'transfer',dtype=object)
    trainhist=histogram(raw_source,sc,source_frame['seg'].to_numpy(),source_frame['y'].to_numpy(),edges,cal)
    targethist=histogram(raw_target,tc,target_frame['seg'].to_numpy(),None,edges,np.ones(target_frame.height,bool))
    recipe={'curves':curves(trainhist,targethist,{'transfer':source_ids.height},{'transfer':target_refs.height},edges)}
    return recipe,{'calibration_source_entities':source_ids.height,'calibration_source_pairs':int(cal.sum()),
        'target_density_entities':target_refs.height,'target_density_pairs':target_frame.height,
        'label_scope':'Source country fold0 partition1 only; target labels never enter calibration.'}

def accepted(frame,raw,recipe):
    co=np.full(frame.height,'transfer',dtype=object)
    calibrated=adjust(raw,co,frame['seg'].to_numpy(),recipe)
    d=frame.with_columns(pl.Series('_transfer',calibrated),pl.Series('raw__transfer',raw.astype(np.float32)))
    return decide(d,'_transfer','expected_f',.05)

def run_transfer(tr,te,refs,rt,features,fitids,tuneids,params,output):
    import lightgbm as lgb
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    features=transfer_features(features)
    if not features:raise ValueError('No country-neutral transfer features')
    if np.intersect1d(fitids,tuneids).size:raise ValueError('Transfer fit/tune owners overlap')
    countries=sorted(str(co) for co in refs['co'].unique())
    if len(countries)!=2:raise ValueError('Transfer needs exactly two labeled countries')
    edges=np.linspace(np.log(.02/.98),np.log(.999/.001),25)
    report={'scope':'Country holdout at new head and calibrator only; frozen upstream models were trained in both countries.',
        'strength':.25,'rounds':182,'features':features,'directions':{},'promoted':False}
    models=[]
    for src,target in (countries,countries[::-1]):
        sr=refs.filter(pl.col('co')==src);targetrefs=refs.filter(pl.col('co')==target)
        sourceframe=tr.filter(pl.col('qid').is_in(sr['rid'].implode()))
        targetframe=tr.filter(pl.col('qid').is_in(targetrefs['rid'].implode()))
        sourcefit=sr.filter(pl.col('rid').is_in(pl.Series(fitids).implode()))['rid']
        fitframe=sourceframe.filter(pl.col('qid').is_in(sourcefit.implode())&
            ((pl.col('own')<0)|pl.col('own').is_in(sourcefit.cast(pl.Int64).implode())))
        if not fitframe.height:raise ValueError('Empty source-country transfer fit')
        print(f'transfer {src}->{target}: fitting {fitframe.height:,} pairs from {len(sourcefit):,} source owners',flush=True)
        p=dict(params);p['num_threads']=int(os.environ.get('ER_THREADS','48'))
        model=lgb.train(p,lgb.Dataset(fitframe.select(features).to_numpy(),label=fitframe['y'].to_numpy(),init_score=gate_offset(fitframe)),num_boost_round=182)
        model.save_model(str(output/f'transfer-{src}.txt'));models.append(model)
        targetanchors=targetrefs.filter(pl.col('rid').is_in(pl.Series(tuneids).implode())).select(pl.col('rid').alias('qid'),'deg')
        metrics={};calibrators={}
        for name,rs,rr in [('baseline_gate',None,None),('residual_gate',predict(model,sourceframe,features),predict(model,targetframe,features))]:
            sourcep,targetp=probability(sourceframe,rs),probability(targetframe,rr)
            recipe,scope=transferred_recipe(sourceframe,targetframe,sr,targetrefs,sourcep,targetp,edges)
            metrics[name]=decode.score(accepted(targetframe,targetp,recipe),targetanchors)
            calibrators[name]={'recipe':recipe,'scope':scope}
        base,new=metrics['baseline_gate'],metrics['residual_gate']
        passes=new['macro_f05']>=base['macro_f05']+5e-6 and new['pair_precision']>=base['pair_precision']-.0001
        print(f'transfer {src}->{target}: baseline {base["macro_f05"]:.9f}, residual {new["macro_f05"]:.9f}, pass={passes}',flush=True)
        report['directions'][src+'->'+target]={'fit_entities':len(sourcefit),'fit_pairs':fitframe.height,
            'fit_positive':int(fitframe['y'].sum()),'tune_entities':targetanchors.height,'metrics':metrics,
            'passes':passes,'calibrators':calibrators}
    report['promoted']=all(d['passes'] for d in report['directions'].values())
    franceids=rt.filter(pl.col('co').str.to_lowercase().is_in(['fr','france']))['rid']
    french=te.filter(pl.col('qid').is_in(franceids.implode()))
    sel=None
    if report['promoted'] and french.height:
        pooled=tr.filter(pl.col('qid').is_in(refs['rid'].implode()))
        source_residual=sum(predict(m,pooled,features) for m in models)/len(models)
        french_residual=sum(predict(m,french,features) for m in models)/len(models)
        recipe,scope=transferred_recipe(pooled,french,refs,rt.filter(pl.col('rid').is_in(franceids.implode())),
            probability(pooled,source_residual),probability(french,french_residual),edges)
        sel=accepted(french,probability(french,french_residual),recipe)
        report['france']={'recipe':recipe,'scope':scope,'accepted_pairs':sel.height,'labeled_accuracy':None}
    else:
        report['reason']='Both country directions must improve macro F0.5 by at least 5e-6 with precision loss at most .0001. Frozen newest France remains selected.'
    (output/'transfer_report.json').write_text(json.dumps(report,indent=2))
    return sel,report

'exercise score preparation, residual fit, exact fallback and real tsv validation'
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import polars as pl
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from latest_fusion.pipeline import prepare,fit,sha256

def test_complete_frozen_fusion(tmp_path,monkeypatch):
    monkeypatch.setenv('ER_THREADS','2')
    data=tmp_path/'data';new=tmp_path/'newest';new.mkdir();data.mkdir()
    (data/'meta.json').write_text('{"layout":"fixture"}')
    exp=tmp_path/'expected.json';exp.write_text((data/'meta.json').read_text())
    old={name:tmp_path/name for name in ('hybrid','graph','friend')}
    for p in old.values():p.mkdir()
    prepared=tmp_path/'prepared';n=240
    for split in ('train','test'):
        d=data/split;d.mkdir()
        q=np.arange(n,dtype=np.uint32)
        countries=['us' if i%2==0 else 'india' for i in q]
        if split=='test':countries[-12:]=['france']*12
        refs=pl.DataFrame({'rid':q,'eid':[f'{split}-r{i}' for i in q],
            'nm':[f'company {i}' for i in q],'ad':['0007 main road']*n,'co':countries,
            'fold':[0 if i<200 else 1 for i in q],'deg':[1]*n})
        refs.write_parquet(d/'ref.parquet')
        targets=pl.DataFrame({'rid':q,'eid':[f'{split}-t{i}' for i in q],
            'nm':[f'company {i}' for i in q],'ad':['99 lane 0007']*n,'co':countries,'own':q.astype(np.int64)})
        targets[:n//2].write_parquet(d/'s2.parquet');targets[n//2:].write_parquet(d/'s3.parquet')
        pairs=pl.DataFrame({'qid':np.r_[q,(q+2)%n],'tid':np.r_[q,q],
            'stack_prob':np.r_[np.full(n,.99),np.full(n,.002)],'gate_prob':np.r_[np.full(n,.99),np.full(n,.002)],
            'neural_prob':np.r_[np.full(n,.98),np.full(n,.002)],'co':[countries[i] for i in np.r_[q,(q+2)%n]],
            'fold':[0 if i<200 else 1 for i in np.r_[q,(q+2)%n]],'own':np.r_[q,q].astype(np.int64),'seg':[0]*(2*n),
            'y':np.r_[np.ones(n,dtype=np.uint8),np.zeros(n,dtype=np.uint8)]})
        for i in range(15):pairs=pairs.with_columns(pl.col('neural_prob').alias(f'np_m{i}'))
        src=new/f'stack-{split}.parquet';pairs.write_parquet(src)
        src.with_suffix('.json').write_text(json.dumps({'data_meta_sha256':sha256(data/'meta.json'),'score_sha256':sha256(src)}))
        for name,column in [('hybrid','p'),('graph','head'),('friend','p2')]:

            scores=pl.concat([pairs.select('qid','tid',pl.col('stack_prob').alias(column)),
                pl.DataFrame({'qid':[1],'tid':[3],column:[.2]},schema={'qid':pl.UInt32,'tid':pl.UInt32,column:pl.Float64})],how='vertical_relaxed')
            filename=('val_pred.parquet' if split=='train' else 'test_pred.parquet') if name=='friend' else ('validation_predictions.parquet' if split=='train' else 'test_predictions.parquet')
            scores.write_parquet(old[name]/filename)
        prepare(split,data,new,old['hybrid'],old['graph'],old['friend'],prepared,exp)
        frame=pl.read_parquet(prepared/f'prepared-{split}.parquet')
        extra=frame.filter((pl.col('qid')==1)&(pl.col('tid')==3))
        assert extra['newest_present'].item()==0 and extra['seg'].item()==0
        assert frame.height==2*n+1
    recipe=tmp_path/'recipe.json'
    recipe.write_text(json.dumps({'known_score':'stack_prob','unseen_gate':True,'countries':['us','india'],
        'partition':{'modulus':3,'remainder':1},'edges':[-4,0,4],
        'curves':{f'{co}|{seg}':{'centres':[-12,0,12],'posterior':[0,.5,1]} for co in ('us','india','france') for seg in range(3)}}))
    meta=tmp_path/'metadata.json';meta.write_text(json.dumps({'params':{'objective':'binary','verbosity':-1,'num_leaves':3,'min_data_in_leaf':10,'learning_rate':.0537,'lambda_l2':5.716}}))
    report=fit(data,prepared/'prepared-train.parquet',prepared/'prepared-test.parquet',meta,tmp_path/'result',recipe)
    assert report['validation']['source1']==n
    assert report['validation']['candidate_pairs']==2*n+1
    assert report['fit_entities']>0 and report['tune_entities']>0
    assert (tmp_path/'result'/'residual_model.txt').exists()
    assert report['changes']['added_by_country'].get('france',0)==0
    assert report['changes']['removed_by_country'].get('france',0)==0

    (new/'stack-train.json').write_text(json.dumps({'data_meta_sha256':sha256(data/'meta.json'),'score_sha256':'wrong'}))
    with pytest.raises(ValueError,match='content hash mismatch'):
        prepare('train',data,new,old['hybrid'],old['graph'],old['friend'],tmp_path/'rejected',exp)

'real specialist fitting, frozen export, and orchestrator fallback checks'
import json
from pathlib import Path
import sys
import numpy as np
import polars as pl
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from latest_fusion.rescue_pipeline import run,proposal_pool,fit_heads,score_actions,risk_floor,export_frozen_additions
from latest_fusion.pipeline import sha256


@pytest.mark.parametrize('competition',[False,True])
def test_exact_fallback_and_additive_export(tmp_path,monkeypatch,competition):
    from test_latest_fusion_pipeline import test_complete_frozen_fusion
    test_complete_frozen_fusion(tmp_path,monkeypatch)
    incumbent=tmp_path/'result'
    report=json.loads((incumbent/'report.json').read_text()); report['selected']='residual_0.5'
    (incumbent/'report.json').write_text(json.dumps(report))
    config=json.loads((ROOT/'configs'/('competition_rescue.json' if competition else 'recall_rescue.json')).read_text())
    config['incumbent_matching_sha256']=report['matching_sha256']
    path=tmp_path/'rescue.json';path.write_text(json.dumps(config))
    out=tmp_path/'rescued'
    res=run(tmp_path/'data',tmp_path/'prepared/prepared-train.parquet',tmp_path/'prepared/prepared-test.parquet',incumbent,path,out)
    assert not res['promoted'] and not res['submission_changed']
    assert sha256(out/'output/matching_results.tsv')==report['matching_sha256']
    assert res['validation']['source1']==240


    accepted=pl.read_parquet(incumbent/'accepted_test.parquet')
    omitted=accepted.filter(pl.col('qid')==0).head(1)
    assert omitted.height==1
    frozen=accepted.join(omitted.select('tid'),on='tid',how='anti')
    frozen.write_parquet(incumbent/'accepted_test.parquet')
    addition=omitted.select('qid','tid',pl.col('p').alias('score'))
    exported=tmp_path/'positive';exported.mkdir()
    details,delta=export_frozen_additions(tmp_path/'data',tmp_path/'prepared/prepared-test.parquet',incumbent,exported,addition)
    final=pl.read_parquet(exported/'accepted_test.parquet')
    assert delta['added_pairs']==1 and delta['removed_pairs']==0 and delta['changed_target_owners']==0
    assert delta['added_by_country'].get('france',0)==0
    assert frozen.select('qid','tid').join(final.select('qid','tid'),on=['qid','tid'],how='anti').height==0
    assert final.height==accepted.height


def test_pool_and_risk_float32():
    pool=pl.DataFrame({'qid':[1,2,3,4],'tid':[10,10,20,30],'co':['us']*4,
        **{c:[.8,.001,.99,.001] for c in ['inc_p','newest','gate','neural','hybrid','graph','friend']}})
    kept=proposal_pool(pool,pl.DataFrame({'tid':[20]}),['us'],.05)
    assert kept['qid'].to_list()==[1,2]
    actions=pl.DataFrame({'qid':[1,2],'tid':[10,11],'co':['us']*2,'y':[0,1],'score':pl.Series([.7,.9],dtype=pl.Float32)})
    floor,report=risk_floor(actions,pl.DataFrame({'qid':[1,2]}))
    assert floor>float(np.float32(.7)) and floor==float(np.nextafter(np.float32(.7),np.float32(np.inf)))
    assert actions.filter(pl.col('score')>=floor)['y'].to_list()==[1]


def test_real_owner_cv_heads_and_scoring(tmp_path,monkeypatch):
    monkeypatch.setenv('ER_THREADS','2')
    owners=np.arange(120,dtype=np.uint32)

    pool=pl.DataFrame({'qid':np.repeat(owners,2),'tid':np.arange(240,dtype=np.uint32),
        'co':['us']*240,'own':np.repeat(owners.astype(np.int64),2),
        'y':np.tile([1,0],120),'evidence':np.tile([1.,-1.],120)})
    foreign=pl.DataFrame({'qid':[0],'tid':[999],'co':['us'],'own':[999],'y':[1],'evidence':[100.]},schema=pool.schema)
    cfg={'minimum_fit_positive':10,'minimum_fit_negative':10,'folds':3,'rounds':12,'early_stopping':3,
        'lgbm':{'objective':'binary','metric':'binary_logloss','verbosity':-1,'num_leaves':3,'min_data_in_leaf':3,'learning_rate':.2}}
    models,summary=fit_heads(pl.concat([pool,foreign]),owners,['evidence'],cfg,tmp_path)
    assert len(models)==3 and summary['pairs']==240 and summary['positive']==120
    assert all((tmp_path/f'rescue_model_{i}.txt').exists() for i in range(3))

    cands=pl.DataFrame({'qid':[1,2,1,2],'tid':[10,10,11,11],'co':['us']*4,'y':[1,0,1,0],'evidence':[1.,-1.,.5,-.5]})
    scored=score_actions(cands,['evidence'],models)
    assert scored['qid'].to_list()==[1,1]
    assert scored['score'].min()>.5
    flipped=score_actions(cands.with_columns((1-pl.col('y')).alias('y')),['evidence'],models)
    assert scored.select('qid','tid','score').equals(flipped.select('qid','tid','score'))


def test_each_head_tie_abstains_even_with_unique_ensemble_minimum():
    class FixedHead:
        def __init__(self,values): self.values=np.asarray(values,dtype=np.float32)
        def predict(self,features,**kwargs): return self.values
    pool=pl.DataFrame({'qid':[1,2,1,2,3],'tid':[10,10,20,20,30],
                       'co':['us']*5,'evidence':[0.]*5})


    heads=[FixedHead([.9,.9,.9,.7,.8]),FixedHead([.8,.7,.8,.6,.9])]
    res=score_actions(pool,['evidence'],heads)
    assert res['tid'].to_list()==[20,30]
    assert res['qid'].to_list()==[1,3]
    assert res['score'].to_list()==[float(np.float32(.8))]*2

    assert score_actions(pool,['evidence'],heads[::-1]).select('qid','tid','score').equals(res.select('qid','tid','score'))
    margins=score_actions(pool,['evidence'],heads,return_margins=True)
    assert margins['min_head_margin'].to_list()==pytest.approx([.2,.8])


def test_full_competition_rescue_branch_with_real_models(tmp_path,monkeypatch):
    from test_latest_fusion_pipeline import test_complete_frozen_fusion
    test_complete_frozen_fusion(tmp_path,monkeypatch)
    incumbent=tmp_path/'result'
    report=json.loads((incumbent/'report.json').read_text())
    report['selected']='residual_0.5'


    report['methods']['residual_0.5']['tune'].update(threshold=2.,macro_f05=0.)
    (incumbent/'report.json').write_text(json.dumps(report))
    config=json.loads((ROOT/'configs/competition_rescue.json').read_text())
    config.update(incumbent_matching_sha256=report['matching_sha256'],minimum_fit_positive=1,
                  minimum_fit_negative=1,rounds=8,early_stopping=2,orphan_fraction=1.)
    config['lgbm'].update(num_leaves=3,min_data_in_leaf=1,learning_rate=.2)
    path=tmp_path/'competition.json';path.write_text(json.dumps(config))
    output=tmp_path/'competition'
    res=run(tmp_path/'data',tmp_path/'prepared/prepared-train.parquet',tmp_path/'prepared/prepared-test.parquet',incumbent,path,output)
    assert len(res['training']['folds'])==3
    assert 'family_selection' in res and 'evidence_gate' in res
    assert not res['promoted'] and not res['submission_changed']
    assert sha256(output/'output/matching_results.tsv')==report['matching_sha256']

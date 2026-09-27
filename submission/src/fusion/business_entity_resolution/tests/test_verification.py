import json
from pathlib import Path
import sys
import numpy as np
import polars as pl

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from innovation.prepare import select_targets,run as prepare
from innovation.verify import view_text
from innovation.common import replace_scores
from innovation.train import run as fit,training_partitions
from test_stack import make_fixture


def test_target_budget_never_uses_labels_and_keeps_competitors():
    p = pl.DataFrame({'qid':[0,1,0,1],'tid':[0,0,1,1],'p':[.7,.6,.99,.01],'y':[1,0,1,0]})
    t = pl.DataFrame({'rid':[0,1],'ad':['street',''],'co':['us','india']})
    a,_ = select_targets(p,t,max_targets=1)
    b,_ = select_targets(p.with_columns((1-pl.col('y')).alias('y')),t,max_targets=1)
    assert a.select('qid','tid').equals(b.select('qid','tid'))
    assert a['tid'].to_list() == [1,1]


def test_views_remove_only_intended_fields():
    d = pl.DataFrame({'nm1':['ACME'],'ad1':['1 Rue'],'co1':['france'],'cn1':['acme'],'ca1':['1 rue']})
    assert view_text(d,1,'name') == ['name: ACME\naddress: \ncountry: france']
    assert view_text(d,1,'address') == ['name: \naddress: 1 Rue\ncountry: france']
    assert view_text(d,1,'canonical') == ['name: acme\naddress: 1 rue\ncountry: ']


def test_score_patch_retains_unverified_candidates():
    p = pl.DataFrame({'qid':[0,1,2],'tid':[0,0,1],'p':[.1,.2,.3]})
    patched = replace_scores(p,p.head(1).with_columns(pl.lit(.9).alias('p'))).sort('qid')
    assert patched['p'].to_list() == [.9,.2,.3]


def test_owner_partitions_exclude_all_heldout_targets_and_group_siblings():
    refs = pl.DataFrame({'rid':np.arange(200,dtype=np.uint32),'fold':[0]*100+[1]*100})
    fit_ids = refs.filter((pl.col('fold')==0)&((pl.col('rid').hash(1033)%5)!=0))['rid'].to_list()
    tune_ids = refs.filter((pl.col('fold')==0)&((pl.col('rid').hash(1033)%5)==0))['rid'].to_list()
    a,b = fit_ids[:2]
    frame = pl.DataFrame({'qid':[a,b,a,b,a,b,a], 'tid':[0,0,1,1,2,2,3],
        'own':[a,a,a,a,100,100,tune_ids[0]], 'fold':[0]*7}).with_columns(pl.col('qid','tid').cast(pl.UInt32))
    allowed,groups = training_partitions(frame,refs)
    assert allowed.tolist() == [True,True,True,True,False,False,False]
    assert len(set(groups[:4])) == 1

    assert not allowed[4:].any()


def test_decoy_groups_use_target_ids_and_holdout_candidate_rows_stay_excluded():
    refs = pl.DataFrame({'rid':np.arange(200,dtype=np.uint32),'fold':[0]*100+[1]*100})
    a = refs.filter((pl.col('fold')==0)&((pl.col('rid').hash(1033)%5)!=0))['rid'][0]
    tids = pl.DataFrame({'tid':np.arange(1000,dtype=np.uint32)}).filter((pl.col('tid').hash(713)%100)<8)['tid'].head(2).to_list()
    frame = pl.DataFrame({'qid':[a,a,100], 'tid':[tids[0],tids[1],tids[0]],'own':[-1]*3,'fold':[0,0,1]}).with_columns(pl.col('qid','tid').cast(pl.UInt32))
    allowed,groups = training_partitions(frame,refs)
    assert allowed.tolist() == [True,True,False]
    assert groups[0] == groups[2]


def test_cloud_preparation_training_and_complete_export(tmp_path):
    data,roots = make_fixture(tmp_path,160)
    parent,prepared,verified = [tmp_path/n for n in ('parent','prepared','verified')]
    parent.mkdir()
    for split in ('train','test'):
        p = pl.read_parquet(Path(roots[split][0])/'parts/part_0.parquet').select('qid','tid',
            pl.col('prob').alias('base'),pl.col('prob').alias('head'),*(['y'] if split=='train' else []))
        p.write_parquet(parent/('validation_predictions.parquet' if split=='train' else 'test_predictions.parquet'))
    (parent/'report.json').write_text(json.dumps({'selected_head_weight':1.,'selected_decoder':{'rule':'top1_threshold','threshold':.5}}))
    prepare(data,parent,prepared,max_targets=10_000)


    train_features = pl.read_parquet(prepared/'train'/'features.parquet')
    refs = pl.read_parquet(data/'train'/'ref.parquet')
    fit_ids = refs.filter((pl.col('fold')==0)&((pl.col('rid').hash(1033)%5)!=0))['rid'].to_list()
    extra = []
    for qid,other in zip(fit_ids,fit_ids[1:]+fit_ids[:1]):
        row = train_features.filter((pl.col('qid')==qid)&(pl.col('own')==qid)).head(1)
        extra.append(row.with_columns(pl.lit(other,dtype=pl.UInt32).alias('qid'),pl.lit(0,dtype=pl.UInt8).alias('y')))
    train_features = pl.concat([train_features,*extra],how='vertical_relaxed')
    train_features.write_parquet(prepared/'train'/'features.parquet')
    for split in ('train','test'):
        (verified/split).mkdir(parents=True)
        ff = pl.read_parquet(prepared/split/'features.parquet')
        p = ff['parent_p'].to_numpy()
        logits = np.log(p/(1-p))
        ff.select('qid','tid').with_columns([pl.Series('ce_'+v+'_lg',logits) for v in ('full','name','address','canonical')]).write_parquet(verified/split/'part_00000.parquet')
    (verified/'_SUCCESS').write_text('ok')
    (verified/'verification.json').write_text(json.dumps({'synthetic':True}))
    fit(data,parent,prepared,verified,tmp_path/'result',rounds=8)
    report = json.loads((tmp_path/'result/report.json').read_text())
    assert report['export']['candidate_pairs'] == 32*2*3
    assert (tmp_path/'result/bundle/multiview_2.txt').exists()
    columns = json.loads((tmp_path/'result/bundle/features.json').read_text())
    assert not {'y','own','fold','co','qid','tid'} & set(columns['multiview'])
    assert 'ce_canonical_lg' not in columns['direct']
    assert 'ce_canonical_lg' in columns['multiview']

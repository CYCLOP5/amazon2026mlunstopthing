import json
from pathlib import Path
import sys

import numpy as np
import polars as pl

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from graph_resolution.ranker import add_null,rank_margin,training_groups,fit_models
from graph_resolution.retrieve import seed_owners,bridge_candidates
from graph_resolution.pipeline import run
from er.stack.features import normalize
from test_stack import make_fixture


def test_null_candidate_and_probability_competition():
    pairs = pl.DataFrame({'qid':pl.Series([0,1,0],dtype=pl.UInt32),'tid':[0,0,1],
        'prob':[.8,.2,.3],'y':pl.Series([1,0,0],dtype=pl.UInt8)})
    augmented = add_null(pairs)
    assert augmented.filter(pl.col('is_null') == 1)['y'].to_list() == [0,1]
    margin = rank_margin(augmented,np.zeros(augmented.height,dtype=float))
    prob = 1/(1+np.exp(-margin))
    np.testing.assert_allclose(prob[:3],1/3)
    np.testing.assert_allclose(prob[3:],1/2)


def test_frozen_graph_seeds_ignore_labels():
    p = pl.DataFrame({'qid':[0,1],'tid':[0,0],'prob':[.999,.1],
        'gate_prob':[.999,.1],'neural_prob':[.999,.1],'y':[1,0]})
    a = seed_owners(p)
    b = seed_owners(p.with_columns((1-pl.col('y')).alias('y')))
    assert a.equals(b)
    assert a.height == 1


def test_target_group_training_excludes_audit_and_tuning():
    refs = pl.DataFrame({'rid':pl.Series(range(100),dtype=pl.UInt32),'fold':[i%2 for i in range(100)]})
    tg = pl.DataFrame({'rid':pl.Series(range(101),dtype=pl.UInt32),'own':list(range(100))+[-1]})
    groups = training_groups(tg,refs)
    fit = groups.filter(pl.col('_fit'))['tid'].to_list()
    tune = set(refs.filter((pl.col('fold')==0)&((pl.col('rid').hash(1033)%5)==0))['rid'].to_list())
    assert not set(fit) & tune
    assert all(t == 100 or t%2 == 0 for t in fit)


def test_bridge_does_not_use_query_as_its_own_alias():
    refs = normalize(pl.DataFrame({'rid':pl.Series([0,1],dtype=pl.UInt32),
        'nm':['Acme plumbing','Other shop'],'ad':['100 River Road','900 Lake Road'],'co':['us','us']}))
    tg = normalize(pl.DataFrame({'rid':pl.Series([0],dtype=pl.UInt32),
        'nm':['Acme plumbing'],'ad':['100 River Road'],'co':['us']}))
    p = pl.DataFrame({'qid':pl.Series([0,1],dtype=pl.UInt32),'tid':pl.Series([0,0],dtype=pl.UInt32),
        'prob':[.8,.1],'gate_prob':[.9,.1],'neural_prob':[.7,.1]})
    seeds = pl.DataFrame({'qid':pl.Series([0],dtype=pl.UInt32),'sid':pl.Series([0],dtype=pl.UInt32),'confidence':[.999]})
    out,_ = bridge_candidates(p,refs,tg,seeds,lambda _:None)
    assert out['bridge_alias_hits'].sum() == 0
    assert out.filter(pl.col('qid')==0)['bridge_direct_hit'][0] == 1


def test_graph_pipeline_saves_report_bundle_and_complete_output(tmp_path):
    data,roots = make_fixture(tmp_path,160)
    sc = {'data':str(data),'train_roots':roots['train'],'test_roots':roots['test'],
        'lexical_train':[],'lexical_test':[],'rounds':8}
    run(sc,tmp_path/'result',tmp_path/'work')
    report = json.loads((tmp_path/'result/report.json').read_text())
    assert report['audit_entities'] == 80
    assert (tmp_path/'result/bundle/rank_2.txt').exists()
    assert (tmp_path/'result/bundle/binary_2.txt').exists()
    output = pl.read_csv(tmp_path/'result/output/matching_results.tsv',separator='\t')
    assert output.height == 32
    assert 'paired_bootstrap_95_ci' in report['comparison']

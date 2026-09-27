from pathlib import Path
import sys
import json
import shutil
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import polars as pl
from latest_fusion.tuning import split_search_check, top_pairs, apply_policy, select_policy, promotion
from latest_fusion import decoder


def frame():
    return pl.DataFrame({'qid':[1,2,1,3], 'tid':[10,10,11,12], 'p':[.8,.9,.99,.49], 'raw':[.8,.9,.99,.49], 'y':[0,1,1,0], 'co':['us']*4})


def test_split_stable_disjoint():
    refs=pl.DataFrame({'rid':range(100), 'deg':[1]*100, 'co':['us']*100})
    a,b=split_search_check(refs,np.arange(100))
    c,d=split_search_check(refs.reverse(),np.arange(100)[::-1])
    assert set(a['qid']) == set(c['qid'])
    assert set(b['qid']) == set(d['qid'])
    assert not set(a['qid']) & set(b['qid'])
    assert a.height+b.height == 100


def test_competitors_preserved():
    f=frame()
    assert top_pairs(f).filter(pl.col('tid')==10)['qid'].item()==2
    anchors=pl.DataFrame({'qid':[1], 'deg':[1], 'co':['us']})
    policy,_=select_policy(f,anchors,expected_f_floors=())
    assert apply_policy(f,policy).filter(pl.col('tid')==10).height in (0,1)
    assert apply_policy(f,policy).filter((pl.col('tid')==10)&(pl.col('qid')==1)).height==0


def test_expected_f_matches_exact_and_raw_tie():
    f=frame().with_columns(pl.when(pl.col('tid')==10).then(.9).otherwise(pl.col('p')).alias('p'))
    keep,_=decoder.choose(f['qid'].to_numpy(),f['tid'].to_numpy(),f['p'].to_numpy(),raw=f['raw'].to_numpy(),floor=.05,exact=64)
    assert apply_policy(f,{'rule':'expected_f','threshold':.05}).select('qid','tid').equals(f.filter(pl.Series(keep)).select('qid','tid'))


def test_empty_and_singleton_check():
    empty=frame().head(0).select('qid','tid','p','y')
    anchors=pl.DataFrame({'qid':[1], 'deg':[0], 'co':['us']})
    res=promotion(empty,empty,anchors)
    assert res['candidate']['macro_f05']==1
    assert res['paired_gain']==0
    assert not res['passed']


def test_loss_rejected():
    f=frame().select('qid','tid','p','y')
    anchors=pl.DataFrame({'qid':[1,2,3], 'deg':[1,1,0], 'co':['us']*3})
    assert not promotion(f.head(0),f.filter(pl.col('y')==1),anchors)['passed']


def test_paired_alignment_independent_of_accepted_row_order():
    anchors=pl.DataFrame({'qid':[3,1,2], 'deg':[1,1,1], 'co':['us']*3})
    accepted=pl.DataFrame({'qid':[1,2,3], 'tid':[1,2,3], 'p':[1.,1.,1.], 'y':[1,0,1]})
    res=promotion(accepted,accepted.reverse(),anchors)
    assert res['paired_gain']==0 and res['paired_ci95']==[0,0]


@pytest.mark.parametrize('force_promotion,strength,competition',[(False,.5,False),(True,.5,False),(True,.65,False),(False,.5,True)])
def test_tuning_orchestrator_real_fit_calibration_export(tmp_path,monkeypatch,force_promotion,strength,competition):
    from test_latest_fusion_pipeline import test_complete_frozen_fusion
    from latest_fusion import tune_pipeline as pipeline
    from latest_fusion.pipeline import sha256


    test_complete_frozen_fusion(tmp_path,monkeypatch)
    incumbent=tmp_path/'result'
    report=json.loads((incumbent/'report.json').read_text())
    report['selected']='residual_0.5'


    assert report['methods']['residual_0.5']['tune']['rule']=='top1_threshold'
    (incumbent/'report.json').write_text(json.dumps(report))
    shutil.copyfile(tmp_path/'recipe.json',tmp_path/'latest_fusion_calibration.json')
    config={'incumbent_method':'residual_0.5', 'incumbent_matching_sha256':report['matching_sha256'],
            'incumbent_job':'fixture','incumbent_leaderboard_user_reported':.98805,
            'strengths':[strength], 'expected_f_floors':[.05,.8],
            'minimum_gain':-1 if force_promotion else .00001,
            'max_precision_loss':.00015,'max_recall_loss':.0001,
            'heads':[{'name':'small','rounds':3,'params':{'num_leaves':3,'min_data_in_leaf':3}}]}
    if strength == .65:
        config.update(feasible_policy_search=True, promotion_family_size=3, density_country_guards=True)
        config['heads'][0]['weight_mode']='owner_missing_address'
    if competition:
        config.update(training_protocol='target_group_competition', orphan_fraction=1.)
    path=tmp_path/'tuning.json';path.write_text(json.dumps(config))
    if force_promotion:
        monkeypatch.setattr(pipeline,'promotion',lambda *args,**kwargs:{'passed':True,'scope':'Fixture gate only; export branch exercise.'})
    output=tmp_path/'tuned'
    res=pipeline.run(tmp_path/'data',tmp_path/'prepared'/'prepared-train.parquet',
                        tmp_path/'prepared'/'prepared-test.parquet',incumbent,tmp_path/'metadata.json',path,output)
    assert (output/'small_model.txt').exists()
    assert res['validation']['source1']==240
    assert res['promoted']==force_promotion
    if competition:
        assert res['training_population']['protected_candidate_pairs']==0
        assert res['training_population']['positive']>0
        assert (output/'training_population.json').exists()
    if force_promotion:
        assert (output/'selected_calibration.json').exists()
        assert res['locked_candidate']['strength']==strength
        assert res['changes']['added_by_country'].get('france',0)==0
        assert res['changes']['removed_by_country'].get('france',0)==0
    else:
        assert sha256(output/'output'/'matching_results.tsv')==report['matching_sha256']


def test_policy_selects_lower_scoring_feasible_threshold():
    f=pl.DataFrame({'qid':[1,2], 'tid':[10,20], 'p':[.7,.9], 'raw':[.7,.9], 'y':[1,1], 'co':['us']*2})
    anchors=pl.DataFrame({'qid':[1,2], 'deg':[1,1], 'co':['us']*2})
    best,metric=select_policy(f,anchors,expected_f_floors=())
    chosen,guarded=select_policy(f,anchors,expected_f_floors=(),
                               feasibility=lambda accepted,anchors: accepted.height==1)
    assert apply_policy(f,best).height==2
    assert apply_policy(f,chosen).height==1
    assert guarded['macro_f05'] < metric['macro_f05']


def test_policy_no_feasible_option_returns_unconstrained_diagnostic():
    f=frame()
    anchors=pl.DataFrame({'qid':[1,2,3], 'deg':[1,1,0], 'co':['us']*3})
    best,metric=select_policy(f,anchors,expected_f_floors=())
    chosen,guarded=select_policy(f,anchors,expected_f_floors=(),feasibility=lambda accepted,anchors: False)
    assert chosen==best
    assert guarded==metric


def test_family_interval_can_reject_nominal_positive_gain():
    anchors=pl.DataFrame({'qid':list(range(10)), 'deg':[1]*10, 'co':['us']*10})
    accepted=pl.DataFrame({'qid':list(range(10)), 'tid':list(range(10)), 'p':[1.]*10, 'y':[1]*10})
    cand=accepted.head(8)
    incumbent=accepted.tail(2)
    nominal=promotion(cand,incumbent,anchors)
    adjusted=promotion(cand,incumbent,anchors,family_size=3)
    assert nominal['checks']['positive_lower_bound']
    assert not adjusted['checks']['positive_lower_bound']
    assert adjusted['paired_ci95']==nominal['paired_ci95']
    assert adjusted['paired_family_interval'][0] < adjusted['paired_ci95'][0]
    assert adjusted['family_size']==3
    assert adjusted['confidence_level']==pytest.approx(1-.05/3)


@pytest.mark.parametrize('family_size,alpha',[(0,.05),(True,.05),(1,0),(1,1)])
def test_invalid_family_parameters_rejected(family_size,alpha):
    anchors=pl.DataFrame({'qid':[1], 'deg':[0], 'co':['us']})
    empty=frame().head(0).select('qid','tid','p','y')
    with pytest.raises(ValueError):
        promotion(empty,empty,anchors,family_size=family_size,confidence_alpha=alpha)

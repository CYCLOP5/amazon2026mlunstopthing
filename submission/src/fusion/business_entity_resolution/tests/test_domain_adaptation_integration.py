import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import polars as pl
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from latest_fusion.tune_pipeline import prepare_domain_weights


def test_uniform_head_does_not_fit_or_read_target(tmp_path):
    weights, report = prepare_domain_weights([{'weight_mode':'uniform'}], pl.DataFrame({'y':[1]}), tmp_path/'missing.parquet',tmp_path)
    assert weights is report is None
    assert not (tmp_path/'domain_adaptation.json').exists()


def test_domain_fit_once_for_multiple_heads_preserves_source_order_and_writes_report(tmp_path, monkeypatch):
    frame=pl.DataFrame({'qid':[9,2,7], 'y':[1,0,1], 'own':[9,-1,7]})
    calls=[]
    def fit(source, test, output):
        calls.append((source,test,output))
        return np.array([.5,1.,1.5]), {'target_labels_used':False,'fixture':True}
    monkeypatch.setitem(sys.modules,'latest_fusion.domain_adaptation',SimpleNamespace(fit_weights=fit))
    weights,report=prepare_domain_weights([{'weight_mode':'uniform'},{'weight_mode':'domain_ratio'},{'weight_mode':'domain_ratio'}],frame,tmp_path/'test.parquet',tmp_path)
    assert len(calls)==1
    assert calls[0][0].equals(frame)
    assert calls[0][1]==tmp_path/'test.parquet'
    assert weights.tolist()==[.5,1.,1.5]
    assert json.loads((tmp_path/'domain_adaptation.json').read_text())==report


@pytest.mark.parametrize('bad', [[1.], [1.,0.,1.], [1.,np.nan,1.]])
def test_invalid_or_misaligned_weights_fail_closed(tmp_path, monkeypatch, bad):
    monkeypatch.setitem(sys.modules,'latest_fusion.domain_adaptation',SimpleNamespace(fit_weights=lambda *args:(bad,{})))
    with pytest.raises(ValueError,match='aligned'):
        prepare_domain_weights([{'weight_mode':'domain_ratio'}],pl.DataFrame({'x':[1,2,3]}),'test.parquet',tmp_path)


def test_bounded_config_preserves_original_protocol_and_identical_heads():
    cfg=json.loads((ROOT/'configs/domain_adaptation.json').read_text())
    uniform, weighted=cfg['heads']
    assert uniform['params']==weighted['params']
    assert uniform['rounds']==weighted['rounds']==300
    assert [uniform['weight_mode'],weighted['weight_mode']]==['uniform','domain_ratio']
    assert cfg['strengths']==[.5] and cfg['incumbent_strengths']==[]
    assert cfg['fixed_incumbent_policy'] and cfg['expected_f_floors']==[]
    assert 'training_protocol' not in cfg
    assert cfg['minimum_gain']==.0001
    assert cfg['max_precision_loss']==cfg['max_recall_loss']==0
    assert cfg['feasible_policy_search'] and cfg['density_country_guards'] and cfg['owner_country_guards']


def test_fixed_policy_prevents_decoder_search_and_legacy_path_remains(monkeypatch):
    from latest_fusion import tune_pipeline as pipeline
    top = pl.DataFrame({'qid':[1,2], 'tid':[10,20], 'p':[.9,.7], 'raw':[.9,.7], 'y':[1,0]})
    anchors = pl.DataFrame({'qid':[1,2], 'deg':[1,0], 'co':['us','india']})
    calls=[]
    def search(*args, **kwargs):
        calls.append(kwargs)
        return {'rule':'top1_threshold','threshold':.7}, {'fixture':True}
    monkeypatch.setattr(pipeline,'select_policy',search)
    incumbent={'rule':'top1_threshold','threshold':.8}
    policy,metrics=pipeline.candidate_policy({'fixed_incumbent_policy':True},top,anchors,incumbent)
    assert policy==incumbent and policy is not incumbent
    assert metrics['pairs']==1
    assert calls==[]
    legacy,_=pipeline.candidate_policy({'expected_f_floors':[.05],'feasible_policy_search':True},top,anchors,incumbent,feasibility=lambda *args:True)
    assert legacy['threshold']==.7 and len(calls)==1
    assert calls[0]['expected_f_floors']==(.05,) and callable(calls[0]['feasibility'])


def test_no_supported_shift_skips_both_heads_without_changing_legacy_configs():
    from latest_fusion.tune_pipeline import supported_domain_heads
    heads=[{'name':'uniform','weight_mode':'uniform'},{'name':'domain','weight_mode':'domain_ratio'}]
    sel,rejection=supported_domain_heads(heads,{'weighted_source_rows':0})
    assert sel==[] and 'unit-weight fallback' in rejection
    sel,rejection=supported_domain_heads(heads,{'weighted_source_rows':12})
    assert sel is heads and rejection is None
    sel,rejection=supported_domain_heads(heads,None)
    assert sel is heads and rejection is None

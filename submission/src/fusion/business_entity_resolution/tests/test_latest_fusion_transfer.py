'check transfer-label isolation and safe failure of promotion'
from pathlib import Path
import sys
import numpy as np
import polars as pl
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from latest_fusion.transfer import transferred_recipe,transfer_features,run_transfer

def test_target_labels_do_not_change_calibrator():
    n=180;q=np.arange(n,dtype=np.uint32)
    src=pl.DataFrame({'qid':q,'tid':q,'fold':[0]*n,'seg':[0]*n,'y':(q%2).astype(np.uint8)})
    refs=pl.DataFrame({'rid':q,'fold':[0]*n})
    target=src.with_columns(pl.lit(0).alias('y'))
    raw=np.linspace(.01,.99,n);edges=np.linspace(-4,4,25)
    a,scope=transferred_recipe(src,target,refs,refs,raw,raw,edges)
    b,_=transferred_recipe(src,target.with_columns(pl.lit(1).alias('y')),refs,refs,raw,raw,edges)
    assert a==b and 'target labels never' in scope['label_scope']

def test_two_direction_proxy_rejects_without_gains(tmp_path,monkeypatch):
    monkeypatch.setenv('ER_THREADS','2')
    n=240;q=np.arange(n,dtype=np.uint32)
    co=['us' if i%2 else 'india' for i in q]
    refs=pl.DataFrame({'rid':q,'co':co,'fold':[0]*n,'deg':[1]*n})
    tr=pl.DataFrame({'qid':q,'tid':q,'own':q.astype(np.int64),'co':co,'fold':[0]*n,'seg':[0]*n,'y':[1]*n,'gate':[.99]*n,'structural':[1.]*n})
    te=tr.drop('y','own','fold').with_columns(pl.lit('france').alias('co'))
    rt=refs.with_columns(pl.lit('france').alias('co'))
    fitids=q[::4];tuneids=q[1::4]
    sel,report=run_transfer(tr,te,refs,rt,['structural','newest_logit','np_m0_logit','neural_present','gate_neural_gap'],

        q[np.arange(n)%4<2],q[np.arange(n)%4>=2],{'objective':'binary','verbosity':-1,'num_leaves':3,'min_data_in_leaf':10},tmp_path)
    assert sel is None and report['promoted'] is False
    assert report['features']==['structural']
    assert len(report['directions'])==2
    assert len(list(tmp_path.glob('transfer-*.txt')))==2
    assert transfer_features(['gate_logit','member_spread','new_old_gap','graph_target_max'])==['gate_logit','graph_target_max']

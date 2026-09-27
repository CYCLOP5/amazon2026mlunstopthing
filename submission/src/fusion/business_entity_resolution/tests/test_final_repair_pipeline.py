'action conflict, owner isolation, and macro selection integration checks'
from pathlib import Path
import sys
import numpy as np
import polars as pl
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'scripts')]
from final_repair.pipeline import actions_for, apply_actions, fit_repair_risk, select_repairs


def refs(qids):
    return pl.DataFrame({'rid': qids, 'co': ['us']*len(qids)})


def proposals(qids, tids):
    return pl.DataFrame({'qid': qids, 'tid': tids, 'rule': ['full_identity']*len(qids), 'rank': [1]*len(qids)})


def test_action_conflicts_and_confident_owner_protection():
    p = proposals([0,1,0,0,0], [0,0,1,2,3])
    base = pl.DataFrame({'qid':[1,1,0], 'tid':[1,2,3], 'p':[.995,.8,.6]})
    a = actions_for(p, base, refs([0,1]))
    assert a['tid'].to_list() == [2]
    assert a['action'].to_list() == ['replace']
    res = apply_actions(base,a).sort('tid')
    assert res['qid'].to_list() == [1,0,0]
    assert res['tid'].n_unique() == res.height


def test_fit_excludes_proposed_true_and_old_heldout_owners():

    actions = pl.DataFrame({'qid':[0,2,0,0], 'tid':[0,1,2,3], 'old_qid':[None,None,None,2],
        'co':['us']*4, 'cell':['full_identity|add']*4, 'action':['add']*4})
    owners = pl.DataFrame({'tid':[0,1,2,3], 'own':[0,2,2,0]})
    risk = fit_repair_risk(actions,owners,refs([0,1,2]),pl.DataFrame({'qid':[0,1]}))
    row = risk['us|full_identity|add']
    assert row['n'] == 1 and row['correct'] == 1 and not row['eligible']


def test_selector_uses_macro_gain_and_does_not_fit_on_tune_truth():
    q = np.arange(121)
    r = refs(q)
    owner = pl.DataFrame({'tid':q,'own':q})
    base = pl.DataFrame(schema={'qid':pl.Int64,'tid':pl.Int64,'p':pl.Float64})
    action = actions_for(proposals(q,q),base,r)
    fit = pl.DataFrame({'qid':q[:120]})
    tune = pl.DataFrame({'qid':[120], 'deg':[1], 'co':['us']})
    policies, transferable, report = select_repairs(base,action,owner,r,fit,tune)
    assert policies == {'us':['full_identity|add']} and transferable == []
    assert report['fit_action_risk']['us|full_identity|add']['n'] == 120
    corrupted = owner.with_columns(pl.when(pl.col('tid') == 120).then(-1).otherwise(pl.col('own')).alias('own'))
    rejected,_, second = select_repairs(base,action,corrupted,r,fit,tune)
    assert rejected == {'us':[]}
    assert second['fit_action_risk'] == report['fit_action_risk']


def test_full_pipeline_tiny_parquet_and_empty_proposals(tmp_path, monkeypatch):
    import json
    from final_repair import pipeline
    roots = {k:tmp_path/k for k in ('data','hybrid','friend','graph','out')}
    for folder in roots.values():
        folder.mkdir()
    for split,n in [('train',120),('test',9)]:
        folder = roots['data']/split
        folder.mkdir()
        co = ['us' if i%2 else 'india' for i in range(n)]
        if split == 'test':
            co[-1]='france'
        r=pl.DataFrame({'rid':np.arange(n,dtype=np.uint32),'eid':[f'S1_{split}_{i}' for i in range(n)],
            'nm':['Reference distinct '+str(i) for i in range(n)],'ad':['12 River Street']*n,'co':co})
        if split=='train':
            r=r.with_columns(pl.Series('fold',[(i//2)%2 for i in range(n)]),pl.lit(2).alias('deg'))
        r.write_parquet(folder/'ref.parquet')
        for sr in (2,3):
            t=pl.DataFrame({'rid':np.arange((sr-2)*n,(sr-1)*n,dtype=np.uint32),
                'eid':[f'S{sr}_{split}_{i}' for i in range(n)],'nm':['Unrelated target '+str(i) for i in range(n)],
                'ad':['98 Other Avenue']*n,'co':co})
            if split=='train':
                t=t.with_columns(pl.Series('own',np.arange(n,dtype=np.int64)))
            t.write_parquet(folder/f's{sr}.parquet')
        scores=pl.DataFrame({'qid':np.tile(np.arange(n,dtype=np.uint32),2),
            'tid':np.arange(2*n,dtype=np.uint32),'p':np.full(2*n,.9999,dtype=np.float32)})
        scores.rename({'p':'p2'}).write_parquet(roots['friend']/('val_pred.parquet' if split=='train' else 'test_pred.parquet'))
        scores.rename({'p':'head'}).write_parquet(roots['graph']/('validation_predictions.parquet' if split=='train' else 'test_predictions.parquet'))
        scores.write_parquet(roots['hybrid']/('validation_predictions.parquet' if split=='train' else 'test_predictions.parquet'))
    (roots['friend']/'report.json').write_text(json.dumps({'selected':'raw','methods':{'raw':{'score':'p2','tune':{'rule':'expected_f','threshold':.5}}}}))
    (roots['hybrid']/'report.json').write_text(json.dumps({'decision':{'rule':'expected_f','threshold':.3}}))
    monkeypatch.setattr(pipeline,'EXPECTED_RERANKER_MATCHES',18)
    report=pipeline.run(roots['data'],roots['hybrid'],roots['friend'],roots['graph'],roots['out'])
    assert (roots['out']/'_SUCCESS').read_text() == 'complete\n'
    assert report['validation']['matches']==18
    assert report['test']['identity_candidates']['proposals']==0
    assert (roots['out']/'output'/'matching_results.tsv').is_file()
    assert (roots['out']/'output'/'candidate_pairs.tsv').is_file()
    assert (roots['out']/'report.json').is_file()


def test_transfer_bundle_rejects_one_negative_country():
    from final_repair.pipeline import gate_transfer_bundle
    base=pl.DataFrame({'qid':[0,1], 'tid':[0,1], 'p':[.9,.9]})
    owners=pl.DataFrame({'tid':[0,1,2], 'own':[0,1,0]})

    a=pl.DataFrame({'qid':[0,0], 'tid':[2,1], 'cell':['full_identity|add']*2})
    tune=pl.DataFrame({'qid':[0,1], 'deg':[2,1], 'co':['us','india']})
    sel,evidence=gate_transfer_bundle(['full_identity|add'],{'reference':base},{'reference':a},owners,tune)
    assert sel==[] and not evidence['passed']
    assert evidence['comparisons']['reference']['india']['macro_gain']<0

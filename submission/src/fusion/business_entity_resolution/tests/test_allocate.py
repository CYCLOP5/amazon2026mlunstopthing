'behavioral checks for fit-only count priors and competing-owner moves'
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location('allocate',Path(__file__).parents[1]/'innovation'/'allocate.py')
allocation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(allocation)


def pairs(rows):
    return pl.DataFrame(rows,schema=['qid','tid','p'],orient='row').with_columns(pl.col('qid','tid').cast(pl.UInt32))


def prior():

    return {'maximum':3,'tail_ratio':.5,'probabilities':[[.05,.9,.04,.01],[.05,.9,.04,.01]]}


def test_joint_move_can_choose_second_owner_and_preserves_confident_edge():
    d = pairs([(0,0,.999),(1,0,.001),(0,1,.60),(1,1,.55)])
    accepted = d.filter(((pl.col('qid') == 0) & (pl.col('tid') == 0)) |
                        ((pl.col('qid') == 0) & (pl.col('tid') == 1)))
    res,info = allocation.allocate(d,accepted,prior(),2,1.)
    assert set(res.select('qid','tid').iter_rows()) == {(0,0),(1,1)}
    assert info['moves'] == 1
    assert res['tid'].n_unique() == res.height


def test_zero_strength_is_exact_parent_fallback():
    d = pairs([(0,0,.4),(1,0,.35)])
    accepted = d.head(1)
    res,info = allocation.allocate(d,accepted,prior(),1,0.)
    assert res.equals(accepted)
    assert info['fallback']


def test_prior_ignores_tuning_audit_and_decoy_labels():
    refs = pl.DataFrame({'rid':range(100),'fold':[0]*50+[1]*50}).with_columns(pl.col('rid').cast(pl.UInt32))
    truth = pl.DataFrame({'own':range(100),'sr':[2]*100})
    first = allocation.learn_prior(refs,truth)
    held = refs.filter((pl.col('fold') != 0) | ((pl.col('rid').hash(1033)%5) == 0))['rid'].to_list()
    altered = pl.concat([truth,pl.DataFrame({'own':held*20+[-1]*30,'sr':[2]*(len(held)*20+30)})])
    assert allocation.learn_prior(refs,altered) == first
    assert np.isfinite(allocation.log_count(first,1000,0))
    assert allocation.log_count(first,1000,0) < allocation.log_count(first,24,0)


def test_null_option_and_source_counts_are_separate():
    d = pairs([(0,0,.8),(0,1,.55),(0,2,.55)])
    res,_ = allocation.allocate(d,d,prior(),2,1.)


    assert set(res.select('qid','tid').iter_rows()) == {(0,0),(0,2)}


def test_candidate_and_target_limits_freeze_skipped_assignments():
    d = pairs([(0,0,.6),(1,0,.5),(2,0,.4),(0,1,.7)])
    accepted = d.filter(pl.col('qid') == 0)
    problem = allocation.prepare_problem(d,accepted,max_candidates=2,max_targets=1)
    assert problem[0]['tid'].unique().to_list() == [1]
    res,_ = allocation.allocate(d,accepted,prior(),2,1.,problem)
    assert (0,0) in set(res.select('qid','tid').iter_rows())
    assert not res.join(d,on=['qid','tid'],how='anti').height


def test_component_cap_freezes_entire_large_chain():
    d = pairs([(0,0,.6),(1,0,.5),(1,1,.6),(2,1,.5)])
    accepted = d.filter(pl.col('p') == .6)
    problem = allocation.prepare_problem(d,accepted,max_component=1)
    assert problem[0].height == 0
    res,_ = allocation.allocate(d,accepted,prior(),2,1.,problem)
    assert res.equals(accepted)


def test_run_replays_parent_and_exports_full_pool(tmp_path):
    from test_stack import make_fixture
    sys.path.insert(0,str(Path(__file__).parents[1]))
    data,roots = make_fixture(tmp_path,160)
    parent = tmp_path/'refinement'
    parent.mkdir()
    for split in ('train','test'):
        d = pl.read_parquet(Path(roots[split][0])/'parts'/'part_0.parquet')
        d = d.select('qid','tid',pl.col('prob').alias('base'),pl.col('prob').alias('head'),
                     *(['y'] if split == 'train' else []))
        d.write_parquet(parent/('validation_predictions.parquet' if split == 'train' else 'test_predictions.parquet'))
    (parent/'report.json').write_text(json.dumps({'selected_head_weight':1.,
        'selected_decoder':{'rule':'top1_threshold','threshold':.5}}))
    report = allocation.run(data,parent,tmp_path/'allocation')
    assert report['selected_strength'] == 0.
    assert report['comparison']['paired_delta'] == 0.
    output = tmp_path/'allocation'/'output'
    matches = pl.read_csv(output/'matching_results.tsv',separator='\t')
    cands = pl.read_csv(output/'candidate_pairs.tsv',separator='\t')
    assert matches.height == cands.height == 32
    assert report['export']['candidate_pairs'] == 32*2*3
    assert report['export']['matches'] == 32*2

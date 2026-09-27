import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from er.decision import best_k
from er.stack import decode
from er.stack.enhanced import consensus_features, occupancy_features
from er.stack.inputs import load_pairs
from er.stack.pipeline import run, tuning_anchors, export


def test_expected_f_matches_enumeration():
    p = np.array([.91, .66, .23, .05])
    values = []
    for k in range(len(p)+1):
        total = 0
        for y in itertools.product([0, 1], repeat=len(p)):
            y = np.array(y)
            prob = np.prod(np.where(y, p, 1-p))
            f = float(k == 0) if not y.sum() else 1.25*y[:k].sum()/max(k+.25*y.sum(), 1e-9)
            total += prob*f
        values.append(total)
    assert best_k(p) == int(np.argmax(values))


def test_abstention_threshold_replays_even_at_probability_one():
    pairs = pl.DataFrame({'qid':[0], 'tid':[0], 'p':[1.0], 'y':[0]})
    anchors = pl.DataFrame({'qid':[0], 'deg':[0]})
    rule = decode.curve(pairs, anchors)
    accepted = decode.apply('top1_threshold', pairs, rule['threshold'])
    assert accepted.height == 0
    assert decode.score(accepted, anchors)['macro_f05'] == 1


def test_consensus_excludes_self_and_canonicalizes_numbers():
    pairs = pl.DataFrame({'qid':[0,0,0], 'tid':[0,1,2], 'is_s3':[0,1,1]})
    conf = pl.DataFrame({'qid':[0,0], 'sid':[0,1], 's_is_s3':[0,1], 's_hn_exact':[1,1], 's_lg':[8.,8.]})
    attrs = pl.DataFrame({'rid':[0,1,2], 'name_core':['acme']*3, 'addr_can':['a','a',''], 'hn1':['00513','513','']})
    out = consensus_features(pairs, conf, attrs).sort('tid')
    assert out['sib_numeric_known'].to_list() == [1,1,0]
    assert out['sib_numeric_agree'].to_list() == [1,1,0]
    assert out['sib_numeric_agree_cross'].to_list() == [1,1,0]
    assert out['sib_addr_exact'].to_list() == [1,1,0]


def test_occupancy_counts_other_records_without_source_capacity_rule():
    df = pl.DataFrame({'qid':[0,0,0,1], 'tid':[0,1,2,0], 'is_s3':[0,0,1,0], 'p1':[.99,.95,.9,.01]})
    out = occupancy_features(df).sort('qid','tid')
    assert out['sib_others_p90_all'].to_list() == [2,2,2,0]
    assert out['sib_others_p90_source'].to_list() == [1,1,0,0]
    assert out['sib_soft_mass_source'][0] == pytest.approx(.95)


def test_export_keeps_all_inference_candidates_and_all_entities(tmp_path):
    df = pl.DataFrame({'qid':[0,0], 'tid':[0,1], 'p2':[.99,.00001], 'in_neural':[1,0]})
    refs = pl.DataFrame({'rid':[0,1], 'eid':['S1_a','S1_b']})
    tg = pl.DataFrame({'rid':[0,1], 'eid':['S2_a','S3_a']})
    export(df, 'p2', 'top1_threshold', .5, refs, tg, str(tmp_path))
    assert (tmp_path/'candidate_pairs.tsv').read_text().splitlines() == [
        'source1_entity_id\tcandidate_entity_ids', 'S1_a\tS2_a,S3_a', 'S1_b\t']
    assert (tmp_path/'matching_results.tsv').read_text().splitlines()[-1] == 'S1_b\t'


def make_fixture(root, n=160):
    data = root/'data'
    roots = {}
    for split, count in [('train',n), ('test',32)]:
        folder = data/split
        folder.mkdir(parents=True)

        ref = pl.DataFrame({'rid':np.arange(count,dtype=np.uint32),
            'eid':[f'S1_{split}_{i}' for i in range(count)],
            'nm':[f'Acme business {i}' for i in range(count)],
            'ad':[f'{100+i} River road' for i in range(count)], 'co':['us']*count})
        if split == 'train':
            ref = ref.with_columns(pl.Series('fold',[i%2 for i in range(count)]), pl.lit(2).alias('deg'))
        ref.write_parquet(folder/'ref.parquet')
        for sr in (2,3):
            target = ref.select('nm','ad','co').with_columns(
                pl.Series('rid',np.arange((sr-2)*count,(sr-1)*count,dtype=np.uint32)),
                pl.Series('eid',[f'S{sr}_{split}_{i}' for i in range(count)]))
            if split == 'train':
                target = target.with_columns(pl.Series('own',list(range(count))))
            target.write_parquet(folder/f's{sr}.parquet')
        rows = []
        for tid in range(count*2):
            for qid in (tid%count, (tid+1)%count, (tid+3)%count):
                y = int(qid == tid%count)
                p = (.65 + (tid%7)*.04) if y else (.03 + (tid%7)*.06)
                row = {'qid':qid, 'tid':tid, 'prob':p, 'gate_prob':p, 'neural_prob':p}
                if split == 'train': row['y'] = y
                rows.append(row)
        score = root/f'{split}_scores'
        (score/'parts').mkdir(parents=True)
        pl.DataFrame(rows).write_parquet(score/'parts/part_0.parquet')
        np.save(score/'parts/part_0.npy',np.arange(count*2))
        (score/'manifest.json').write_text(json.dumps({'config_sha256':'fixture', 'parts':[
            {'name':'parts/part_0.parquet', 'coverage':'parts/part_0.npy', 'pairs':len(rows)}]}))
        roots[split] = [str(score)]
    return data, roots


def fixture_config(data, roots):
    return {'stack': {'data':str(data), 'train_roots':roots['train'], 'test_roots':roots['test'],
        'require_complete':True, 'eval_folds':[0,1], 'fit_fold':0, 'audit_fold':1, 'cv':2, 'seed':42,
        'text_groups':['name','address','numbers'], 'chunk_rows':100, 'confident_p':.5,
        'lexical_train':[], 'lexical_test':[], 'lexical_top_k':5, 'lexical_min_rel':.6,
        'rules':False, 'analysis':True, 'enhanced_features':True, 'tune_holdout_buckets':5,
        'exact_rescue':True, 'exact_max_owners':8,
        'fit':{'num_boost_round':8,'early_stopping_rounds':3},
        'lgbm':{'num_leaves':7,'min_data_in_leaf':3,'learning_rate':.2,'seed':7}}}


@pytest.mark.parametrize("variant", ["control", "v4", "recall"])
def test_training_bundle_inference_and_heldout_label_isolation(tmp_path, variant):
    data, roots = make_fixture(tmp_path)
    cfg = fixture_config(data, roots)
    cfg["stack"]["enhanced_features"] = variant != "control"
    cfg["stack"]["exact_rescue"] = variant == "recall"
    path = SimpleNamespace(exp=str(tmp_path/'fit'),exp_name='fixture',output=str(tmp_path/'fit/output'))
    run(cfg,path)
    bundle = tmp_path/'fit/stack/bundle'
    assert (bundle/'round1/fold_0.txt').exists()
    assert (bundle/'round2/fold_0.txt').exists()
    original = pl.read_parquet(tmp_path/'fit/stack/val_pred.parquet')
    cfg['stack']['inference_bundle'] = str(bundle)

    cfg['stack']['train_roots'] = ['/does/not/exist']
    replay = SimpleNamespace(exp=str(tmp_path/'replay'),exp_name='replay',output=str(tmp_path/'replay/output'))
    run(cfg,replay)
    for name in ('matching_results.tsv','candidate_pairs.tsv'):
        assert (Path(path.output)/name).read_bytes() == (Path(replay.output)/name).read_bytes()
    import runpy
    validate = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/validate_outputs.py'))['validate']
    assert validate(data,Path(replay.output))['source1'] == 32

    cfg['stack']['inference_bundle'] = ''
    cfg['stack']['train_roots'] = roots['train']
    anchors = pl.read_parquet(data/'train/ref.parquet').rename({'rid':'qid'})
    tune = set(tuning_anchors(anchors,cfg['stack'])['qid'].to_list())
    audit = set(anchors.filter(pl.col('fold') == 1)['qid'].to_list())


    cached_work = tmp_path/'fit/stack'
    scores_path = cached_work/'r1_train.parquet'
    scores = pl.read_parquet(scores_path).with_columns(
        pl.when(pl.col('qid').is_in(list(tune|audit))).then(1-pl.col('y')).otherwise(pl.col('y')).alias('y'))
    scores.write_parquet(scores_path)
    cfg["stack"]["work"] = str(cached_work)
    changed = SimpleNamespace(exp=str(tmp_path/'changed'),exp_name='changed',output=str(tmp_path/'changed/output'))
    run(cfg,changed)
    second = pl.read_parquet(tmp_path/'changed/stack/val_pred.parquet')
    assert not original.sort('qid','tid')['y'].equals(second.sort('qid','tid')['y'])
    a,b = [x.sort('qid','tid').select('p1','p2').to_numpy() for x in (original,second)]
    np.testing.assert_array_equal(a,b)


def test_bad_coverage_is_rejected(tmp_path):
    _, roots = make_fixture(tmp_path,16)
    cov = Path(roots['train'][0])/'parts/part_0.npy'
    np.save(cov,np.array([-1,0,1]))
    with pytest.raises(ValueError,match='outside'):
        load_pairs(roots['train'],32)


def test_exact_retrieval_no_empty_common_or_cross_country_keys():
    from er.stack.recall import add_exact_candidates
    refs = pl.DataFrame({'rid':[0,1,2,3], 'co':['us','us','us','france'],
        'addr_can':['100 main','200 main','300 main','100 main'], 'name_core':['chain','chain','solo','solo']})
    targets = pl.DataFrame({'rid':[0,1,2,3], 'co':['us']*4,
        'addr_can':['100 main','','',''], 'name_core':['invented','solo','chain','']})
    pairs = pl.DataFrame({'qid':[1], 'tid':[0], 'prob':[.1], 'neural_prob':[.1], 'gate_prob':[.1]})
    out, info = add_exact_candidates(pairs, refs, targets, max_owners=1)
    assert set(out.select('qid','tid').iter_rows()) == {(1,0),(0,0),(2,1)}
    assert info['exact_new_pairs'] == 2
    assert out.filter(pl.col('qid')==0)['in_neural'][0] == 0

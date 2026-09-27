'real artifact replay, fitted heads, frozen inference and submission validation'
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from er.stack import decode
from er.stack.inputs import load_refs, load_targets
from er.stack.pipeline import export
from latest_fusion.pipeline import sha256
from latest_fusion.structured_pipeline import CORE, METHOD, assert_same_pairs, assert_same_pool, replay, run
from latest_fusion.tuning import fit_population


def make_fixture(root, n=1200):
    data, prepared, inc = root/'data', root/'prepared', root/'incumbent'
    prepared.mkdir(parents=True); inc.mkdir()
    q = np.arange(n, dtype=np.uint32)
    for split in ('train', 'test'):
        folder = data/split; folder.mkdir(parents=True)
        countries = ['us' if i % 4 < 2 else 'india' for i in q]
        if split == 'test': countries[-24:] = ['france']*24
        ref = pl.DataFrame({'rid': q, 'eid': [f'{split}-r{i}' for i in q],
            'co': countries, 'nm': [f'company group{i//2} club' for i in q],
            'ad': [f'{i%2+24} lake road' for i in q],
            'fold': [0 if i < 240 else 1 for i in q], 'deg': [1]*n})
        ref.write_parquet(folder/'ref.parquet')
        tg = pl.DataFrame({'rid': np.arange(2*n, dtype=np.uint32),
            'eid': [f'{split}-t{i}' for i in range(2*n)], 'co': countries*2,
            'nm': ref['nm'].to_list()+[f'company group{i//2} committee' for i in q],
            'ad': ref['ad'].to_list()*2, 'own': np.r_[q.astype(np.int64), np.full(n,-1)]})
        tg[:n].write_parquet(folder/'s2.parquet'); tg[n:].write_parquet(folder/'s3.parquet')
        ids = np.r_[q, q^1, q]
        newest = np.r_[np.full(n,.9999), np.full(n,.001), np.full(n,.001)].astype(np.float32)
        frame = pl.DataFrame({'qid': ids, 'tid': np.r_[q,q,q+n].astype(np.uint32),
            'co': [countries[i] for i in ids], 'newest': newest, 'gate': newest,
            'seg': np.zeros(3*n,dtype=np.uint8),
            'fold': [0 if i<240 else 1 for i in ids],
            'own': np.r_[q.astype(np.int64),q.astype(np.int64),np.full(n,-1)],
            'y': np.r_[np.ones(n,dtype=np.uint8), np.zeros(2*n,dtype=np.uint8)]})
        frame = frame.with_columns(*[pl.lit(0., dtype=pl.Float32).alias(c) for c in CORE if c!='baseline_p'])
        frame = frame.with_columns((pl.col('newest')/(1-pl.col('newest'))).log().alias('newest_logit'))
        if split=='test': frame=frame.drop('own','y','fold')
        frame.write_parquet(prepared/f'prepared-{split}.parquet')
        (prepared/f'features-{split}.json').write_text('["newest_logit"]')
    tr = pl.read_parquet(prepared/'prepared-train.parquet')
    model = lgb.train({'objective':'binary', 'num_leaves':3, 'verbosity':-1, 'num_threads':2},
                      lgb.Dataset(tr.select('newest_logit').to_numpy(),label=tr['y'].to_numpy()),num_boost_round=3)
    model.save_model(str(inc/'residual_model.txt'))
    recipe = {'curves': {f'{co}|{s}': {'centres':[-12,0,12],'posterior':[0,.5,1]}
                        for co in ('us','india','france') for s in range(3)}}
    prev = {'selected':METHOD, 'methods':{METHOD:{'tune':{'threshold':.7842838168144226}}},
                'calibration':{'methods':{METHOD:{'train':recipe,'test':recipe}}},
                'coverage':{'train':{'union_pairs':3*n}, 'test':{'union_pairs':3*n}}}
    refs = load_refs(str(data),'train')
    tr.select('qid','tid','y').write_parquet(inc/'validation_predictions.parquet')
    _, _, accepted = replay(tr, model, ['newest_logit'], prev, 'train')
    _, ids = fit_population(refs)
    anchors = refs.filter(pl.col('rid').is_in(pl.Series(ids).implode())).select(pl.col('rid').alias('qid'),'deg')
    prev['methods'][METHOD]['tune'].update(decode.score(accepted,anchors))
    te = pl.read_parquet(prepared/'prepared-test.parquet')
    te.select('qid','tid').write_parquet(inc/'test_predictions.parquet')
    te, _, accepted = replay(te,model,['newest_logit'],prev,'test')
    accepted.write_parquet(inc/'accepted_test.parquet')
    export(te,'baseline_p','fixture',0.,load_refs(str(data),'test'),load_targets(str(data),'test'),str(inc/'output'),accepted)
    prev['matching_sha256'] = sha256(inc/'output'/'matching_results.tsv')
    (inc/'report.json').write_text(json.dumps(prev))
    return data, prepared, inc, prev['matching_sha256']


@pytest.mark.parametrize('mode',['contrastive','set_utility','directional'])
def test_complete_experiment_exact_fallback(tmp_path, monkeypatch, mode):
    monkeypatch.setenv('ER_THREADS','2')
    data, prepared, inc, checksum = make_fixture(tmp_path)
    report = run(data,prepared/'prepared-train.parquet',prepared/'prepared-test.parquet',inc,
                 tmp_path/'result',mode,expected_sha=checksum)
    assert report['validation']['source1']==1200
    assert report['validation']['candidate_pairs']==3600
    assert report['protocol']['target_overlap']==0
    assert report['protocol']['calibration_check_target_overlap']==0
    assert report['matching_sha256']==checksum
    assert report['changed'] is False
    assert report['baseline_original_tune']['macro_f05']==1.


def test_pair_guard_detects_owner_change_and_duplicates():
    frame=pl.DataFrame({'qid':[1,2],'tid':[7,8]})
    assert_same_pairs(frame.reverse(),frame,'fixture')
    with pytest.raises(ValueError,match='differs'):
        assert_same_pairs(pl.DataFrame({'qid':[1,3],'tid':[7,8]}),frame,'fixture')
    with pytest.raises(ValueError,match='multiple owners'):
        same=pl.DataFrame({'qid':[1,2],'tid':[7,7]})
        assert_same_pairs(same,same,'fixture')


def test_candidate_guard_detects_changed_rejected_pair(tmp_path):
    frame = pl.DataFrame({'qid':[1,2,3], 'tid':[7,7,8]})
    path = tmp_path/'scores.parquet'; frame.write_parquet(path)
    assert_same_pool(frame.reverse(),path)
    with pytest.raises(ValueError,match='Full candidate pool'):
        assert_same_pool(pl.DataFrame({'qid':[1,4,3], 'tid':[7,7,8]}),path)


def test_saved_heads_replay_and_protocol_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv('ER_THREADS','2')
    data, prepared, inc, checksum = make_fixture(tmp_path)
    sources = {}
    for family in ('contrastive','directional'):
        sources[family] = tmp_path/family
        run(data,prepared/'prepared-train.parquet',prepared/'prepared-test.parquet',inc,
            sources[family],family,expected_sha=checksum)
    report = run(data,prepared/'prepared-train.parquet',prepared/'prepared-test.parquet',inc,
                 tmp_path/'replayed','additions',expected_sha=checksum,pretrained_heads=sources)
    assert report['changed'] is False
    assert report['matching_sha256']==checksum
    assert all(not s['retrained'] for s in report['experiment']['sources'].values())
    source_report = sources['contrastive']/'report.json'
    invalid = json.loads(source_report.read_text()); invalid['protocol']['seed'] += 1
    source_report.write_text(json.dumps(invalid))
    with pytest.raises(ValueError,match='pretrained head benchmark/protocol differs'):
        run(data,prepared/'prepared-train.parquet',prepared/'prepared-test.parquet',inc,
            tmp_path/'rejected_replay','additions',expected_sha=checksum,pretrained_heads=sources)


def test_changed_actions_use_real_export_and_validate(tmp_path, monkeypatch):
    from latest_fusion import contrastive_experiment as module

    monkeypatch.setenv('ER_THREADS', '2')
    data, prepared, inc, checksum = make_fixture(tmp_path)


    monkeypatch.setattr(module, 'run', lambda *args, **kwargs: {'report': {'fixture_action': 'remove_us_pair'}})

    def remove_one(bundle, frame, refs, targets, baseline, **kwargs):
        us = refs.filter(pl.col('co') == 'us').select(pl.col('rid').alias('qid'))
        removed = baseline.join(us, on='qid', how='semi').head(1).select('qid', 'tid')
        assert removed.height == 1
        return baseline.join(removed, on=['qid', 'tid'], how='anti'), {'removed_pairs': 1}

    monkeypatch.setattr(module, 'infer', remove_one)
    output = tmp_path / 'result'
    report = run(data, prepared/'prepared-train.parquet', prepared/'prepared-test.parquet',
                 inc, output, 'contrastive', expected_sha=checksum)
    assert report['changed'] is True
    assert report['changes']['removed_pairs'] == 1
    assert report['changes']['added_pairs'] == 0
    assert report['changes']['removed_by_country'] == {'us': 1}
    assert report['matching_sha256'] != checksum
    assert report['submission_status'] == 'candidate_requires_review'
    assert report['validation'] == {'source1': 1200, 'matches': 1199, 'candidate_pairs': 3600}
    assert (output/'output'/'export.json').exists()
    accepted = pl.read_parquet(output/'accepted_test.parquet')
    base = pl.read_parquet(inc/'accepted_test.parquet').select('qid', 'tid', 'p')
    refs = load_refs(str(data), 'test')
    france = refs.filter(pl.col('co') == 'france').select(pl.col('rid').alias('qid'))
    assert accepted.join(france, on='qid', how='semi').sort('qid').equals(
        base.join(france, on='qid', how='semi').sort('qid'))
    pool = pl.read_parquet(prepared/'prepared-test.parquet', columns=['qid', 'tid'])
    assert accepted.select('qid', 'tid').join(pool, on=['qid', 'tid'], how='anti').height == 0


def test_us_india_mode_rejects_french_action_before_export(tmp_path, monkeypatch):
    from latest_fusion import contrastive_experiment as module

    monkeypatch.setenv('ER_THREADS', '2')
    data, prepared, inc, checksum = make_fixture(tmp_path)
    monkeypatch.setattr(module, 'run', lambda *args, **kwargs: {'report': {'fixture_action': 'invalid_france_removal'}})

    def remove_france(bundle, frame, refs, targets, baseline, **kwargs):
        france = refs.filter(pl.col('co') == 'france').select(pl.col('rid').alias('qid'))
        removed = baseline.join(france, on='qid', how='semi').head(1).select('qid', 'tid')
        assert removed.height == 1
        return baseline.join(removed, on=['qid', 'tid'], how='anti'), {'removed_pairs': 1}

    monkeypatch.setattr(module, 'infer', remove_france)
    output = tmp_path / 'result'
    with pytest.raises(ValueError, match='changed a French decision'):
        run(data, prepared/'prepared-train.parquet', prepared/'prepared-test.parquet',
            inc, output, 'contrastive', expected_sha=checksum)
    assert not (output/'output'/'matching_results.tsv').exists()

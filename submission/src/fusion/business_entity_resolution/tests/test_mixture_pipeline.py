from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import polars as pl
import pytest
from latest_fusion import mixture_pipeline as m


def fixture():
    winners = pl.DataFrame({'qid': [1,2,3,4,5], 'tid': [10,11,12,13,14],
        'co': ['us','us','india','india','france'], 'p': [.9,.8,.9,.8,.9],
        'raw': [.9,.8,.9,.8,.9], 'y': [1,0,1,0,1]})
    anchors = pl.DataFrame({'qid': [1,2,3,4], 'co': ['us','us','india','india'], 'deg': [1,0,1,0]})
    return winners, anchors


def test_metadata_joins_after_global_ownership_and_derives_source():
    w,_ = fixture()
    extra = w.select('qid','tid').with_columns(pl.lit('owned').alias('seg'), pl.lit(0).alias('target_address_empty'))
    res = m.winner_metadata(w, extra, 12)
    assert res.select('qid','tid').equals(w.select('qid','tid'))
    assert res['is_s3'].to_list() == [0,0,1,1,1]


def test_source_is_only_fold_zero_partition_one_and_unlabeled_projection():
    ids = np.arange(100, dtype=np.uint32)
    w = pl.DataFrame({'qid': ids, 'fold': [0]*50+[1]*50, 'y': [1]*100, 'own': ids, 'co': ['us']*100})
    src = m.source_population(w)
    assert src.height > 0
    assert (src['fold'].to_numpy() == 0).all()
    assert (m.partition(src['qid'].to_numpy()) == 1).all()
    assert not {'y','own','fold'} & set(m.unlabeled(w).columns)


def test_threshold_changes_acceptance_but_freezes_owners_and_france():
    w,_ = fixture()
    base = w.select('qid','tid','p','y')
    changed = m.frozen_accept(m.blend(w, [1,.01,1,.01,.01], 1), {'threshold': .5}, base)
    assert changed['tid'].to_list() == [10,12,14]
    assert changed.filter(pl.col('tid') == 14)['p'][0] == base.filter(pl.col('tid') == 14)['p'][0]
    assert changed.select('qid','tid').join(w.select('qid','tid'), on=['qid','tid'], how='anti').is_empty()


def test_guard_requires_precision_recall_and_country_metrics():
    w,a = fixture()
    base = w.select('qid','tid','p','y')
    better = base.filter(pl.col('y') == 1)
    assert m.guard(better, base, a, 1e-5, True)['passed']

    bad = base.filter(pl.col('tid').is_in([10,13,14]))
    res = m.guard(bad, base, a, 0, True)
    assert not res['passed']
    assert not res['checks']['india_pair_recall']


def test_search_locks_one_policy_without_check_access():
    w,a = fixture()
    base = w.select('qid','tid','p','y')
    locked, cands = m.lock_policy(w, np.array([.99,.01,.99,.01,.9]), base, a)
    assert locked is not None
    assert len(cands) == 3
    original_policy = dict(locked['policy'])
    check = a.with_columns(pl.lit(2).alias('deg'))
    accepted = m.frozen_accept(m.blend(w, [.99,.01,.99,.01,.9], locked['policy']['weight']), locked['policy'], base)
    m.guard(accepted, base, check, 1e-5, True)
    assert locked['policy'] == original_policy


def test_rejects_invalid_posterior():
    w,_ = fixture()
    with pytest.raises(ValueError, match='posterior'):
        m.blend(w, [np.nan]*5, .5)


def test_output_cannot_overwrite_inputs_or_existing_artifacts(tmp_path):
    src = tmp_path/'source'
    src.mkdir()
    for output in (src, src/'nested', tmp_path):
        with pytest.raises(ValueError, match='overlaps'):
            m.prepare_output(output, [src])
    output = tmp_path/'output'
    m.prepare_output(output, [src])
    artifact = output/'report.json'
    artifact.write_text('preserved')
    with pytest.raises(ValueError, match='empty'):
        m.prepare_output(output, [src])
    assert artifact.read_text() == 'preserved'


def test_exact_search_retains_feasible_cut_when_unconstrained_optimum_loses_country_precision():



    rows = pl.DataFrame({'qid': [1,2,3,4,5,6], 'tid': [10,11,12,13,14,15],
        'co': ['us']*4+['india']*2, 'p': [.9,.9,.9,.8,.95,.7],
        'raw': [.9,.9,.9,.8,.95,.7], 'y': [0,0,0,1,1,1]})
    anchors = pl.DataFrame({'qid': [1,2,3,4,5,6], 'co': ['us']*4+['india']*2, 'deg': [0,0,0,1,1,1]})
    baseline = rows.filter(pl.col('tid') != 15).select('qid','tid','p','y')
    unconstrained = m.decode.curve(rows.select('qid','p','y'), anchors)['threshold']
    bad = m.frozen_accept(rows, {'threshold': unconstrained}, baseline)
    assert not m.guard(bad, baseline, anchors, require_country_f=True)['passed']
    cut, diagnostics = m.constrained_threshold(rows, baseline, anchors, .7842838168144226)
    good = m.frozen_accept(rows, {'threshold': cut}, baseline)
    assert diagnostics['feasible_cut_found']
    assert cut == .7
    assert m.guard(good, baseline, anchors, require_country_f=True)['passed']


@pytest.mark.parametrize('promote', [False, True])
def test_run_artifact_protocol_and_real_export(tmp_path, monkeypatch, promote):
    'Use real parquet, metadata, calibration fitting/adaptation, and TSV validation'
    import json
    from latest_fusion import mixture_calibration as calibration
    from er.stack.pipeline import export as real_export
    from er.stack.inputs import load_refs, load_targets
    data, prepared, incumbent, output = [tmp_path/name for name in ('data','prepared','incumbent','result')]
    prepared.mkdir()
    (incumbent/'output').mkdir(parents=True)

    groups = [[200+4*g+i for i in range(4)] for g in range(3)]
    source_ids = [int(q) for q in np.arange(100, dtype=np.uint32) if m.partition(np.array([q], dtype=np.uint32))[0] == 1][:4]
    ids = source_ids + sum(groups, []) + [999]
    countries = ['us','us','india','india']*4 + ['france']
    labels = [1,0,1,0]*4 + [1]
    folds = [0]*4+[1]*12+[1]
    refs = pl.DataFrame({'rid': ids, 'eid': [f'r{q}' for q in ids], 'nm': ['name']*17,
        'ad': ['address']*17, 'co': countries, 'fold': folds, 'deg': labels}).with_columns(pl.col('rid').cast(pl.UInt32))
    tids = list(range(17))
    targets = pl.DataFrame({'rid': tids, 'eid': [f't{t}' for t in tids], 'nm': ['target']*17,
        'ad': ['address']*17, 'co': countries}).with_columns(pl.col('rid').cast(pl.UInt32))
    frame = pl.DataFrame({'qid': ids, 'tid': tids, 'co': countries, 'y': labels,
        'own': [q if y else -1 for q,y in zip(ids,labels)], 'fold': folds,
        'seg': [0]*17, 'target_address_empty': [0]*17, 'baseline_p': [.9]*17,
        'raw': [.99 if y else .01 for y in labels]}).with_columns(pl.col('qid','tid').cast(pl.UInt32))
    test_frame = frame.drop('y','own','fold')
    for split in ('train','test'):
        (data/split).mkdir(parents=True)
        (refs if split == 'train' else refs.drop('fold','deg')).write_parquet(data/split/'ref.parquet')
        s2_count = 6 if split == 'train' else 8
        targets.head(s2_count).write_parquet(data/split/'s2.parquet')
        targets.tail(17-s2_count).write_parquet(data/split/'s3.parquet')
    train, test = prepared/'train.parquet', prepared/'test.parquet'
    frame.write_parquet(train)
    test_frame.write_parquet(test)
    for split in ('train','test'):
        (prepared/f'features-{split}.json').write_text('[]')
    frame.select('qid','tid').write_parquet(incumbent/'validation_predictions.parquet')
    test_frame.select('qid','tid').write_parquet(incumbent/'test_predictions.parquet')
    baseline = frame.select('qid','tid',pl.col('baseline_p').alias('p'),'y')
    baseline.drop('y').write_parquet(incumbent/'accepted_test.parquet')
    real_export(test_frame, 'baseline_p', 'fixture', 0., load_refs(str(data),'test'), load_targets(str(data),'test'), str(incumbent/'output'), accepted=baseline.drop('y'))
    baseline_bytes = (incumbent/'output'/'matching_results.tsv').read_bytes()
    digest = m.sha256(incumbent/'output'/'matching_results.tsv')
    monkeypatch.setattr(m, 'INCUMBENT_SHA', digest)
    (incumbent/'residual_model.txt').write_text('fixture upstream booster')
    tune_ids = groups[0]+groups[1]
    tune_anchors = refs.filter(pl.col('rid').is_in(tune_ids)).select(pl.col('rid').alias('qid'),'deg','co')
    (incumbent/'report.json').write_text(json.dumps({'selected': m.METHOD, 'matching_sha256': digest,
        'methods': {m.METHOD: {'tune': {**m.decode.score(baseline,tune_anchors),'threshold': .7842838168144226}}}}))
    monkeypatch.setattr(m.lgb, 'Booster', lambda **kwargs: object())
    def replay_fixture(f, model, features, previous, split):
        top = f.select('qid','tid','co','raw', *(['y'] if split == 'train' else []), pl.col('baseline_p').alias('p'))
        return f, top, top.select('qid','tid','p', *(['y'] if split == 'train' else []))
    monkeypatch.setattr(m, 'replay', replay_fixture)
    monkeypatch.setattr(m, 'fit_population', lambda r: (np.array([],dtype=np.uint32), np.array(tune_ids,dtype=np.uint32)))
    monkeypatch.setattr(m, 'split_search_check', lambda r, ids: tuple(r.filter(pl.col('rid').is_in(group)).select(pl.col('rid').alias('qid'),'deg','co') for group in groups[:2]))
    monkeypatch.setattr(m, 'split_references', lambda r: (None,None,r.filter(pl.col('rid').is_in(groups[2])),{}))
    real_adapt = calibration.adapt
    seen = []
    def checked_adapt(bundle, winners):
        assert not {'y','own','fold'} & set(winners.columns)
        s2_count = 6 if not seen else 8
        assert winners['is_s3'].to_list() == [0]*s2_count+[1]*(17-s2_count)
        seen.append(winners.height)
        return real_adapt(bundle,winners)
    monkeypatch.setattr(calibration, 'adapt', checked_adapt)
    def policy_signal(bundle, winners, recipe=None):
        assert not {'y','own','fold'} & set(winners.columns)
        return winners['raw'].to_numpy() if promote else winners['p'].to_numpy()
    monkeypatch.setattr(calibration, 'predict', policy_signal)
    report = m.run(data, train, test, incumbent, output)
    assert seen == [17,17]
    assert report['promoted'] is promote
    accepted = pl.read_parquet(output/'accepted_test.parquet')
    assert accepted.filter(pl.col('qid') == 999).select('qid','tid','p').equals(baseline.filter(pl.col('qid') == 999).drop('y'))
    assert accepted['tid'].n_unique() == accepted.height
    assert report['validation']['candidate_pairs'] == 17
    assert not report['selection']['test_labels_used']
    assert (output/'mixture_recipes.json').exists()
    if promote:
        assert accepted.height == sum(labels)
        assert report['selection']['locked']['family'] in ('em','empirical')
        assert report['selection']['evaluation']['owner_check']['treatment']['passed']
        assert report['selection']['evaluation']['structured_check']['treatment']['passed']
    else:
        assert report['selection']['locked'] is None
        assert (output/'output'/'matching_results.tsv').read_bytes() == baseline_bytes
        assert report['matching_sha256'] == digest

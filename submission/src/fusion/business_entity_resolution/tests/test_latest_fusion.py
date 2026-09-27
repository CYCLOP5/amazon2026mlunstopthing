import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]

import numpy as np
import polars as pl
import pytest

from latest_fusion.features import prepare_features, SCORES
from latest_fusion.calibration import adjust, calibrate_methods, decide, partition


def records():
    refs = pl.DataFrame({'rid':[0,1,2], 'co':['france']*3,
        'nm':['Étoile SARL','Étoile SARL','Autre SAS'],
        'ad':['10 avenue de Paris','12 avenue de Paris','3 rue du Sud']}).with_columns(pl.col('rid').cast(pl.UInt32))
    target = pl.DataFrame({'rid':[0,1,2,3], 'co':['france']*4,
        'nm':['Etoile SARL Groupe','Etoile Groupe SARL','Autre SAS',''],
        'ad':['10 av de Paris','12 avenue de Paris','3 rue du Sud','']}).with_columns(pl.col('rid').cast(pl.UInt32))
    pairs = pl.DataFrame({'qid':[0,1,0,1,2,2], 'tid':[0,0,1,1,2,3]}).with_columns(pl.col('qid','tid').cast(pl.UInt32))
    for col in SCORES:
        pairs = pairs.with_columns(pl.Series(col,[.9,.8,.5,.8,.99,None],dtype=pl.Float32))
    return refs,target,pairs


def test_features_use_full_owner_competition_and_french_word_position():
    refs,tg,pairs = records()
    d, ff = prepare_features(pairs,refs,tg,chunk_size=2)
    assert not set(ff)&{'qid','tid','y','own','co','fold'}
    assert d['ref_name_twins_log'][0] == pytest.approx(np.log(3))
    assert d['extra_after_legal'][0] == 1
    assert d['extra_before_legal'][2] == 1
    assert d['newest_owner_margin'][0] == pytest.approx(.1)
    assert d['newest_behind_best'][1] == pytest.approx(-.1)
    assert d['street_jaccard'][0] == 1
    assert d['newest_present'][-1] == 0
    assert all(d[c].is_finite().all() for c in ff)


def test_features_ignore_label_values_and_preserve_order():
    refs,tg,pairs = records()
    left, names = prepare_features(pairs.with_columns(pl.lit(0).alias('y')),refs,tg)
    right, names2 = prepare_features(pairs.with_columns(pl.lit(1).alias('y')),refs,tg)
    assert names == names2 and left.select(names).equals(right.select(names))
    assert left.select('qid','tid').equals(pairs.select('qid','tid'))


def test_features_reject_bad_keys_and_probabilities():
    refs,tg,pairs = records()
    with pytest.raises(ValueError,match='Duplicate'):
        prepare_features(pl.concat([pairs,pairs.head(1)]),refs,tg)
    with pytest.raises(ValueError,match='Invalid probabilities'):
        prepare_features(pairs.with_columns(pl.lit(2.).alias('graph')),refs,tg)


def test_exact_decoder_uses_raw_tie_break_and_excludes_old_only_baseline():
    d = pl.DataFrame({'qid':[0,1,2], 'tid':[0,0,1], 'baseline':[.9,.9,.99],
                     'raw_baseline':[.7,.8,.99], 'newest_present':[1.,1.,0.]})
    accepted = decide(d,'baseline',baseline_only=True)
    assert set(accepted.select('qid','tid').rows()) == {(1,0)}


def calibration_fixture(tmp_path):
    refs = pl.DataFrame({'rid':range(120),'co':['india']*120,'fold':[0]*120})
    rtest = pl.DataFrame({'rid':[0,1], 'co':['india','france']})
    raw = np.tile([.02,.2,.7,.999],30)
    tr = pl.DataFrame({'qid':range(120),'tid':range(120), 'baseline':raw, 'candidate':raw,
        'gate':raw,'newest_present':[1.]*120,'seg':[0]*120,'y':(raw>.5).astype(np.uint8)})
    te = pl.DataFrame({'qid':[0,1], 'tid':[0,1], 'baseline':[.8,.999],'candidate':[.9,.999],
        'gate':[.8,.2],'newest_present':[1.,1.],'seg':[0,0]})
    recipe = {'known_score':'stack_prob','unseen_gate':True, 'countries':['india'],
        'partition':{'modulus':3,'remainder':1},'edges':[-3.,0.,3.],
        'curves':{f'{co}|{seg}':{'centres':[-12.,0.,12.],'posterior':[0.,.5,1.]}
            for co in ('india','france') for seg in range(3)}}
    path = tmp_path/'recipe.json'; path.write_text(json.dumps(recipe))
    return refs,rtest,tr,te,recipe,path


def test_calibration_excludes_tune_audit_labels_and_keeps_france_gate(tmp_path):
    refs,rtest,tr,te,recipe,path = calibration_fixture(tmp_path)
    a,b,report = calibrate_methods(tr,te,refs,rtest,['baseline','candidate'],path)
    changed = tr.with_columns(pl.when(pl.Series(partition(tr['qid']) != 1)).then(1-pl.col('y')).otherwise(pl.col('y')).alias('y'))
    a2,b2,report2 = calibrate_methods(changed,te,refs,rtest,['baseline','candidate'],path)
    assert report == report2
    assert a.select('baseline','candidate').equals(a2.select('baseline','candidate'))
    assert b.select('baseline','candidate').equals(b2.select('baseline','candidate'))
    assert b['raw_baseline'][1] == pytest.approx(.2)
    assert b['raw_candidate'][1] == pytest.approx(.2)
    exp = adjust(np.array([.8,.2]),np.array(['india','france']),np.array([0,0]),recipe)
    np.testing.assert_array_equal(b['baseline'].to_numpy(),exp)


def test_latest_decoder_matches_brute_force():
    from latest_fusion.decoder import check
    check()

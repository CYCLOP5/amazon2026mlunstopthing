'reference fidelity and country isolation checks'
from pathlib import Path
import sys
import numpy as np
import polars as pl
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final_repair.reference import blend, selected_decoder, segments, fit, apply, calibrated_predictions


def test_missing_score_blend_and_source_filter():
    f = pl.DataFrame({'qid':[0,1,2], 'tid':[0,1,2], 'p2':[.9,.0009,.8], 'y':[1,0,1]})
    g = pl.DataFrame({'qid':[2,3], 'tid':[2,3], 'head':[.6,.7], 'y':[1,0]})
    d = blend(f,g).sort('qid')
    assert d['qid'].to_list() == [0,2,3]
    lg = lambda p: np.log(p/(1-p))
    assert d['p'][0] == pytest.approx(1/(1+np.exp(-(.75*lg(.9)+.25*lg(.0001)))))
    assert d['p'][1] == pytest.approx(1/(1+np.exp(-(.75*lg(.8)+.25*lg(.6)))))
    with pytest.raises(ValueError, match='Conflicting'):
        blend(f,g.with_columns(pl.lit(0).alias('y')))


def test_actual_selected_score_and_decoder():
    r = {'selected':'first', 'methods':{'first':{'score':'p1','tune':{'rule':'expected_f','threshold':.3}},
         'second':{'score':'p2','tune':{'rule':'top1_threshold','threshold':.7}}}}
    assert selected_decoder(r) == {'score':'p1','rule':'expected_f','threshold':.3}
    with pytest.raises(ValueError):
        selected_decoder({'methods':r['methods']})


def test_house_number_matches_later_number_exact_reference_semantics():
    p = pl.DataFrame({'qid':[0,0,0], 'tid':[0,1,2], 'p':[.8,.8,.8]})
    r = pl.DataFrame({'qid':[0], 'ad':['0012 Rue Example']})
    t = pl.DataFrame({'tid':[0,1,2], 'ad':['75001 Paris 12 Rue Example','13 Rue Example','No number']})
    assert segments(p,r,t)['seg'].to_list() == ['equal','conflict','unknown']


def test_calibration_fit_apply_identity_and_country_rejection():
    p = pl.DataFrame({'qid':[0,1,2], 'tid':[0,1,2], 'p':[.2,.8,.9999],
                      'y':[0,1,1], 'co':['us']*3})
    r = pl.DataFrame({'qid':[0,1,2], 'ad':['12 Rue']*3})
    t = pl.DataFrame({'tid':[0,1,2], 'ad':['12 Rue']*3})
    segmented = segments(p,r,t)
    tables = fit(segmented,segmented,{'us':3},{'us':3})
    got, same_tables = calibrated_predictions(p,p,r,t,r,t,{'us':3},{'us':3})
    assert same_tables == tables
    assert got.equals(apply(segmented,tables))
    assert np.isfinite(got['p'].to_numpy()).all()
    assert np.diff(tables['us|equal']['posterior']).min() >= 0
    assert tables['us|equal']['a'] == pytest.approx(1)
    with pytest.raises(ValueError, match='Unlabeled'):
        calibrated_predictions(p,p.with_columns(pl.lit('france').alias('co')),r,t,r,t,{'us':3},{'france':3})
    with pytest.raises(ValueError, match='No labeled'):
        fit(segmented,segmented.with_columns(pl.lit('france').alias('co')),{'us':3},{'france':3})

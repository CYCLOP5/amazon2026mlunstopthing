from pathlib import Path
import sys

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latest_fusion.rescue_features import build_rescue_features


def fixture():
    refs=pl.DataFrame({'rid':[1,2,3,4],'co':['us','us','us','india'],'nm':['rare common llc','common inc','common ltd','rare'],'ad':['']*4})
    targets=pl.DataFrame({'rid':[10,11],'co':['us','us'],'nm':['rare common','common'],'ad':['','']})
    pool=pl.DataFrame({'qid':[2,1,3,1],'tid':[10,10,11,11],'co':['us']*4,'np_m0':[.9,.9,None,.8],'np_m1':[.2,.8,None,None],'inc_p':[.1,.7,.1,.8],'label':[0,1,0,1]})
    return pool,refs,targets


def test_consensus_rarity_order_labels():
    p,r,t=fixture()
    d,n=build_rescue_features(p,r,t,[])
    assert d.select('qid','tid').equals(p.select('qid','tid'))
    assert d['rescue_np_m0_vote'].to_list()==[0,0,0,1]
    assert d['rescue_member_vote_fraction'].to_list()==[0,.5,0,1]
    assert d['rescue_shared_idf_sum'][1]>d['rescue_shared_idf_sum'][0]
    assert d['rescue_rescue_shared_idf_sum_margin'][1]>0
    assert 'label' not in n and 'qid' not in n
    other,_=build_rescue_features(p.with_columns(pl.lit(999).alias('label')),r,t,[])
    assert d.select(n).equals(other.select(n))
    assert all(d[c].dtype==pl.Float32 for c in n)

    other,_=build_rescue_features(p,r.filter(pl.col('co')=='us'),t,[])
    assert d['rescue_shared_idf_sum'].equals(other['rescue_shared_idf_sum'])


def test_duplicate_rejected():
    p,r,t=fixture()
    with pytest.raises(ValueError,match='Duplicate'):
        build_rescue_features(pl.concat([p,p.head(1)]),r,t,[])


def test_missing_ids_and_empty_names():
    p,r,t=fixture()
    with pytest.raises(ValueError,match='missing'):
        build_rescue_features(p,r.filter(pl.col('rid')!=1),t,[])
    d,n=build_rescue_features(p,r.with_columns(pl.lit(None,dtype=pl.String).alias('nm')),t.with_columns(pl.lit('').alias('nm')),[])
    assert d['rescue_shared_idf_sum'].sum()==0
    assert d.select(pl.all_horizontal([pl.col(c).is_finite() for c in n]).all()).item()


def test_unknown_extra_and_country_alignment():
    p,r,t=fixture()
    unknown=t.with_columns(pl.when(pl.col('rid')==10).then(pl.lit('rare common unseen')).otherwise(pl.col('nm')).alias('nm'))
    d,n=build_rescue_features(p,r,unknown,[])
    import math
    assert d['rescue_extra_idf_sum'][1]==pytest.approx(math.log(4)+1)
    assert d['rescue_member_present_count'].to_list()==[2,2,0,1]
    with pytest.raises(ValueError,match='country'):
        build_rescue_features(p,r,t.with_columns(pl.lit('india').alias('co')),[])

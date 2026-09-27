from pathlib import Path
import sys
import polars as pl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from final_repair.identity import normalize_records, propose


def records(names,addresses,countries=None):
    return pl.DataFrame({'rid':list(range(len(names))), 'nm':names,'ad':addresses,
                         'co':countries or ['france']*len(names)})


def test_accents_dotted_legal_and_french_distinctions():
    d=normalize_records(records(['Café Étoile S.A.R.L.','Etoile Café SARL','Alpha SAS','Alpha LLC'],
                                ['12 av. St Étienne Paris','Paris 12 avenue Saint Etienne',
                                 '12 rue Sainte Marie Paris','12 rue Saint Marie Paris']))
    assert d['name_full'][0] == d['name_full'][1]
    assert d['address_key'][0] == d['address_key'][1]
    assert d['legal_form'].to_list() == ['sarl','sarl','sas','llc']
    assert d['address_key'][2] != d['address_key'][3]
    x=normalize_records(records(['Alpha']*4,['12 bis rue Victor Paris','12 ter rue Victor Paris',
                                             '12/14 rue Victor Paris','12 14 rue Victor Paris']))
    assert x['address_key'].n_unique() == 4


def test_distinct_street_and_reference_ambiguity():
    r=records(['Cafe Etoile SARL','Cafe Etoile SARL'],['12 rue Victor Paris','12 rue Hugo Paris'])
    t=records(['Cafe Etoile SARL','Cafe Etoile SARL'],['12 rue Victor Paris',''])
    p,report=propose(r,t)
    assert p.filter((pl.col('rule')=='full_identity') & (pl.col('tid')==0))['qid'].to_list()==[0]
    assert p.filter(pl.col('tid')==1).height==0
    r=records(['Cafe Etoile SARL']*2,['12 rue Victor Paris']*2)
    assert propose(r,t)[0].height==0


def test_missing_address_country_uniqueness_and_legal_conflict():
    r=records(['Cafe Etoile SARL','Cafe Etoile SARL','Alpha LLC'],
              ['12 rue Victor Paris']*3,['france','us','france'])
    t=records(['Cafe Etoile SARL','Alpha SARL','Alpha'],['','',''])
    p,report=propose(r,t)
    assert set(p.filter(pl.col('tid')==0)['qid'].to_list())=={0}
    assert p.filter(pl.col('tid')==1).height==0
    assert set(p.filter(pl.col('tid')==2)['qid'].to_list())=={2}


def test_close_name_and_city_only_address_exclusion():
    r=records(['Grand Cafe Etoile SARL','Alpha Market LLC'],['12 rue Victor Paris','75001 Paris'])
    t=records(['Grand Cafe Etoilee SARL','Alpha Market LLC'],['12 rue Victor Paris','75001 Paris'])
    p,report=propose(r,t)
    assert p.filter(pl.col('rule')=='unique_address_close_name')['qid'].to_list()==[0]
    assert p.filter(pl.col('tid')==1).height==0


def test_reordered_legal_tokens_and_all_dotted_forms():
    d=normalize_records(records(['Saint-Herblain SARL Amicale','Saint-Herblain Amicale SARL',
                                'DUNKERQUE SAS ATELIERS','Ateliers Dunkerque S.A.S.',
                                'Alpha E.U.R.L.','E.U.R.L. Alpha','Alpha S.A.S.U.','S.A.S.U. Alpha'],
                                ['25 rue Victor Paris']*8))
    for i in (0,2,4,6):
        assert d['name_full'][i] == d['name_full'][i+1]
    assert d['legal_form'].to_list() == ['sarl','sarl','sas','sas','eurl','eurl','sasu','sasu']


def test_alias_identity_country_noise_and_number_prefix():
    r=records(['Allez & SARL Groupe','Allez & Fils SARL','Club Etoile SARL','Amicale Etoile SARL'],
              ['Nº 25 Rue du Contour du Sud Dunkerque','25 Rue Victor Paris',
               '18 rue Victor Paris','18 rue Victor Paris'])
    t=records(['Allez & Cie SARL','Club Etoile SARL','Union Etoile SARL'],
              ['25 Rue du Contour du Sud Dunkerque','18 rue Victor Paris','18 rue Victor Paris'])
    p,_=propose(r,t)
    assert p.filter((pl.col('rule')=='alias_identity') & (pl.col('tid')==0))['qid'].to_list()==[0]
    assert p.filter(pl.col('tid')==2).height==0
    d=normalize_records(records(['Alpha']*4,['Nº 25 rue Victor Paris','N°25 rue Victor Paris',
                                             'No. 25 rue Victor Paris','25 rue Victor Paris']))
    assert d['address_key'].n_unique()==1
    n=normalize_records(records(['Allez Groupe SARL','Allez Group LLC','Allez Enterprises LTD'],
                                ['12 rue Victor Paris']*3,['france','us','india']))
    assert n['name_alias'].to_list()==['allez']*3
    wrong=normalize_records(records(['Allez Groupe LLC'],['12 rue Victor Paris'],['us']))
    assert wrong['name_alias'][0]=='allez groupe'


def test_alias_reference_ambiguity_is_not_broken_by_legal_form():
    r=records(['Allez Groupe SARL','Allez Cie SARL'],['25 rue Victor Paris']*2)
    t=records(['Allez Fils SARL'],['25 rue Victor Paris'])
    p,_=propose(r,t)
    assert p.filter(pl.col('rule')=='alias_identity').height==0

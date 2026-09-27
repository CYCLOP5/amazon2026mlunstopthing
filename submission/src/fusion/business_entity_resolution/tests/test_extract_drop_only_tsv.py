import hashlib
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from extract_drop_only_tsv import HEADER,reconstruct


def test_intersection_removes_only_and_preserves_order(tmp_path):
    a=HEADER+b'r1\tb,a,c\nr2\tx\nr3\t\n'
    b=HEADER+b'r1\tc,b,z\nr2\t\nr3\tk\n'
    old,new,out=[tmp_path/x for x in ('old.tsv','new.tsv','out.tsv')]
    old.write_bytes(a);new.write_bytes(b)
    res=reconstruct(old,new,out,hashlib.sha256(a).hexdigest(),hashlib.sha256(b).hexdigest(),2)
    assert out.read_bytes()==HEADER+b'r1\tb,c\nr2\t\nr3\t\n'
    assert res['removed_pairs']==2 and res['ignored_mixed_additions']==2
    assert res['added_pairs']==0 and res['source_rows']==3


@pytest.mark.parametrize('invalid',['hash','id_order','count'])
def test_invalid_inputs_do_not_publish_file(tmp_path,invalid):
    a=HEADER+b'r1\tx,y\n';b=HEADER+(b'r2\tx\n' if invalid=='id_order' else b'r1\tx\n')
    old,new,out=[tmp_path/x for x in ('old.tsv','new.tsv','out.tsv')]
    old.write_bytes(a);new.write_bytes(b)
    with pytest.raises(ValueError):
        reconstruct(old,new,out,'wrong' if invalid=='hash' else hashlib.sha256(a).hexdigest(),
                    hashlib.sha256(b).hexdigest(),2 if invalid=='count' else 1)
    assert not out.exists() and not out.with_name('out.tsv.partial').exists()

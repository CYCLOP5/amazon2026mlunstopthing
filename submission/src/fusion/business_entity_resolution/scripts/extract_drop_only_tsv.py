'reconstruct only the deletions of a mixed submission, preserving baseline order'
import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

HEADER = b'source1_entity_id\tmatched_entity_ids\n'
BASE_SHA = '1d1196df3840ced54b7b1af1f2e7649990d0c6183302f184eddcbda4da1ec00c'
MIXED_SHA = '0f91f8ec752bfba46a30aecc15488ee8b7a31a835de8ed808944db1f8a4e16ef'


def reconstruct(baseline, mixed, output, expected_baseline_sha, expected_mixed_sha, expected_removed=None):
    baseline, mixed, output = map(Path, (baseline, mixed, output))
    if output.exists() or output.resolve() in (baseline.resolve(), mixed.resolve()):
        raise ValueError('Output must be a new file separate from both sources')
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name+'.partial')
    hashes = [hashlib.sha256(), hashlib.sha256(), hashlib.sha256()]
    stats = {'source_rows':0, 'baseline_matches':0, 'mixed_matches':0,
             'retained_matches':0, 'removed_pairs':0, 'ignored_mixed_additions':0,
             'changed_reference_rows':0}
    try:
        with baseline.open('rb') as left, mixed.open('rb') as right, partial.open('xb') as dest:
            for handle, h in zip((left,right),hashes):
                line=handle.readline();h.update(line)
                if line != HEADER:
                    raise ValueError('Unexpected submission header')
            dest.write(HEADER);hashes[2].update(HEADER)
            for a,b in zip_longest(left,right):
                if a is None or b is None:
                    raise ValueError('Source row counts differ')
                hashes[0].update(a);hashes[1].update(b)
                aa,bb=a.rstrip(b'\n').split(b'\t'),b.rstrip(b'\n').split(b'\t')
                if len(aa)!=2 or len(bb)!=2 or aa[0]!=bb[0] or not aa[0]:
                    raise ValueError('Source reference IDs/order or TSV fields differ')
                old=aa[1].split(b',') if aa[1] else []
                new=bb[1].split(b',') if bb[1] else []
                if len(old)!=len(set(old)) or len(new)!=len(set(new)) or b'' in old or b'' in new:
                    raise ValueError('Duplicate or empty target inside a match list')
                membership=set(new)
                kept=[tid for tid in old if tid in membership]
                line=aa[0]+b'\t'+b','.join(kept)+b'\n'
                dest.write(line);hashes[2].update(line)
                stats['source_rows']+=1
                stats['baseline_matches']+=len(old);stats['mixed_matches']+=len(new)
                stats['retained_matches']+=len(kept)
                stats['removed_pairs']+=len(old)-len(kept)
                stats['ignored_mixed_additions']+=len(membership-set(old))
                stats['changed_reference_rows']+=int(kept!=old)
        if hashes[0].hexdigest()!=expected_baseline_sha or hashes[1].hexdigest()!=expected_mixed_sha:
            raise ValueError('Source submission checksum differs')
        if expected_removed is not None and stats['removed_pairs']!=expected_removed:
            raise ValueError('Removal count differs from requested variant')
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {**stats, 'baseline_sha256':hashes[0].hexdigest(),
            'mixed_sha256':hashes[1].hexdigest(), 'matching_sha256':hashes[2].hexdigest(),
            'bytes':output.stat().st_size, 'added_pairs':0,
            'operation':'baseline intersection mixed, preserving baseline reference and target order',
            'leaderboard_score':None, 'score_note':'Friend reports .9893 for a drop-only file; exact artifact identity requires confirmation'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','mixed','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-baseline-sha',default=BASE_SHA)
    parser.add_argument('--expected-mixed-sha',default=MIXED_SHA)
    parser.add_argument('--expected-removed',type=int,default=11161)
    args=vars(parser.parse_args())
    result=reconstruct(**args)
    args['output'].with_suffix('.provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

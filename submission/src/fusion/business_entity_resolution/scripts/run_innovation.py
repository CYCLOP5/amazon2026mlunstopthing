'cloud entry point for verification and joint owner allocation experiments'
import argparse
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT),str(PROJECT/'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['prepare','verify','verify-worker','train','allocate','diagnose'])
    for name in ('data','parent','prepared','model','verified','output'):
        ap.add_argument('--'+name,type=Path)
    ap.add_argument('--max-targets',type=int,default=500_000)
    ap.add_argument('--topk',type=int,default=8)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--rank',type=int,default=0)
    ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--rounds',type=int,default=1200)
    args = ap.parse_args()
    os.environ.setdefault('ER_THREADS','0.85')
    os.environ.setdefault('ER_MIN_FREE_GB','12')
    import er.safe
    if args.stage == 'prepare':
        from innovation.prepare import run
        run(args.data,args.parent,args.output,args.max_targets,args.topk)
    elif args.stage == 'verify':
        from innovation.verify import run
        run(args.prepared,args.model,args.output,args.workers,args.batch)
    elif args.stage == 'verify-worker':
        from innovation.verify import worker
        worker(args.prepared,args.model,args.output,args.rank,args.workers,args.batch)
    elif args.stage == 'train':
        from innovation.train import run
        run(args.data,args.parent,args.prepared,args.verified,args.output,args.rounds)
    elif args.stage == 'allocate':
        from innovation.allocate import run
        run(args.data,args.parent,args.output)
    else:
        from innovation.diagnose import run
        run(args.data,args.parent,args.output)


if __name__ == '__main__':
    main()

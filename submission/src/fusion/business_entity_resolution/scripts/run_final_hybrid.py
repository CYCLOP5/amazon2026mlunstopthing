'run bounded hybrid retrieval, reused neural scorers, and residual heads'
import argparse
import os
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'gpu', 'dense-worker', 'score-worker', 'fit'])
    for name in ('data', 'parent', 'prepared', 'lexical', 'lexical-train', 'lexical-test',
                 'old-model', 'verified', 'expert', 'hybrid', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--split', choices=['train', 'test', 'both'], default='both')
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=800)
    args = parser.parse_args()
    required = {'prepare': ('data', 'prepared', 'output'),
        'gpu': ('prepared', 'lexical_train', 'lexical_test', 'old_model', 'verified', 'expert', 'output'),
        'dense-worker': ('lexical', 'output'),
        'score-worker': ('prepared', 'lexical', 'old_model', 'verified', 'expert', 'output'),
        'fit': ('data', 'parent', 'prepared', 'hybrid', 'output')}
    for name in required[args.stage]:
        if getattr(args, name) is None:
            parser.error(f'{args.stage} requires --{name.replace("_", "-")}')
    for name in ('topk', 'threads', 'workers', 'batch', 'rounds'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if not 0 <= args.rank < args.workers:
        parser.error('--rank must be within --workers')
    os.environ.setdefault('ER_THREADS', '0.8' if args.stage in ('prepare', 'fit') else '4')
    os.environ.setdefault('ER_MIN_FREE_GB', '12')
    import er.safe
    if args.stage == 'prepare':
        from final_hybrid.prepare import run
        run(args.data, args.prepared, args.output, args.split, args.topk, args.threads)
    elif args.stage == 'gpu':
        from final_hybrid.gpu import run
        with tempfile.TemporaryDirectory(prefix='hybrid_lexical_') as work:
            merged = Path(work)
            for split, root in [('train', args.lexical_train), ('test', args.lexical_test)]:
                if not (root/f'_{split}_SUCCESS').is_file():
                    raise RuntimeError(f'Lexical {split} preparation is incomplete')
                (merged/split).symlink_to((root/split).resolve(), target_is_directory=True)
            run(args.prepared, merged, args.old_model, args.verified, args.expert,
                args.output, args.workers, args.batch)
    elif args.stage == 'dense-worker':
        from final_hybrid.gpu import dense_worker
        dense_worker(args.lexical, args.output, args.rank, args.workers, args.batch)
    elif args.stage == 'score-worker':
        from final_hybrid.gpu import score_worker
        score_worker(args.prepared, args.lexical, args.old_model, args.verified, args.expert,
                     args.output, args.rank, args.workers, args.batch)
    else:
        from final_hybrid.fit import run
        run(args.data, args.parent, args.prepared, args.hybrid, args.output, args.rounds)


if __name__ == '__main__':
    main()

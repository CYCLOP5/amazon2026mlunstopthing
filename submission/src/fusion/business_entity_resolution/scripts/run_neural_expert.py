'prepare, train, score and evaluate an independent bounded neural expert'
import argparse
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'gpu', 'train-worker', 'score-worker', 'fit'])
    for name in ('data', 'parent', 'prepared', 'training', 'expert', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--max-pairs', type=int, default=600_000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--train-seconds', type=int, default=2400)
    parser.add_argument('--rounds', type=int, default=1000)
    args = parser.parse_args()
    required = {'prepare': ('data', 'prepared', 'output'),
                'gpu': ('training', 'prepared', 'output'),
                'train-worker': ('training', 'output'),
                'score-worker': ('prepared', 'output'),
                'fit': ('data', 'parent', 'prepared', 'expert', 'output')}
    for name in required[args.stage]:
        if getattr(args, name) is None:
            parser.error(f'{args.stage} requires --{name}')
    for name in ('workers', 'batch', 'epochs', 'train_seconds', 'rounds'):
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.max_pairs < 20 or not 0 <= args.rank < args.workers:
        parser.error('--max-pairs must be >=20 and --rank must be within --workers')
    os.environ.setdefault('ER_THREADS', '0.85' if args.stage in ('prepare', 'fit') else '4')
    os.environ.setdefault('ER_MIN_FREE_GB', '12')
    import er.safe
    if args.stage == 'prepare':
        from innovation.neural_expert import prepare
        prepare(args.data, args.prepared, args.output, args.max_pairs)
    elif args.stage == 'gpu':
        from innovation.neural_gpu import gpu
        gpu(args.training, args.prepared, args.output, args.workers, args.batch, args.epochs, args.train_seconds)
    elif args.stage == 'train-worker':
        from innovation.neural_gpu import train_worker
        train_worker(args.training, args.output, args.batch, args.epochs, args.train_seconds)
    elif args.stage == 'score-worker':
        from innovation.neural_gpu import score_worker
        score_worker(args.prepared, args.output, args.rank, args.workers, args.batch)
    else:
        from innovation.expert_fit import run
        run(args.data, args.parent, args.prepared, args.expert, args.output, args.rounds)


if __name__ == '__main__':
    main()

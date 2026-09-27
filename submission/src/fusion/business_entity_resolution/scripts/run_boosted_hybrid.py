'run cached feature augmentation, four independent booster workers, and selection'
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'gpu', 'preflight', 'worker', 'select'])
    for name in ('data', 'hybrid-result', 'features-train', 'features-test', 'models', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--split', choices=['train', 'test'])
    parser.add_argument('--head', choices=['xgboost_independent', 'catboost_independent',
                                         'xgboost_residual', 'catboost_residual'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--rounds', type=int, default=700)
    args = parser.parse_args()
    required = {'prepare': ('data', 'hybrid_result', 'output', 'split'),
        'gpu': ('data', 'features_train', 'features_test', 'output'),
        'worker': ('data', 'features_train', 'features_test', 'output', 'head'),
        'select': ('data', 'hybrid_result', 'features_train', 'features_test', 'models', 'output'),
        'preflight': ()}
    for name in required[args.stage]:
        if getattr(args, name) is None:
            parser.error(f'{args.stage} requires --{name.replace("_", "-")}')
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    os.environ.setdefault('ER_THREADS', '0.8' if args.stage in ('prepare', 'select') else '16')
    os.environ.setdefault('ER_MIN_FREE_GB', '12')
    import er.safe
    if args.stage == 'prepare':
        from boosted_hybrid.features import run
        run(args.data, args.hybrid_result, args.output, args.split)
    else:
        from boosted_hybrid import pipeline
        if args.stage == 'preflight':
            pipeline.gpu_preflight()
        elif args.stage == 'gpu':
            pipeline.gpu(args.data, args.features_train, args.features_test, args.output, args.rounds)
        elif args.stage == 'worker':
            pipeline.worker(args.data, args.features_train, args.features_test,
                args.output, args.head, args.rounds, args.device)
        else:
            pipeline.select(args.data, args.hybrid_result, args.features_train,
                args.features_test, args.models, args.output)


if __name__ == '__main__':
    main()

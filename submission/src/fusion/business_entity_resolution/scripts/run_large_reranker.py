'prepare, adapt, score and select the bounded multilingual 4b experiment'
import argparse
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','train','train-worker','score','score-worker','fit','preflight'])
    for name in ('data','hybrid-result','features-train','features-test','friend','graph',
                 'prepared','trained','scored','scored-extra','output'):
        parser.add_argument('--'+name,type=Path)
    for name,default in [('max-targets',120000),('max-pairs',800000),('workers',4),
                         ('rank',0),('shard-index',0),('shards',2),('epochs',2),
                         ('train-seconds',2400),('rounds',500)]:
        parser.add_argument('--'+name,type=int,default=default)
    parser.add_argument('--batch',type=int)
    args = parser.parse_args()
    required = {'prepare':('data','hybrid_result','features_train','features_test','friend','graph','output'),
        'train':('prepared','output'), 'train-worker':('prepared','output'),
        'score':('prepared','trained','output'), 'score-worker':('prepared','trained','output'),
        'fit':('data','hybrid_result','features_train','features_test','friend','graph',
               'prepared','scored','scored_extra','output'), 'preflight':()}
    for key in required[args.stage]:
        if getattr(args,key) is None:
            parser.error(f'{args.stage} requires --{key.replace("_","-")}')
    for key in ('max_targets','max_pairs','workers','shards','epochs','train_seconds','rounds'):
        if getattr(args,key) < 1:
            parser.error(f'--{key.replace("_","-")} must be positive')
    if not 0 <= args.rank < args.workers or not 0 <= args.shard_index < args.shards:
        parser.error('Invalid rank or shard index')
    if args.batch is not None and args.batch < 1:
        parser.error('--batch must be positive')
    os.environ.setdefault('ER_THREADS','0.8' if args.stage in ('prepare','fit') else '4')
    os.environ.setdefault('ER_MIN_FREE_GB','12')
    import er.safe
    if args.stage == 'prepare':
        from large_reranker.prepare import prepare
        prepare(args.data,args.hybrid_result,args.features_train,args.features_test,
                args.friend,args.graph,args.output,args.max_targets,args.max_pairs)
    elif args.stage == 'fit':
        from large_reranker.gpu import gather
        from large_reranker.fit import run
        with tempfile.TemporaryDirectory(prefix='large_scores_') as work:
            merged = gather(args.prepared,[args.scored,args.scored_extra],work)
            run(args.data,args.hybrid_result,args.features_train,args.features_test,
                args.friend,args.graph,args.prepared,merged,args.output,args.rounds)
    else:
        from large_reranker import gpu
        if args.stage == 'train':
            gpu.train(args.prepared,args.output,args.workers,args.batch or 8,args.epochs,args.train_seconds)
        elif args.stage == 'train-worker':
            gpu.train_worker(args.prepared,args.output,args.batch or 8,args.epochs,args.train_seconds)
        elif args.stage == 'score':
            gpu.score(args.prepared,args.trained,args.output,args.workers,args.batch or 48,args.shard_index,args.shards)
        elif args.stage == 'score-worker':
            gpu.score_worker(args.prepared,args.trained,args.output,args.rank,args.workers,
                             args.batch or 48,args.shard_index,args.shards)
        else:
            gpu.preflight()


if __name__ == '__main__':
    main()

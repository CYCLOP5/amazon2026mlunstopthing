'Run the independent alias graph + owner/abstention ranking experiment'
import argparse
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT),str(PROJECT/'src')]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for field in ('data','scores','lexical','output','work'):
        ap.add_argument('--'+field,type=Path,required=True)
    ap.add_argument('--rounds',type=int,default=1800)
    ap.add_argument('--graph-predictions',type=Path,help='Run learned-sibling refinement using this completed graph output')
    args = ap.parse_args()
    os.environ.setdefault('ER_THREADS','0.9')
    os.environ.setdefault('ER_MIN_FREE_GB','12')
    import er.safe
    from run_stack import score_roots,lexical_roots
    if args.graph_predictions:
        from graph_resolution.refine import run
    else:
        from graph_resolution.pipeline import run
    sc = {'data':str(args.data),'rounds':args.rounds}
    if args.graph_predictions:
        sc['graph_predictions'] = str(args.graph_predictions)
    for split in ('train','test'):
        sc[f'{split}_roots'] = score_roots(args.scores,split)
        sc[f'lexical_{split}'] = lexical_roots(args.lexical,split)
    run(sc,args.output,args.work)


if __name__ == '__main__':
    main()

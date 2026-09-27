'Prepare frozen score unions or fit/export the latest CPU fusion'
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
os.environ.setdefault('ER_THREADS','48')
import er.safe

def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='stage',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--split',choices=['train','test'],required=True)
    for name in ('data','newest','hybrid','graph','friend','output','expected-meta'): prepare.add_argument('--'+name,type=Path,required=True)
    fit=sub.add_parser('fit')
    for name in ('data','train','test','metadata','output','calibration'):fit.add_argument('--'+name,type=Path,required=True)
    args=vars(ap.parse_args());stage=args.pop('stage')
    from latest_fusion import pipeline
    stop=__import__('threading').Event()
    def heartbeat():
        while not stop.wait(30):pipeline.log('HEARTBEAT '+stage)
    __import__('threading').Thread(target=heartbeat,daemon=True).start()
    try:getattr(pipeline,stage)(**args)
    finally:stop.set()

if __name__=='__main__':main()

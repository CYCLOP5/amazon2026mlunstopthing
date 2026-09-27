'run frozen-owner conditional mixture calibration on azure artifacts'
import argparse
import os
from pathlib import Path
import sys
import threading
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
os.environ.setdefault('ER_THREADS', '48')
import er.safe


def main():
    from latest_fusion.mixture_pipeline import run
    from latest_fusion.pipeline import log
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('data', 'train', 'test', 'incumbent', 'output'):
        parser.add_argument('--'+field, type=Path, required=True)
    args = vars(parser.parse_args())
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(30):
            log('HEARTBEAT mixture calibration')
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        run(**args)
    finally:
        stop.set()


if __name__ == '__main__':
    main()

'fit a strictly additive recall specialist using frozen azure scores'
import argparse
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
os.environ.setdefault('ER_THREADS', '48')
import er.safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'train', 'test', 'incumbent', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    from latest_fusion.rescue_pipeline import run
    from latest_fusion.pipeline import log
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(30):
            log('HEARTBEAT additive recall experiment')

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        run(**vars(parser.parse_args()))
    finally:
        stopped.set()


if __name__ == '__main__':
    main()

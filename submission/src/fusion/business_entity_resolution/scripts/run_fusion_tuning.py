'run one bounded cpu search using completed fusion artifacts in azure'
import argparse
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
os.environ.setdefault('ER_THREADS', '48')
import er.safe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'train', 'test', 'incumbent', 'metadata', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = vars(parser.parse_args())
    from latest_fusion.tune_pipeline import run
    from latest_fusion.pipeline import log
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            log('HEARTBEAT bounded fusion tuning')

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        run(**args)
    finally:
        stop.set()


if __name__ == '__main__':
    main()

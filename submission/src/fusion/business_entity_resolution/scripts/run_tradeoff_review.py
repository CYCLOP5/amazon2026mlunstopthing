'replay a saved trial at a fixed strength and policy; no retraining'
import argparse
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'train', 'test', 'completed', 'metadata', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--trial', required=True)
    parser.add_argument('--expected-sha', required=True)
    args = vars(parser.parse_args())
    from latest_fusion.tradeoff_review import run
    from latest_fusion.pipeline import log
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            log('HEARTBEAT fixed trade-off review; no training or threshold search')

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        run(**args)
    finally:
        stop.set()


if __name__ == '__main__':
    main()

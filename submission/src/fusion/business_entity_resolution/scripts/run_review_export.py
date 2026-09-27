'export an unchanged locked candidate whose sole rejection was uncertainty'
import argparse
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'train', 'test', 'incumbent', 'completed', 'metadata', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    from latest_fusion.review_export import run
    from latest_fusion.pipeline import log
    args = vars(parser.parse_args())
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            log('HEARTBEAT locked experimental export; no training or policy search')

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        run(**args)
    finally:
        stop.set()


if __name__ == '__main__':
    main()

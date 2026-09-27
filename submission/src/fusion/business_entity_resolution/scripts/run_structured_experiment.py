'run one controlled structural experiment on the completed fusion artifacts'
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
    from latest_fusion.structured_pipeline import MODES, run
    from latest_fusion.pipeline import log
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('data', 'train', 'test', 'incumbent', 'output'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--contrastive-bundle', type=Path)
    parser.add_argument('--directional-bundle', type=Path)
    args = vars(parser.parse_args())
    sources = {family: args.pop(family+'_bundle') for family in ('contrastive', 'directional')}
    if args['mode'] == 'additions':
        if any(path is None for path in sources.values()):
            parser.error('additions requires --contrastive-bundle and --directional-bundle')
        args['pretrained_heads'] = sources
    elif any(path is not None for path in sources.values()):
        parser.error('pretrained bundle arguments are only used by additions')
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(30):
            log('HEARTBEAT structured experiment '+args['mode'])
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        run(**args)
    finally:
        stop.set()


if __name__ == '__main__':
    main()

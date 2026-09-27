'bounded cpu experiment reusing the completed graph pool and request shards'
import argparse
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'src')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'parent', 'prepared', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--rounds', type=int, default=1000)
    parser.add_argument('--work', type=Path)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or not 1 <= args.rounds <= 2000:
        parser.error('workers must be 1..32 and rounds 1..2000')
    os.environ.setdefault('ER_THREADS', '0.85')
    os.environ.setdefault('ER_MIN_FREE_GB', '12')
    import er.safe
    from innovation.edit_channel import run
    run(args.data, args.parent, args.prepared, args.output, args.workers, args.rounds, args.work)


if __name__ == '__main__':
    main()

'run the final reference-calibration and direct identity repair experiment'
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
os.environ.setdefault('ER_THREADS', '48')
os.environ.setdefault('ER_MIN_FREE_GB', '12')
import er.safe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'hybrid', 'friend', 'graph', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    args = ap.parse_args()
    from final_repair.pipeline import run
    run(**vars(args))


if __name__ == '__main__':
    main()

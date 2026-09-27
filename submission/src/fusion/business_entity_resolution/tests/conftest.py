import os
import sys
from pathlib import Path

os.environ.setdefault('ER_THREADS', '2')
os.environ.setdefault('ER_MIN_FREE_GB', '0')
os.environ.setdefault('POLARS_MAX_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

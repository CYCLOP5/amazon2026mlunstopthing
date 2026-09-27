'shared helpers for experiments that reuse the completed graph refinement'
import json
from pathlib import Path
import time
import numpy as np
import polars as pl

START = time.monotonic()


def log(message):
    print(f'[{(time.monotonic()-START)/60:6.1f} min] {message}', flush=True)


def read_parent(parent, split):
    parent = Path(parent)
    report = json.loads((parent/'report.json').read_text())
    weight = report['selected_head_weight']
    name = 'validation_predictions.parquet' if split == 'train' else 'test_predictions.parquet'
    frame = pl.read_parquet(parent/name)
    return frame.select('qid', 'tid', ((1-weight)*pl.col('base')+weight*pl.col('head')).alias('p'),
                        *(['y'] if 'y' in frame.columns else [])), report


def logit(values):
    p = np.clip(values, 1e-6, 1-1e-6)
    return np.log(p/(1-p))


def replace_scores(parent, updated):
    'all unverified candidate pairs retain their exact previous score'
    return parent.join(updated.select('qid', 'tid', pl.col('p').alias('_new')), on=['qid', 'tid'],
        how='left').with_columns(pl.coalesce('_new', 'p').alias('p')).drop('_new')

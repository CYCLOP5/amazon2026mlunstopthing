"""materialize the frozen direction-rule parameters from the recorded fitting pool"""

import argparse as ap
import json
from pathlib import Path as path
from tempfile import TemporaryDirectory

import polars as pl

from finish import fit_swaps, prep, sha


def fit(data, pool, cfg, out):
    spec = json.loads(cfg.read_text())
    with TemporaryDirectory(prefix='france-rule-fit-') as temporary:
        records = data / 'test' if (data / 'test').is_dir() else data
        if (records / 'ref.parquet').is_file():
            for name, expected in spec['prepared_sha256'].items():
                if sha(records / name) != expected:
                    raise ValueError('rule fitting data differs: ' + name)
        else:
            for name, info in spec['data'].items():
                if sha(records / name) != info['sha256']:
                    raise ValueError('rule fitting raw data differs: ' + name)
            records = prep(records, path(temporary) / 'records')
        refs = pl.read_parquet(records / 'ref.parquet', columns=['rid', 'nm', 'ad', 'co']).filter(pl.col('co') == 'france')
        refs = refs.select(pl.col('rid').alias('qid'), pl.col('nm').alias('rn'), pl.col('ad').alias('ra'))
        targets = pl.concat([pl.scan_parquet(records / f's{i}.parquet').select(
            pl.col('rid').alias('tid'), pl.col('nm').alias('tn'), pl.col('ad').alias('ta')) for i in (2, 3)])
        pairs = fit_swaps(refs, targets, pl.read_parquet(pool)).sort('wa', 'wb')
    result = {'algorithm': 'france-category-direction-v1', 'data': spec['data'],
              'source_pool_sha256': sha(pool), 'minimum_support': 20, 'maximum_forward_share': .7,
              'pairs': pairs.to_dicts()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'pairs': pairs.height, 'model_sha256': sha(out)}))


if __name__ == '__main__':
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=path, required=True)
    p.add_argument('--pool', type=path, required=True)
    p.add_argument('--config', type=path, required=True)
    p.add_argument('--out', type=path, required=True)
    a = p.parse_args()
    fit(a.data, a.pool, a.config, a.out)

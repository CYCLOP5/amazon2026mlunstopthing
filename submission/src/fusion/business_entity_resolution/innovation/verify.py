'four gpu workers run the frozen cross-encoder on independent input shards'
import json
import os
from pathlib import Path
import subprocess
import sys
import time

VIEWS = ('full','name','address','canonical')


def view_text(rows, side, view):
    if view == 'canonical':
        return [f'name: {n or ""}\naddress: {a or ""}\ncountry: ' for n,a in rows.select(f'cn{side}',f'ca{side}').iter_rows()]
    out = []
    for n,a,c in rows.select(f'nm{side}',f'ad{side}',f'co{side}').iter_rows():
        if view == 'name':
            a = ''
        elif view == 'address':
            n = ''
        out.append(f'name: {n or ""}\naddress: {a or ""}\ncountry: {c or ""}')
    return out


def worker(prepared, model, output, rank, workers, batch):
    import numpy as np
    import polars as pl
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(4)
    root, output, model = Path(prepared), Path(output), Path(model)
    meta = json.loads((model/'neural_metadata.json').read_text())
    if meta['parameters'] > 8_000_000_000:
        raise ValueError('Model exceeds challenge limit')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU verification requires CUDA')
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    net = AutoModelForSequenceClassification.from_pretrained(model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    print(f'GPU {rank}: loaded frozen E5 on {torch.cuda.get_device_name(0)}; max_length={meta["configuration"]["maxlen"]}',flush=True)
    jobs = [(split,f) for split in ('train','test') for f in sorted((root/split/'requests').glob('*.parquet'))]
    start = time.monotonic()
    count = 0
    for task,(split,path) in enumerate(jobs):
        if task%workers != rank:
            continue
        dest = output/split/path.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            continue
        rows = pl.read_parquet(path)
        res = rows.select('qid','tid')
        for view in VIEWS:
            a,b = view_text(rows,1,view),view_text(rows,2,view)
            scores = []
            with torch.inference_mode():
                for off in range(0,len(a),batch):
                    x = tok(a[off:off+batch],b[off:off+batch],padding=True,truncation=True,
                            max_length=meta['configuration']['maxlen'],return_tensors='pt')
                    x = {k:v.to('cuda') for k,v in x.items()}
                    scores.append(net(**x).logits[:,0].float().cpu().numpy())
            res = res.with_columns(pl.Series('ce_'+view+'_lg',np.concatenate(scores).astype(np.float32)))
        tmp = dest.with_suffix('.tmp')
        res.write_parquet(tmp)
        tmp.replace(dest)
        count += rows.height
        print(f'GPU {rank}: {split}/{path.name}; {count:,} pairs x4views; {(time.monotonic()-start)/60:.1f} min',flush=True)


def run(prepared,model,output,workers=4,batch=128):
    prepared,output = Path(prepared),Path(output)
    if not (prepared/'_SUCCESS').exists():
        raise RuntimeError('Preparation did not finish')
    output.mkdir(parents=True,exist_ok=True)
    children = []
    for rank in range(workers):
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(rank),TOKENIZERS_PARALLELISM='true',
                   OMP_NUM_THREADS='4',POLARS_MAX_THREADS='4',RAYON_NUM_THREADS='4')
        cmd = [sys.executable,str(Path(__file__).resolve().parents[1]/'scripts/run_innovation.py'),
            'verify-worker','--prepared',str(prepared),'--model',str(model),'--output',str(output),
            '--rank',str(rank),'--workers',str(workers),'--batch',str(batch)]
        children.append(subprocess.Popen(cmd,env=env))
    try:
        while children:
            for child in list(children):
                res = child.poll()
                if res is not None:
                    if res:
                        raise RuntimeError(f'GPU worker exited with code {res}')
                    children.remove(child)
            if children:
                time.sleep(2)
    except BaseException:
        for child in children:
            child.terminate()
        for child in children:
            child.wait()
        raise
    import polars as pl
    counts = {}
    for split in ('train','test'):
        exp = pl.scan_parquet(str(prepared/split/'features.parquet')).select(pl.len()).collect().item()
        got = pl.scan_parquet(str(output/split/'*.parquet')).select(pl.len()).collect().item()
        if got != exp:
            raise ValueError(f'{split}: {got} scores for {exp} requested pairs')
        counts[split] = got
    (output/'verification.json').write_text(json.dumps({'views':VIEWS,'pairs':counts,'workers':workers,
        'model_retrained':False,'max_length':'original model metadata'},indent=2))
    (output/'_SUCCESS').write_text('complete\n')

'one bounded four-gpu fine-tune and inference pass for an independent expert'
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from innovation.neural_expert import BASE_MODEL, BASE_REVISION, pair_texts


def encoder(model_path, pretrained=False):
    import torch
    from transformers import AutoModel
    class Matcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_path, add_pooling_layer=False,
                revision=BASE_REVISION if pretrained else None, attn_implementation='sdpa',
                use_safetensors=True)
            self.head = torch.nn.Linear(self.encoder.config.hidden_size, 1)

        def forward(self, batch):
            hidden = self.encoder(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
            return self.head(pooled).squeeze(-1)
    net = Matcher()
    if not pretrained:
        from safetensors.torch import load_file
        net.head.load_state_dict(load_file(str(Path(model_path)/'head.safetensors')))
    return net


def train_worker(training, output, batch=128, epochs=2, train_seconds=2400):
    import numpy as np
    import polars as pl
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from transformers import AutoTokenizer
    from safetensors.torch import save_file
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    dist.init_process_group('nccl')
    torch.manual_seed(8701)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    net = encoder(BASE_MODEL, pretrained=True)
    total_parameters = sum(p.numel() for p in net.parameters())
    if total_parameters > 8_000_000_000:
        raise ValueError('Model exceeds challenge size limit')
    if len(net.encoder.encoder.layer) != 24:
        raise ValueError('Expected the pinned 24-layer multilingual E5-large architecture')
    for parameter in net.encoder.parameters():
        parameter.requires_grad_(False)
    for layer in net.encoder.encoder.layer[-8:]:
        layer.requires_grad_(True)
    net = net.cuda(rank)
    fitted_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    train = pl.read_parquet(Path(training)/'fit.parquet').to_dicts()
    valid = pl.read_parquet(Path(training)/'valid.parquet').to_dicts() if rank == 0 else []
    sampler = torch.utils.data.distributed.DistributedSampler(
        train, num_replicas=world, rank=rank, shuffle=True, seed=8701, drop_last=True)
    def collate(indices):
        return indices
    loader = torch.utils.data.DataLoader(train, batch_size=batch, sampler=sampler,
                                        collate_fn=collate, drop_last=False)
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                  lr=2e-5, weight_decay=.01)
    total_steps = max(epochs*len(loader), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        min(1., (s+1)/max(total_steps//20, 1))*.5*(1+math.cos(math.pi*min(s, total_steps)/total_steps)))
    rng = np.random.default_rng(8701+rank)
    started = time.monotonic()
    best_loss, best_epoch, history, step, stop = float('inf'), 0, [], 0, False
    if rank == 0:
        print(f'EXPERT CUDA: {world} GPUs; parameters={total_parameters:,}; trainable={fitted_parameters:,}; '
              f'fit={len(train):,}; holdout={len(valid):,}; upper 8 layers only', flush=True)
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        net.train()
        for rows in loader:
            text = [pair_texts(row, mask_shared=bool(rng.random() < .3)) for row in rows]
            x = tok([p[0] for p in text], [p[1] for p in text], padding=True, truncation=True,
                    max_length=192, return_tensors='pt').to(rank)
            y = torch.tensor([row['y'] for row in rows], dtype=torch.float32, device=rank)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = net(x)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite expert training loss')
            loss.backward()
            if step == 0 and rank == 0:
                if any(p.grad is not None for p in net.module.encoder.embeddings.parameters()):
                    raise RuntimeError('Frozen embedding parameters received gradients')
                if net.module.head.weight.grad is None or not torch.isfinite(net.module.head.weight.grad).all():
                    raise RuntimeError('Expert classifier gradient preflight failed')
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            step += 1
            if rank == 0 and (step == 1 or step%100 == 0):
                print(f'EXPERT epoch={epoch+1} step={step}/{total_steps} loss={loss.item():.5f} '
                      f'elapsed={(time.monotonic()-started)/60:.1f}min', flush=True)
            flag = torch.tensor(int(time.monotonic()-started > train_seconds), device=rank)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item():
                stop = True
                break
        dist.barrier()
        if rank == 0:
            net.module.eval()
            losses = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for off in range(0, len(valid), batch):
                    rows = valid[off:off+batch]
                    text = [pair_texts(row) for row in rows]
                    x = tok([p[0] for p in text], [p[1] for p in text], padding=True,
                            truncation=True, max_length=192, return_tensors='pt').to(rank)
                    y = torch.tensor([row['y'] for row in rows], dtype=torch.float32, device=rank)
                    losses.extend(F.binary_cross_entropy_with_logits(net.module(x).float(), y,
                                                                     reduction='none').cpu().tolist())
            metric = float(np.mean(losses))
            if not math.isfinite(metric):
                raise RuntimeError('Nonfinite expert internal holdout loss')
            history.append({'epoch': epoch+1, 'logloss': metric, 'steps': step})
            print(f'EXPERT true-owner internal holdout: {history[-1]}', flush=True)
            if metric < best_loss:
                best_loss, best_epoch = metric, epoch+1
                dest = Path(output)/'model'
                dest.mkdir(parents=True, exist_ok=True)
                net.module.encoder.save_pretrained(dest, safe_serialization=True)
                tok.save_pretrained(dest)
                save_file({k: v.detach().cpu().contiguous() for k, v in net.module.head.state_dict().items()},
                          str(dest/'head.safetensors'))
            report = {'model': BASE_MODEL, 'revision': BASE_REVISION, 'license': 'MIT',
                'parameters': total_parameters, 'trainable_parameters': fitted_parameters,
                'trainable_encoder_layers': 8, 'epochs': history, 'selected_epoch': best_epoch,
                'budget_stop': stop, 'train_seconds_limit': train_seconds, 'max_length': 192,
                'preparation': json.loads((Path(training)/'preparation.json').read_text()),
                'text_protocol': 'address first; shared-name masking at fit only; mean-pool cross-encoder'}
            (Path(output)/'expert_report.json').write_text(json.dumps(report, indent=2))
        dist.barrier()
        if stop:
            break
    dist.destroy_process_group()


def score_worker(prepared, output, rank, workers, batch):
    import numpy as np
    import polars as pl
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    model = Path(output)/'model'
    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    net = encoder(model).to(device='cuda', dtype=torch.bfloat16).eval()
    tasks = [(split, path) for split in ('train', 'test')
             for path in sorted((Path(prepared)/split/'requests').glob('*.parquet'))]
    for index, (split, path) in enumerate(tasks):
        if index%workers != rank:
            continue
        frame = pl.read_parquet(path)
        rows = frame.to_dicts()
        res = np.empty(frame.height, dtype=np.float32)

        order = np.argsort([len(str(r['nm1']))+len(str(r['ad1']))+len(str(r['nm2']))+len(str(r['ad2']))
                            for r in rows], kind='stable')
        with torch.inference_mode():
            for off in range(0, len(order), batch):
                ids = order[off:off+batch]
                text = [pair_texts(rows[i]) for i in ids]
                x = tok([p[0] for p in text], [p[1] for p in text], padding=True,
                        truncation=True, max_length=192, return_tensors='pt').to('cuda')
                res[ids] = net(x).float().cpu().numpy()
        if not np.isfinite(res).all():
            raise RuntimeError('Nonfinite expert inference score')
        dest = Path(output)/split/path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix('.tmp')
        frame.select('qid', 'tid').with_columns(pl.Series('expert_lg', res)).write_parquet(temporary)
        temporary.replace(dest)
        print(f'EXPERT GPU{rank} scored {split}/{path.name}: {frame.height:,} pairs', flush=True)


def gpu(training, prepared, output, workers=4, batch=128, epochs=2, train_seconds=2400):
    import polars as pl
    import torch
    from transformers import AutoTokenizer, AutoModel
    if torch.cuda.device_count() != workers:
        raise RuntimeError(f'Expected {workers} GPUs, found {torch.cuda.device_count()}')
    if not (Path(training)/'_SUCCESS').exists():
        raise RuntimeError('Neural preparation incomplete')
    Path(output).mkdir(parents=True, exist_ok=True)
    print(f'GPU preflight: torch {torch.__version__}, CUDA {torch.version.cuda}, '
          f'devices {[torch.cuda.get_device_name(i) for i in range(workers)]}', flush=True)

    AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    net = AutoModel.from_pretrained(BASE_MODEL, revision=BASE_REVISION, use_safetensors=True)
    del net
    script = str(Path(__file__).resolve().parents[1]/'scripts/run_neural_expert.py')
    env = dict(os.environ, OMP_NUM_THREADS='4', POLARS_MAX_THREADS='4', RAYON_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='true')
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node', str(workers), script, 'train-worker', '--training', str(training),
        '--output', str(output), '--batch', str(batch), '--epochs', str(epochs),
        '--train-seconds', str(train_seconds)], env=env, check=True)
    children = []
    for rank in range(workers):
        children.append(subprocess.Popen([sys.executable, script, 'score-worker', '--prepared', str(prepared),
            '--output', str(output), '--rank', str(rank), '--workers', str(workers), '--batch', str(batch)],
            env=dict(env, CUDA_VISIBLE_DEVICES=str(rank))))
    try:
        while children:
            for child in list(children):
                code = child.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'Expert scoring worker failed: {code}')
                    children.remove(child)
            if children:
                time.sleep(2)
    except BaseException:
        for child in children:
            child.terminate()
        for child in children:
            child.wait()
        raise
    coverage = {}
    for split in ('train', 'test'):
        exp = pl.read_parquet(Path(prepared)/split/'features.parquet', columns=['qid', 'tid'])
        got = pl.read_parquet(str(Path(output)/split/'*.parquet'))
        if (exp.height != got.height or got.select('qid', 'tid').n_unique() != got.height or
            exp.join(got.select('qid', 'tid'), on=['qid', 'tid'], how='anti').height or
            not got['expert_lg'].is_finite().all()):
            raise RuntimeError(f'{split}: expert coverage failed')
        coverage[split] = got.height
    report_path = Path(output)/'expert_report.json'
    report = json.loads(report_path.read_text())
    report['inference_coverage'] = coverage
    report_path.write_text(json.dumps(report, indent=2))
    (Path(output)/'_SUCCESS').write_text('complete\n')
    print(f'Expert inference coverage PASS: {coverage}', flush=True)

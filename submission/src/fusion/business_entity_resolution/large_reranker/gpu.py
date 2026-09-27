'budgeted lora adaptation and independent replicated inference on azure gpus'
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from large_reranker.protocol import MODEL, REVISION, MAX_LENGTH, MAX_PARAMETERS
from large_reranker.protocol import encode_rows, answer_tokens, worker_tasks


def log(message):
    print(message, flush=True)


def load_base(path):
    import torch
    from transformers import AutoModelForCausalLM
    net = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', use_safetensors=True)
    count = sum(p.numel() for p in net.parameters())
    if count > MAX_PARAMETERS:
        raise ValueError(f'Model exceeds parameter limit: {count:,}')
    net.config.use_cache = False
    return net, count


def tensor_batch(tokenizer, ids, device):
    return tokenizer.pad({'input_ids': ids}, padding=True, return_tensors='pt').to(device)


def logits(net, batch, yes, no):

    values = net(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :]
    return values[:, yes].float() - values[:, no].float()


def preflight():
    'Exercise PEFT/Qwen CUDA training, last-token scoring and model replay'
    import tempfile
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(92)
    config = Qwen3Config(vocab_size=128,hidden_size=64,intermediate_size=128,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=32,
        tie_word_embeddings=True,attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    base = Qwen3ForCausalLM(config).to(device='cuda',dtype=torch.bfloat16)
    with tempfile.TemporaryDirectory(prefix='qwen_preflight_') as work:
        base.save_pretrained(Path(work)/'base')
        net = get_peft_model(base,LoraConfig(task_type=TaskType.CAUSAL_LM,r=4,
            lora_alpha=8,target_modules=['q_proj','v_proj']))
        net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        net.enable_input_require_grads()
        x = {'input_ids':torch.randint(0,128,(4,12),device='cuda'),
             'attention_mask':torch.ones(4,12,device='cuda',dtype=torch.long)}
        y = torch.tensor([0.,1.,0.,1.],device='cuda')
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.01)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits(net,x,3,4),y)
        loss.backward()
        if not any(p.grad is not None and torch.any(p.grad != 0) for p in net.parameters() if p.requires_grad):
            raise RuntimeError('PEFT gradient preflight failed')
        optimizer.step()
        net.eval()
        with torch.inference_mode():
            expected = logits(net,x,3,4)
        net.save_pretrained(Path(work)/'adapter')
        replay_base = Qwen3ForCausalLM.from_pretrained(Path(work)/'base',torch_dtype=torch.bfloat16).cuda()
        replay = PeftModel.from_pretrained(replay_base,Path(work)/'adapter').eval()
        with torch.inference_mode():
            torch.testing.assert_close(logits(replay,x,3,4),expected,atol=.015,rtol=.015)
            merged = replay.merge_and_unload().eval()
            torch.testing.assert_close(logits(merged,x,3,4),expected,atol=.03,rtol=.03)
    log('LARGE actual CUDA Qwen/LoRA gradient, adapter replay and merge preflight PASS')


def train_worker(prepared, output, batch=8, epochs=2, train_seconds=2400):
    import datetime
    import numpy as np
    import polars as pl
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from transformers import AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    torch.manual_seed(1927)
    root = Path(output)
    tok = AutoTokenizer.from_pretrained(root/'base_model', local_files_only=True, padding_side='left')
    tok.pad_token = tok.eos_token
    yes, no = answer_tokens(tok)
    base, count = load_base(root/'base_model')
    net = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16,
        lora_alpha=32, lora_dropout=.05, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    if sum(p.numel() for p in net.parameters()) > MAX_PARAMETERS:
        raise ValueError('Model plus adapters exceeds parameter limit')
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    net.enable_input_require_grads()
    net = net.to(rank)
    fitted = sum(p.numel() for p in net.parameters() if p.requires_grad)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], broadcast_buffers=False)
    train = pl.read_parquet(Path(prepared)/'training/fit.parquet').to_dicts()
    valid = pl.read_parquet(Path(prepared)/'training/valid.parquet').to_dicts()
    if len(train) < world * batch or not valid:
        raise ValueError('Insufficient isolated training/validation examples')

    train_ids = encode_rows(tok, train)
    valid_ids = encode_rows(tok, valid)
    sampler = torch.utils.data.distributed.DistributedSampler(range(len(train)),
        num_replicas=world, rank=rank, shuffle=True, seed=1927, drop_last=True)
    loader = torch.utils.data.DataLoader(list(range(len(train))), batch_size=batch,
        sampler=sampler, drop_last=False)
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
        lr=8e-5, weight_decay=.01)
    total_steps = max(epochs * len(loader), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        min(1., (s+1)/max(total_steps//20, 1)) * .5 * (1+math.cos(math.pi*min(s,total_steps)/total_steps)))
    best_loss, selected_step, history = float('inf'), 0, []
    step, stopped = 0, False
    started = time.monotonic()

    def evaluate_and_save():
        nonlocal best_loss, selected_step
        net.module.eval()
        total = torch.zeros(2, device=rank, dtype=torch.float64)
        indices = list(range(rank, len(valid), world))
        with torch.inference_mode():
            for start in range(0, len(indices), batch*2):
                ix = indices[start:start+batch*2]
                x = tensor_batch(tok, [valid_ids[i] for i in ix], rank)
                y = torch.tensor([valid[i]['y'] for i in ix], device=rank, dtype=torch.float32)
                loss = F.binary_cross_entropy_with_logits(logits(net.module,x,yes,no), y, reduction='sum')
                total[0] += loss.double()
                total[1] += len(ix)
        dist.all_reduce(total)
        value = (total[0]/total[1]).item()
        if not math.isfinite(value):
            raise RuntimeError('Nonfinite isolated neural validation loss')
        history.append({'step': step, 'logloss': value, 'elapsed_seconds': time.monotonic()-started})
        if value < best_loss:
            best_loss, selected_step = value, step
            if rank == 0:
                net.module.save_pretrained(root/'adapter', safe_serialization=True)
                tok.save_pretrained(root/'adapter')
        if rank == 0:
            log(f'LARGE isolated owner holdout {history[-1]}; selected_step={selected_step}')
            report = {'model': MODEL, 'revision': REVISION, 'license': 'Apache-2.0',
                'parameters': count, 'trainable_parameters': fitted, 'adapter_rank': 16,
                'selected_step': selected_step, 'selected_is_frozen': selected_step == 0,
                'history': history, 'fit_pairs': len(train), 'valid_pairs': len(valid),
                'train_seconds_limit': train_seconds, 'budget_stop': stopped, 'max_length': MAX_LENGTH,
                'selection': 'isolated fold-2 owner holdout logloss; includes frozen step-zero control',
                'source': 'challenge labels only; no supplied test labels or external examples'}
            (root/'training_report.json').write_text(json.dumps(report, indent=2))
        dist.barrier()
        net.train()

    if rank == 0:
        log(f'LARGE {world} GPUs; base={count:,}, trainable={fitted:,}; train={len(train):,}; valid={len(valid):,}')
    evaluate_and_save()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for idx in loader:
            ix = idx.tolist()
            x = tensor_batch(tok, [train_ids[i] for i in ix], rank)
            y = torch.tensor([train[i]['y'] for i in ix], device=rank, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(logits(net,x,yes,no), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite adapter loss')
            loss.backward()
            if step == 0:
                grads = [p.grad for n,p in net.named_parameters() if 'lora_B' in n and p.grad is not None]
                if not grads or not any(torch.isfinite(g).all() and torch.any(g != 0) for g in grads):
                    raise RuntimeError('Adapter gradient check failed')
                if any(p.grad is not None for p in net.parameters() if not p.requires_grad):
                    raise RuntimeError('Frozen base received gradients')
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            step += 1
            flag = torch.tensor(int(time.monotonic()-started >= train_seconds), device=rank)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stopped = bool(flag.item())
            if rank == 0 and (step == 1 or step % 50 == 0):
                log(f'LARGE epoch={epoch+1} step={step}/{total_steps} loss={loss.item():.5f} elapsed={(time.monotonic()-started)/60:.1f}min')
            if step % 250 == 0 or stopped:
                evaluate_and_save()
            if stopped:
                break
        if not history or history[-1]['step'] != step:
            evaluate_and_save()
        if stopped:
            break
    if rank == 0:
        (root/'_TRAIN_SUCCESS').write_text('complete\n')
    dist.barrier()
    dist.destroy_process_group()


def train(prepared, output, workers=4, batch=8, epochs=2, train_seconds=2400):
    import torch
    from huggingface_hub import snapshot_download
    if torch.cuda.device_count() != workers:
        raise RuntimeError(f'Expected {workers} CUDA devices')
    if not (Path(prepared)/'_SUCCESS').is_file():
        raise RuntimeError('Pair/training preparation incomplete')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    log(f'LARGE CUDA {torch.__version__} {torch.version.cuda}; GPUs={[torch.cuda.get_device_name(i) for i in range(workers)]}')
    script = Path(__file__).resolve().parents[1]/'scripts/run_large_reranker.py'
    subprocess.run([sys.executable,str(script),'preflight'],check=True,
                   env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'))

    snapshot_download(MODEL, revision=REVISION, local_dir=root/'base_model',
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', 'LICENSE', 'README.md'])
    env = dict(os.environ, OMP_NUM_THREADS='4', POLARS_MAX_THREADS='4',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
    subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node', str(workers), str(script), 'train-worker', '--prepared', str(prepared),
        '--output', str(output), '--batch', str(batch), '--epochs', str(epochs),
        '--train-seconds', str(train_seconds)], env=env, check=True)
    if not (root/'_TRAIN_SUCCESS').is_file():
        raise RuntimeError('Training completion marker missing')


def score_worker(prepared, trained, output, rank, workers, batch=48, shard_index=0, shards=2):
    import numpy as np
    import polars as pl
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    torch.set_num_threads(4)
    root = Path(trained)
    tok = AutoTokenizer.from_pretrained(root/'adapter', local_files_only=True, padding_side='left')
    tok.pad_token = tok.eos_token
    yes, no = answer_tokens(tok)
    base, _ = load_base(root/'base_model')
    net = PeftModel.from_pretrained(base, root/'adapter', local_files_only=True)
    if sum(p.numel() for p in net.parameters()) > MAX_PARAMETERS:
        raise ValueError('Loaded model plus adapter exceeds parameter limit')
    net = net.merge_and_unload().to('cuda').eval()
    tasks = [(split,p) for split in ('train','test')
             for p in sorted((Path(prepared)/split/'requests').glob('*.parquet'))]
    started, processed = time.monotonic(), 0
    for split, path in worker_tasks(tasks,rank,workers,shard_index,shards):
        frame = pl.read_parquet(path)
        ids = encode_rows(tok, frame.to_dicts())
        order = np.argsort([len(x) for x in ids], kind='stable')
        result = np.empty(frame.height, dtype=np.float32)
        offset, size = 0, batch
        with torch.inference_mode():
            while offset < len(order):
                ix = order[offset:offset+size]
                try:
                    x = tensor_batch(tok, [ids[i] for i in ix], 'cuda')
                    result[ix] = logits(net,x,yes,no).cpu().numpy()
                    del x
                    offset += len(ix)
                except torch.cuda.OutOfMemoryError:
                    if size == 1:
                        raise

                    if 'x' in locals():
                        del x
                    torch.cuda.empty_cache()
                    size = max(1,size//2)
                    log(f'LARGE GPU{rank} reduced scoring batch to {size}')
        if not np.isfinite(result).all():
            raise RuntimeError('Nonfinite larger-model score')
        dest = Path(output)/split/path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix('.tmp')
        frame.select('qid','tid').with_columns(pl.Series('large_lg',result)).write_parquet(tmp)
        tmp.replace(dest)
        processed += frame.height
        log(f'LARGE shard{shard_index}/GPU{rank} {split}/{path.name} {frame.height:,} pairs; cumulative {processed/max(time.monotonic()-started,1):.1f} pairs/sec')


def score(prepared, trained, output, workers=4, batch=48, shard_index=0, shards=2):
    import torch
    if torch.cuda.device_count() != workers or not (Path(trained)/'_TRAIN_SUCCESS').is_file():
        raise RuntimeError('CUDA device count or training completion check failed')
    Path(output).mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1]/'scripts/run_large_reranker.py'
    children = []
    try:
        for rank in range(workers):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='4',
                       POLARS_MAX_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
            children.append(subprocess.Popen([sys.executable,str(script),'score-worker',
                '--prepared',str(prepared),'--trained',str(trained),'--output',str(output),
                '--rank',str(rank),'--workers',str(workers),'--batch',str(batch),
                '--shard-index',str(shard_index),'--shards',str(shards)],env=env))
        while children:
            for child in list(children):
                code = child.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f'Larger-model inference worker exited {code}')
                    children.remove(child)
            if children:
                time.sleep(2)
    except BaseException:
        for child in children:
            child.terminate()
        for child in children:
            child.wait()
        raise
    (Path(output)/'shard.json').write_text(json.dumps({'shard_index':shard_index,'shards':shards}))
    (Path(output)/'_SUCCESS').write_text('complete\n')


def gather(prepared, roots, output):
    'validate disjoint distributed outputs before the downstream cpu fit'
    import polars as pl
    dest = Path(output)
    dest.mkdir(parents=True,exist_ok=True)
    manifests = []
    for root in roots:
        if not (Path(root)/'_SUCCESS').is_file():
            raise RuntimeError('Incomplete inference shard')
        manifests.append(json.loads((Path(root)/'shard.json').read_text()))
    n = manifests[0]['shards']
    if any(m['shards'] != n for m in manifests) or sorted(m['shard_index'] for m in manifests) != list(range(n)):
        raise ValueError('Inference shard manifest coverage mismatch')
    coverage = {}
    for split in ('train','test'):
        expected = pl.read_parquet(Path(prepared)/split/'features.parquet',columns=['qid','tid'])
        paths = [p for root in roots for p in sorted((Path(root)/split).glob('*.parquet'))]
        if not paths:
            raise ValueError(f'No {split} score shards')
        got = pl.concat([pl.read_parquet(p) for p in paths])
        if (got.height != expected.height or got.select('qid','tid').n_unique() != got.height or
            expected.join(got.select('qid','tid'),on=['qid','tid'],how='anti').height or
            got['large_lg'].null_count() or not got['large_lg'].is_finite().all()):
            raise ValueError(f'{split}: incomplete/duplicate/nonfinite neural coverage')
        (dest/split).mkdir(exist_ok=True)
        got.write_parquet(dest/split/'part_00000.parquet')
        coverage[split] = got.height
    (dest/'coverage.json').write_text(json.dumps(coverage,indent=2))
    (dest/'_SUCCESS').write_text('complete\n')
    return dest

import argparse as ap
import hashlib as hh
import json
import os
import random
import tempfile as tf
from pathlib import Path as path

import numpy as np
import polars as pl


ver = 1
mod0 = "intfloat/multilingual-e5-base"
rev0 = "d128750597153bb5987e10b1c3493a34e5a4502a"
src0 = path(__file__).resolve().parents[1] / "reports/model_sources.json"
fmt = "name: {nm}\naddress: {ad}\ncountry: {co}"


def _sha(p):
    h = hh.sha256()
    with path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _json(p):
    try:
        return json.loads(path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"invalid json {p}") from e


def _write(p, z):
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(z, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    t.replace(p)


def _pq(d, p):
    p = path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + ".tmp")
    d.write_parquet(t, compression="zstd")
    t.replace(p)


def _relative(p, base):
    return os.path.relpath(path(p).resolve(), path(base).resolve())


def _need(d, cs, what):
    miss = set(cs) - set(d.columns)
    if miss:
        raise ValueError(f"{what} missing {sorted(miss)}")


def text(d):
    _need(d, {"nm", "ad", "co"}, "text rows")
    return [fmt.format(nm=str(n or ""), ad=str(a or ""), co=str(c or ""))
            for n, a, c in d.select("nm", "ad", "co").iter_rows()]


def _parts(run):
    m = _json(run / "metrics.json")
    ps = m.get("parts")
    if not isinstance(ps, list) or not ps:
        raise ValueError("run has no candidate parts")
    out = []
    for x in ps:
        p = (run / str(x)).resolve()
        if (not isinstance(x, str) or p.parent != run.resolve() or p.suffix != ".parquet" or
                not p.is_file()):
            raise ValueError(f"invalid candidate part {x}")
        out.append(p)
    if len(set(out)) != len(out):
        raise ValueError("repeated candidate part")
    return m, out


def _run(data, run):
    data, run = path(data).resolve(), path(run).resolve()
    m, ps = _parts(run)
    fold, co = m.get("fold"), m.get("country")
    if fold not in (0, 1, 2) or not isinstance(co, str) or not co:
        raise ValueError("run needs one valid country and fold")
    a, q = pl.read_parquet(run / "anchors.parquet"), pl.read_parquet(run / "queries.parquet")
    _need(a, {"rid", "co", "fold", "deg", "nm", "ad"}, "anchors")
    _need(q, {"rid", "co", "sr", "own", "nm", "ad"}, "queries")
    if a.is_empty() or q.is_empty() or a["rid"].n_unique() != len(a) or q["rid"].n_unique() != len(q):
        raise ValueError("empty or duplicate run ids")
    if a.filter((pl.col("co") != co) | (pl.col("fold") != fold)).height:
        raise ValueError("anchor fold or country mismatch")
    if q.filter((pl.col("co") != co) | ~pl.col("sr").is_in([2, 3])).height:
        raise ValueError("query country or source mismatch")
    ref = pl.read_parquet(data / "train/ref.parquet")
    _need(ref, {"rid", "nm", "ad", "co", "fold", "deg"}, "prepared references")
    current = ref.filter(pl.col("co") == co)
    pool = current.filter(pl.col("fold") == 2) if fold == 2 else current
    if pool.is_empty():
        raise ValueError("empty reference pool")
    ac = a.select("rid", "co", "fold", "deg", "nm", "ad").join(
        current.select("rid", "co", "fold", "deg", "nm", "ad"), on="rid", how="left", suffix="_data")
    if ac["co_data"].null_count() or ac.filter(
            (pl.col("co") != pl.col("co_data")) | (pl.col("fold") != pl.col("fold_data")) |
            (pl.col("deg") != pl.col("deg_data")) | (pl.col("nm") != pl.col("nm_data")) |
            (pl.col("ad") != pl.col("ad_data"))).height:
        raise ValueError("anchors are stale or not owned by this data")
    auth = []
    for sr in (2, 3):
        ids = q.filter(pl.col("sr") == sr)["rid"].to_list()
        if ids:
            auth.append(pl.scan_parquet(data / "train" / f"s{sr}.parquet").filter(
                pl.col("rid").is_in(ids)).select("rid", "sr", "own", "co", "nm", "ad").collect(engine="streaming"))
    curq = pl.concat(auth) if auth else q.head(0)
    qc = q.select("rid", "sr", "own", "co", "nm", "ad").join(
        curq, on="rid", how="left", suffix="_data")
    if len(curq) != len(q) or qc["sr_data"].null_count() or qc.filter(
            (pl.col("sr") != pl.col("sr_data")) | (pl.col("own") != pl.col("own_data")) |
            (pl.col("co") != pl.col("co_data")) | (pl.col("nm") != pl.col("nm_data")) |
            (pl.col("ad") != pl.col("ad_data"))).height:
        raise ValueError("queries are stale or ownership disagrees with data")
    sel = a["rid"].implode()
    if q.filter((pl.col("own") >= 0) & ~pl.col("own").is_in(sel)).height:
        raise ValueError("query belongs to an unselected anchor")
    got = q.filter(pl.col("own").is_in(sel)).group_by("own").len().rename({"own": "rid", "len": "n"})
    if a.select("rid", "deg").join(got, on="rid", how="left").with_columns(pl.col("n").fill_null(0)).filter(
            pl.col("deg") != pl.col("n")).height:
        raise ValueError("run omits a selected positive query")
    return {"data": data, "dir": run, "metrics": m, "parts": ps, "fold": fold, "country": co,
            "anchors": a, "queries": q, "pool": pool}


def _candidates(r, dense=None):
    fs = []
    seen = set()
    for j, p in enumerate(r["parts"] + ([path(dense).resolve()] if dense else [])):
        d = pl.read_parquet(p)
        _need(d, {"tid", "qid"}, f"candidate {p}")
        if d.select("tid", "qid").n_unique() != len(d):
            raise ValueError(f"duplicate candidate pair {p}")
        if j < len(r["parts"]):
            tids = set(d["tid"].to_list())
            if tids & seen:
                raise ValueError(f"query split across candidate parts {p}")
            seen.update(tids)
        d = d.with_row_index("order").with_columns(pl.lit(str(p)).alias("source"))
        fs.append(d)
    d = pl.concat(fs, how="diagonal_relaxed").with_row_index("order_all")
    d = d.unique(subset=["tid", "qid"], keep="first", maintain_order=True)
    q = r["queries"].select(pl.col("rid").alias("tid"), pl.col("own").alias("own_data"),
                              pl.col("sr").alias("sr_data"), "nm", "ad", "co")
    a = r["pool"].select(pl.col("rid").alias("qid"), pl.col("nm").alias("anm"),
                           pl.col("ad").alias("aad"), pl.col("co").alias("aco"))
    d = d.join(q, on="tid", how="left", validate="m:1").join(a, on="qid", how="left", validate="m:1")
    if d["own_data"].null_count() or d["anm"].null_count() or d.filter(pl.col("co") != pl.col("aco")).height:
        raise ValueError("candidate contains a stale id or cross-country pair")
    y = (pl.col("qid").cast(pl.Int64) == pl.col("own_data")).cast(pl.UInt8)
    if "own" in d.columns and d.filter(pl.col("own") != pl.col("own_data")).height:
        raise ValueError("candidate ownership disagrees with current data")
    if "sr" in d.columns and d.filter(pl.col("sr") != pl.col("sr_data")).height:
        raise ValueError("candidate source disagrees with current data")
    if "y" in d.columns and d.filter(pl.col("y").cast(pl.UInt8) != y).height:
        raise ValueError("candidate label disagrees with current data")
    return d.with_columns(y.alias("y")).drop("own_data", "sr_data", *[x for x in ("own", "sr") if x in d.columns])


def _score(d):
    cs = [x for x in ("ns", "ads", "en", "ea", "ds") if x in d.columns]
    if not cs:
        return np.zeros(len(d), dtype=np.float32)
    z = np.zeros(len(d), dtype=np.float32)
    for x in cs:
        v = d[x].fill_null(0).cast(pl.Float32).to_numpy()
        z = np.maximum(z, np.nan_to_num(v, nan=0.0))
    return z


def _select_train(d, r, hard, rnd, seed):
    rows = []
    miss = r["queries"].filter(pl.col("own") >= 0).join(
        d.select("tid", "qid"), left_on=["rid", "own"], right_on=["tid", "qid"], how="anti")
    if len(miss):
        a = r["pool"].select(pl.col("rid").alias("qid"), pl.col("nm").alias("anm"),
                               pl.col("ad").alias("aad"), pl.col("co").alias("aco"))
        x = miss.select(pl.col("rid").alias("tid"), "own", "sr", "nm", "ad", "co").join(
            a, left_on="own", right_on="qid", how="left").with_columns(pl.col("own").cast(pl.UInt32).alias("qid"))
        if x["anm"].null_count():
            raise ValueError("positive owner is outside fold2 pool")
        x = x.with_columns(pl.lit(1, dtype=pl.UInt8).alias("y"), pl.lit("injected_positive").alias("source"))
        d = pl.concat([d.drop("order_all"), x], how="diagonal_relaxed").with_row_index("order_all")
    d = d.with_columns(pl.Series("hardness", _score(d)))
    for _, g in d.group_by("tid", maintain_order=True):
        pos = g.filter(pl.col("y") == 1)
        neg = g.filter(pl.col("y") == 0).sort(["hardness", "order_all"], descending=[True, False])
        take = neg.head(hard)
        rest = neg.slice(len(take))
        if rnd and len(rest):
            ix = list(range(len(rest)))
            random.Random(seed + int(g["tid"][0])).shuffle(ix)
            take = pl.concat([take, rest.gather(pl.Series(ix[:min(rnd, len(ix))], dtype=pl.UInt32))])
        rows.extend([pos, take])
    return pl.concat(rows, how="diagonal_relaxed") if rows else d.head(0)


def prepare(data, run, out, dense=None, hard=8, rnd=2, seed=42):
    if hard < 0 or rnd < 0:
        raise ValueError("negative counts must be nonnegative")
    r = _run(data, run)
    d = _candidates(r, dense)
    original = len(d)
    if r["fold"] == 2:
        d = _select_train(d, r, hard, rnd, seed)
    d = d.select("tid", "qid", "y", "nm", "ad", "co", "anm", "aad", "aco").with_columns(
        pl.Series("text_a", text(d.select(pl.col("anm").alias("nm"), pl.col("aad").alias("ad"), pl.col("aco").alias("co")))),
        pl.Series("text_b", text(d.select("nm", "ad", "co"))))
    d = d.rename({"y": "label"}).select("text_a", "text_b", "label", "qid", "tid")
    out = path(out).resolve()
    _pq(d, out / "pairs.parquet")
    src = {"data_meta_sha256": _sha(r["data"] / "meta.json"), "run_metrics_sha256": _sha(r["dir"] / "metrics.json"),
           "anchors_sha256": _sha(r["dir"] / "anchors.parquet"), "queries_sha256": _sha(r["dir"] / "queries.parquet"),
           "parts": [{"name": p.name, "sha256": _sha(p)} for p in r["parts"]]}
    if dense:
        src["dense"] = {"path": _relative(dense, out), "sha256": _sha(dense)}
    m = {"version": ver, "kind": "neural-pairs", "fold": r["fold"], "country": r["country"], "seed": seed,
         "selection": "all_candidates" if r["fold"] != 2 else {"hard": hard, "random": rnd, "missed_positive": "included"},
         "source": src, "input_pairs": original, "pairs": len(d), "positive_pairs": int(d["label"].sum()),
         "pairs_sha256": _sha(out / "pairs.parquet"), "text_format": fmt}
    _write(out / "manifest.json", m)
    return m


def _prepared(p, fold):
    p = path(p).resolve()
    p = p / "pairs.parquet" if p.is_dir() else p
    m = _json(p.parent / "manifest.json")
    if m.get("kind") != "neural-pairs" or m.get("fold") != fold or m.get("pairs_sha256") != _sha(p):
        raise ValueError(f"{p} is not a current fold{fold} neural pair file")
    d = pl.read_parquet(p)
    _need(d, {"text_a", "text_b", "label", "qid", "tid"}, "pair data")
    if d.is_empty() or d.filter(~pl.col("label").is_in([0, 1])).height:
        raise ValueError("invalid pair labels")
    return d, m, p


def source(p, model, rev):
    for x in _json(p):
        if x.get("model") == model and x.get("revision") == rev:
            if str(x.get("license", "")).lower() not in {"mit", "apache-2.0", "apache2", "apache 2.0"}:
                raise ValueError("model source license is not mit or apache-2.0")
            n = x.get("parameters")
            if n is not None and int(n) > 8_000_000_000:
                raise ValueError("model source exceeds 8b parameters")
            return x
    raise ValueError("model revision is absent from source metadata")


class pairs:
    def __init__(self, d):
        self.a = d["text_a"].to_list()
        self.b = d["text_b"].to_list()
        self.y = d["label"].to_numpy()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"text_a": self.a[i], "text_b": self.b[i], "labels": float(self.y[i])}


class collate:
    def __init__(self, tok, maxlen=384):
        self.tok, self.maxlen = tok, maxlen

    def __call__(self, xs):
        import torch
        ys = [x["labels"] for x in xs]
        z = self.tok([x["text_a"] for x in xs], [x["text_b"] for x in xs],
                     truncation=True, max_length=self.maxlen, padding=True, return_tensors="pt")
        z["labels"] = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
        return z


def _device(dev):
    import torch
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but unavailable")
    if dev not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return dev


def _trainer():
    try:
        import accelerate
        from transformers import Trainer, TrainingArguments
    except ImportError as e:
        raise RuntimeError("training requires accelerate>=0.26.0 in the neural environment") from e
    return Trainer, TrainingArguments, accelerate


def train(trains, vals, out, model=mod0, revision=rev0, sources=src0, batch=8, epochs=1.0,
          lr=2e-5, maxlen=384, seed=42, device="auto", resume=None):
    if batch < 1 or epochs <= 0 or lr <= 0 or maxlen < 8:
        raise ValueError("invalid training options")
    if not trains or not vals:
        raise ValueError("fold2 training and fold0 validation files are required")
    if resume and not all((path(resume) / x).is_file() for x in ("trainer_state.json", "optimizer.pt", "scheduler.pt")):
        raise ValueError("resume requires a complete trainer checkpoint including optimizer and scheduler")
    tx = [_prepared(x, 2) for x in trains]
    vx = [_prepared(x, 0) for x in vals]
    td, vd = [x[0] for x in tx], [x[0] for x in vx]
    tr, va = pl.concat(td), pl.concat(vd)
    if tr["label"].n_unique() != 2:
        raise ValueError("training pairs need both labels")
    si = source(sources, model, revision)
    Trainer, args, ac = _trainer()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, __version__ as tfver, set_seed
    dev = _device(device)
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True, trust_remote_code=False)
    net = AutoModelForSequenceClassification.from_pretrained(model, revision=revision, num_labels=1,
                                                              problem_type="multi_label_classification",
                                                              trust_remote_code=False)
    net.config.num_labels = 1
    net.config.problem_type = "multi_label_classification"
    n = sum(x.numel() for x in net.parameters())
    if n > 8_000_000_000:
        raise ValueError("loaded model exceeds 8b parameters")
    fp16 = dev == "cuda" and not torch.cuda.is_bf16_supported()
    bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()
    out = path(out).resolve()
    kw = {"output_dir": str(out), "per_device_train_batch_size": batch, "per_device_eval_batch_size": batch,
          "num_train_epochs": epochs, "learning_rate": lr, "warmup_ratio": 0.1, "weight_decay": 0.01,
          "seed": seed, "data_seed": seed, "save_strategy": "steps", "save_steps": 500, "save_total_limit": 2,
          "eval_strategy": "epoch", "logging_steps": 50, "report_to": [],
          "remove_unused_columns": False, "fp16": fp16, "bf16": bf16, "save_safetensors": True,
          "dataloader_num_workers": 2 if dev == "cuda" else 0, "dataloader_pin_memory": dev == "cuda"}
    if dev == "cpu":
        kw["use_cpu"] = True
    ta = args(**kw)
    ds, ev = pairs(tr), pairs(va)
    t = Trainer(model=net, args=ta, train_dataset=ds, eval_dataset=ev, data_collator=collate(tok, maxlen))
    t.train(resume_from_checkpoint=str(resume) if resume else None)
    t.save_model()
    t.save_state()
    tok.save_pretrained(out)
    m = {"version": ver, "kind": "neural-cross-encoder", "model": model, "revision": revision, "source": si,
         "parameters": n, "problem_type": "multi_label_classification", "loss": "bce_with_logits",
         "configuration": {"batch": batch, "epochs": epochs, "lr": lr, "maxlen": maxlen, "seed": seed,
                           "device": dev, "precision": "bf16" if bf16 else "fp16" if fp16 else "fp32",
                           "resume": _relative(resume, out) if resume else None, "accelerate": getattr(ac, "__version__", None),
                           "torch": torch.__version__, "transformers": tfver, "sources_sha256": _sha(sources)},
          "inputs": {"train": [{"path": _relative(x[2], out), "pairs_sha256": x[1]["pairs_sha256"]} for x in tx],
                     "validation": [{"path": _relative(x[2], out), "pairs_sha256": x[1]["pairs_sha256"]} for x in vx]},
          "validation": {"label": "candidate pair validation, not official macro f0.5", "pairs": len(ev),
                        "total_pairs": len(va)}}
    _write(out / "neural_metadata.json", m)
    return m


def member_paths(root, meta):
    import re
    root = path(root)
    members = meta.get("members", [])
    if not isinstance(members, list) or not members:
        raise ValueError("ensemble has no members")
    result, names = [], set()
    for member in members:
        name, directory = member.get("name"), member.get("directory")
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names or
                not isinstance(directory, str) or path(directory).name != directory):
            raise ValueError("invalid ensemble member")
        target = root / directory
        if _sha(target / "neural_metadata.json") != member.get("metadata_sha256"):
            raise ValueError("ensemble member metadata changed")
        if _json(target / "neural_metadata.json").get("architecture") == "neural-ensemble-v1":
            raise ValueError("nested neural ensembles are unsupported")
        result.append((name, target))
        names.add(name)
    return result


def load(model_dir, device="auto"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    d = path(model_dir).resolve()
    m = _json(d / "neural_metadata.json")
    if m.get("problem_type") != "multi_label_classification" or m.get("parameters", 8_000_000_001) > 8_000_000_000:
        raise ValueError("invalid neural model metadata")
    dev = _device(device)
    if m.get("architecture") == "neural-ensemble-v1":
        children = {name: load(folder, dev) for name, folder in member_paths(d, m)}
        if sum(x["metadata"]["parameters"] for x in children.values()) != m["parameters"]:
            raise ValueError("ensemble parameter count mismatch")
        return {"members": children, "device": dev, "maxlen": m["configuration"]["maxlen"], "metadata": m}
    tok = AutoTokenizer.from_pretrained(d, local_files_only=True)
    if m.get("architecture") == "mean-pooled-cross-encoder-v1":
        from ce import load as load_mean
        net = load_mean(d)
        if str(dev).startswith("cuda"):
            net.half()
    else:
        net = AutoModelForSequenceClassification.from_pretrained(d, local_files_only=True)
    net.to(dev).eval()
    return {"model": net, "tokenizer": tok, "device": dev, "maxlen": m["configuration"]["maxlen"], "metadata": m}


def predict(bundle, d, batch=32):
    if batch < 1:
        raise ValueError("positive batch required")
    _need(d, {"text_a", "text_b"}, "score pairs")
    if "members" in bundle:
        return aggregate(predict_members(bundle, d, batch))
    import torch
    out = []
    net, tok, dev, maxlen = (bundle[x] for x in ("model", "tokenizer", "device", "maxlen"))
    dt = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast("cuda", dtype=dt, enabled=dev == "cuda"):
        for lo in range(0, len(d), batch):
            x = d.slice(lo, batch)
            z = tok(x["text_a"].to_list(), x["text_b"].to_list(), truncation=True, max_length=maxlen,
                    padding=True, return_tensors="pt")
            z = {k: v.to(dev) for k, v in z.items()}
            out.append(torch.sigmoid(net(**z).logits[:, 0].float()).cpu().numpy())
    return np.concatenate(out).astype(np.float32, copy=False) if out else np.empty(0, np.float32)


def predict_members(bundle, d, batch=32):
    if "members" not in bundle:
        raise ValueError("per-member scoring requires an ensemble")
    scores = {}
    for name, child in bundle["members"].items():
        upper = next(m.get("upper_gate", 1.) for m in bundle["metadata"]["members"] if m["name"] == name)
        if not isinstance(upper, (int, float)) or not 0 < upper <= 1:
            raise ValueError("invalid selective neural threshold")
        if upper < 1:
            if "gate_prob" not in d.columns:
                raise ValueError("selective ensemble requires gate probabilities")
            value = d["gate_prob"].to_numpy().astype(np.float32).copy()
            use = value < upper
            if use.any():
                value[use] = predict(child, d.filter(pl.Series(use)), batch)
        else:
            value = predict(child, d, batch)
        scores["np_" + name] = value
    return scores


def aggregate(values):
    if not values:
        raise ValueError("empty neural score ensemble")
    columns = [np.asarray(x, dtype=np.float64) for x in values.values()]
    if any(x.ndim != 1 or x.shape != columns[0].shape or not np.isfinite(x).all() or ((x < 0) | (x > 1)).any() for x in columns):
        raise ValueError("invalid neural member scores")
    logits = np.mean([np.log(np.clip(x, 1e-6, 1 - 1e-6) / np.clip(1 - x, 1e-6, 1)) for x in columns], axis=0)
    return (1 / (1 + np.exp(-logits))).astype(np.float32)


def score(pair_file, model_dir, out, batch=32, device="auto"):
    pair_file = path(pair_file).resolve()
    pair_file = pair_file / "pairs.parquet" if pair_file.is_dir() else pair_file
    d, _, _ = _prepared(pair_file, _json(pair_file.parent / "manifest.json")["fold"])
    p = predict(load(model_dir, device), d, batch)
    z = d.select("tid", "qid", pl.col("label").alias("y")).with_columns(pl.Series("prob", p))
    _pq(z, out)
    return {"pairs": len(z), "out": str(path(out).resolve()), "sha256": _sha(out)}


def check():
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    import torch
    with tf.TemporaryDirectory() as tmp:
        p, data, run, out = path(tmp), path(tmp) / "data", path(tmp) / "run", path(tmp) / "pairs"
        (data / "train").mkdir(parents=True)
        _write(data / "meta.json", {"check": True})
        ref = pl.DataFrame({"rid": [1, 2, 3], "nm": ["राम bazaar", "other", "val"], "ad": ["1 गली", "x", "z"],
                            "co": ["india"] * 3, "fold": [2, 2, 0], "deg": [1, 0, 1]})
        ref.write_parquet(data / "train/ref.parquet")
        s2 = pl.DataFrame({"rid": [10, 11, 12], "nm": ["राम bazaar", "noise", "val"], "ad": ["1 गली", "y", "z"],
                           "co": ["india"] * 3, "sr": [2] * 3, "own": [1, -1, 3]})
        s2.write_parquet(data / "train/s2.parquet")
        s2.head(0).write_parquet(data / "train/s3.parquet")
        run.mkdir()
        a, q = ref.filter(pl.col("fold") == 2), s2.filter(pl.col("rid").is_in([10, 11]))
        a.write_parquet(run / "anchors.parquet")
        q.write_parquet(run / "queries.parquet")
        pl.DataFrame({"tid": [11], "qid": [2], "ns": [.9], "ads": [.1], "en": [0], "ea": [0], "sr": [2], "own": [-1], "y": [0]}).write_parquet(run / "pairs_00000.parquet")
        _write(run / "metrics.json", {"fold": 2, "country": "india", "parts": ["pairs_00000.parquet"]})
        m = prepare(data, run, out, hard=1, rnd=0)
        d = pl.read_parquet(out / "pairs.parquet")
        assert m["positive_pairs"] == 1 and set(d["label"]) == {0, 1}
        assert "राम" in "\n".join(d["text_a"].to_list() + d["text_b"].to_list())
        assert all(x not in "\n".join(d["text_a"].to_list()) for x in ("rid", "fold", "deg", "S1-"))
        bad = _json(run / "metrics.json"); bad["fold"] = 1; _write(run / "metrics.json", bad)
        try:
            prepare(data, run, p / "bad")
        except ValueError:
            pass
        else:
            raise AssertionError("fold misuse accepted")
        _write(run / "metrics.json", {"fold": 0, "country": "india", "parts": ["pairs_00000.parquet"]})
        a0, q0 = ref.filter(pl.col("fold") == 0), s2.filter(pl.col("rid") == 12)
        a0.write_parquet(run / "anchors.parquet"); q0.write_parquet(run / "queries.parquet")
        pl.DataFrame({"tid": [12], "qid": [1], "ns": [.2], "ads": [.2], "en": [0], "ea": [0], "sr": [2], "own": [3], "y": [0]}).write_parquet(run / "pairs_00000.parquet")
        v = prepare(data, run, p / "val")
        assert v["pairs"] == 1 and v["positive_pairs"] == 0
        vocab = p / "vocab.txt"; vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nname\naddress\ncountry\nराम\n", encoding="utf-8")
        tok = BertTokenizerFast(vocab_file=str(vocab))
        net = BertForSequenceClassification(BertConfig(vocab_size=9, hidden_size=16, num_hidden_layers=1,
                                                        num_attention_heads=2, intermediate_size=32, num_labels=1,
                                                        problem_type="multi_label_classification"))
        x = tok(["name: राम", "name: other"], ["address: 1", "address: 2"], padding=True, return_tensors="pt")
        y = torch.tensor([[1.0], [0.0]])
        old = net.classifier.weight.detach().clone(); loss = net(**x, labels=y).loss; loss.backward()
        torch.optim.AdamW(net.parameters(), lr=1e-2).step(); assert not torch.equal(old, net.classifier.weight)
        net.eval(); z = torch.sigmoid(net(**x).logits).detach(); net.save_pretrained(p / "model"); tok.save_pretrained(p / "model")
        net2 = BertForSequenceClassification.from_pretrained(p / "model").eval()
        got = torch.sigmoid(net2(**x).logits).detach()
        assert torch.allclose(z, got)
        _write(p / "model/neural_metadata.json", {"parameters": sum(v.numel() for v in net.parameters()),
               "problem_type": "multi_label_classification", "configuration": {"maxlen": 32}})
        dd = pl.DataFrame({"text_a": ["name: राम", "name: other"], "text_b": ["address: 1", "address: 2"]})
        assert np.allclose(predict(load(p / "model", "cpu"), dd), z.numpy().ravel())
        try:
            _prepared(p / "val", 1)
        except ValueError:
            pass
        else:
            raise AssertionError("audit fold accepted for training")
        sp = p / "sources.json"
        _write(sp, [{"model": str(p / "model"), "revision": "local", "license": "mit"}])
        trained = train([out], [p / "val"], p / "trained", model=str(p / "model"), revision="local",
                        sources=sp, batch=2, epochs=1, maxlen=32, device="cpu")
        assert trained["loss"] == "bce_with_logits" and trained["validation"]["pairs"] == 1
        assert (p / "trained/trainer_state.json").is_file()
        try:
            train([out], [p / "val"], p / "resume", model=str(p / "model"), revision="local",
                  sources=sp, device="cpu", resume=p / "trained")
        except ValueError as e:
            assert "complete trainer checkpoint" in str(e)
        else:
            raise AssertionError("weights only resume accepted")
    print("checks passed")


def main():
    root = path(__file__).resolve().parents[1]
    pa = ap.ArgumentParser()
    sub = pa.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data", type=path, default=root / "cache/data"); p.add_argument("--run", type=path, required=True)
    p.add_argument("--out", type=path, required=True); p.add_argument("--dense", type=path); p.add_argument("--hard", type=int, default=8)
    p.add_argument("--random", type=int, default=2); p.add_argument("--seed", type=int, default=42)
    t = sub.add_parser("train")
    t.add_argument("--train", type=path, nargs="+", required=True); t.add_argument("--val", type=path, nargs="+", required=True)
    t.add_argument("--out", type=path, required=True); t.add_argument("--model", default=mod0); t.add_argument("--revision", default=rev0)
    t.add_argument("--sources", type=path, default=src0); t.add_argument("--batch", type=int, default=8); t.add_argument("--epochs", type=float, default=1)
    t.add_argument("--lr", type=float, default=2e-5); t.add_argument("--maxlen", type=int, default=384); t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto"); t.add_argument("--resume", type=path)
    s = sub.add_parser("score")
    s.add_argument("--pairs", type=path, required=True); s.add_argument("--model", type=path, required=True); s.add_argument("--out", type=path, required=True)
    s.add_argument("--batch", type=int, default=32); s.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    sub.add_parser("check")
    a = pa.parse_args()
    if a.cmd == "prepare":
        print(json.dumps(prepare(a.data, a.run, a.out, a.dense, a.hard, a.random, a.seed), indent=2))
    elif a.cmd == "train":
        print(json.dumps(train(a.train, a.val, a.out, a.model, a.revision, a.sources, a.batch, a.epochs, a.lr,
                               a.maxlen, a.seed, a.device, a.resume), indent=2))
    elif a.cmd == "score":
        print(json.dumps(score(a.pairs, a.model, a.out, a.batch, a.device), indent=2))
    else:
        check()


if __name__ == "__main__":
    main()

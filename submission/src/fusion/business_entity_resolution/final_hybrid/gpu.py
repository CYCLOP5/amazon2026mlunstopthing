'bounded retrieval union and pair scoring for the final hybrid candidate pool'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import polars as pl


BASE_MODEL = "intfloat/multilingual-e5-base"
BASE_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
EMBED_DIM = 768
EMBED_PROTOCOL = "symmetric query: name/address; mean-pool attention mask; l2-normalized"
RRF_OFFSET = 20
LEXICAL_LANES = ("lex_name", "lex_address")
DENSE_LANES = ("dense_name", "dense_address")
LANES = LEXICAL_LANES + DENSE_LANES


def _atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _atomic_npy(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _rows_fingerprint(frame: pl.DataFrame, id_col: str, text_col: str) -> str:
    'Stable compact fingerprint for cache inputs, including the exact IDs/text'
    hashed = frame.select(id_col, pl.col(text_col).fill_null("")).sort(id_col).hash_rows(seed=7319)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def _ids_fingerprint(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def embedding_cache_meta(refs: pl.DataFrame, queries: pl.DataFrame, field: str,
                         model=BASE_MODEL, revision=BASE_REVISION):
    if field not in ("name", "address"):
        raise ValueError("embedding field must be name or address")
    text_col = "nm" if field == "name" else "ad"
    return {"model": model, "revision": revision, "dimension": EMBED_DIM,
        "field": field, "protocol": EMBED_PROTOCOL,
        "references": {"rows": refs.height, "sha256": _rows_fingerprint(refs, "rid", text_col),
                       "ids_sha256": _ids_fingerprint(refs.sort("rid")["rid"].to_numpy())},
        "queries": {"rows": queries.height, "sha256": _rows_fingerprint(queries, "rid", text_col),
                    "ids_sha256": _ids_fingerprint(queries.sort("rid")["rid"].to_numpy())}}


def save_embedding_cache(directory, refs_ids, refs_vectors, queries_ids, queries_vectors, meta):
    'atomically persist normalized half-precision vectors and aligned ids'
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    exp = (("refs", refs_ids, refs_vectors, meta["references"]),
                ("queries", queries_ids, queries_vectors, meta["queries"]))
    for prefix, raw_ids, raw_vec, spec in exp:
        ids = np.asarray(raw_ids, dtype=np.int64)
        vec = np.asarray(raw_vec, dtype=np.float16)
        if vec.shape != (spec["rows"], meta["dimension"]):
            raise ValueError(f"{prefix} embedding shape mismatch")
        if ids.shape != (spec["rows"],) or (len(ids) and (np.diff(ids) <= 0).any()):
            raise ValueError(f"{prefix} IDs must be sorted, unique, and complete")
        if _ids_fingerprint(ids) != spec["ids_sha256"]:
            raise ValueError(f"{prefix} IDs do not match embedding cache metadata")
        for off in range(0, len(vec), 50_000):
            block = vec[off:off+50_000]
            if not np.isfinite(block).all():
                raise ValueError(f"{prefix} embeddings must be finite")
            norms = np.linalg.norm(block.astype(np.float32), axis=1)
            if len(norms) and not np.allclose(norms, 1., atol=2e-3):
                raise ValueError(f"{prefix} embeddings must be normalized")
        _atomic_npy(root / f"{prefix}.npy", vec)
        _atomic_npy(root / f"{prefix}_ids.npy", ids)
    _atomic_json(root / "meta.json", meta)


def load_embedding_cache(directory, expected_meta, mmap_mode="r"):
    'load a complete cache only when its model and text fingerprints match'
    root = Path(directory)
    meta_path = root / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"Incomplete embedding cache: {root}")
    if json.loads(meta_path.read_text()) != expected_meta:
        raise ValueError(f"Stale embedding cache: {root}")
    out = {}
    for prefix, spec in (("refs", expected_meta["references"]), ("queries", expected_meta["queries"])):
        vec_path, ids_path = root / f"{prefix}.npy", root / f"{prefix}_ids.npy"
        if not vec_path.exists() or not ids_path.exists():
            raise RuntimeError(f"Incomplete {prefix} embedding cache: {root}")
        vec = np.load(vec_path, mmap_mode=mmap_mode, allow_pickle=False)
        ids = np.load(ids_path, mmap_mode=mmap_mode, allow_pickle=False)
        if vec.shape != (spec["rows"], expected_meta["dimension"]) or vec.dtype != np.float16:
            raise ValueError(f"Invalid {prefix} embedding array")
        if ids.shape != (spec["rows"],) or ids.dtype.kind not in "iu":
            raise ValueError(f"Invalid {prefix} ID array")
        if len(ids) and (np.diff(ids.astype(np.int64)) <= 0).any():
            raise ValueError(f"{prefix} IDs must be strictly increasing")
        if _ids_fingerprint(ids) != spec["ids_sha256"]:
            raise ValueError(f"Stale {prefix} ID array")
        out[prefix], out[f"{prefix}_ids"] = vec, ids
    return out


def exact_topk_numpy(query, refs, ref_ids, k=8, query_ids=None, valid_ref=None):
    'small cpu exact cosine search helper; vectors must already be normalized'
    q = np.asarray(query, dtype=np.float32)
    r = np.asarray(refs, dtype=np.float32)
    ids = np.asarray(ref_ids)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1]:
        raise ValueError("query/reference embedding dimensions differ")
    if len(r) != len(ids) or k < 1:
        raise ValueError("invalid reference IDs or top-k")
    keep = np.ones(len(r), dtype=bool) if valid_ref is None else np.asarray(valid_ref, dtype=bool)
    pos = np.flatnonzero(keep)
    if not len(pos):
        return pl.DataFrame(schema={"qid": pl.Int64, "tid": pl.Int64,
                                   "score": pl.Float32, "rank": pl.UInt16})
    take = min(int(k), len(pos))
    vals = q @ r[pos].T
    rows = []
    tids = np.arange(len(q)) if query_ids is None else np.asarray(query_ids)
    for i in range(len(q)):
        order = np.lexsort((ids[pos], -vals[i]))[:take]
        for rank, j in enumerate(order, 1):
            rows.append((int(ids[pos[j]]), int(tids[i]), float(vals[i, j]), rank))
    return pl.DataFrame(rows, schema={"qid": pl.Int64, "tid": pl.Int64,
        "score": pl.Float32, "rank": pl.UInt16}, orient="row")


def _validate_pairs(frame, name, required):
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns {sorted(missing)}")
    if frame.select("qid", "tid").n_unique() != frame.height:
        raise ValueError(f"{name} contains duplicate qid/tid pairs")
    if frame.filter(pl.col("qid").is_null() | pl.col("tid").is_null()).height:
        raise ValueError(f"{name} contains null pair IDs")


def _lane_rank(frame, lane, topk=8):
    if frame.is_empty():
        return pl.DataFrame(schema={"qid": pl.Int64, "tid": pl.Int64,
            lane: pl.Float32, lane + "_rank": pl.UInt16})
    _validate_pairs(frame, lane, ("qid", "tid", lane))
    if frame.schema[lane] not in (pl.Float32, pl.Float64):
        frame = frame.with_columns(pl.col(lane).cast(pl.Float32))
    values = frame[lane].to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{lane} scores must be finite and nonnegative")
    scored = frame.filter(pl.col(lane) > 0).sort(["tid", lane, "qid"], descending=[False, True, False])
    ranked = scored.with_columns(pl.int_range(1, pl.len() + 1).over("tid").cast(pl.UInt16).alias(lane + "_rank"))
    return ranked.filter(pl.col(lane + "_rank") <= topk).select("qid", "tid", lane, lane + "_rank")


def _lexical_lane(frame, lane, topk):
    score_col = lane + "_score" if lane + "_score" in frame.columns else lane
    rank_col = lane + "_rank"
    if score_col not in frame.columns:
        raise ValueError(f"lexical input missing {lane}_score")
    base = frame.select("qid", "tid", pl.col(score_col).cast(pl.Float32).alias(lane))
    if rank_col not in frame.columns:
        return _lane_rank(base, lane, topk)
    _validate_pairs(frame.select("qid", "tid"), lane, ("qid", "tid"))
    values = base[lane].to_numpy()
    ranks = frame[rank_col].cast(pl.Int64).fill_null(0).to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{lane} scores must be finite and nonnegative")
    if (ranks < 0).any() or ((values > 0) & (ranks == 0)).any():
        raise ValueError(f"{lane} ranks must be positive for positive scores")
    ranked = base.with_columns(pl.Series(rank_col, ranks, dtype=pl.UInt32))
    return ranked.filter((pl.col(lane) > 0) & (pl.col(rank_col) <= topk)).select(
        "qid", "tid", lane, pl.col(rank_col).cast(pl.UInt16))


def fuse_candidates(old_pairs: pl.DataFrame, lexical: pl.DataFrame,
                    dense_name: pl.DataFrame, dense_address: pl.DataFrame,
                    lane_topk=8, max_new=12):
    'keep all old prepared pairs; add a deterministic rrf top-k union per target'
    _validate_pairs(old_pairs, "prepared", ("qid", "tid"))
    if max_new < 0 or lane_topk < 1:
        raise ValueError("invalid fusion budget")
    old = old_pairs.select("qid", "tid").with_columns(pl.lit(1, pl.Int8).alias("from_prepared"))
    lanes = []
    lanes.extend([_lexical_lane(lexical, name, lane_topk) for name in LEXICAL_LANES])
    for frame, name in ((dense_name, "dense_name"), (dense_address, "dense_address")):
        lane_frame = frame.select("qid", "tid", pl.col(name).cast(pl.Float32),
                                  pl.col(name + "_rank").cast(pl.UInt16))
        _validate_pairs(lane_frame, name, ("qid", "tid", name, name + "_rank"))
        if not np.isfinite(lane_frame[name].to_numpy()).all():
            raise ValueError(f"{name} scores must be finite")
        lanes.append(lane_frame.filter(pl.col(name + "_rank") > 0)
                     .filter(pl.col(name + "_rank") <= lane_topk))

    all_keys = pl.concat([old.select("qid", "tid")] + [x.select("qid", "tid") for x in lanes],
                         how="vertical_relaxed").unique(["qid", "tid"], maintain_order=True)
    fused = all_keys.join(old, on=["qid", "tid"], how="left", validate="1:1")
    for lane, hits in zip(LANES, lanes):
        fused = fused.join(hits, on=["qid", "tid"], how="left", validate="1:1")
    fills = [pl.col("from_prepared").fill_null(0).cast(pl.Int8)]
    for lane in LANES:
        fills.extend([pl.col(lane).fill_null(0).cast(pl.Float32),
                      pl.col(lane + "_rank").fill_null(0).cast(pl.UInt16)])
    fused = fused.with_columns(fills)
    fused = fused.with_columns(
        *[ (pl.col(lane + "_rank") > 0).cast(pl.Int8).alias("hit_" + lane) for lane in LANES ],
        pl.sum_horizontal(*[(pl.when(pl.col(lane + "_rank") > 0)
            .then(1. / (RRF_OFFSET + pl.col(lane + "_rank"))).otherwise(0.)) for lane in LANES])
            .cast(pl.Float32).alias("rrf"))
    new = (fused.filter(pl.col("from_prepared") == 0)
        .sort(["tid", "rrf", "qid"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("tid").alias("_new_rank"))
        .filter(pl.col("_new_rank") <= max_new).drop("_new_rank"))
    res = pl.concat([fused.filter(pl.col("from_prepared") == 1), new], how="vertical_relaxed")
    res = res.sort("tid", "qid")
    _validate_pairs(res, "fused candidates", ("qid", "tid", "from_prepared", "rrf"))
    return res


def _records(root, split, name):
    path = Path(root) / split / name
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path)
    _validate_records(frame, str(path))
    return frame.sort("rid")


def _validate_records(frame, name):
    needed = {"rid", "nm", "ad", "co"}
    if needed - set(frame.columns):
        raise ValueError(f"{name} missing columns {sorted(needed-set(frame.columns))}")
    if frame["rid"].null_count() or frame["rid"].n_unique() != frame.height:
        raise ValueError(f"{name} has null or duplicate rid")
    if frame["co"].null_count():
        raise ValueError(f"{name} has null country")


def _encode_to_npy(path, texts, tokenizer, model, batch):
    import torch
    started = time.monotonic()
    tmp = path.with_suffix(path.suffix + ".tmp")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                    shape=(len(texts), EMBED_DIM))
    with torch.inference_mode():
        for lo in range(0, len(texts), batch):
            tokens = tokenizer(texts[lo:lo+batch], padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            tokens = {k: v.to("cuda") for k, v in tokens.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            arr[lo:lo+len(tokens["input_ids"])] = pooled.cpu().numpy().astype(np.float16)
            done = lo + len(tokens["input_ids"])
            if lo == 0 or done == len(texts) or (lo // batch) % 100 == 0:
                print(f"FINAL HYBRID encode {path.name}: {done:,}/{len(texts):,}; "
                      f"{(time.monotonic()-started)/60:.1f} min", flush=True)
    arr.flush()
    del arr
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def _embed_task(lexical, output, split, country, field, tokenizer, model, batch, topk=8):
    started = time.monotonic()
    root, out = Path(lexical), Path(output)
    refs, queries = _records(root, split, "refs.parquet"), _records(root, split, "queries.parquet")
    refs = refs.filter(pl.col("co") == country).sort("rid")
    queries = queries.filter(pl.col("co") == country).sort("rid")
    text_col = "nm" if field == "name" else "ad"
    refs = refs.with_columns(pl.col(text_col).fill_null("").cast(pl.String))
    queries = queries.with_columns(pl.col(text_col).fill_null("").cast(pl.String))
    meta = embedding_cache_meta(refs, queries, field)
    cache = out / "embedding" / split / country / field
    try:
        arrays = load_embedding_cache(cache, meta)
    except RuntimeError:
        rtext = [f"query: {field}: {x}" for x in refs[text_col].to_list()]
        qtext = [f"query: {field}: {x}" for x in queries[text_col].to_list()]
        cache.mkdir(parents=True, exist_ok=True)
        _encode_to_npy(cache / "refs.npy", rtext, tokenizer, model, batch)
        _encode_to_npy(cache / "queries.npy", qtext, tokenizer, model, batch)
        save_embedding_cache(cache, refs["rid"].to_numpy(), np.load(cache / "refs.npy", mmap_mode="r"),
            queries["rid"].to_numpy(), np.load(cache / "queries.npy", mmap_mode="r"), meta)
        arrays = load_embedding_cache(cache, meta)
    if field == "name":
        valid_ref = np.ones(refs.height, dtype=bool)
        valid_q = np.ones(queries.height, dtype=bool)
    else:
        valid_ref = refs["ad"].fill_null("").str.strip_chars().ne("").to_numpy()
        valid_q = queries["ad"].fill_null("").str.strip_chars().ne("").to_numpy()

    hits = _exact_topk_torch(arrays["queries"], arrays["refs"], arrays["refs_ids"],
        queries["rid"].to_numpy(), valid_q, valid_ref, topk, batch)
    name = "dense_" + field
    dest = out / split / "retrieval" / country / f"{name}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    hits = hits.rename({"score": name, "rank": name + "_rank"})
    tmp = dest.with_suffix(".tmp")
    hits.write_parquet(tmp)
    tmp.replace(dest)
    return {"split": split, "country": country, "field": field, "refs": refs.height,
            "queries": queries.height, "nonblank_queries": int(valid_q.sum()),
            "retrieved_pairs": hits.height, "seconds": time.monotonic()-started}


def _exact_topk_torch(qvec, rvec, ref_ids, query_ids, valid_queries, valid_refs, k, batch):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Dense candidate retrieval requires CUDA")
    torch.set_num_threads(4)
    rpos = np.flatnonzero(np.asarray(valid_refs, bool))
    qpos = np.flatnonzero(np.asarray(valid_queries, bool))
    ids = np.asarray(ref_ids)
    qids = np.asarray(query_ids)
    if not len(rpos) or not len(qpos):
        return pl.DataFrame(schema={"qid": pl.Int64, "tid": pl.Int64,
            "score": pl.Float32, "rank": pl.UInt16})
    dim, take = rvec.shape[1], min(k, len(rpos))
    rows = []
    free, _ = torch.cuda.mem_get_info()
    reserve = 2 * 1024**3
    full_refs = None
    if np.asarray(rvec).nbytes < free - reserve:
        full_refs = torch.as_tensor(np.asarray(rvec)[rpos], device="cuda", dtype=torch.float16)
    for start in range(0, len(qpos), batch):
        qp = qpos[start:start+batch]
        q = torch.as_tensor(np.asarray(qvec[qp]), device="cuda", dtype=torch.float16)
        best_s = best_i = None
        if full_refs is not None:
            sim = q @ full_refs.T
            best_s, best_i = torch.topk(sim, take, dim=1)
            del sim
        else:
            block = max(take, int(max(1, free-reserve) // max(1, 2*dim + 2*len(qp))))
            for off in range(0, len(rpos), block):
                rp = rpos[off:off+block]
                v = torch.as_tensor(np.asarray(rvec[rp]), device="cuda", dtype=torch.float16)
                sim = q @ v.T
                s, i = torch.topk(sim, min(take, sim.shape[1]), dim=1)
                i = i + off
                if best_s is None:
                    best_s, best_i = s, i
                else:
                    both_s = torch.cat((best_s, s), dim=1)
                    both_i = torch.cat((best_i, i), dim=1)
                    best_s, loc = torch.topk(both_s, min(take, both_s.shape[1]), dim=1)
                    best_i = torch.gather(both_i, 1, loc)
                del v, sim, s, i
        scores = best_s.float().cpu().numpy()
        locs = best_i.cpu().numpy()
        for i, qindex in enumerate(qp):


            order = np.lexsort((ids[rpos[locs[i]]], -scores[i]))
            for rank, j in enumerate(order, 1):
                rows.append((int(ids[rpos[locs[i, j]]]), int(qids[qindex]),
                             float(scores[i, j]), rank))
        if start == 0 or start + len(qp) == len(qpos) or (start // batch) % 100 == 0:
            print(f"FINAL HYBRID exact search: {min(start+len(qp),len(qpos)):,}/"
                  f"{len(qpos):,} queries; top-{take};", flush=True)
        del q, best_s, best_i
    return pl.DataFrame(rows, schema={"qid": pl.Int64, "tid": pl.Int64,
        "score": pl.Float32, "rank": pl.UInt16}, orient="row")


def _read_lane(root, split, country, lane):
    path = Path(root) / split / "retrieval" / country / f"{lane}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def _fuse_split(prepared, lexical, output, split, lane_topk=8, max_new=12, chunk_rows=250_000):
    old_path = Path(prepared) / split / "features.parquet"
    old = pl.read_parquet(old_path, columns=["qid", "tid"])
    lex = pl.read_parquet(Path(lexical) / split / "lexical.parquet")
    lex_required = tuple(lane + "_score" if lane + "_score" in lex.columns else lane
                         for lane in LEXICAL_LANES)
    _validate_pairs(lex, f"{split} lexical", ("qid", "tid", *lex_required))
    records = _records(lexical, split, "queries.parquet")
    target_ids = set(records["rid"].to_list())
    if set(old["tid"].unique().to_list()) != target_ids or set(lex["tid"].unique().to_list()) - target_ids:
        raise ValueError(f"{split} lexical queries must exactly match prepared target IDs")
    country_ids = records.select("rid", "co").rename({"rid": "tid"})
    all_fused = []
    for country in country_ids["co"].unique().sort().to_list():
        tids = country_ids.filter(pl.col("co") == country)["tid"]
        old_c = old.filter(pl.col("tid").is_in(tids.implode()))
        lex_c = lex.filter(pl.col("tid").is_in(tids.implode()))
        fused = fuse_candidates(old_c, lex_c,
            _read_lane(output, split, country, "dense_name"),
            _read_lane(output, split, country, "dense_address"), lane_topk, max_new)
        all_fused.append(fused)
    cands = pl.concat(all_fused, how="vertical_relaxed").sort("tid", "qid")
    _validate_pairs(cands, f"{split} fused candidates", ("qid", "tid"))
    dest = Path(output) / split
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "candidates.parquet.tmp"
    cands.write_parquet(tmp, compression="zstd")
    tmp.replace(dest / "candidates.parquet")

    qcountry = country_ids.rename({"co": "_co"})
    cands = cands.join(qcountry, on="tid", how="left", validate="m:1")
    shard_root = dest / "score_inputs"
    shard_root.mkdir(parents=True, exist_ok=True)
    shards, part = [], 0
    for country in cands["_co"].unique().sort().to_list():
        sub = cands.filter(pl.col("_co") == country).drop("_co")
        for off in range(0, sub.height, chunk_rows):
            name = f"part_{part:05d}.parquet"
            tmp = shard_root / (name + ".tmp")
            sub.slice(off, chunk_rows).write_parquet(tmp, compression="zstd")
            tmp.replace(shard_root / name)
            shards.append(name)
            part += 1
    report = {"pairs": cands.height, "targets": records.height,
        "old_pairs": old.height, "lexical_pairs": lex.height,
        "by_country": cands.join(qcountry, on="tid", how="left").group_by("_co").len().to_dicts(),
        "lane_coverage": {lane: {"pairs": int(cands["hit_" + lane].sum() or 0),
            "targets": int(cands.filter(pl.col("hit_" + lane) == 1)["tid"].n_unique())}
            for lane in LANES},
        "combined_target_coverage": int(cands["tid"].n_unique()),
        "max_candidates_per_target": int(cands.group_by("tid").len()["len"].max()) if cands.height else 0,
        "retrieval_shards": shards, "lane_topk": lane_topk, "max_new_per_target": max_new}
    _atomic_json(dest / "fusion_report.json", report)
    return report


def _read_cached_scores(root, split, score_name, output_name):
    paths = sorted((Path(root) / split).glob("part*.parquet"))
    if not paths:
        raise RuntimeError(f"No cached {output_name} score shards in {root}/{split}")
    frames = [pl.read_parquet(p, columns=["qid", "tid", score_name]) for p in paths]
    scores = pl.concat(frames, how="vertical_relaxed").select(
        "qid", "tid", pl.col(score_name).cast(pl.Float32).alias(output_name))
    _validate_pairs(scores, f"cached {output_name}", ("qid", "tid", output_name))
    if not np.isfinite(scores[output_name].to_numpy()).all():
        raise ValueError(f"Cached {output_name} contains nonfinite scores")
    return scores


def _pair_dense_scores(pairs, country, output, split, batch=16_384):
    froms = {}
    for field in ("name", "address"):
        cache = Path(output) / "embedding" / split / country / field
        meta = json.loads((cache / "meta.json").read_text())
        arrays = load_embedding_cache(cache, meta)
        qids = arrays["queries_ids"].astype(np.int64)
        rids = arrays["refs_ids"].astype(np.int64)
        qpos = np.searchsorted(qids, pairs["tid"].to_numpy().astype(np.int64))
        rpos = np.searchsorted(rids, pairs["qid"].to_numpy().astype(np.int64))
        if ((qpos >= len(qids)).any() or (rpos >= len(rids)).any() or
            not np.array_equal(qids[qpos], pairs["tid"].to_numpy().astype(np.int64)) or
            not np.array_equal(rids[rpos], pairs["qid"].to_numpy().astype(np.int64))):
            raise ValueError("Pair IDs are outside cached dense embedding coverage")
        out = np.empty(pairs.height, dtype=np.float32)
        for off in range(0, pairs.height, batch):
            end = min(off + batch, pairs.height)
            a = np.asarray(arrays["queries"][qpos[off:end]], dtype=np.float32)
            b = np.asarray(arrays["refs"][rpos[off:end]], dtype=np.float32)
            out[off:end] = np.sum(a * b, axis=1)
        froms["dense_" + field] = out
    return pairs.with_columns(*[pl.Series(k, v, dtype=pl.Float32) for k, v in froms.items()])


def _score_pairs_gpu(frame, tokenizer, base_model, base_maxlen, expert_tokenizer,
                     expert_model, batch):
    import torch
    from innovation.neural_expert import pair_texts
    from innovation.verify import view_text
    from innovation.neural_gpu import encoder
    torch.set_num_threads(4)
    out = frame
    for model, tok, column, mode, maxlen in (
        (base_model, tokenizer, "ce_base_lg", "base", base_maxlen),
        (expert_model, expert_tokenizer, "expert_lg", "expert", 192)):
        missing = out[column].is_null().to_numpy()
        indices = np.flatnonzero(missing)
        started = time.monotonic()
        print(f"FINAL HYBRID score {column}: cached={len(out)-len(indices):,}; "
              f"inference={len(indices):,}", flush=True)
        if not len(indices):
            continue
        rs = np.empty(len(indices), dtype=np.float32)
        if mode == "base":
            left, right = view_text(out[indices], 1, "full"), view_text(out[indices], 2, "full")
        else:
            rows = out[indices].to_dicts()
            pairs = [pair_texts(row) for row in rows]
            left, right = [p[0] for p in pairs], [p[1] for p in pairs]
        order = np.argsort([len(a)+len(b) for a, b in zip(left, right)], kind="stable")
        with torch.inference_mode():
            for off in range(0, len(indices), batch):
                ix = order[off:off+batch]
                x = tok([left[i] for i in ix], [right[i] for i in ix], padding=True,
                        truncation=True, max_length=maxlen, return_tensors="pt")
                x = {k: v.to("cuda") for k, v in x.items()}
                logits = _model_scores(model, x, mode)
                rs[ix] = logits
                done = min(off + batch, len(indices))
                if off == 0 or done == len(indices) or (off // batch) % 100 == 0:
                    print(f"FINAL HYBRID score {column}: {done:,}/{len(indices):,}; "
                          f"{(time.monotonic()-started)/60:.1f} min", flush=True)
        vals = out[column].to_numpy().astype(np.float32, copy=True)
        vals[indices] = rs
        out = out.with_columns(pl.Series(column, vals, dtype=pl.Float32))
    if out["ce_base_lg"].null_count() or out["expert_lg"].null_count():
        raise RuntimeError("Neural scoring left missing pair scores")
    if not np.isfinite(out.select("ce_base_lg", "expert_lg").to_numpy()).all():
        raise RuntimeError("Neural scoring produced nonfinite logits")
    return out


def _model_scores(model, tokens, mode):
    'run the hugging face head or the innovation matcher wrapper'
    if mode == "expert":
        return model(tokens).float().cpu().numpy()
    return model(**tokens).logits[:, 0].float().cpu().numpy()


def score_worker(prepared, lexical, old_model, verified, expert, output, rank, workers, batch):
    'score fused candidate shards, reusing exact-key cached pair scores'
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from innovation.neural_gpu import encoder
    from innovation.neural_expert import BASE_MODEL as EXPERT_BASE, BASE_REVISION as EXPERT_REV
    from innovation.verify import view_text
    del view_text
    if not torch.cuda.is_available() or workers < 1 or not 0 <= rank < workers:
        raise RuntimeError("Invalid GPU worker assignment")
    out_root = Path(output)
    model_dir = Path(old_model)
    metadata_path = model_dir / "neural_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"Missing old base model metadata: {metadata_path}")
    base_meta = json.loads(metadata_path.read_text())
    base_maxlen = int(base_meta["configuration"]["maxlen"])
    old_tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    old_net = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    expert_dir = Path(expert) / "model"
    expert_tok = AutoTokenizer.from_pretrained(expert_dir, local_files_only=True)
    expert_net = encoder(expert_dir).to(device="cuda", dtype=torch.bfloat16).eval()
    print(f"FINAL HYBRID GPU{rank}: base={model_dir}; expert={expert_dir}; "
          f"batch={batch}; devices={torch.cuda.get_device_name(0)}", flush=True)
    tasks = []
    for split in ("train", "test"):
        manifest = json.loads((out_root / split / "fusion_report.json").read_text())
        tasks.extend((split, out_root / split / "score_inputs" / name)
                     for name in manifest["retrieval_shards"])
    cache_by_split = {}
    records_by_split = {}
    for split in ("train", "test"):
        cache_by_split[split] = (
            _read_cached_scores(verified, split, "ce_full_lg", "ce_base_lg"),
            _read_cached_scores(expert, split, "expert_lg", "expert_lg"))
        records_by_split[split] = (
            _records(lexical, split, "refs.parquet").select(
                pl.col("rid").alias("qid"), pl.col("nm").alias("nm1"),
                pl.col("ad").alias("ad1"), pl.col("co").alias("co1")),
            _records(lexical, split, "queries.parquet"))
    for task_index, (split, shard) in enumerate(tasks):
        if task_index % workers != rank:
            continue
        pairs = pl.read_parquet(shard)
        scores0, scores1 = cache_by_split[split]
        pairs = pairs.join(scores0, on=["qid", "tid"], how="left", validate="1:1")
        pairs = pairs.join(scores1, on=["qid", "tid"], how="left", validate="1:1")
        reused_scores = int((pairs["ce_base_lg"].is_not_null() & pairs["expert_lg"].is_not_null()).sum())
        ref_text, query_records = records_by_split[split]
        qco = query_records.select(pl.col("rid").alias("tid"), "co")
        countries = pairs.join(qco, on="tid", how="left")["co"].unique().to_list()
        if len(countries) != 1:
            raise ValueError("Scoring shard must contain exactly one country")
        country = str(countries[0])
        pairs = _pair_dense_scores(pairs, country, output, split)
        queries = query_records.select(
            pl.col("rid").alias("tid"), pl.col("nm").alias("nm2"),
            pl.col("ad").alias("ad2"), pl.col("co").alias("co2"))
        pairs = pairs.join(ref_text, on="qid", how="left", validate="m:1").join(
            queries, on="tid", how="left", validate="m:1")
        if pairs.filter(pl.col("co1").is_null() | pl.col("co2").is_null()).height:
            raise ValueError("Unknown record ID in fused pair")
        if pairs.filter(pl.col("co1") != pl.col("co2")).height:
            raise ValueError("Cross-country fused pair")
        pairs = pairs.with_columns(
            pl.when((pl.col("ad1").fill_null("").str.strip_chars() == "") |
                    (pl.col("ad2").fill_null("").str.strip_chars() == ""))
              .then(0.).otherwise(pl.col("dense_address")).cast(pl.Float32).alias("dense_address"))
        pairs = _score_pairs_gpu(pairs, old_tok, old_net, base_maxlen, expert_tok,
                                 expert_net, batch)
        keep = ["qid", "tid", "ce_base_lg", "expert_lg", "dense_name", "dense_address",
                "lex_name", "lex_address", "hit_lex_name", "hit_lex_address",
                "hit_dense_name", "hit_dense_address", "lex_name_rank", "lex_address_rank",
                "dense_name_rank", "dense_address_rank", "rrf", "from_prepared"]
        res = pairs.select(*keep)
        dest = out_root / split / "parts" / shard.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        res.write_parquet(tmp, compression="zstd")
        tmp.replace(dest)
        print(f"FINAL HYBRID GPU{rank}: {split}/{shard.name} scored {res.height:,}; "
              f"reused both cached logits={reused_scores:,}; new pair inferences="
              f"{res.height-reused_scores:,}", flush=True)


def dense_worker(lexical, output, rank, workers, batch):
    'encode pinned e5-base field vectors, then exact-gpu retrieve top eight'
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not torch.cuda.is_available() or workers < 1 or not 0 <= rank < workers:
        raise RuntimeError("Invalid GPU worker assignment")
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model = AutoModel.from_pretrained(BASE_MODEL, revision=BASE_REVISION,
        attn_implementation="sdpa", use_safetensors=True,
        torch_dtype=torch.bfloat16).to("cuda").eval()
    root = Path(lexical)
    tasks = []
    for split in ("train", "test"):
        refs, queries = _records(root, split, "refs.parquet"), _records(root, split, "queries.parquet")
        countries = sorted(set(refs["co"].unique().to_list()) & set(queries["co"].unique().to_list()))
        for country in countries:
            for field in ("name", "address"):
                tasks.append((split, country, field))
    stats = []
    for index, (split, country, field) in enumerate(tasks):
        if index % workers != rank:
            continue
        stats.append(_embed_task(lexical, output, split, country, field, tok, model, batch))
        print(f"FINAL HYBRID GPU{rank}: embedded/retrieved {split}/{country}/{field}", flush=True)
    (Path(output) / "dense_worker" / f"rank_{rank}.json").parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(Path(output) / "dense_worker" / f"rank_{rank}.json", stats)


def _spawn_runner(stage, args, workers, runner=None):
    script = Path(runner) if runner else Path(__file__).resolve().parents[1] / "scripts/run_final_hybrid.py"
    if not script.exists():
        raise FileNotFoundError(f"Final hybrid runner not found: {script}")
    children = []
    env_base = dict(os.environ, TOKENIZERS_PARALLELISM="true", OMP_NUM_THREADS="4",
                    POLARS_MAX_THREADS="4", RAYON_NUM_THREADS="4")
    for rank in range(workers):
        cmd = [sys.executable, str(script), stage, *args, "--rank", str(rank),
               "--workers", str(workers)]
        env = dict(env_base, CUDA_VISIBLE_DEVICES=str(rank))
        children.append(subprocess.Popen(cmd, env=env))
    try:
        while children:
            for child in list(children):
                code = child.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"{stage} worker exited with code {code}")
                    children.remove(child)
            if children:
                time.sleep(2)
    except BaseException:
        for child in children:
            child.terminate()
        for child in children:
            child.wait()
        raise


def _verify_output(output):
    root = Path(output)
    reports = {}
    for split in ("train", "test"):
        base = root / split
        exp = pl.read_parquet(base / "candidates.parquet", columns=["qid", "tid"])
        manifest = json.loads((base / "fusion_report.json").read_text())
        parts = [base / "parts" / name for name in manifest["retrieval_shards"]]
        if any(not p.exists() for p in parts):
            raise RuntimeError(f"Missing expected {split} score part")
        if not parts:
            raise RuntimeError(f"No completed score parts for {split}")
        got = pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed")
        _validate_pairs(got, f"{split} scored union", ("qid", "tid", "ce_base_lg", "expert_lg"))
        if (got.height != exp.height or
            exp.join(got.select("qid", "tid"), on=["qid", "tid"], how="anti").height or
            got.join(exp, on=["qid", "tid"], how="anti").height):
            raise RuntimeError(f"{split} score coverage differs from fused candidate set")
        if not np.isfinite(got.select("ce_base_lg", "expert_lg").to_numpy()).all():
            raise RuntimeError(f"{split} has nonfinite cross-encoder scores")
        reports[split] = {"candidate_pairs": exp.height, "scored_pairs": got.height,
                          "parts": len(parts)}
    dense_tasks = []
    for path in sorted((root / "dense_worker").glob("rank_*.json")):
        dense_tasks.extend(json.loads(path.read_text()))
    _atomic_json(root / "report.json", {"model": BASE_MODEL, "revision": BASE_REVISION,
        "embedding_protocol": EMBED_PROTOCOL, "workers": 4, "coverage": reports,
        "dense_task_timings": dense_tasks,
        "retrieval": "exact chunked GPU cosine top-8 per field and country",
        "fusion": f"old prepared top-8 plus up to 12 new candidates per target; RRF offset {RRF_OFFSET}",
        "neural_models_finetuned": False})
    (root / "_SUCCESS").write_text("complete\n")
    return reports


def _validate_old_score_pools(prepared, verified, expert):
    for split in ("train", "test"):
        keys = pl.read_parquet(Path(prepared) / split / "features.parquet",
                               columns=["qid", "tid"])
        for root, score, name in ((verified, "ce_full_lg", "base"),
                                  (expert, "expert_lg", "expert")):
            got = _read_cached_scores(root, split, score, score)
            if (got.height != keys.height or
                keys.join(got.select("qid", "tid"), on=["qid", "tid"], how="anti").height or
                got.join(keys, on=["qid", "tid"], how="anti").height):
                raise ValueError(f"Cached {name} scores do not exactly cover old {split} prepared pairs")


def run(prepared, lexical, old_model, verified, expert, output, workers=4, batch=128):
    'run dense retrieval, deterministic fusion, and cached-fill pair scoring'
    if workers != 4:
        raise ValueError("Final hybrid GPU stage is configured for exactly four workers")
    if batch < 1:
        raise ValueError("batch must be positive")
    prepared, lexical, output = Path(prepared), Path(lexical), Path(output)
    for split in ("train", "test"):
        for file in ("refs.parquet", "queries.parquet", "lexical.parquet"):
            if not (lexical / split / file).exists():
                raise FileNotFoundError(lexical / split / file)
        if not (prepared / split / "features.parquet").exists():
            raise FileNotFoundError(prepared / split / "features.parquet")
        if not (Path(verified) / split).exists() or not (Path(expert) / split).exists():
            raise FileNotFoundError(f"Missing cached neural score split {split}")
    for path in (Path(verified) / "_SUCCESS", Path(expert) / "_SUCCESS"):
        if not path.exists():
            raise RuntimeError(f"Cached neural scoring incomplete: {path}")
    _validate_old_score_pools(prepared, verified, expert)
    output.mkdir(parents=True, exist_ok=True)
    for done in (output / "_SUCCESS",):
        done.unlink(missing_ok=True)

    from transformers import AutoModel, AutoTokenizer
    AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    warm = AutoModel.from_pretrained(BASE_MODEL, revision=BASE_REVISION,
        attn_implementation="sdpa", use_safetensors=True, torch_dtype="bfloat16")
    del warm
    runner = Path(__file__).resolve().parents[1] / "scripts/run_final_hybrid.py"
    dense_args = ["--lexical", str(lexical), "--output", str(output), "--batch", str(batch)]
    _spawn_runner("dense-worker", dense_args, workers, runner)
    fusion = {}
    for split in ("train", "test"):
        fusion[split] = _fuse_split(prepared, lexical, output, split)
    _atomic_json(output / "fusion_report.json", fusion)
    score_args = ["--prepared", str(prepared), "--lexical", str(lexical),
        "--old-model", str(old_model), "--verified", str(verified), "--expert", str(expert),
        "--output", str(output), "--batch", str(batch)]
    _spawn_runner("score-worker", score_args, workers, runner)
    return _verify_output(output)

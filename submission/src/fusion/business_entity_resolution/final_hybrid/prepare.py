'build bounded, country-scoped lexical candidates for the hybrid reranker'
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import polars as pl
from unidecode import unidecode


def clean_name(value: str | None) -> str:
    'Normalize script/punctuation and make token order irrelevant'
    text = unidecode(value or "").casefold()

    text = re.sub(r"\b(?:[a-z]\.\s*){2,}", lambda m: re.sub(r"[^a-z]", "", m.group()) + " ", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(sorted(tokens))


def clean_address(value: str | None) -> str:
    'keep address words and geography while removing all numeric strings'
    text = unidecode(value or "").casefold()
    text = re.sub(r"\d+", " ", text)
    return " ".join(re.findall(r"[a-z]+", text))


def _read_refs(data: Path, split: str) -> pl.DataFrame:
    return pl.read_parquet(data / split / "ref.parquet", columns=["rid", "nm", "ad", "co"])


def _read_targets(data: Path, split: str) -> pl.DataFrame:
    return pl.concat([
        pl.read_parquet(data / split / f"s{source}.parquet", columns=["rid", "nm", "ad", "co"])
        for source in (2, 3)
    ], how="vertical_relaxed")


def _max_df(n_docs: int, min_df: int) -> int:


    return max(min_df, min(2_000, max(min_df, int(math.floor(0.03 * n_docs)))))


def _country_key(value) -> str:
    return "" if value is None else str(value)


def _retrieve_sparse(query_matrix, ref_matrix, query_ids: np.ndarray, ref_ids: np.ndarray,
                     topk: int, threads: int, progress: str = "retrieval") -> tuple[list, list, list, list]:
    'return top-k rows using sparse-dot-topn in bounded chunks'
    try:
        from sparse_dot_topn import sp_matmul_topn
    except ImportError:
        sp_matmul_topn = None

    qids, rids, scores, ranks = [], [], [], []
    ref_transpose = ref_matrix.T.tocsr()
    starts = range(0, query_matrix.shape[0], 4096)
    total_chunks = (query_matrix.shape[0] + 4095) // 4096
    for chunk_index, start in enumerate(starts, start=1):
        stop = min(start + 4096, query_matrix.shape[0])
        queries = query_matrix[start:stop]
        if sp_matmul_topn is not None:
            products = sp_matmul_topn(queries, ref_transpose, top_n=topk,
                                      threshold=0.0, sort=True, n_threads=threads)
        else:


            if len(ref_ids) > 20_000:
                raise RuntimeError("sparse-dot-topn is required for more than 20,000 references")
            rows = []
            for row in range(queries.shape[0]):
                product = queries[row] @ ref_transpose
                rows.append(product.tocsr())
            from scipy.sparse import vstack
            products = vstack(rows, format="csr") if rows else queries[:0] @ ref_matrix.T

        for row in range(products.shape[0]):
            lo, hi = products.indptr[row], products.indptr[row + 1]
            cols = products.indices[lo:hi]
            vals = products.data[lo:hi]
            if not len(cols):
                continue

            order = np.lexsort((ref_ids[cols], -vals))[:topk]
            for rank, j in enumerate(order, start=1):
                qids.append(ref_ids[cols[j]])
                rids.append(query_ids[start + row])
                scores.append(float(vals[j]))
                ranks.append(rank)
        if chunk_index % 40 == 0 or chunk_index == total_chunks:
            print(f"[{progress}] retrieval chunks {chunk_index:,}/{total_chunks:,}; "
                  f"pairs {len(qids):,}", flush=True)
    return qids, rids, scores, ranks


def _lane(refs: pl.DataFrame, queries: pl.DataFrame, field: str, lane: str,
          topk: int, threads: int, model_dir: Path, report: dict) -> pl.DataFrame:
    from sklearn.feature_extraction.text import TfidfVectorizer

    text_fn = clean_name if lane == "name" else clean_address
    analyzer, ngrams = ("char_wb", (4, 5)) if lane == "name" else ("word", (1, 2))
    all_qid, all_tid, all_score, all_rank = [], [], [], []
    refs = refs.with_columns(pl.col("co").fill_null("").cast(pl.String))
    queries = queries.with_columns(pl.col("co").fill_null("").cast(pl.String))
    country_reports = []

    query_countries = sorted(queries["co"].unique().to_list())
    for country in query_countries:
        r = refs.filter(pl.col("co") == country)
        q = queries.filter(pl.col("co") == country)
        print(f"[prepare:{lane}:{country or '<empty-country>'}] references {r.height:,}; "
              f"queries {q.height:,}", flush=True)
        country_item = {"country": country, "reference_count": r.height, "query_count": q.height,
                        "candidate_pairs": 0, "empty_vocabulary": False}
        if not r.height or not q.height:
            country_item["empty_vocabulary"] = True
            country_reports.append(country_item)
            continue

        min_df = 1 if r.height < 200 else 2
        max_df = _max_df(r.height, min_df)
        vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=ngrams, min_df=min_df, max_df=max_df,
            max_features=600_000, dtype=np.float32, sublinear_tf=True, norm="l2",
            lowercase=False, token_pattern=r"(?u)\b\w\w+\b",
        )
        ref_text = [text_fn(v) for v in r[field].to_list()]
        query_text = [text_fn(v) for v in q[field].to_list()]
        try:
            ref_matrix = vectorizer.fit_transform(ref_text).tocsr()
        except ValueError as exc:
            if "empty vocabulary" not in str(exc).lower() and "after pruning" not in str(exc).lower():
                raise
            country_item["empty_vocabulary"] = True
            country_item["vocabulary_error"] = str(exc)
            country_reports.append(country_item)
            continue
        query_matrix = vectorizer.transform(query_text).tocsr()
        print(f"[prepare:{lane}:{country or '<empty-country>'}] fitted vocabulary "
              f"{len(vectorizer.vocabulary_):,}; transform nnz ref/query "
              f"{ref_matrix.nnz:,}/{query_matrix.nnz:,}", flush=True)

        slug = re.sub(r"[^a-z0-9]+", "_", country.casefold()).strip("_") or "empty"
        digest = hashlib.sha1(country.encode("utf-8")).hexdigest()[:8]
        model_path = model_dir / f"{lane}_{slug}_{digest}.joblib"
        import joblib
        joblib.dump(vectorizer, model_path, compress=3)

        got = _retrieve_sparse(query_matrix, ref_matrix, q["rid"].to_numpy(), r["rid"].to_numpy(),
                               topk, threads, progress=f"prepare:{lane}:{country or '<empty-country>'}")
        qid, tid, score, rank = got
        all_qid.extend(qid)
        all_tid.extend(tid)
        all_score.extend(score)
        all_rank.extend(rank)
        country_item.update({"candidate_pairs": len(qid), "vocabulary_size": len(vectorizer.vocabulary_),
                             "min_df": min_df, "max_df": max_df,
                             "vectorizer": str(model_path.relative_to(model_dir.parents[1]))})
        country_reports.append(country_item)

    report["lanes"][lane] = {
        "analyzer": analyzer, "ngram_range": list(ngrams), "fit_scope": "actual Source-1 references, within country",
        "max_features": 600_000, "sublinear_tf": True, "country_reports": country_reports,
    }
    return pl.DataFrame({"qid": all_qid, "tid": all_tid,
                         f"lex_{lane}_score": all_score, f"lex_{lane}_rank": all_rank},
                        schema={"qid": refs.schema["rid"], "tid": queries.schema["rid"],
                                f"lex_{lane}_score": pl.Float32, f"lex_{lane}_rank": pl.UInt8})


def prepare_split(data: Path, prepared: Path, output: Path, split: str,
                  topk: int = 8, threads: int = 24) -> dict:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    if topk < 1 or threads < 1:
        raise ValueError("topk and threads must be positive")
    refs = _read_refs(data, split).select("rid", "nm", "ad", "co")
    targets = _read_targets(data, split).select("rid", "nm", "ad", "co")
    feature_path = prepared / split / "features.parquet"
    sel = pl.read_parquet(feature_path, columns=["tid"]).select(pl.col("tid").unique())
    if sel["tid"].null_count():
        raise ValueError(f"{feature_path} contains null target IDs")
    queries = targets.join(sel.rename({"tid": "rid"}), on="rid", how="inner")
    if queries.height != sel.height:
        missing = sel.join(queries.select("rid"), left_on="tid", right_on="rid", how="anti").height
        raise ValueError(f"Selected target coverage mismatch: {missing} IDs missing from raw data")

    split_root = output / split
    model_dir = split_root / "index"
    split_root.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    refs.write_parquet(split_root / "refs.parquet", compression="zstd")
    queries.write_parquet(split_root / "queries.parquet", compression="zstd")

    report = {
        "split": split, "topk_per_lane": topk, "threads": threads,
        "reference_rows": refs.height, "query_rows": queries.height,
        "selected_target_ids": sel.height,
        "provenance": "TF-IDF vocabularies and IDF fit on raw unlabeled Source-1 references, separately per actual country",
        "preprocessing": {"unicode": "unidecode", "name": "dotted initials joined; sorted tokens; char_wb 4-5",
                          "address": "all digit strings removed; word 1-2 grams"},
        "lanes": {},
    }
    name = _lane(refs, queries, "nm", "name", topk, threads, model_dir, report)
    address = _lane(refs, queries, "ad", "address", topk, threads, model_dir, report)
    lexical = (name.join(address, on=["qid", "tid"], how="full", coalesce=True)
               .with_columns(pl.col("lex_name_score", "lex_address_score").fill_null(0).cast(pl.Float32),
                             pl.col("lex_name_rank", "lex_address_rank").fill_null(0).cast(pl.UInt8))
               .sort("tid", "qid"))
    if lexical.select("qid", "tid").n_unique() != lexical.height:
        raise ValueError("Lexical retrieval emitted duplicate qid/tid pairs")
    lexical.write_parquet(split_root / "lexical.parquet", compression="zstd")
    report["lexical_pairs"] = lexical.height
    report["name_lane_pairs"] = name.height
    report["address_lane_pairs"] = address.height
    report_path = output / f"preparation_{split}.json"
    report_path.write_text(json.dumps(report, indent=2))
    (output / f"_{split}_SUCCESS").write_text("complete\n")
    return report


def run(data, prepared, output, split="train", topk=8, threads=24):
    'Prepare train/test independently, or both sequentially in one process'
    data, prepared, output = Path(data), Path(prepared), Path(output)
    splits = ("train", "test") if split == "both" else (split,)
    reports = {part: prepare_split(data, prepared, output, part, topk, threads) for part in splits}
    if split == "both":
        (output / "_SUCCESS").write_text("complete\n")
    return reports[split] if split != "both" else reports

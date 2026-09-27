'versioned name-change and raw-number evidence for the dense gate'

import hashlib as hh
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher as seq
from pathlib import Path as path

import numpy as np
import polars as pl

import infer
import norm2
import rfeat
import tfeat

backend = "hybrid-v3"
fillers = {"center", "services", "service", "partners", "france"}
stop = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "and", "the", "of"}
ff = ["g_exact", "g_add", "g_drop", "g_typo", "g_swap_place", "g_swap_move", "g_swap_filler",
      "g_swap_common", "g_swap_rare", "g_multi", "g_missing", "g_extra", "g_missing_df", "g_extra_df",
      "g_extra_df_min", "g_extra_filler", "g_acronym", "g_domain", "g_alias", "g_initials_agree",
      "g_brand", "g_source_tokens", "g_target_tokens", "g_house_conflict", "g_house_typo", "g_house_last_delta"]
alias = re.compile(r"\b(?:dba|d/b/a|d\.b\.a|f/k/a|fka|formerly|aka|a/k/a|nee|née|doing business|trading as|t/a)\b", re.I)
domain = re.compile(r"\.(?:com|fr|net|org)\b|^www\.|^[#@]", re.I)


def names(dense):
    return rfeat.names(dense) + ff


def contract(meta):
    dense = meta.get("dense_features", [])
    exp = names(dense)
    if meta.get("feature_backend") != backend or meta.get("feature_names") != exp:
        raise ValueError("name-change gate feature contract mismatch")
    return exp, dense


def initials(text, exclude=False):
    return "".join(w[0] for w in re.findall(r"[a-z0-9]+", text.lower()) if not exclude or w not in stop)


def describe(rn, tn, raw_r, raw_t, ra, ta, freq):
    s, t = rn.split(), tn.split()
    ss, ts = set(s), set(t)
    missing, extra = [w for w in s if w not in ts], [w for w in t if w not in ss]
    z = np.zeros(len(ff), np.float32)
    if not missing and not extra:
        z[0] = 1
    elif not missing:
        z[1] = 1
    elif not extra:
        z[2] = 1
    elif len(missing) == len(extra) == 1:
        if seq(None, missing[0], extra[0]).ratio() >= .75:
            z[3] = 1
        else:
            same = len(s) == len(t) and s.index(missing[0]) == t.index(extra[0])
            z[4 if same else 5] = 1
            z[6 if extra[0] in fillers else 7 if freq.get(extra[0], 0) >= 20 else 8] = 1
    else:
        z[9] = 1
    z[10:16] = [len(missing), len(extra), math.log1p(max((freq.get(w, 0) for w in missing), default=0)),
                math.log1p(max((freq.get(w, 0) for w in extra), default=0)),
                math.log1p(min((freq.get(w, 0) for w in extra), default=0)), float(bool(extra) and set(extra) <= fillers)]
    is_domain = bool(domain.search(raw_t))
    is_alias = bool(alias.search(raw_t))
    compact = re.sub(r"[^a-z0-9]", "", raw_t.lower())
    is_acronym = not is_domain and bool(re.fullmatch(r"[A-Z0-9](?:[. -]?[A-Z0-9]){1,7}\.?", raw_t.strip()))
    init = {initials(raw_r), initials(raw_r, True), initials(rn), initials(rn, True)} - {""}
    agrees = (is_acronym and compact in init) or (is_domain and any(len(v) >= 2 and
              re.sub(r"^(?:www\.)|\.(?:com|fr|net|org).*$|^[#@]", "", raw_t.lower()) == v for v in init))
    raw_words = re.findall(r"[a-z]+", raw_t.lower())
    brand = len(raw_words) == 1 and len(raw_words[0]) >= 6 and raw_words[0] not in ss and freq.get(raw_words[0], 0) < 5
    z[16:23] = [is_acronym, is_domain, is_alias, agrees, brand, len(s), len(t)]
    a = [v.lstrip("0") or "0" for v in re.findall(r"(?<![a-z0-9])\d+", ra.lower())]
    b = [v.lstrip("0") or "0" for v in re.findall(r"(?<![a-z0-9])\d+", ta.lower())]
    if a and b and a[0] not in b:
        z[23] = 1
        for v in b:
            short, long = sorted((a[0], v), key=len)
            one = len(a[0]) == len(v) and len(v) >= 2 and sum(x != y for x, y in zip(a[0], v)) == 1
            dropped = len(long) - len(short) == 1 and len(short) >= 2 and any(long[:i] + long[i + 1:] == short for i in range(len(long)))
            z[24] = max(z[24], one or dropped)
            if len(a[0]) == len(v) and a[0][:-1] == v[:-1]:
                delta = abs(int(a[0][-1]) - int(v[-1]))
                z[25] = delta if z[25] == 0 else min(z[25], delta)
    return z


def frequencies(data, split, normalizer, cache):
    src = {"data": infer._sha(path(data) / "meta.json"), "normalizer": infer._sha(normalizer),
              "sources": {p: infer._sha(path(__file__).parent / p) for p in ("gfeat.py", "norm2.py", "tfeat.py", "tm_prep.py", "tm_rules.py")},
              "split": split}
    key = hh.sha256(json.dumps(src, sort_keys=True).encode()).hexdigest()
    dest = path(cache) / "name-change" / f"{key}.json"
    if dest.exists():
        old = infer._json(dest)
        if old.get("source") != src:
            raise ValueError("name-change frequency cache changed")
        return old["frequency"]
    refs = pl.read_parquet(path(data) / split / "ref.parquet", columns=["rid", "nm", "ad", "co"])
    normalized = tfeat.prep(norm2.apply(refs, norm2.load(normalizer, data)))
    res = {}
    for (country,), group in normalized.group_by("country"):
        count = Counter()
        for text in group["name_core"]:
            count.update(set(text.split()))
        res[country] = dict(count)
    infer._write(dest, {"source": src, "frequency": res})
    return res


def prep(ref, data, split, normalizer, cache):
    st = rfeat.prep(ref, data, split, normalizer, cache)
    st["raw_refs"] = ref.select(pl.col("rid").alias("qid"), pl.col("nm").alias("raw_r"), pl.col("ad").alias("raw_ra"))
    st["frequency"] = frequencies(data, split, normalizer, cache)
    return st


def rows(st, queries, pairs):
    q = queries.filter(pl.col("rid").is_in(pairs["tid"].unique().implode()))
    qt = tfeat.prep(norm2.apply(q, st["mapping"]))
    refs = st["base"].select(pl.col("entity_id").alias("qid"), pl.col("country").alias("co"),
                             pl.col("name_core").alias("rn"), pl.col("addr_can").alias("ra"))
    d = (pairs.select("qid", "tid").join(refs, on="qid", how="left", maintain_order="left", validate="m:1")
         .join(st["raw_refs"], on="qid", how="left", maintain_order="left", validate="m:1")
         .join(qt.select(pl.col("entity_id").alias("tid"), pl.col("name_core").alias("tn"), pl.col("addr_can").alias("ta")), on="tid", how="left", maintain_order="left", validate="m:1")
         .join(q.select(pl.col("rid").alias("tid"), pl.col("nm").alias("raw_t"), pl.col("ad").alias("raw_ta")), on="tid", how="left", maintain_order="left", validate="m:1"))
    if d.null_count().sum_horizontal().sum():
        raise ValueError("name-change feature join lost a record")
    return d


def make(st, queries, pairs, threads=1, dense=()):
    base, _ = rfeat.make(st, queries, pairs, threads, dense)
    if not len(pairs):
        return np.empty((0, len(names(dense))), np.float32), names(dense)
    d = rows(st, queries, pairs)
    values = np.asarray([describe(rn, tn, r, t, ra, ta, st["frequency"].get(co, {}))
                         for co, rn, tn, r, t, ra, ta in d.select("co", "rn", "tn", "raw_r", "raw_t", "raw_ra", "raw_ta").iter_rows()], dtype=np.float32)
    return np.column_stack((base, values)), names(dense)


def check():
    import tempfile
    import train
    a = describe("lyon club", "lyon comite", "Lyon Club", "Lyon Comite", "10 rue x", "10 rue x", {"comite": 30})
    assert a[4] == a[7] == 1 and a[23] == 0
    a = describe("alpha beta gamma", "abzz", "Alpha Beta Gamma", "ABZZ", "10 main", "10 main", {})
    assert a[18] == 0 and a[16] == 1 and a[19] == 0
    a = describe("alpha beta gamma", "abg", "Alpha Beta Gamma", "A.B.G.", "601 main", "60 main", {})
    assert a[19] == a[23] == a[24] == 1
    a = describe("alpha beta", "ab", "Alpha Beta", "ab.fr", "8 rue", "cite1 8 rue", {})
    assert a[17] == a[19] == 1 and a[23] == 0
    with tempfile.TemporaryDirectory() as tmp:
        root = path(tmp)
        data = infer._check_data(root)
        for split in ("train", "test"):
            for sr, ids in ((2, [0, 1]), (3, [2, 3])):
                file = data / split / f"s{sr}.parquet"
                d = pl.read_parquet(file).with_columns(pl.Series("rid", ids, dtype=pl.UInt32))
                infer._pq(d, file)
        normalizer = root / "normalizer.json"
        norm2.fit(data, normalizer)
        ref = pl.read_parquet(data / "train/ref.parquet")
        target = pl.concat([pl.read_parquet(data / "train" / f"s{sr}.parquet") for sr in (2, 3)])
        pairs = pl.DataFrame({"qid": [0, 1, 0, 3], "tid": [0, 0, 2, 3], "ns": [.9, .2, .8, .1],
                              "ads": [.9, .1, .8, .1], "ds_e5_small": [.9, .2, .8, .1]}).with_columns(pl.col("qid", "tid").cast(pl.UInt32))
        state = prep(ref, data, "train", normalizer, root / "cache")
        x, fs = make(state, target, pairs, dense=["ds_e5_small"])
        assert x.shape == (4, len(fs)) and np.isfinite(x).all()
        assert np.array_equal(x[:1, -len(ff):], make(state, target, pairs.head(1), dense=["ds_e5_small"])[0][:, -len(ff):])
        assert train._contract({"feature_backend": backend, "feature_names": fs, "dense_features": ["ds_e5_small"]}) == (fs, ["ds_e5_small"])
    print("name-change feature checks passed")


if __name__ == "__main__":
    check()

'fold-safe learned token normalization for newly trained feature contracts'
import argparse as ap
import collections as ct
import re
from pathlib import Path as path

import polars as pl

import infer


native = re.compile(r"[\u0900-\u0dff]")
token = re.compile(r"[^\s,.;:()\[\]{}&+/\\#\"'_|-]+")
departments = {"nord": "hauts-de-france", "pas de calais": "hauts-de-france",
               "gironde": "nouvelle aquitaine", "loire atlantique": "pays de la loire"}
suffix = r"(?i)\b(?:compagnie|fils|fr[eè]res|frs|associ[eé]s|groupe|d[eé]veloppement)\b"


def words(value):
    return token.findall((value or "").lower())


def learn(pairs, minimum=2):
    names, joint, na, la = ct.defaultdict(ct.Counter), ct.Counter(), ct.Counter(), ct.Counter()
    for name, address, ref_name, ref_address in pairs:
        a, b = words(name), words(ref_name)
        if len(a) == len(b):
            for x, y in zip(a, b):
                if native.search(x) and not native.search(y):
                    names[x][y] += 1
        a = {t for t in words(address) if native.search(t)}
        if a:
            b = {t for t in words(ref_address) if not native.search(t) and not t.isdigit()}
            na.update(a)
            la.update(b)
            joint.update((x, y) for x in a for y in b)
    res = {}
    for name, counts in names.items():
        best, count = max(counts.items(), key=lambda z: (z[1], z[0]))
        if count >= minimum and count >= .5 * counts.total():
            res[name] = best
    best = {}
    for (x, y), count in joint.items():
        dice = 2 * count / (na[x] + la[y])
        if count >= minimum and (dice, y) > best.get(x, (.3, "")):
            best[x] = dice, y
    for x, (_, y) in best.items():
        res.setdefault(x, y)
    return res


def training_pairs(data):
    refs = pl.scan_parquet(data / "train/ref.parquet").filter(pl.col("fold") == 2).select(
        pl.col("rid").cast(pl.Int32).alias("own"), pl.col("nm").alias("ref_name"), pl.col("ad").alias("ref_address"))
    targets = pl.concat([pl.scan_parquet(data / "train" / f"s{i}.parquet").select("own", "nm", "ad") for i in (2, 3)])
    targets = targets.filter((pl.col("own") >= 0) & (pl.col("nm").str.contains(native.pattern) | pl.col("ad").str.contains(native.pattern)))
    return targets.join(refs, on="own", how="inner").select("nm", "ad", "ref_name", "ref_address").collect(engine="streaming")


def fit(data, out, minimum=2):
    data, out = path(data).resolve(), path(out).resolve()
    if minimum < 1 or out.exists():
        raise ValueError("invalid minimum or existing normalization model")
    pairs = training_pairs(data)
    mapping = learn(pairs.iter_rows(), minimum)
    model = {"version": 1, "kind": "fold-safe-token-normalization", "fit_folds": [2],
             "data_meta_sha256": infer._sha(data / "meta.json"), "pairs": len(pairs),
             "minimum": minimum, "mapping": mapping}
    infer._write(out, model)
    return model


def load(p, data=None):
    model = infer._json(path(p))
    if (model.get("version") != 1 or model.get("kind") != "fold-safe-token-normalization" or
            model.get("fit_folds") != [2] or not isinstance(model.get("mapping"), dict) or
            any(not isinstance(k, str) or not isinstance(v, str) for k, v in model["mapping"].items())):
        raise ValueError("invalid fold-safe normalization model")
    if data and model["data_meta_sha256"] != infer._sha(path(data) / "meta.json"):
        raise ValueError("normalization model uses different prepared data")
    return model["mapping"]


def translate(values, mapping):
    values = values.fill_null("")
    unique = values.filter(values.str.contains(native.pattern)).unique()
    if len(unique):
        converted = [token.sub(lambda m: mapping.get(m.group().lower(), m.group()), text) for text in unique]
        values = values.replace(unique, converted)
    return values


def france_address(value):

    parts = value.split(",")
    for i in range(1, len(parts)):
        key = re.sub(r"[-\s]+", " ", parts[i].strip().lower())
        if key in departments:
            parts[i] = " " + departments[key]
    return ",".join(parts)


def apply(records, mapping):
    if {"nm", "ad", "co"} - set(records.columns):
        raise ValueError("normalization needs raw name, address and country")
    records = records.with_columns(translate(records["nm"], mapping).alias("nm"), translate(records["ad"], mapping).alias("ad"))
    fr = records["co"].str.to_lowercase().eq("france")
    if fr.any():
        names = records["nm"].str.replace_all(suffix, " ").str.replace_all(r"\s+", " ").str.strip_chars()
        addresses = records["ad"].filter(fr).unique()
        lut = {a: france_address(a) for a in addresses}
        records = records.with_columns(pl.when(fr).then(names).otherwise(pl.col("nm")).alias("nm"),
                                       pl.when(fr).then(pl.col("ad").replace(lut)).otherwise(pl.col("ad")).alias("ad"))
    return records


def check():
    import tempfile as tf
    pairs = [("राम", "नदी सड़क", "ram", "river road")] * 3
    mapping = learn(pairs)
    assert mapping["राम"] == "ram"
    frame = pl.DataFrame({"nm": ["राम", "ciel ecole & fils", "fils bakery"],
                          "ad": ["नदी सड़क", "27 rue du nord, nord", "27 rue du nord, nord"], "co": ["india", "france", "us"]})
    out = apply(frame, mapping)
    assert out["nm"][0] == "ram" and "fils" not in out["nm"][1] and out["nm"][2] == "fils bakery"
    assert out["ad"][1] == "27 rue du nord, hauts-de-france" and out["ad"][2] == frame["ad"][2]
    with tf.TemporaryDirectory() as tmp:
        data = infer._check_data(path(tmp))
        for sr in (2, 3):
            p = data / "train" / f"s{sr}.parquet"
            d = pl.read_parquet(p).with_columns(pl.lit("राम").alias("nm"))
            d.write_parquet(p)
        assert training_pairs(data).is_empty()
        p = data / "train/s2.parquet"
        d = pl.read_parquet(p).with_columns(pl.lit(3).cast(pl.Int32).alias("own"))
        d.write_parquet(p)
        assert len(training_pairs(data)) == len(d)
    print("fold-safe normalization checks passed")


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=path, default=path("cache/data"))
    p.add_argument("--out", type=path, default=path("artifacts/norm2.json"))
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.check:
        check()
    else:
        model = fit(a.data, a.out)
        print({"pairs": model["pairs"], "tokens": len(model["mapping"]), "out": str(a.out)})

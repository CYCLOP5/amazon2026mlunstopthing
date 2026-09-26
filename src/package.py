#!/usr/bin/env python3
"""build a checked, reproducible final submission archive."""

import argparse as ap
import hashlib as hh
import json
import math
import os
import re
import shutil
import tempfile as tf
import zipfile as zf
from pathlib import Path as path

try:
    from validate import validate
except ModuleNotFoundError:
    from src.validate import validate


lim = 8_000_000_000
lic = {"mit", "apache-2.0", "apache2", "apache 2.0"}
tok = {
    "config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "sentencepiece.bpe.model", "spiece.model", "vocab.txt", "vocab.json", "merges.txt",
}


def sha(p):
    h = hh.sha256()
    with path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jsha(x):
    return hh.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def load(p):
    try:
        return json.loads(path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"invalid json {p}") from e


def rel(p, base):
    try:
        return path(p).resolve().relative_to(path(base).resolve())
    except ValueError as e:
        raise ValueError(f"path escapes its root: {p}") from e


def file(p, why):
    p = path(p)
    if not p.is_file() or p.is_symlink():
        raise ValueError(f"missing or unsafe {why}: {p}")
    return p


def add(out, arc, p):
    if not isinstance(arc, str) or arc.startswith("/") or ".." in path(arc).parts:
        raise ValueError(f"unsafe archive path {arc}")
    p = file(p, arc)
    if arc in out:
        raise ValueError(f"duplicate archive path {arc}")
    out[arc] = p


def srcs(root, out):
    d = path(root) / "src"
    if not d.is_dir():
        raise ValueError(f"missing source directory {d}")
    bad = {".aws", ".azure", ".git", ".venv", "__pycache__", "artifacts", "cache", "credentials", "data", "models", "output"}
    for p in sorted(d.rglob("*")):
        r = rel(p, d)
        if (not p.is_file() or p.is_symlink() or any(x in bad for x in r.parts) or
                p.name.lower() in {"credentials.json", "secrets.json"} or p.name.startswith(".env") or
                p.suffix in {".pyc", ".pyo", ".pem", ".key", ".parquet", ".tsv"}):
            continue
        add(out, f"code/business_entity_resolution/src/{r.as_posix()}", p)
    if not any(x.startswith("code/business_entity_resolution/src/") for x in out):
        raise ValueError("no source files")


def doc(p, what, filled=False):
    p = file(p, what)
    s = p.read_text(encoding="utf-8").strip()
    if not s:
        raise ValueError(f"empty {what}")
    if filled and any(x in s for x in ("[Your Team Name]", "[List all team members]", "[Date]", "[total]")):
        raise ValueError("methodology document still has template placeholders")
    return p


def env(root, out):
    root = path(root)
    got = False
    for n in ("pyproject.toml", "uv.lock", ".python-version"):
        p = root / n
        if p.is_file():
            add(out, f"code/business_entity_resolution/{n}", p)
            got = True
    if not got:
        p = root / "requirements.txt"
        if not p.is_file():
            raise ValueError("missing pyproject.toml/uv.lock/.python-version or requirements.txt")
        add(out, "code/business_entity_resolution/requirements.txt", p)


def gate(d, out):
    d = path(d)
    m = load(d / "metadata.json")
    ms, fs = m.get("models"), m.get("model_files")
    if not isinstance(ms, list) or not ms or not isinstance(fs, dict):
        raise ValueError("gate metadata has no declared models")
    add(out, "code/business_entity_resolution/models/gate/metadata.json", d / "metadata.json")
    for n in ms:
        x = fs.get(n)
        if not isinstance(n, str) or not isinstance(x, str) or path(x).name != x:
            raise ValueError("invalid gate model file")
        add(out, f"code/business_entity_resolution/models/gate/{x}", d / x)
    if m.get("feature_backend") in ("hybrid-v2", "hybrid-v3"):
        from norm2 import load as load_normalizer
        normalizer = m.get("normalizer", {})
        name = normalizer.get("file")
        if not isinstance(name, str) or path(name).name != name or sha(d / name) != normalizer.get("sha256"):
            raise ValueError("invalid packaged gate normalizer")
        load_normalizer(d / name)
        add(out, f"code/business_entity_resolution/models/gate/{name}", d / name)
    return m


def neural(d, out, prefix="code/business_entity_resolution/models/neural"):
    d = path(d)
    m = load(d / "neural_metadata.json")
    n = m.get("parameters")
    s = m.get("source")
    if not isinstance(n, int) or not 0 <= n <= lim:
        raise ValueError("invalid neural parameter count")
    if m.get("problem_type") != "multi_label_classification":
        raise ValueError("invalid neural model metadata")
    if not isinstance(s, dict) or str(s.get("license", "")).lower() not in lic:
        raise ValueError("neural model license is not mit or apache-2.0")
    add(out, f"{prefix}/neural_metadata.json", d / "neural_metadata.json")
    if m.get("architecture") == "neural-ensemble-v1":
        from neural import member_paths
        children = [neural(folder, out, f"{prefix}/{folder.name}") for _, folder in member_paths(d, m)]
        if sum(x["parameters"] for x in children) != n:
            raise ValueError("packaged ensemble parameter count mismatch")
        return {**m, "sources": [x["source"] for x in children]}
    ws = sorted(x for x in d.iterdir() if x.is_file() and not x.name.startswith(("optimizer", "rng")) and
                (x.suffix == ".safetensors" or x.name.startswith("pytorch_model") and x.suffix == ".bin"))
    if not ws:
        raise ValueError("neural model has no inference weights")
    for p in ws:
        add(out, f"{prefix}/{p.name}", p)
    for n in sorted(tok):
        if (d / n).is_file():
            add(out, f"{prefix}/{n}", d / n)
    if f"{prefix}/config.json" not in out:
        raise ValueError("neural model has no config.json")
    return m


def sources(p, nn, retrievers):
    xs = load(p)
    if not isinstance(xs, list):
        raise ValueError("model source report is not a list")
    by = {(x.get("model"), x.get("revision")): x for x in xs if isinstance(x, dict)}

    def one(model, revision):
        x = by.get((model, revision))
        if not isinstance(model, str) or not isinstance(revision, str) or x is None:
            raise ValueError("selected model is absent from model source report")
        if str(x.get("license", "")).lower() not in lic:
            raise ValueError("source report has an ineligible model license")
        if not isinstance(x.get("parameters"), int) or not 0 <= x["parameters"] <= lim:
            raise ValueError("source report has an invalid model parameter count")
        return {k: x[k] for k in ("model", "revision", "license", "parameters", "source") if k in x}

    rs = [one(x.get("model"), x.get("revision")) for x in retrievers]
    ns = [one(s.get("model"), s.get("revision")) for s in nn.get("sources", [nn["source"]])]
    if sum(x["parameters"] for x in rs) + nn["parameters"] > lim:
        raise ValueError("selected retrieval and neural models exceed 8b parameters")
    return rs, [*rs, *ns]


def calibration(p, gate_dir, neural_dir):
    try:
        import match
    except ModuleNotFoundError:
        from src import match

    cal = load(file(p, "final calibration"))
    if not isinstance(cal, dict):
        raise ValueError("final calibration is not an object")
    segmented = cal.get("kind") == "segmented-postprocessor"
    tune = cal.get("tune")
    cov, sel, cfg = cal.get("train_coverage"), tune.get("selected") if isinstance(tune, dict) else None, cal.get("config")
    if segmented:
        from post import validate_recipe
        validate_recipe(cal)
        sel = {"decoder": "expected_f05", "threshold": cal.get("floor")}
        if (cal.get("version") != 1 or not isinstance(cal.get("exact_limit"), int) or not 1 <= cal["exact_limit"] <= 256 or
                not isinstance(cal.get("curves"), dict) or not cal["curves"]):
            raise ValueError("invalid segmented postprocessor")
    if (cal.get("kind") not in ("full-corpus-calibration", "segmented-postprocessor") or not isinstance(cov, dict) or cov.get("complete") is not True or
            not isinstance(cov.get("targets"), int) or cov["targets"] < 1):
        raise ValueError("calibration does not declare complete full-target training coverage")
    decoders = {"expected_f05"} if segmented else {"plain_threshold", "target_top1_then_threshold"}
    if (not isinstance(sel, dict) or sel.get("decoder") not in decoders or
            isinstance(sel.get("threshold"), bool) or not isinstance(sel.get("threshold"), (int, float)) or
            not math.isfinite(sel["threshold"])):
        raise ValueError("calibration has no selected decoder and finite cutoff")
    if not isinstance(cfg, dict) or cal.get("config_sha256") != jsha(cfg):
        raise ValueError("invalid calibration configuration fingerprint")
    _, names, dense, gi = match._gate(gate_dir)
    _, ni = match._neural(neural_dir)
    gate_cfg = {**gi, "feature_names": names, "dense_features": dense, "score_version": match.block.sv}
    if cfg.get("gate") != gate_cfg or cfg.get("neural") != ni:
        raise ValueError("calibration and supplied model fingerprints mismatch")
    numeric = cfg.get("numeric")
    blend = numeric.get("blend") if isinstance(numeric, dict) else None
    weight = blend.get("neural_weight") if isinstance(blend, dict) else None
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("calibration has an invalid neural weight")
    model = {"sha256": jsha({"gate": gi["sha256"], "neural": ni["sha256"], "neural_weight": weight})}
    if cfg.get("stack"):
        if cal.get("stack_model", cfg["stack"]).get("sha256") != cfg["stack"]["sha256"]:
            raise ValueError("stack model and calibration configuration differ")
        model = {"sha256": jsha({"base": model["sha256"], "stack": cfg["stack"]["sha256"]})}
    if cfg.get("model") != model or cal.get("model_sha256") != model["sha256"]:
        raise ValueError("calibration model fingerprint mismatch")
    retrievers = cfg.get("retrievers")
    if not isinstance(retrievers, list) or not retrievers or not all(isinstance(x, dict) for x in retrievers):
        raise ValueError("calibration has no selected retrievers")
    match._retrievers(retrievers)
    return cal, retrievers


def hf(cache, xs, out):
    cache = path(cache).resolve()
    if not cache.is_dir():
        raise ValueError(f"missing hf cache {cache}")
    for x in xs:
        model, rev = x["model"], x["revision"]
        if path(rev).name != rev:
            raise ValueError("unsafe hf revision")
        name = "models--" + model.replace("/", "--")
        d = cache / name / "snapshots" / rev
        if not d.is_dir() or d.is_symlink():
            raise ValueError(f"missing safe hf snapshot {d}")
        before = len(out)
        for base, ds, fs in os.walk(d, followlinks=False):
            ds[:] = sorted(x for x in ds if not (path(base) / x).is_symlink())
            for n in sorted(fs):
                p = path(base) / n
                if not p.is_file():
                    continue
                try:
                    rel(p.resolve(), cache)
                except ValueError as e:
                    raise ValueError(f"hf snapshot link escapes cache: {p}") from e
                r = p.relative_to(d).as_posix()
                add(out, f"code/business_entity_resolution/models/hf/{name}/snapshots/{rev}/{r}", p.resolve())
        if len(out) == before:
            raise ValueError(f"empty hf snapshot {d}")


def trained_retrievers(retrievers, roots, out):
    from retr import bundle
    trained = {}
    for root in roots:
        meta, digest = bundle(root)
        if digest in trained:
            raise ValueError("duplicate trained retriever")
        trained[digest] = path(root), meta
    specs, used = [], set()
    for row in retrievers:
        spec = {"model": row["model"], "revision": row["revision"]}
        digest = row.get("checkpoint_sha256")
        if digest:
            if digest not in trained:
                raise ValueError("selected trained retriever weights were not supplied")
            root, meta = trained[digest]
            if meta["model"] != row["model"] or meta["base_revision"] != row["revision"]:
                raise ValueError("trained retriever origin mismatch")
            folder = f"code/business_entity_resolution/models/retrievers/{digest}"
            for name in [*meta["files"], "retriever.json"]:
                add(out, f"{folder}/{name}", root / name)
            spec.update({"checkpoint": folder, "checkpoint_sha256": digest})
            used.add(digest)
        specs.append(spec)
    if set(trained) != used:
        raise ValueError("unselected trained retriever supplied")
    return specs


def reverse_assets(retrievers, specs, roots, out):
    import reverse
    import hybrid
    available, used = {}, set()
    for root in roots:
        metadata, digest = reverse.config(root)
        if digest in available:
            raise ValueError("duplicate reverse cache")
        available[digest] = path(root), metadata
    for row, spec in zip(retrievers, specs):
        digest = row.get("reverse_contract_sha256")
        if not digest:
            continue
        if digest not in available:
            raise ValueError("selected reverse retrieval cache was not supplied")
        root, metadata = available[digest]
        if metadata["model_sha256"] != spec.get("checkpoint_sha256"):
            raise ValueError("reverse retrieval encoder differs")
        source = out.get("code/business_entity_resolution/src/reverse.py")
        if source is None or sha(source) != metadata["code_sha256"]:
            raise ValueError("reverse retrieval code snapshot differs")
        folder = f"code/business_entity_resolution/models/reverse/{digest}"
        add(out, folder + "/config.json", root / "config.json")
        for split, countries in metadata["countries"].items():
            for country in countries:
                name = f"{split}_{hybrid.tag(country)}"
                directory = root / name
                meta = load(directory / "metadata.json")
                if (meta.get("contract_sha256") != digest or meta.get("split") != split or meta.get("country") != country or
                        set(meta.get("files", {})) != {"qids.npy", "ranks.npy", "offsets.npy"}):
                    raise ValueError("reverse retrieval coverage is incomplete")
                for filename, expected in meta["files"].items():
                    if sha(file(directory / filename, filename)) != expected:
                        raise ValueError("reverse retrieval data changed")
                    add(out, f"{folder}/{name}/{filename}", directory / filename)
                add(out, f"{folder}/{name}/metadata.json", directory / "metadata.json")
        spec.update({"reverse_root": folder, "reverse_contract_sha256": digest})
        used.add(digest)
    if set(available) != used:
        raise ValueError("unselected reverse cache supplied")


def manifest(files, prov, offline, retrievers=None):
    z = {"format": 2, "archive_root": "code/business_entity_resolution", "calibration": "models/calibration.json",
          "files": [], "offline_retriever_weights": offline, "selected_model_provenance": prov,
          "retriever_specs": retrievers or []}
    for n, p in sorted(files.items()):
        z["files"].append({"path": n, "size": p.stat().st_size, "sha256": sha(p)})
    z["package_manifest"] = {"path": "package_manifest.json", "sha256": "excluded"}
    return (json.dumps(z, indent=2, sort_keys=True) + "\n").encode()


def write(dst, files, man):
    part = dst.with_name(dst.name + ".part")
    if dst.exists() or part.exists():
        raise ValueError(f"refusing to overwrite {dst if dst.exists() else part}")
    try:
        with zf.ZipFile(part, "x", zf.ZIP_DEFLATED, allowZip64=True, compresslevel=9) as z:
            for n, p in sorted(files.items()):
                info = zf.ZipInfo(n, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zf.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                with p.open("rb") as a, z.open(info, "w", force_zip64=True) as b:
                    shutil.copyfileobj(a, b, 1 << 20)
            info = zf.ZipInfo("package_manifest.json", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zf.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with z.open(info, "w", force_zip64=True) as f:
                f.write(man)
        part.replace(dst)
    except Exception:
        part.unlink(missing_ok=True)
        raise


def build(matching, candidate, test_dir, repo_root, code_root, readme, methodology, gate_dir,
          neural_dir, calibration_file, output_zip, team_name=None, hf_cache=None, trained=None, stack_dir=None, reverse_roots=None):
    matching, candidate, test_dir = map(path, (matching, candidate, test_dir))
    repo, code, dst = map(path, (repo_root, code_root, output_zip))
    if team_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", team_name):
        raise ValueError("unsafe team name")
    if team_name and dst.name != f"{team_name}_submission.zip":
        raise ValueError("output zip must match team name")
    validate(matching, candidate, test_dir)
    out = {}
    add(out, "output/matching_results.tsv", matching)
    add(out, "output/candidate_pairs.tsv", candidate)
    srcs(code, out)
    add(out, "code/business_entity_resolution/README.md", doc(readme, "README"))
    env(code, out)
    pp = repo / "reports/model_sources.json"
    add(out, "code/business_entity_resolution/reports/model_sources.json", file(pp, "model source report"))
    gate(gate_dir, out)
    nn = neural(neural_dir, out)
    cal, retrievers = calibration(calibration_file, gate_dir, neural_dir)
    selected_stack = cal.get("stack_model") or cal.get("config", {}).get("stack")
    if selected_stack:
        from stack2 import bundle
        if stack_dir is None:
            raise ValueError("selected pairwise stack weights were not supplied")
        stack, digest = bundle(stack_dir)
        if digest != selected_stack["sha256"]:
            raise ValueError("pairwise stack weights differ from calibration")
        base = dict(cal["config"])
        base.pop("stack")
        base["model"] = {"sha256": jsha({"gate": base["gate"]["sha256"], "neural": base["neural"]["sha256"],
                                        "neural_weight": base["numeric"]["blend"]["neural_weight"]})}
        if jsha(base) != stack["source_config_sha256"] or base["data_meta_sha256"] != stack["data_meta_sha256"]:
            raise ValueError("pairwise stack was fitted to different base scores")
        for name in [*stack["files"], "metadata.json"]:
            add(out, f"code/business_entity_resolution/models/stack/{name}", path(stack_dir) / name)
    elif stack_dir is not None:
        raise ValueError("unselected pairwise stack supplied")
    if cal.get("rules"):
        rule_code = out.get("code/business_entity_resolution/src/frule.py")
        if rule_code is None or sha(rule_code) != cal["rules"]["code_sha256"]:
            raise ValueError("name-pattern rule implementation differs from calibration")
        normalizers = [p for name, p in out.items() if name.endswith("/normalizer.json")]
        if not any(sha(p) == cal["rules"]["normalizer_sha256"] for p in normalizers):
            raise ValueError("name-pattern normalization model was not supplied")
    add(out, "code/business_entity_resolution/models/calibration.json", file(calibration_file, "final calibration"))
    selected, prov = sources(pp, nn, retrievers)
    specs = trained_retrievers(retrievers, trained or [], out)
    reverse_assets(retrievers, specs, reverse_roots or [], out)
    frozen = [s for s, r in zip(selected, retrievers) if not r.get("checkpoint_sha256")]
    if frozen and hf_cache is None:
        raise ValueError("hf cache is required for selected retrievers")
    if frozen:
        hf(hf_cache, frozen, out)
    if (repo / "plan.md").is_file():
        add(out, "plan.md", repo / "plan.md")
    for name in ("readme.md", "e5-mit.txt", "qwen3-apache-2.0.txt"):
        p = repo / "licenses" / name
        if p.is_file():
            add(out, "code/business_entity_resolution/licenses/" + name, p)
    add(out, "Documentation_template.md", doc(methodology, "methodology document", filled=True))
    man = manifest(out, prov, bool(selected), specs)
    dst.parent.mkdir(parents=True, exist_ok=True)
    write(dst, out, man)
    return {"zip": str(dst), "sha256": sha(dst), "files": len(out) + 1,
            "offline_retriever_weights": bool(selected)}


def put(p, s):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def check():
    with tf.TemporaryDirectory() as t:
        try:
            import match
        except ModuleNotFoundError:
            from src import match

        rs = match.hybrid.specs()
        r = path(t) / "repo"
        c, raw, out = r, r / "raw", r / "out"
        pp = c / "reports/model_sources.json"
        put(c / "src/run.py", "print('ok')\n")
        put(c / "README.md", "run src/run.py\n")
        put(c / "pyproject.toml", "[project]\nname = 'x'\nversion = '0'\n")
        put(pp, json.dumps([
            {"model": rs[0]["model"], "revision": rs[0]["revision"], "license": "mit", "parameters": 278044162},
            {"model": rs[1]["model"], "revision": rs[1]["revision"], "license": "apache-2.0", "parameters": 595776512},
            {"model": rs[2]["model"], "revision": rs[2]["revision"], "license": "mit", "parameters": 559890432},
            {"model": "org/model", "revision": "rev", "license": "mit", "parameters": 2},
        ]))
        put(c / "plan.md", "plan\n")
        put(c / "licenses/e5-mit.txt", "test license notice\n")
        hd = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        put(raw / "test_source1.tsv", hd + "S1-a\ta\ta\tUS\n")
        put(raw / "test_source2.tsv", hd + "S2-a\ta\ta\tUS\n")
        put(raw / "test_source3.tsv", hd)
        put(out / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a\n")
        put(out / "candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids\nS1-a\tS2-a\n")
        gd = r / "gate"; gd.mkdir()
        put(gd / "gate.bin", "gate")
        put(gd / "checkpoint.bin", "skip")
        put(gd / "metadata.json", json.dumps({"score_version": match.block.sv, "feature_names": match.feat.ff,
                                                "dense_features": [], "models": ["lgb"],
                                                "model_files": {"lgb": "gate.bin"}}))
        nd = r / "neural"; nd.mkdir()
        put(nd / "model.safetensors", "weights")
        put(nd / "config.json", "{}")
        put(nd / "tokenizer.json", "{}")
        put(nd / ".env", "secret")
        put(nd / "neural_metadata.json", json.dumps({"parameters": 2, "problem_type": "multi_label_classification", "source": {"model": "org/model", "revision": "rev", "license": "mit"}}))
        nn = load(nd / "neural_metadata.json")
        (nd / "checkpoint-1").mkdir()
        put(nd / "checkpoint-1/model.safetensors", "checkpoint")
        _, names, dense, gi = match._gate(gd)
        _, ni = match._neural(nd)
        data = r / "data"; put(data / "meta.json", "{}")
        cfg, config_sha = match._cfg(data, {**gi, "feature_names": names, "dense_features": dense,
                                            "score_version": match.block.sv}, ni, rs, 1, 2, 3, .6)
        cal = {"version": 1, "kind": "full-corpus-calibration", "config": cfg, "config_sha256": config_sha,
               "model_sha256": cfg["model"]["sha256"], "train_coverage": {"complete": True, "targets": 4},
               "tune": {"selected": {"decoder": "plain_threshold", "threshold": .5}}}
        cp = r / "calibration.json"; put(cp, json.dumps(cal))
        put(r / "Documentation_template.md", "# methodology\nfilled\n")
        cache = r / "hf"
        sns = []
        for i, x in enumerate(cfg["retrievers"]):
            name = "models--" + x["model"].replace("/", "--")
            blob = cache / name / "blobs" / f"weight-{i}"
            put(blob, f"weight-{i}")
            sn = cache / name / "snapshots" / x["revision"]
            sn.mkdir(parents=True)
            (sn / "weights.bin").symlink_to(f"../../blobs/weight-{i}")
            sns.append(sn)
        z = build(out / "matching_results.tsv", out / "candidate_pairs.tsv", raw, r, c, c / "README.md",
                  r / "Documentation_template.md", gd, nd, cp, r / "team_submission.zip", "team", cache)
        with zf.ZipFile(z["zip"]) as x:
            ns = x.namelist()
            man = json.loads(x.read("package_manifest.json"))
            assert x.testzip() is None
            assert "code/business_entity_resolution/models/calibration.json" in ns
            assert x.read("code/business_entity_resolution/licenses/e5-mit.txt") == b"test license notice\n"
            assert all(f"code/business_entity_resolution/models/hf/{p.parents[1].name}/snapshots/{p.name}/weights.bin" in ns
                       for p in sns)
            assert x.read(f"code/business_entity_resolution/models/hf/{sns[0].parents[1].name}/snapshots/{sns[0].name}/weights.bin") == b"weight-0"
            assert not any("bge" in n for n in ns)
            assert not any("checkpoint" in n or ".env" in n for n in ns)
            assert man["offline_retriever_weights"] is True
            assert man["selected_model_provenance"][0]["model"] == rs[0]["model"]
            wp = f"code/business_entity_resolution/models/hf/models--{rs[0]['model'].replace('/', '--')}/snapshots/{rs[0]['revision']}/weights.bin"
            assert next(x for x in man["files"] if x["path"] == wp)["sha256"] == sha(cache / ("models--" + rs[0]["model"].replace("/", "--")) / "blobs/weight-0")
        try:
            build(out / "matching_results.tsv", out / "candidate_pairs.tsv", raw, r, c, c / "README.md",
                  r / "Documentation_template.md", gd, nd, cp, r / "nohf.zip")
        except ValueError as e:
            assert "hf cache" in str(e)
        else:
            raise AssertionError("missing hf cache accepted")
        shutil.rmtree(sns[-1])
        try:
            build(out / "matching_results.tsv", out / "candidate_pairs.tsv", raw, r, c, c / "README.md",
                  r / "Documentation_template.md", gd, nd, cp, r / "missing.zip", hf_cache=cache)
        except ValueError as e:
            assert "snapshot" in str(e)
        else:
            raise AssertionError("missing selected revision accepted")
        put(out / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-no\n")
        try:
            build(out / "matching_results.tsv", out / "candidate_pairs.tsv", raw, r, c, c / "README.md",
                  r / "Documentation_template.md", gd, nd, cp, r / "bad.zip")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid output accepted")
        put(out / "matching_results.tsv", "source1_entity_id\tmatched_entity_ids\nS1-a\tS2-a\n")
        cal["train_coverage"]["complete"] = False
        badcal = r / "partial.json"; put(badcal, json.dumps(cal))
        try:
            build(out / "matching_results.tsv", out / "candidate_pairs.tsv", raw, r, c, c / "README.md",
                  r / "Documentation_template.md", gd, nd, badcal, r / "partial.zip")
        except ValueError:
            pass
        else:
            raise AssertionError("partial calibration accepted")
        for p in (c / "README.md", r / "Documentation_template.md"):
            q = p.with_suffix(p.suffix + ".old")
            p.replace(q)
            try:
                build(out / "matching_results.tsv", out / "candidate_pairs.tsv", raw, r, c, p,
                      r / "Documentation_template.md", gd, nd, cp, r / (p.name + ".zip"))
            except ValueError:
                pass
            else:
                raise AssertionError("missing required document accepted")
            q.replace(p)
        outside = r / "outside"; put(outside, "no")
        (sns[0] / "bad.bin").symlink_to(outside)
        try:
            build(out / "matching_results.tsv", out / "candidate_pairs.tsv", raw, r, c, c / "README.md",
                  r / "Documentation_template.md", gd, nd, cp, r / "unsafe.zip", hf_cache=cache)
        except ValueError:
            pass
        else:
            raise AssertionError("external hf link accepted")
        bad = load(pp)
        bad[0]["license"] = "bsd"
        put(pp, json.dumps(bad))
        try:
            sources(pp, nn, cfg["retrievers"])
        except ValueError as e:
            assert "license" in str(e)
        else:
            raise AssertionError("ineligible source license accepted")
        bad[0]["license"] = "mit"
        bad[0]["parameters"] = lim + 1
        put(pp, json.dumps(bad))
        try:
            sources(pp, nn, cfg["retrievers"])
        except ValueError as e:
            assert "parameter" in str(e)
        else:
            raise AssertionError("oversized source accepted")
    print("check: passed")


def main(argv=None):
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true")
    p.add_argument("--matching", type=path)
    p.add_argument("--candidate", type=path)
    p.add_argument("--test-dir", "--raw-test-dir", dest="test_dir", type=path)
    p.add_argument("--repo-root", type=path)
    p.add_argument("--code-root", type=path)
    p.add_argument("--readme", type=path)
    p.add_argument("--methodology", type=path)
    p.add_argument("--gate-model-dir", type=path)
    p.add_argument("--neural-model-dir", type=path)
    p.add_argument("--calibration", type=path)
    p.add_argument("--output-zip", type=path)
    p.add_argument("--team-name")
    p.add_argument("--hf-cache", type=path)
    p.add_argument("--trained-retriever", type=path, action="append", default=[])
    p.add_argument("--stack-model-dir", type=path)
    p.add_argument("--reverse-root", type=path, action="append", default=[])
    a = p.parse_args(argv)
    if a.check:
        check()
        return 0
    need = ("matching", "candidate", "test_dir", "repo_root", "code_root", "readme", "methodology",
            "gate_model_dir", "neural_model_dir", "calibration", "output_zip")
    if any(getattr(a, x) is None for x in need):
        p.error("all package inputs are required")
    print(json.dumps(build(a.matching, a.candidate, a.test_dir, a.repo_root, a.code_root, a.readme, a.methodology,
                            a.gate_model_dir, a.neural_model_dir, a.calibration, a.output_zip, a.team_name,
                             a.hf_cache, a.trained_retriever, a.stack_model_dir, a.reverse_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

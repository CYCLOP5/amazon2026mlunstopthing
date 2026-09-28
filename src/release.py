'build and verify the two final submission archives'
import argparse as ap
import hashlib as hh
import json
import zipfile as zf
from pathlib import Path as path


def sha(p):
    with p.open("rb") as f:
        return hh.file_digest(f, "sha256").hexdigest()


def sources(root):
    suffixes = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".lock", ".sh", ".mmd", ".svg"}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if rel.parts[0] in {'models', 'resume', 'training', '.model-downloads'} or rel.as_posix() in {'reproduction_manifest.json', 'model-fetch.json'}:
            continue
        if not p.is_file() or any(k in rel.parts for k in ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox", ".venv", ".git", "assets", "output")):
            continue
        if p.name in {"Ml_Challenge.txt", "Documentation_template.md"} or p.name.startswith(".env"):
            continue
        if p.suffix in suffixes or p.name in {".python-version", ".gitignore", ".amlignore"}:
            if p.is_symlink() and not p.resolve().is_relative_to(root.resolve()):
                raise ValueError("source symlink escapes the project")
            yield p, rel.as_posix()


def doc(source, spec):
    score = str(spec["reported_public_score"]) if spec["reported_public_score"] is not None else "not separately recorded; parent sprint2: 0.990284"
    text = (source / "Documentation_template.md").read_text()
    for k, v in {"variant": spec["variant"], "matches": f'{spec["matches"]:,}', "score": score}.items():
        text = text.replace("{{" + k + "}}", v)
    if "{{" in text:
        raise ValueError("unresolved documentation field")
    return text


def verify(file):
    with zf.ZipFile(file) as z:
        entries = z.namelist()
        required = {"output/matching_results.tsv", "output/candidate_pairs.tsv", "code/business_entity_resolution/src/finish.py",
                    "code/business_entity_resolution/README.md", "code/business_entity_resolution/requirements.txt",
                    "code/business_entity_resolution/configs/release.json", "Documentation_template.md", "manifest.json"}
        if len(entries) != len(set(entries)) or not required.issubset(entries):
            raise ValueError("invalid archive layout")
        if any(n.startswith("/") or ".." in path(n).parts for n in entries):
            raise ValueError("unsafe archive path")
        inv = json.loads(z.read("manifest.json"))
        if set(entries) != set(inv["files"]) | {"manifest.json"}:
            raise ValueError("archive inventory differs")
        for name, info in inv["files"].items():
            with z.open(name) as f:
                digest = hh.file_digest(f, "sha256").hexdigest()
            if digest != info["sha256"] or z.getinfo(name).file_size != info["bytes"]:
                raise ValueError("archive member differs: " + name)
        spec = json.loads(z.read("code/business_entity_resolution/configs/release.json"))
        for name, digest in spec["output_sha256"].items():
            if inv["files"]["output/" + name]["sha256"] != digest:
                raise ValueError("submitted output hash differs")
        if inv.get('reproduction') == 'trained-models-and-source':
            prefix = 'code/business_entity_resolution/'
            assets = json.loads(z.read(prefix + 'reproduction_manifest.json'))
            for name, info in assets['files'].items():
                record = inv['files'].get(prefix + name)
                if record is None or any(record[k] != info[k] for k in ('sha256', 'bytes')):
                    raise ValueError('reproduction artifact differs: ' + name)
            if prefix + 'models/collective/models/reranker.txt' not in inv['files']:
                raise ValueError('missing fitted final model')
        if inv.get('reproduction') == 'kaggle-hosted-trained-models':
            prefix = 'code/business_entity_resolution/'
            cfg = json.loads(z.read(prefix + 'configs/kaggle.json'))
            if not cfg.get('public_verified') or not isinstance(cfg.get('version'), int) or cfg['version'] < 1:
                raise ValueError('unverified or unpinned Kaggle model release')
            if inv['files'][prefix + 'reproduction_manifest.json']['sha256'] != cfg['payload_manifest_sha256']:
                raise ValueError('hosted model inventory differs')
            if not {prefix + 'src/fetch_assets.py', prefix + 'src/fetch_models.sh'}.issubset(entries):
                raise ValueError('missing automatic model fetcher')
            if file.stat().st_size > cfg['unstop_zip_max_bytes']:
                raise ValueError('archive exceeds the Unstop size limit')
    return {"archive": str(file), "bytes": file.stat().st_size, "sha256": sha(file), "files": len(entries),
            "variant": spec["variant"], "outputs": spec["output_sha256"], "verified": True,
            "crc": "verified during complete member reads", "reproduction": inv.get('reproduction', 'score-replay')}


def build(source, assets, matching, candidate, config, dest, payload=None, kaggle_assets=None):
    if payload is not None and kaggle_assets is not None:
        raise ValueError('choose embedded or Kaggle-hosted assets')
    spec = json.loads(config.read_text())
    for name, file in (("matching_results.tsv", matching), ("candidate_pairs.tsv", candidate)):
        if sha(file) != spec["output_sha256"][name]:
            raise ValueError("input is not the submitted file: " + str(file))
    asset = json.loads((assets / "manifest.json").read_text())
    if set(asset["files"]) != {"collective.parquet", "france.parquet", "swap-pool.parquet"}:
        raise ValueError("incomplete replay assets")
    for name, info in asset["files"].items():
        if sha(assets / name) != info["sha256"]:
            raise ValueError("replay asset differs: " + name)
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / "Amazites_submission.zip"
    temporary = dest / "Amazites_submission.zip.part"
    if final.exists() or temporary.exists():
        raise FileExistsError(final)
    prefix = "code/business_entity_resolution/"
    inv = {"schema": 1, "team": "Amazites", "variant": spec["variant"], "files": {}}
    if payload is not None:
        inv['reproduction'] = 'trained-models-and-source'
        import importlib.util
        module_spec = importlib.util.spec_from_file_location('reproduction', source / 'src/reproduce.py')
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.verify(payload.resolve())
        payload_manifest = json.loads((payload / 'manifest.json').read_text())
    if kaggle_assets is not None:
        cfg = json.loads((source / 'configs/kaggle.json').read_text())
        if not cfg.get('public_verified') or sha(kaggle_assets / 'manifest.json') != cfg['payload_manifest_sha256']:
            raise ValueError('Kaggle publication must be verified before packaging')
        inv['reproduction'] = 'kaggle-hosted-trained-models'
        inv['model_dataset'] = {'dataset': cfg['dataset'], 'version': cfg['version'], 'manifest_sha256': cfg['payload_manifest_sha256']}
    with zf.ZipFile(temporary, "w", compression=zf.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        def add(file, name):
            inv["files"][name] = {"sha256": sha(file), "bytes": file.stat().st_size}
            z.write(file, name, compress_type=zf.ZIP_STORED if file.suffix in {'.parquet', '.npy', '.npz'} else zf.ZIP_DEFLATED,
                    compresslevel=1 if file.suffix in {'.safetensors', '.bin'} else 6)

        def text(value, name):
            raw = value.encode()
            inv["files"][name] = {"sha256": hh.sha256(raw).hexdigest(), "bytes": len(raw)}
            z.writestr(name, raw)

        for file, rel in sources(source):
            if rel != "configs/release.json":
                add(file, prefix + rel)
        text(json.dumps(spec, indent=2) + "\n", prefix + "configs/release.json")
        for name in sorted(asset["files"]):
            add(assets / name, prefix + "assets/" + name)
        add(assets / "manifest.json", prefix + "assets/manifest.json")
        if payload is not None:
            for name, expected in sorted(payload_manifest['files'].items()):
                add(payload / name, prefix + name)
                if any(inv['files'][prefix + name][k] != expected[k] for k in ('sha256', 'bytes')):
                    raise ValueError('payload changed during packaging: ' + name)
            add(payload / 'manifest.json', prefix + 'reproduction_manifest.json')
        if kaggle_assets is not None:
            add(kaggle_assets / 'manifest.json', prefix + 'reproduction_manifest.json')
        add(matching, "output/matching_results.tsv")
        add(candidate, "output/candidate_pairs.tsv")
        text(doc(source, spec), "Documentation_template.md")
        z.writestr("manifest.json", json.dumps(inv, indent=2) + "\n")
    receipt = verify(temporary)
    temporary.replace(final)
    receipt["archive"] = str(final)
    (dest / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    return receipt


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=path, default=path("submission"))
    p.add_argument("--assets", type=path, default=path("artifacts/package-assets"))
    p.add_argument("--sprint2", type=path, default=path("artifacts/release-inputs/sprint2/matching_results.tsv"))
    p.add_argument("--final", type=path, default=path("artifacts/release-inputs/final-france/matching_results.tsv"))
    p.add_argument("--candidate", type=path, default=path("artifacts/release-inputs/candidate_pairs.tsv"))
    p.add_argument("--out", type=path, default=path("artifacts/final-packages"))
    p.add_argument("--verify", type=path)
    p.add_argument("--render-doc", type=path)
    p.add_argument('--payload', type=path)
    p.add_argument('--kaggle-assets', type=path)
    p.add_argument('--final-only', action='store_true')
    a = p.parse_args()
    if a.verify:
        print(json.dumps(verify(a.verify), indent=2))
    elif a.render_doc:
        a.render_doc.write_text(doc(a.source, json.loads((a.source / "configs/release.json").read_text())))
    else:
        if not a.final_only:
            build(a.source, a.assets, a.sprint2, a.candidate, a.source / "configs/release.json", a.out / "sprint2", a.payload, a.kaggle_assets)
        build(a.source, a.assets, a.final, a.candidate, a.source / "configs/final.json", a.out / "final-france", a.payload, a.kaggle_assets)

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
    suffixes = {".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".lock", ".sh"}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
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
        if z.testzip() is not None:
            raise ValueError("archive crc check failed")
    return {"archive": str(file), "bytes": file.stat().st_size, "sha256": sha(file), "files": len(entries),
            "variant": spec["variant"], "outputs": spec["output_sha256"], "verified": True}


def build(source, assets, matching, candidate, config, dest):
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
    with zf.ZipFile(temporary, "w", compression=zf.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        def add(file, name):
            inv["files"][name] = {"sha256": sha(file), "bytes": file.stat().st_size}
            z.write(file, name, compress_type=zf.ZIP_STORED if file.suffix == ".parquet" else zf.ZIP_DEFLATED)

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
    p.add_argument("--sprint2", type=path, default=path.home() / "Downloads/Amazites_sprint_2_matching_results.tsv")
    p.add_argument("--final", type=path, default=path.home() / "Downloads/Amazites_final_france_matching_results.tsv")
    p.add_argument("--candidate", type=path, default=path.home() / "Downloads/Amazites_submission/output/candidate_pairs.tsv")
    p.add_argument("--out", type=path, default=path("artifacts/final-packages"))
    p.add_argument("--verify", type=path)
    p.add_argument("--render-doc", type=path)
    a = p.parse_args()
    if a.verify:
        print(json.dumps(verify(a.verify), indent=2))
    elif a.render_doc:
        a.render_doc.write_text(doc(a.source, json.loads((a.source / "configs/release.json").read_text())))
    else:
        build(a.source, a.assets, a.sprint2, a.candidate, a.source / "configs/release.json", a.out / "sprint2")
        build(a.source, a.assets, a.final, a.candidate, a.source / "configs/final.json", a.out / "final-france")

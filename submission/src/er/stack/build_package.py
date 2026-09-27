'assemble the final submission package <team>_submission.zip (streaming copy + zip, low memory)'
import argparse
import hashlib
import os
import shutil
import zipfile

SKIP_DIRS = {"__pycache__", ".venv", ".git", ".pytest_cache", "cache", "artifacts", "work", "experiments",
             "_scan_inputs", "final_submission", "node_modules", "stackwork"}
SKIP_FILES = {"CONTEXT_HANDOFF.md", "RUNS_HANDOFF.md", "Documentation_template.md"}
SKIP_EXT = {".parquet", ".npy", ".tsv", ".pt", ".bin", ".safetensors", ".zip", ".pyc"}
MAX_BYTES = 20_000_000


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def copy_tree(src, dst, skipped):
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for f in sorted(files):
            p = os.path.join(root, f)
            if f in SKIP_FILES or os.path.splitext(f)[1].lower() in SKIP_EXT or os.path.getsize(p) > MAX_BYTES:
                skipped.append(p)
                continue
            q = os.path.join(dst, os.path.relpath(p, src))
            os.makedirs(os.path.dirname(q), exist_ok=True)
            shutil.copyfile(p, q)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-output", required=True, help="dir with the final matching_results.tsv + candidate_pairs.tsv")
    ap.add_argument("--ours", default=".", help="code/business_entity_resolution (ours)")
    ap.add_argument("--neural", required=True, help="Varun's repo root (amazon2026mlunstopthing-impl)")
    ap.add_argument("--fusion", required=True, help="Raj's repo root (Amazites_submission snapshot)")
    ap.add_argument("--docs", required=True, help="dir with README.md, MODELS.md, requirements.txt, Documentation_template.md")
    ap.add_argument("--members", help="ensemble member metadata json -> src/neural_v2/reports/final_ensemble_members.json")
    ap.add_argument("--out", required=True, help="package folder; the zip is written next to it as <out>.zip")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists - remove it first")
    code = os.path.join(a.out, "code", "business_entity_resolution")
    skipped, counts = [], {}

    os.makedirs(os.path.join(a.out, "output"))
    sums = {}
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        shutil.copyfile(os.path.join(a.final_output, f), os.path.join(a.out, "output", f))
        sums[f] = md5(os.path.join(a.out, "output", f))
    shutil.copyfile(os.path.join(a.docs, "Documentation_template.md"), os.path.join(a.out, "Documentation_template.md"))

    os.makedirs(code)
    for f in ("README.md", "MODELS.md", "requirements.txt"):
        shutil.copyfile(os.path.join(a.docs, f), os.path.join(code, f))
    counts["ours"] = copy_tree(os.path.join(a.ours, "src"), os.path.join(code, "src"), skipped)
    counts["ours"] += copy_tree(os.path.join(a.ours, "configs"), os.path.join(code, "configs"), skipped)
    counts["ours"] += copy_tree(os.path.join(a.ours, "azure"), os.path.join(code, "azure"), skipped)
    os.makedirs(os.path.join(code, "docs"))
    for src, dst in (("FINAL_BLEND_README.md", "FINAL_BLEND_README.md"), ("FRANCE_0986_README.md", "FRANCE_0986_README.md"),
                     ("README_0989.md", "COUNTRY_MERGE_README.md"), ("README.md", "OUR_LEXICAL_PIPELINE.md")):
        shutil.copyfile(os.path.join(a.ours, src), os.path.join(code, "docs", dst))
    counts["neural_v2"] = copy_tree(a.neural, os.path.join(code, "src", "neural_v2"), skipped)
    counts["fusion"] = copy_tree(a.fusion, os.path.join(code, "src", "fusion"), skipped)
    if a.members:
        shutil.copyfile(a.members, os.path.join(code, "src", "neural_v2", "reports", "final_ensemble_members.json"))
    with open(os.path.join(code, "src", "fusion", "NAMING.md"), "w", encoding="utf-8") as f:
        f.write("# Naming note\n\nIn this part's code and docs, \"friend\" (for example `--friend`, `blend_friend`, "
                "\"friend's final blend\") means teammate Shivsharan Sanjawad's run-6 stacker and blend "
                "(`src/er` in this package), not code from outside the team.\n")
    with open(os.path.join(a.out, "MANIFEST.txt"), "w", encoding="utf-8") as f:
        for k, v in sums.items():
            f.write(f"output/{k}  md5 {v}\n")
        for k, v in counts.items():
            f.write(f"code files ({k}): {v}\n")
        f.write(f"skipped (data/binaries/handoffs/old templates): {len(skipped)}\n")

    zpath = a.out.rstrip("/\\") + ".zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, dirs, files in os.walk(a.out):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, a.out))
    print("output md5:", sums, flush=True)
    print("code files:", counts, "skipped:", len(skipped), flush=True)
    for p in skipped:
        if os.path.getsize(p) > 1_000_000:
            print("  skipped large:", p, os.path.getsize(p), flush=True)
    print("zip:", zpath, os.path.getsize(zpath), "bytes", flush=True)


if __name__ == "__main__":
    main()

'Prepare or explicitly submit three model/retrieval innovation jobs on existing CPUs'
from __future__ import annotations

import argparse
import json
import os
import re

from launch_recall_campaign import (
    BASELINE, COMPUTES, INPUTS, ROOT, SAFE_JOB_QUERY, azure_job, timestamp, write_json,
)


METHODS = ("reciprocal_retrieval", "collective_graph", "gated_ensemble")
TEST_FILES = (
    "test_model_innovation.py", "test_innovation_retrieval.py", "test_innovation_graph.py",
    "test_innovation_models.py",
)


def make_spec(campaign, method, index, directory):
    if method not in METHODS:
        raise ValueError(f"Unknown model innovation: {method}")
    name = f"amazites-model-{method.replace('_', '-')}-{campaign}"
    tests = " ".join(f"business_entity_resolution/tests/{name}" for name in TEST_FILES)
    return {
        "$schema": "https://azuremlschemas.azureedge.net/latest/commandJob.schema.json",
        "type": "command", "name": name, "display_name": name,
        "experiment_name": "amazon-ml-2026",
        "description": f"Model innovation {method}; comparison against frozen latest-fusion benchmark.",
        "tags": {"campaign": campaign, "method": method, "baseline": BASELINE},
        "compute": f"azureml:{COMPUTES[index]}",
        "code": os.path.relpath(ROOT, directory),
        "environment": {"image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"},
        "resources": {"instance_count": 1, "shm_size": "32g"},
        "limits": {"timeout": 7200},
        "environment_variables": {
            "ER_THREADS": "48", "POLARS_MAX_THREADS": "48",
            "OMP_NUM_THREADS": "48", "OPENBLAS_NUM_THREADS": "1",
        },
        "inputs": {
            key: {"type": "uri_folder",
                  "path": f"azureml://datastores/workspaceblobstore/paths/{path}",
                  "mode": "ro_mount" if key == "incumbent" else "download"}
            for key, path in INPUTS.items()
        },
        "outputs": {"out": {"type": "uri_folder", "mode": "upload"}},
        "command": (
            "set -eu; curl -LsSf https://astral.sh/uv/0.7.16/install.sh | sh; "
            'export PATH="$HOME/.local/bin:$PATH"; uv sync --frozen --python 3.13; '
            "ER_THREADS=2 POLARS_MAX_THREADS=2 ER_MIN_FREE_GB=0 "
            f"uv run --frozen --python 3.13 python -m pytest {tests} -q --disable-warnings; "
            "uv run --frozen --python 3.13 python "
            "business_entity_resolution/scripts/run_model_innovation.py "
            f"--mode {method} "
            '--data "${{inputs.data}}" --train "${{inputs.train}}/prepared-train.parquet" '
            '--test "${{inputs.test}}/prepared-test.parquet" '
            '--incumbent "${{inputs.incumbent}}" --output "${{outputs.out}}"'
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="20260927-04")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--credit-authorization", help="Record existing user authorization to use redeemed sponsorship credit.")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", args.campaign):
        parser.error("campaign must contain 1-41 lowercase letters, digits, or hyphens")
    if args.submit and not (args.credit_authorization or "").strip():
        parser.error("--submit requires recorded --credit-authorization")
    directory = ROOT / "business_entity_resolution/azure/model_innovation" / args.campaign
    manifest_path = ROOT / "artifacts/model_innovation" / args.campaign / "manifest.json"
    jobs = []
    for index, method in enumerate(METHODS):
        spec = make_spec(args.campaign, method, index, directory)
        path = directory / f"{method}.yml"
        if path.exists() and json.loads(path.read_text()) != spec:
            raise RuntimeError(f"Refusing to replace differing existing job spec {path}")
        write_json(path, spec)
        jobs.append({"method": method, "name": spec["name"], "compute": spec["compute"],
                     "spec": str(path.relative_to(ROOT)), "status": "Prepared", "accepted": False})
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if [job["name"] for job in manifest["jobs"]] != [job["name"] for job in jobs]:
            raise RuntimeError("Existing manifest has a different campaign job set")
    else:
        manifest = {
            "campaign": args.campaign, "created_at": timestamp(), "baseline": BASELINE,
            "baseline_leaderboard_f05": 0.98805,
            "compute_policy": "Three existing CPU workers only; no resource creation or scale-setting changes",
            "per_job_timeout_seconds": 7200, "jobs": jobs,
        }
    write_json(manifest_path, manifest)
    if args.validate:
        for job in manifest["jobs"]:
            res = azure_job("validate", "--file", str(ROOT / job["spec"]))
            if res.get("result") != "Succeeded":
                raise RuntimeError(f"Schema validation failed: {job['name']}: {res}")
            job["schema_validation"] = "Succeeded"
            write_json(manifest_path, manifest)
    if args.submit:
        base = azure_job("show", "--name", BASELINE, "--query", SAFE_JOB_QUERY)
        if base.get("status") != "Completed":
            raise RuntimeError(f"Expected completed benchmark: {base}")
        existing = set(azure_job("list", "--all-results", "true", "--query", "[].name"))
        collisions = [job["name"] for job in manifest["jobs"]
                      if job["name"] in existing and not job["accepted"]]
        if collisions:
            raise RuntimeError(f"Existing unrecorded jobs need inspection before launch: {collisions}")
        manifest["credit_authorization"] = args.credit_authorization
        manifest["launch_started_at"] = timestamp()
        write_json(manifest_path, manifest)
        for job in manifest["jobs"]:
            if job["accepted"]:
                continue
            res = azure_job("create", "--file", str(ROOT / job["spec"]), "--query", SAFE_JOB_QUERY)
            if res.get("name") != job["name"]:
                raise RuntimeError(f"Unexpected accepted Azure job: {res}")
            job.update(res, accepted=True, accepted_at=timestamp())
            write_json(manifest_path, manifest)
            print(json.dumps({key: job[key] for key in ("name", "status", "accepted")}), flush=True)
        for job in manifest["jobs"]:
            job.update(azure_job("show", "--name", job["name"], "--query", SAFE_JOB_QUERY))
            job["status_checked_at"] = timestamp()
            write_json(manifest_path, manifest)
        manifest["launch_finished_at"] = timestamp()
        write_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path.relative_to(ROOT)), "jobs": len(jobs),
                      "accepted": sum(job["accepted"] for job in manifest["jobs"])}), flush=True)


if __name__ == "__main__":
    main()

'prepare eight independent recall experiments; submit only with explicit opt-in'
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "honest_leaf", "crossview_consensus", "hard_negative", "sibling_bridge",
    "orphan_factor", "source_corroboration", "monotone_verifier", "country_consensus",
)
COMPUTES = ("amazites-graph64", "amazites-graph64", "amazites-cpu-v4")
RESOURCE_GROUP = "vnjhaveri-rg"
WORKSPACE = "mlworkloads"
BASELINE = "amazites-latest-fusion-20260927-01"
INPUTS = {
    "data": "LocalUpload/34424852455351d9d8c06dba0370c88bccc8b762e34f31033722ffd91d6c7a36/learned-assets/data/",
    "train": "azureml/c4da719f-d934-45ca-84a7-738a0810e1a8/out/",
    "test": "azureml/e1279250-583f-4921-a2cf-b310ffaef6ec/out/",
    "incumbent": "azureml/1e2ecccd-ac04-456e-8266-ddae08ce5f26/out/",
}
SAFE_JOB_QUERY = "{name:name,status:status,compute:compute,studio_url:services.Studio.endpoint}"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_spec(campaign: str, method: str, index: int, directory: Path) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown campaign method: {method}")
    name = f"amazites-recall-{method.replace('_', '-')}-{campaign}"
    return {
        "$schema": "https://azuremlschemas.azureedge.net/latest/commandJob.schema.json",
        "type": "command",
        "name": name,
        "display_name": name,
        "experiment_name": "amazon-ml-2026",
        "description": f"Recall campaign {campaign}: {method}; benchmark frozen at leaderboard F0.5 0.98805.",
        "tags": {"campaign": campaign, "method": method, "baseline": BASELINE},
        "compute": f"azureml:{COMPUTES[index % len(COMPUTES)]}",
        "code": os.path.relpath(ROOT, directory),
        "environment": {"image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"},
        "resources": {"instance_count": 1, "shm_size": "32g"},
        "limits": {"timeout": 3600},
        "environment_variables": {
            "ER_THREADS": "48", "POLARS_MAX_THREADS": "48",
            "OMP_NUM_THREADS": "48", "OPENBLAS_NUM_THREADS": "1",
        },
        "inputs": {
            key: {
                "type": "uri_folder",
                "path": f"azureml://datastores/workspaceblobstore/paths/{path}",
                "mode": "ro_mount" if key == "incumbent" else "download",
            }
            for key, path in INPUTS.items()
        },
        "outputs": {"out": {"type": "uri_folder", "mode": "upload"}},
        "command": (
            "set -eu; "
            "curl -LsSf https://astral.sh/uv/0.7.16/install.sh | sh; "
            'export PATH="$HOME/.local/bin:$PATH"; '
            "uv sync --frozen --python 3.13; "
            "ER_THREADS=2 POLARS_MAX_THREADS=2 ER_MIN_FREE_GB=0 "
            "uv run --frozen --python 3.13 python -m pytest "
            "business_entity_resolution/tests/test_recall_campaign.py "
            "business_entity_resolution/tests/test_recall_campaign_a.py "
            "business_entity_resolution/tests/test_recall_campaign_b.py "
            "business_entity_resolution/tests/test_structured_protocol.py "
            "business_entity_resolution/tests/test_additions_experiment.py -q --disable-warnings; "
            "uv run --frozen --python 3.13 python "
            "business_entity_resolution/scripts/run_structured_experiment.py "
            f"--mode {method} "
            '--data "${{inputs.data}}" '
            '--train "${{inputs.train}}/prepared-train.parquet" '
            '--test "${{inputs.test}}/prepared-test.parquet" '
            '--incumbent "${{inputs.incumbent}}" '
            '--output "${{outputs.out}}"'
        ),
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def azure_job(command: str, *args: str) -> dict | list:
    completed = subprocess.run(
        ["az", "ml", "job", command, *args,
         "--resource-group", RESOURCE_GROUP, "--workspace-name", WORKSPACE,
         "--only-show-errors", "--output", "json"],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode:
        raise RuntimeError(f"Azure job {command} failed: {completed.stderr[-6000:]}")
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--validate", action="store_true", help="Validate specs using Azure; does not launch jobs.")
    parser.add_argument("--submit", action="store_true", help="Submit the eight jobs to Azure after review.")
    parser.add_argument("--credit-authorization", help="Record the user's confirmation that this campaign is covered by sponsorship credit.")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", args.campaign):
        parser.error("campaign must contain 1-41 lowercase letters, digits, or hyphens")
    if args.submit and not (args.credit_authorization or "").strip():
        parser.error("--submit requires --credit-authorization because remaining sponsorship credit is not API-visible")

    directory = ROOT / "business_entity_resolution/azure/recall_campaign" / args.campaign
    manifest_path = ROOT / "artifacts/recall_campaign" / args.campaign / "manifest.json"
    specs = [make_spec(args.campaign, method, index, directory) for index, method in enumerate(METHODS)]
    jobs = []
    for method, spec in zip(METHODS, specs, strict=True):
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
            "compute_policy": "Existing CPU clusters only; no creation or scaling changes; max three concurrent nodes",
            "per_job_timeout_seconds": 3600, "jobs": jobs,
        }
    write_json(manifest_path, manifest)

    if args.validate:
        for job in manifest["jobs"]:
            res = azure_job("validate", "--file", str(ROOT / job["spec"]))
            if res.get("result") != "Succeeded":
                raise RuntimeError(f"Azure schema validation failed: {job['name']}: {res}")
            job["schema_validation"] = "Succeeded"
            write_json(manifest_path, manifest)

    if args.submit:
        base = azure_job("show", "--name", BASELINE, "--query", SAFE_JOB_QUERY)
        if base.get("status") != "Completed":
            raise RuntimeError(f"Expected completed benchmark: {base}")
        existing_names = set(azure_job("list", "--all-results", "true", "--query", "[].name"))
        unexpected = [job["name"] for job in manifest["jobs"]
                      if job["name"] in existing_names and not job["accepted"]]
        if unexpected:
            raise RuntimeError(f"Existing unrecorded jobs require inspection before any launch: {unexpected}")
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

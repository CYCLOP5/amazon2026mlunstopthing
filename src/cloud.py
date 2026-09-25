import argparse as ap
import json
import re
import shlex
import shutil
import stat
import subprocess
import sys
from decimal import Decimal as dc
from decimal import ROUND_CEILING as rc
from pathlib import Path as path
from types import SimpleNamespace as ns
from uuid import uuid4

import budget as bd


pr = "amazon-ml-2026"
im = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04@sha256:7481fbfbbc1c7d7ab0e9e4633180c809413bd77e108b605d6e4d3a60aa35bf67"
pf = {
    "cpu": ("Standard_E16ds_v4", "dedicated", False),
    "gpu": ("Standard_NC24ads_A100_v4", "low_priority", True),
}
end = {"completed", "failed", "canceled", "cancelled", "notresponding"}
rx = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
skip = {".venv", ".azure", "__pycache__", "artifacts", "cache", "data", "dataset", "models", "output"}


class err(Exception):
    pass


def die(x):
    raise err(x)


def tag(run):
    return {"project": pr, "owner": "src-cloud", "run": run}


def own(x, j):
    return getattr(x, "tags", None) == tag(j["run"])


def term(x):
    return str(x or "").lower() in end


def pairs(xs):
    out = {}
    for x in xs:
        if "=" not in x:
            die("input must be name=path")
        n, p = x.split("=", 1)
        q = path(p).expanduser()
        if not rx.fullmatch(n) or n in out or not p or not q.is_dir():
            die("inputs must be unique names and existing folders")
        out[n] = q.resolve()
    return out


def spec(a, run, ins):
    try:
        hrs = bd.num(a.hours, True)
        rate = bd.num(a.rate, True)
        fixed = bd.num(a.fixed)
    except bd.err as e:
        die(str(e))
    if hrs > dc("6"):
        die("hours must be at most 6")
    size, tier, gpu = pf[a.profile]
    neural = gpu or a.neural
    timeout = int((hrs * 3600).to_integral_value(rounding=rc))
    if not timeout:
        die("hours must be at least one second")
    return {
        "run": run,
        "compute": run,
        "job": run + "-job",
        "size": size,
        "tier": tier,
        "neural": neural,
        "hours": hrs,
        "rate": rate,
        "fixed": fixed,
        "timeout": timeout,
        "min": 0,
        "max": 1,
        "idle": 120,
        "input_mode": "download",
        "output_mode": "upload",
        "inputs": ins,
        "command": a.command,
        "tags": tag(run),
    }


def shell(s):
    groups = " --group neural" if s["neural"] else ""
    # Install uv and synchronize the selected worker dependencies in one shell.
    return " ; ".join((
        "set -eu",
        "curl --fail --location --silent --show-error https://astral.sh/uv/0.12.18/install.sh -o /tmp/uv-install.sh",
        "sh /tmp/uv-install.sh",
        'export PATH="$HOME/.local/bin:$PATH"',
        "uv sync --frozen --no-default-groups" + groups,
        "sh -lc " + shlex.quote(s["command"]),
    ))


def ignored(q):
    n = q.name.lower()
    return (
        any(x in skip for x in q.parts)
        or n.startswith(".env")
        or any(x in n for x in ("secret", "credential", "password"))
        or q.suffix in {".key", ".pem", ".pfx", ".p12", ".pyc", ".pt", ".bin", ".safetensors"}
    )


def snapshot(root, dst):
    cmd = ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    try:
        raw = subprocess.run(cmd, check=True, capture_output=True).stdout.decode().split("\0")
    except (OSError, subprocess.CalledProcessError, UnicodeError):
        die("cannot list git-ignored code snapshot")
    dst.mkdir(parents=True, mode=0o700)
    for x in raw:
        if not x:
            continue
        q = path(x)
        src = root / q
        if ignored(q) or not src.is_file() or src.is_symlink():
            continue
        to = dst / q
        to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, to)
    if not (dst / "pyproject.toml").is_file() or not (dst / "uv.lock").is_file():
        die("snapshot needs pyproject.toml and uv.lock")
    return dst


def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    q.chmod(stat.S_IRUSR | stat.S_IWUSR)
    q.replace(p)


def read(p):
    try:
        x = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(x, dict) or x.get("project") != pr or not isinstance(x.get("run"), str):
            raise ValueError
        return x
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        die("corrupt cloud journal")


def reserve(a, s):
    with bd.lock(a.ledger):
        ld = bd.load(a.ledger, a.cap)
        bd.reserve(ld, s["run"], s["rate"], s["hours"] + dc("1"), s["fixed"], False, bd.now())
        bd.save(a.ledger, ld)


def close(a, j):
    with bd.lock(a.ledger):
        ld = bd.load(a.ledger)
        for x in ld["items"]:
            if x["name"] == j["run"] and x["status"] == "active":
                bd.close(ld, j["run"], "deleted", bd.now())
                bd.save(a.ledger, ld)
                return
            if x["name"] == j["run"]:
                return
        die("journal reservation is missing")


def client(a):
    try:
        from azure.ai.ml import MLClient
        from azure.identity import AzureCliCredential
    except ImportError:
        die("install the cloud dependency group")
    return MLClient(AzureCliCredential(), a.subscription, a.group, a.workspace)


def entities(s, code):
    from azure.ai.ml import Input, Output, command
    from azure.ai.ml.entities import AmlCompute, Environment

    comp = AmlCompute(
        name=s["compute"], size=s["size"], tier=s["tier"], tags=s["tags"],
        min_instances=s["min"], max_instances=s["max"], idle_time_before_scale_down=s["idle"],
        ssh_public_access_enabled=False,
    )
    ins = {n: Input(type="uri_folder", path=str(p), mode=s["input_mode"]) for n, p in s["inputs"].items()}
    job = command(
        name=s["job"], display_name=s["run"], experiment_name=pr, tags=s["tags"],
        code=str(code), command=shell(s), compute="azureml:" + s["compute"],
        environment=Environment(image=im), inputs=ins,
        outputs={"out": Output(type="uri_folder", mode=s["output_mode"])}, timeout=s["timeout"],
    )
    return comp, job


def cancel(ml, j):
    from azure.core.exceptions import ResourceNotFoundError

    name = j.get("job")
    if not name:
        return
    try:
        x = ml.jobs.get(name)
    except ResourceNotFoundError:
        return
    if not own(x, j):
        die("job ownership tag does not match journal")
    if not term(getattr(x, "status", None)):
        ml.jobs.begin_cancel(name).result()


def cleanup(ml, j):
    from azure.core.exceptions import ResourceNotFoundError

    cancel(ml, j)
    try:
        x = ml.compute.get(j["compute"])
    except ResourceNotFoundError:
        return
    if not own(x, j):
        die("compute ownership tag does not match journal")
    ml.compute.begin_delete(j["compute"]).result()
    try:
        ml.compute.get(j["compute"])
    except ResourceNotFoundError:
        return
    die("compute deletion was not verified")


def run(a):
    ins = pairs(a.input)
    if "${{outputs.out}}" not in a.command:
        die("command must write to ${{outputs.out}}")
    if a.profile == "gpu":
        print("warning: spot nodes can be preempted; checkpoint survival is not guaranteed", file=sys.stderr)
    root = path(__file__).resolve().parents[1]
    run = "aml26-" + a.profile + "-" + uuid4().hex[:10]
    s = spec(a, run, ins)
    jp = root / "artifacts/cloud" / (run + ".json")
    j = {
        "project": pr, "run": run, "compute": s["compute"], "job": s["job"],
        "subscription": a.subscription, "group": a.group, "workspace": a.workspace, "state": "new",
        "ledger": str(a.ledger), "out": str(a.out),
    }
    write(jp, j)
    code = snapshot(root, jp.parent / (run + "-code"))
    ml = client(a)
    reserve(a, s)
    j["state"] = "reserved"
    write(jp, j)
    primary = clean = None
    try:
        comp, job = entities(s, code)
        ml.compute.begin_create_or_update(comp).result()
        j["state"] = "compute_created"
        write(jp, j)
        got = ml.jobs.create_or_update(job)
        j["job"] = got.name
        j["state"] = "job_submitted"
        write(jp, j)
        ml.jobs.stream(got.name)
        got = ml.jobs.get(got.name)
        if str(getattr(got, "status", "")).lower() != "completed":
            die("job ended with " + str(getattr(got, "status", "unknown")))
        a.out.mkdir(parents=True, exist_ok=True)
        ml.jobs.download(got.name, download_path=a.out, output_name="out")
        j["state"] = "downloaded"
        write(jp, j)
    except Exception as e:
        primary = e
    try:
        cleanup(ml, j)
        close(a, j)
        j["state"] = "deleted"
        write(jp, j)
    except Exception as e:
        clean = e
        j["state"] = "cleanup_failed"
        write(jp, j)
    if clean:
        die("cleanup failed; reservation remains active: " + str(clean))
    if primary:
        raise primary
    print(json.dumps({"job": j["job"], "compute": j["compute"], "out": str(a.out)}, indent=2))


def recover(a):
    jp = a.journal
    j = read(jp)
    if (j.get("subscription") != a.subscription or j.get("group") != a.group
            or j.get("workspace") != a.workspace):
        die("Azure scope does not match journal")
    try:
        cleanup(client(a), j)
        close(a, j)
        j["state"] = "deleted"
        write(jp, j)
    except Exception as e:
        j["state"] = "cleanup_failed"
        write(jp, j)
        die("cleanup failed; reservation remains active: " + str(e))
    print(json.dumps({"compute": j["compute"], "state": "deleted"}, indent=2))


def check():
    a = ns(profile="gpu", hours="2", rate="3.50", fixed="10", neural=False, command="echo ok")
    s = spec(a, "aml26-gpu-check", {"train": path("/tmp")})
    assert s["tier"] == "low_priority" and s["neural"] and s["timeout"] == 7200
    assert (s["min"], s["max"], s["idle"], s["input_mode"], s["output_mode"]) == (0, 1, 120, "download", "upload")
    c, j = entities(s, path("."))
    assert (c.min_instances, c.max_instances, c.idle_time_before_scale_down) == (0, 1, 120)
    assert c.enable_node_public_ip and not c.ssh_public_access_enabled
    assert j.component.inputs["train"]["mode"] == "download"
    assert j.component.outputs["out"]["mode"] == "upload" and j.limits.timeout == 7200
    assert "--group cloud" not in shell(s) and "--group neural" in shell(s)
    assert s["hours"] + dc("1") == dc("3") and im.startswith("mcr.microsoft.com/")
    ld = bd.new("500")
    try:
        bd.new("500.01")
        assert False
    except bd.err:
        pass
    t = bd.now()
    bd.reserve(ld, s["run"], s["rate"], s["hours"] + dc("1"), s["fixed"], False, t)
    assert ld["items"][0]["reserved"] == "20.50"
    j = {"run": s["run"]}
    assert own(ns(tags=tag(s["run"])), j) and not own(ns(tags={}), j)
    print("checks passed")


def common(p, sub=True):
    p.add_argument("--subscription", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--ledger", type=path, default=path(__file__).resolve().parents[1] / "artifacts/budget.json")
    if sub:
        p.add_argument("--journal", type=path, required=True)


def main():
    p = ap.ArgumentParser(description="one tagged azure ml command job")
    p.add_argument("--check", action="store_true")
    su = p.add_subparsers(dest="cmd")
    r = su.add_parser("run")
    common(r, False)
    r.add_argument("--profile", choices=tuple(pf), required=True)
    r.add_argument("--hours", required=True, help="runtime limit, at most 6")
    r.add_argument("--rate", required=True, help="current usd per hour estimate")
    r.add_argument("--fixed", default="10", help="staging/storage/egress allowance")
    r.add_argument("--cap", default="500")
    r.add_argument("--input", action="append", default=[], help="name=local-folder")
    r.add_argument("--command", required=True, help="shell template using azure inputs and outputs.out")
    r.add_argument("--out", type=path, required=True)
    r.add_argument("--neural", action="store_true")
    x = su.add_parser("recover")
    common(x)
    z = su.add_parser("status")
    z.add_argument("--journal", type=path, required=True)
    a = p.parse_args()
    if a.check:
        check()
        return
    if a.cmd == "status":
        print(json.dumps(read(a.journal), indent=2, sort_keys=True))
        return
    if not a.cmd:
        p.error("choose run, recover, or status")
    try:
        if a.cmd == "run":
            run(a)
        else:
            recover(a)
    except (err, bd.err, OSError, subprocess.CalledProcessError) as e:
        p.exit(2, "error: " + str(e) + "\n")


if __name__ == "__main__":
    main()

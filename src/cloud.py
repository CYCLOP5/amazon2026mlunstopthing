import argparse as ap
import json
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from decimal import Decimal as dc
from decimal import ROUND_CEILING as rc
from pathlib import Path as path
from types import SimpleNamespace as ns
from uuid import uuid4
from urllib.parse import unquote as uq

import budget as bd


pr = "amazon-ml-2026"
im = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04@sha256:7481fbfbbc1c7d7ab0e9e4633180c809413bd77e108b605d6e4d3a60aa35bf67"
pf = {
    "cpu": ("Standard_E16ds_v4", "dedicated", False),
    "cpu64": ("Standard_E64ds_v4", "dedicated", False),
    "gpu": ("Standard_NC24ads_A100_v4", "low_priority", True),
    "gpu2": ("Standard_NC48ads_A100_v4", "low_priority", True),
    "gpu4": ("Standard_NC96ads_A100_v4", "low_priority", True),
}
end = {"completed", "failed", "canceled", "cancelled"}
rx = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
skip = {".venv", ".azure", ".cortexkit", "__pycache__", "artifacts", "cache", "data", "dataset", "models", "output"}


class err(Exception):
    pass


def die(x):
    raise err(x)


def tag(run):
    return {"project": pr, "owner": "src-cloud", "run": run}


def own(x, j):
    return getattr(x, "tags", None) == tag(j["run"])


def state(x):
    return str(getattr(x, "value", x) or "").lower()


def term(x):
    return state(x) in end


def pairs(xs):
    out = {}
    for x in xs:
        if "=" not in x:
            die("input must be name=path")
        n, p = x.split("=", 1)
        if not rx.fullmatch(n) or n in out or not p:
            die("inputs must have unique valid names")
        if p.startswith("azureml://datastores/"):
            location(p)
            out[n] = p
            continue
        q = path(p).expanduser()
        if not q.is_dir():
            die("inputs must be existing folders or azure ml datastore uris")
        out[n] = q.resolve()
    return out


def spec(a, run, ins):
    try:
        hrs = bd.num(a.hours, True)
        rate = None if getattr(a, "uncapped", False) else bd.num(a.rate, True)
        fixed = bd.num(a.fixed)
    except bd.err as e:
        die(str(e))
    if hrs > dc("12"):
        die("hours must be at most 12")
    om = getattr(a, "output_mode", "upload")
    if om not in {"upload", "rw_mount"}:
        die("invalid output mode")
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
        "output_mode": om,
        "inputs": ins,
        "command": a.command,
        "tags": tag(run),
    }


def shell(s):
    groups = " --group neural" if s["neural"] else ""
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
        bd.reserve(ld, s["run"], s["rate"], s["hours"] + dc("1"), s["fixed"], getattr(a, "final", False), bd.now())
        bd.save(a.ledger, ld)


def close(a, j):
    if j.get("uncapped"):
        return
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
        code=str(code), command=shell(s), compute=s["compute"],
        environment=Environment(image=im), inputs=ins,
        outputs={"out": Output(type="uri_folder", mode=s["output_mode"], path=s.get("out_uri"))}, timeout=s["timeout"],
    )
    return comp, job


def wait_job(ml, name, due, interval=20):
    last = None
    while True:
        job = ml.jobs.get(name)
        st = state(job.status)
        if st != last:
            print("job state", name, st, flush=True)
            last = st
        if term(st):
            return job
        if time.monotonic() >= due:
            die("job controller deadline reached")
        time.sleep(interval)


def location(uri):
    m = re.fullmatch(r"azureml://datastores/([^/]+)/paths/(.+)", uri or "")
    if not m:
        die("unsupported output datastore uri")
    return m[1], uq(m[2]).rstrip("/") + "/"


def blobs(ml, uri):
    from azure.core.credentials import AzureNamedKeyCredential, AzureSasCredential
    from azure.identity import AzureCliCredential
    from azure.storage.blob import BlobServiceClient

    name, pre = location(uri)
    ds = ml.datastores.get(name, include_secrets=True)
    cr = getattr(ds, "credentials", None)
    sas = getattr(cr, "sas_token", None)
    key = getattr(cr, "account_key", None)
    cred = AzureSasCredential(sas) if sas else (AzureNamedKeyCredential(ds.account_name, key) if key else AzureCliCredential())
    svc = BlobServiceClient("https://" + ds.account_name + ".blob.core.windows.net", credential=cred,
                            max_single_get_size=64 * 1024**2, max_chunk_get_size=16 * 1024**2)
    cc = svc.get_container_client(ds.container_name)
    return cc, pre


def download(ml, uri, dest, recursive=True):
    cc, pre = blobs(ml, uri)
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for obj in cc.list_blobs(name_starts_with=pre, include=["metadata"]):
        rel = obj.name[len(pre):]
        md = obj.metadata or {}
        if not rel or rel.endswith("/") or md.get("hdi_isfolder", "").lower() == "true":
            continue
        if not recursive and "/" in rel:
            continue
        p = (dest / rel).resolve()
        if not p.is_relative_to(dest):
            die("unsafe artifact path")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        with tmp.open("wb") as f:
            cc.download_blob(obj.name, max_concurrency=8).readinto(f)
        tmp.replace(p)
        n += 1
    if not n:
        die("completed job has no output artifacts")
    print("downloaded artifacts", n, flush=True)
    return n


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
    run = "aml26-" + a.profile + "-r" + uuid4().hex[:10]
    s = spec(a, run, ins)
    jp = root / "artifacts/cloud" / (run + ".json")
    j = {
        "project": pr, "run": run, "compute": s["compute"], "job": s["job"],
        "subscription": a.subscription, "group": a.group, "workspace": a.workspace, "state": "new",
        "ledger": str(a.ledger), "out": str(a.out),
    }
    if getattr(a, "uncapped", False):
        j["uncapped"] = True
    write(jp, j)
    code = snapshot(root, jp.parent / (run + "-code"))
    ml = client(a)
    ds = ml.datastores.get_default()
    s["out_uri"] = "azureml://datastores/" + ds.name + "/paths/" + pr + "/" + run + "/out/"
    j["out_uri"] = s["out_uri"]
    if not j.get("uncapped"):
        reserve(a, s)
    due = time.monotonic() + (float(s["hours"]) + 1) * 3600
    j["state"] = "planned" if j.get("uncapped") else "reserved"
    write(jp, j)
    print(j["state"], run, "creating", s["compute"], flush=True)
    primary = clean = None
    try:
        comp, job = entities(s, code)
        ml.compute.begin_create_or_update(comp).result()
        j["state"] = "compute_created"
        write(jp, j)
        print("compute ready", s["compute"], "submitting job", flush=True)
        got = ml.jobs.create_or_update(job)
        j["job"] = got.name
        j["state"] = "job_submitted"
        write(jp, j)
        print("job submitted", got.name, flush=True)
        got = wait_job(ml, got.name, due)
        j["job_status"] = state(got.status)
        if state(got.status) != "completed":
            die("job ended with " + state(got.status))
        if not getattr(a, "no_download", False):
            a.out.mkdir(parents=True, exist_ok=True)
            download(ml, s["out_uri"], a.out)
        else:
            cc, pre = blobs(ml, s["out_uri"])
            if not any(not x.name.endswith("/") and (x.metadata or {}).get("hdi_isfolder", "").lower() != "true"
                       for x in cc.list_blobs(name_starts_with=pre, include=["metadata"])):
                die("completed job has no output artifacts")
        j["downloaded"] = not getattr(a, "no_download", False)
        j["state"] = "downloaded" if j["downloaded"] else "outputs_remote"
        j["outcome"] = "completed"
        write(jp, j)
    except BaseException as e:
        primary = e
        j["outcome"] = "failed"
        j["error_type"] = type(e).__name__
        write(jp, j)
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
    print(json.dumps({"job": j["job"], "compute": j["compute"], "out": str(a.out) if j["downloaded"] else None,
                      "out_uri": j["out_uri"]}, indent=2))


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
    for profile, size in [("gpu2", "Standard_NC48ads_A100_v4"), ("gpu4", "Standard_NC96ads_A100_v4")]:
        q = spec(ns(**{**vars(a), "profile": profile, "hours": "12"}), "check", {})
        assert q["size"] == size and q["timeout"] == 43200 and q["max"] == 1
    try:
        spec(ns(**{**vars(a), "hours": "12.01"}), "check", {})
        assert False
    except err:
        pass
    c, j = entities(s, path("."))
    assert (c.min_instances, c.max_instances, c.idle_time_before_scale_down) == (0, 1, 120)
    assert c.enable_node_public_ip and not c.ssh_public_access_enabled
    assert j.component.inputs["train"]["mode"] == "download"
    assert j.compute == s["compute"]
    assert j.component.outputs["out"]["mode"] == "upload" and j.limits.timeout == 7200
    assert "--group cloud" not in shell(s) and "--group neural" in shell(s)
    assert s["hours"] + dc("1") == dc("3") and im.startswith("mcr.microsoft.com/")
    cpu = spec(ns(profile="cpu64", hours="2", rate=None, fixed="0", neural=False, command="true", uncapped=True), "check", {})
    assert cpu["size"] == "Standard_E64ds_v4" and cpu["rate"] is None and not cpu["neural"]
    close(ns(), {"uncapped": True})
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
    assert term(ns(value="Completed")) and not term(ns(value="Running"))
    seq = iter([ns(status=ns(value="Running")), ns(status=ns(value="Completed"))])
    got = wait_job(ns(jobs=ns(get=lambda _: next(seq))), "check", time.monotonic() + 1, 0)
    assert state(got.status) == "completed"
    assert location("azureml://datastores/store/paths/project/run/out/") == ("store", "project/run/out/")
    a.output_mode = "rw_mount"
    _, mounted = entities(spec(a, "aml26-gpu-check", {"train": path("/tmp")}), path("."))
    assert mounted.component.outputs["out"]["mode"] == "rw_mount"
    u = "azureml://datastores/store/paths/project/model/"
    assert pairs(["model=" + u]) == {"model": u}
    _, remote = entities(spec(a, "aml26-gpu-check", {"model": u}), path("."))
    assert remote.inputs["model"].path == u
    from unittest.mock import patch
    from types import SimpleNamespace as obj
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cc = obj(list_blobs=lambda **kw: [obj(name="p/model", metadata={"hdi_isfolder": "true"}),
                                        obj(name="p/model/config.json", metadata=None)],
                 download_blob=lambda *a, **kw: obj(readinto=lambda f: f.write(b"{}")))
        ml = obj(datastores=obj(get=lambda *a, **kw: obj(account_name="a", container_name="c", credentials=obj(sas_token="test"))))
        with patch("azure.storage.blob.BlobServiceClient", return_value=obj(get_container_client=lambda x: cc)):
            assert download(ml, "azureml://datastores/s/paths/p/", path(tmp)) == 1
        assert (path(tmp) / "model/config.json").read_text() == "{}"
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
    r.add_argument("--hours", required=True, help="runtime limit, at most 12")
    r.add_argument("--rate", help="current usd per hour estimate for budgeted runs")
    r.add_argument("--uncapped", action="store_true", help="explicitly authorized run without a monetary cap")
    r.add_argument("--fixed", default="10", help="staging/storage/egress allowance")
    r.add_argument("--cap", default="500")
    r.add_argument("--final", action="store_true", help="allow the reserved final inference allocation within the same hard cap")
    r.add_argument("--input", action="append", default=[], help="name=local-folder")
    r.add_argument("--command", required=True, help="shell template using azure inputs and outputs.out")
    r.add_argument("--out", type=path, required=True)
    r.add_argument("--neural", action="store_true")
    r.add_argument("--output-mode", choices=["upload", "rw_mount"], default="upload")
    r.add_argument("--no-download", action="store_true", help="keep outputs in the recorded datastore uri for downstream jobs")
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

import argparse as ap
import fcntl
import json
import os
import tempfile
from contextlib import contextmanager as cm
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import timezone as tz
from decimal import Decimal as dc
from decimal import InvalidOperation as ie
from decimal import ROUND_UP as ru
from decimal import localcontext as lc
from pathlib import Path as path


z = dc("0")
h = dc("3600")
fi = dc("75.00")
co = dc("25.00")
rv = fi + co


class err(Exception):
    pass


def num(x, pos=False):
    try:
        n = dc(str(x))
    except (ie, ValueError):
        raise err("invalid number") from None
    if not n.is_finite() or (n <= z if pos else n < z):
        raise err("invalid number")
    return n


def mon(x):
    try:
        with lc() as c:
            c.prec = 50
            return x.quantize(dc("0.01"), rounding=ru)
    except ie:
        raise err("invalid amount") from None


def iso(x):
    if not isinstance(x, str) or not x.endswith("Z"):
        raise err("invalid utc timestamp")
    try:
        d = dt.fromisoformat(x[:-1] + "+00:00")
    except ValueError:
        raise err("invalid utc timestamp") from None
    if d.tzinfo is None or d.utcoffset() != td():
        raise err("invalid utc timestamp")
    return d


def now():
    return dt.now(tz.utc)


def out(d):
    return d.astimezone(tz.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def cost(rt, hr, fx):
    try:
        with lc() as c:
            c.prec = 50
            return mon(rt * hr + fx)
    except ie:
        raise err("invalid amount") from None


def new(cp):
    return {"cap": str(mon(num(cp, True))), "items": []}


def chk(ld):
    if not isinstance(ld, dict) or set(ld) != {"cap", "items"} or not isinstance(ld["items"], list):
        raise err("corrupt ledger")
    cp = mon(num(ld["cap"], True))
    ns = set()
    for x in ld["items"]:
        if not isinstance(x, dict) or set(x) - {"name", "phase", "start", "deadline", "rate", "hours", "fixed", "reserved", "status", "closed", "elapsed_estimate", "state"}:
            raise err("corrupt ledger")
        try:
            nm = x["name"]
            ph = x["phase"]
            st = iso(x["start"])
            dl = iso(x["deadline"])
            rt = num(x["rate"], True)
            hr = num(x["hours"], True)
            fx = mon(num(x["fixed"]))
            rs = mon(num(x["reserved"], True))
            ss = x["status"]
        except (KeyError, TypeError, err):
            raise err("corrupt ledger") from None
        if not isinstance(nm, str) or not nm or nm != nm.strip() or nm in ns or ph not in {"experiment", "final"} or dl <= st or rs != cost(rt, hr, fx) or ss not in {"active", "closed"}:
            raise err("corrupt ledger")
        ns.add(nm)
        if ss == "closed":
            try:
                cl = iso(x["closed"])
                ac = mon(num(x["elapsed_estimate"]))
                sy = x["state"]
            except (KeyError, TypeError, err):
                raise err("corrupt ledger") from None
            if cl < st or sy not in {"deallocated", "deleted"}:
                raise err("corrupt ledger")
        elif any(k in x for k in ("closed", "elapsed_estimate", "state")):
            raise err("corrupt ledger")
    return cp


@cm
def lock(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(fd, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def load(p, cp=None):
    if not p.exists():
        return new("500" if cp is None else cp)
    try:
        ld = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        raise err("corrupt ledger") from None
    old = chk(ld)
    if cp is not None and old != mon(num(cp, True)):
        raise err("cap differs from ledger")
    return ld


def save(p, ld):
    chk(ld)
    fd, q = tempfile.mkstemp(prefix="." + p.name + ".", dir=p.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(ld, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(q, p)
        fd = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        raise err("ledger write failed") from None
    finally:
        if os.path.exists(q):
            os.unlink(q)


def accrued(x, nw):
    st = iso(x["start"])
    hr = max(z, dc(str((nw - st).total_seconds())) / h)
    return cost(num(x["rate"], True), hr, mon(num(x["fixed"])))


def stat(ld, nw):
    cp = chk(ld)
    sp = z
    ac = z
    rs = z
    for x in ld["items"]:
        if x["status"] == "closed":
            sp += mon(num(x["elapsed_estimate"]))
        else:
            a = accrued(x, nw)
            ac += a
            rs += max(mon(num(x["reserved"], True)), a)
    cm = mon(sp + rs)
    av = mon(max(z, cp - cm))
    return {
        "cap": str(cp),
        "estimated_spent": str(mon(sp + ac)),
        "active_reserved_worst_case": str(mon(rs)),
        "committed_worst_case": str(cm),
        "hard_available": str(av),
        "experiment_available": str(mon(max(z, av - rv))),
        "final_inference_reserve": str(fi),
        "contingency_reserve": str(co),
    }


def reserve(ld, nm, rt, hr, fx, fn, nw):
    cp = chk(ld)
    if not isinstance(nm, str) or not nm or nm != nm.strip() or any(x["name"] == nm for x in ld["items"]):
        raise err("duplicate or invalid name")
    rt = num(rt, True)
    hr = num(hr, True)
    fx = mon(num(fx))
    rs = cost(rt, hr, fx)
    try:
        dl = nw + td(seconds=float(hr * h))
    except (OverflowError, ValueError):
        raise err("invalid hours") from None
    if dl <= nw:
        raise err("invalid hours")
    s = stat(ld, nw)
    av = num(s["hard_available"])
    if rs > av or (not fn and rs + rv > av):
        raise err("budget exceeded")
    ld["items"].append({"name": nm, "phase": "final" if fn else "experiment", "start": out(nw), "deadline": out(dl), "rate": str(rt), "hours": str(hr), "fixed": str(fx), "reserved": str(rs), "status": "active"})


def close(ld, nm, sy, nw):
    chk(ld)
    if sy not in {"deallocated", "deleted"}:
        raise err("invalid externally verified state")
    for x in ld["items"]:
        if x["name"] == nm:
            if x["status"] != "active":
                raise err("reservation is not active")
            if nw < iso(x["start"]):
                raise err("close time precedes start")
            x.update({"status": "closed", "closed": out(nw), "elapsed_estimate": str(accrued(x, nw)), "state": sy})
            return
    raise err("unknown reservation")


def check():
    ld = new("500")
    t = dt(2026, 1, 1, tzinfo=tz.utc)
    reserve(ld, "a", "10", "1", "0", False, t)
    try:
        reserve(ld, "a", "10", "1", "0", False, t)
        assert False
    except err:
        pass
    try:
        reserve(ld, "b", "400", "1", "0", False, t)
        assert False
    except err:
        pass
    for x in ("NaN", "-1"):
        try:
            reserve(ld, "x" + x, x, "1", "0", False, t)
            assert False
        except err:
            pass
    close(ld, "a", "deallocated", t + td(hours=2))
    x = ld["items"][0]
    assert dc(x["elapsed_estimate"]) > dc(x["reserved"])
    print("checks passed")


def main():
    pa = ap.ArgumentParser()
    root = path(__file__).resolve().parents[1]
    pa.add_argument("--ledger", type=path, default=root / "artifacts/budget.json")
    pa.add_argument("--check", action="store_true")
    su = pa.add_subparsers(dest="cmd")
    r = su.add_parser("reserve")
    r.add_argument("name")
    r.add_argument("--rate", required=True)
    r.add_argument("--hours", required=True)
    r.add_argument("--fixed", required=True)
    r.add_argument("--cap")
    r.add_argument("--final", action="store_true")
    su.add_parser("status")
    c = su.add_parser("close")
    c.add_argument("name")
    c.add_argument("--state", required=True, choices=("deallocated", "deleted"))
    a = pa.parse_args()
    if a.check:
        check()
        return
    if not a.cmd:
        pa.error("choose reserve, status, or close")
    try:
        with lock(a.ledger):
            nw = now()
            ld = load(a.ledger, a.cap if a.cmd == "reserve" else None)
            if a.cmd == "reserve":
                reserve(ld, a.name, a.rate, a.hours, a.fixed, a.final, nw)
                save(a.ledger, ld)
            elif a.cmd == "close":
                close(ld, a.name, a.state, nw)
                save(a.ledger, ld)
            print(json.dumps(stat(ld, nw), indent=2))
    except err as e:
        pa.exit(2, "error: " + str(e) + "\n")


if __name__ == "__main__":
    main()

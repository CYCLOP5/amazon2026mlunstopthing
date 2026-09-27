'resource limits so the pipeline never freezes a 16 gb laptop'
import os

def _threads(spec: str) -> int:
    "'0' -> half the logical cores; '0.75' -> 75% of them; '16' -> 16"
    n_cpu = os.cpu_count() or 2
    v = float(spec or 0)
    if v <= 0:
        return max(1, n_cpu // 2)
    if v < 1:
        return max(1, int(n_cpu * v))
    return int(v)


THREADS = _threads(os.environ.get("ER_THREADS", "0"))
MIN_FREE_GB = float(os.environ.get("ER_MIN_FREE_GB", "1.5"))
PRIORITY = os.environ.get("ER_PRIORITY", "normal").lower()

for var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "RAYON_NUM_THREADS"):
    os.environ.setdefault(var, str(THREADS))

try:
    import psutil

    if PRIORITY == "low":
        _p = psutil.Process()
        _p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
except Exception:
    psutil = None


def free_gb() -> float:
    return psutil.virtual_memory().available / 2**30 if psutil else float("inf")


def guard(where: str = ""):
    'abort cleanly (instead of swapping the machine to death) when ram runs low'
    f = free_gb()
    if f < MIN_FREE_GB:
        raise MemoryError(f"[safe] only {f:.1f} GB RAM free at {where!r} - aborting to protect the system. "
                          f"Close other programs and re-run: finished work is kept.")

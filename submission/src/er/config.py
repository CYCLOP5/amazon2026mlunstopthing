'configuration: toml files with inheritance + command-line overrides'
import copy
import hashlib
import json
import os
import tomllib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "default.toml")


def _merge(base: dict, over: dict) -> dict:
    'deep merge; a table containing  _replace = true  replaces the base table entirely'
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and v.get("_replace"):
            out[k] = {kk: copy.deepcopy(vv) for kk, vv in v.items() if kk != "_replace"}
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: str) -> dict:
    with open(path, "rb") as fh:
        d = tomllib.load(fh)
    parent = d.pop("extends", None)
    if parent:
        d = _merge(_load(os.path.normpath(os.path.join(os.path.dirname(path), parent))), d)
    return d


def _parse_value(text: str):
    try:
        return tomllib.loads(f"v = {text}")["v"]
    except tomllib.TOMLDecodeError:
        return text


def set_dotted(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str | None = None, overrides=()) -> dict:
    cfg = _load(os.path.abspath(path or DEFAULT_CONFIG))
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        set_dotted(cfg, k.strip(), _parse_value(v.strip()))
    return cfg


def settings_hash(*parts) -> str:
    'stable short hash of settings (used to key cached stage outputs)'
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]

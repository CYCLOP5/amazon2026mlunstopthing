"""shorten local names without renaming public or checkpoint interfaces"""
import argparse as ap
import ast
import hashlib as hh
import json
import re
from pathlib import Path as path

import libcst as cst
from libcst.metadata import FunctionScope, MetadataWrapper, ScopeProvider

names = {"probability": "prob", "probabilities": "probs", "threshold": "cut", "thresholds": "cuts",
         "features": "ff", "feature_names": "fnames", "metadata": "meta", "references": "refs",
         "reference": "ref", "candidates": "cands", "candidate": "cand", "calibration": "cal",
         "validation": "val", "source": "src", "destination": "dst", "result": "res", "results": "rs",
         "predictions": "pred", "parameters": "params", "configuration": "cfg", "baseline": "base",
         "training": "tr", "selected": "sel", "expected": "exp", "actual": "got", "previous": "prev"}


class edit(cst.CSTTransformer):
    def __init__(self, changes):
        self.changes = changes

    def leave_Name(self, original_node, updated_node):
        return updated_node.with_changes(value=self.changes[id(original_node)]) if id(original_node) in self.changes else updated_node

    def leave_Comment(self, original_node, updated_node):
        keep = r"copyright|license|spdx|type:|noqa|pragma|fmt:|ruff:|pyright:|mypy:|coding[:=]|ponytail:|^#!"
        return updated_node if re.search(keep, original_node.value, re.I) else cst.RemoveFromParent()

    def leave_TrailingWhitespace(self, original_node, updated_node):
        return updated_node.with_changes(whitespace=cst.SimpleWhitespace("")) if updated_node.comment is None else updated_node

    def leave_EmptyLine(self, original_node, updated_node):
        return updated_node.with_changes(indent=False, whitespace=cst.SimpleWhitespace("")) if updated_node.comment is None else updated_node

    def doc(self, body):
        items = list(body)
        if not items or not isinstance(items[0], cst.SimpleStatementLine) or len(items[0].body) != 1:
            return body
        first = items[0].body[0]
        if not isinstance(first, cst.Expr) or not isinstance(first.value, cst.SimpleString):
            return body
        text = first.value.evaluated_value
        if not isinstance(text, str) or re.search(r">>>|https?://|copyright|licen[cs]e|citation|adapted|based on", text, re.I):
            return body
        short = next((s.strip() for s in text.splitlines() if s.strip()), "")
        if "`" not in short and not re.search(r"[/\\]|\w+\(", short):
            short = short.lower()
        short = short.rstrip(" .")
        if re.search(r"\b(?:teammate|friend)\b", short, re.I):
            short = re.sub(r"\b(?:teammate|friend)\b", "team", short, flags=re.I)
        items[0] = items[0].with_changes(body=[first.with_changes(value=cst.SimpleString(repr(short)))])
        return items

    def leave_Module(self, original_node, updated_node):
        return updated_node.with_changes(body=self.doc(updated_node.body))

    def leave_FunctionDef(self, original_node, updated_node):
        b = updated_node.body
        return updated_node.with_changes(body=b.with_changes(body=self.doc(b.body))) if isinstance(b, cst.IndentedBlock) else updated_node

    def leave_ClassDef(self, original_node, updated_node):
        b = updated_node.body
        return updated_node.with_changes(body=b.with_changes(body=self.doc(b.body))) if isinstance(b, cst.IndentedBlock) else updated_node


class debug(cst.CSTVisitor):
    found = False

    def visit_FormattedStringExpression(self, node):
        self.found |= node.equal is not None


def rewrite(text):
    tree = ast.parse(text)
    dynamic = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {"locals", "vars", "eval", "exec"} for n in ast.walk(tree))
    wrapper = MetadataWrapper(cst.parse_module(text))
    dbg = debug()
    wrapper.module.visit(dbg)
    scopes = wrapper.resolve(ScopeProvider)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    changes, aliases = {}, {}
    if not dynamic and not dbg.found and not re.search(r"f_locals|f_globals|getargvalues", text):
        for scope in set(scopes.values()):
            if not isinstance(scope, FunctionScope):
                continue
            for old, new in names.items():
                if new in used:
                    continue
                assignments = list(scope.assignments[old])
                if not assignments or any(not isinstance(a.node, cst.Name) for a in assignments):
                    continue
                aliases[old] = new
                for assignment in assignments:
                    changes[id(assignment.node)] = new
                    for ref in assignment.references:
                        if isinstance(ref.node, cst.Name):
                            changes[id(ref.node)] = new
    changed = wrapper.module.visit(edit(changes)).code
    compile(changed, "<styled>", "exec")
    return changed, aliases


def check():
    samples = [
        "def f(value):\n    result = value + 1\n    return result\n",
        "def f(value):\n    result = value + 1\n    return f'answer {result}'\n",
        "def f(value):\n    result = value + 1\n    return f'{result=}'\n",
        "def f(value):\n    result = value + 1\n    return locals()\n",
        "def f(value):\n    result = value + 1\n    def g(res):\n        return result + res\n    return g(2)\n",
    ]
    for source in samples:
        before, after = {}, {}
        changed, _ = rewrite(source)
        exec(source, before)
        exec(changed, after)
        assert before["f"](2) == after["f"](2)
    assert "res =" in rewrite(samples[0])[0]
    assert "result=" in rewrite(samples[2])[0]
    print("scope-aware style checks passed")


def run(roots, report, apply=False):
    records = []
    for root in roots:
        for file in sorted(root.rglob("*.py")):
            if "__pycache__" in file.parts or "student_resource" in file.parts or file.name == "code_style.py":
                continue
            old = file.read_text()
            new, aliases = rewrite(old)
            if old == new:
                continue
            records.append({"path": str(file), "before": hh.sha256(old.encode()).hexdigest(), "after": hh.sha256(new.encode()).hexdigest(), "local_aliases": aliases})
            if apply:
                file.write_text(new)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"changed_files": len(records), "applied": apply, "files": records}, indent=2))
    print("style files", len(records), "applied", apply)


if __name__ == "__main__":
    p = ap.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=path)
    p.add_argument("--report", type=path, required=True)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    run(a.roots, a.report, a.apply)

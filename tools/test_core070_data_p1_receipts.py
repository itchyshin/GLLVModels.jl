"""Negative controls for `tools/core070_data_p1_receipts.py --check` provenance.

The tracked tree must pass --check. Then, without touching any file, one case
receipt is served to check() with a tampered provenance field (glvmodels_commit
set to forty zeros, then glvmodels_src_tree, then glvmodels_worktree_dirty), and
check() must exit nonzero naming the field each time.

Usage: python3 tools/test_core070_data_p1_receipts.py
"""
import contextlib
import copy
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core070_data_p1_receipts as mod  # noqa: E402

REAL_LOAD = mod.load
TARGET = "CORE070-DATA-OFF-"  # any data offset case receipt


def run_check():
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.check()
    except SystemExit as e:
        return (e.code or 0), buf.getvalue()
    return 0, buf.getvalue()


def serve_tampered(field, value):
    hit = []

    def load(p):
        obj = REAL_LOAD(p)
        if not hit and Path(p).parent.name == "cases" and Path(p).stem.startswith(TARGET):
            obj = copy.deepcopy(obj)
            obj[field] = value
            hit.append(Path(p).stem)
        return obj
    mod.load = load
    return hit


def main():
    code, out = run_check()
    assert code == 0 and "CORE070_DATA_P1_RECEIPTS_CURRENT" in out, f"clean tree must pass --check:\n{out}"
    for field, value in (("glvmodels_commit", "0" * 40), ("glvmodels_src_tree", "0" * 40),
                         ("glvmodels_worktree_dirty", ["src/fit.jl"])):
        hit = serve_tampered(field, value)
        try:
            code, out = run_check()
        finally:
            mod.load = REAL_LOAD
        assert hit, f"no {TARGET}* case receipt was served"
        assert code != 0 and "STALE" in out, f"tampered {field} on {hit[0]} passed --check:\n{out}"
        needle = "dirty tree" if field == "glvmodels_worktree_dirty" else field
        assert needle in out, f"tampered {field}: --check failed but did not name it:\n{out}"
        print(f"ok  tampered {field} on {hit[0]} -> --check STALE")
    # Campaign rows (clause C3/C4/C5 plus a ruling) are carried verbatim; an unknown row without
    # them is still refused, and a campaign row whose clause is stripped is refused too.
    for label, mutate, want_ok in (
            ("unknown row with no clause", lambda rows: rows.append({"source_id": "data/NOT-A-P0-ROW"}), False),
            ("campaign row with its clause removed",
             lambda rows: next(r for r in rows if r.get("clause")).pop("clause"), False)):
        def load(path, _m=mutate):
            obj = REAL_LOAD(path)
            if Path(path).name == "case-map-data.json":
                obj = copy.deepcopy(obj)
                _m(obj["rows"])
            return obj
        mod.load = load
        try:
            code, out = run_check()
        finally:
            mod.load = REAL_LOAD
        assert (code == 0) == want_ok and "re-derivation refused" in out, f"{label}: --check did not refuse:\n{out}"
        print(f"ok  {label} -> --check refuses it")
    n_campaign = sum(1 for r in REAL_LOAD(mod.ROOT / mod.casemap_rel("data"))["rows"] if r.get("clause"))
    assert n_campaign > 0, "expected campaign rows in case-map-data.json"
    print(f"ok  {n_campaign} campaign rows carried verbatim by the clean --check")
    print("TEST_CORE070_DATA_P1_RECEIPTS_OK")


if __name__ == "__main__":
    main()

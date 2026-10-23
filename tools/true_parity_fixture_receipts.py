#!/usr/bin/env python3
"""Convert tracked R-at-P1 vs Julia TOML fixtures into true-parity numeric receipts.

Python 3 stdlib only. Deterministic: re-running writes byte-identical files.

What this does, and does not do
  * Every number in a receipt is COPIED from a tracked fixture (r_value / julia_value). Nothing
    is recomputed, rounded or typed here. The only derived field is `abs_diff`, which the
    checker (tools/true_parity_check.mjs, comparisonCaseDiff) recomputes itself and requires
    to agree.
  * A tolerance is READ from the existing assertion in the Julia test file (the line is named
    in SOURCES and its text is regex-matched, so a moved or edited assertion fails this tool
    rather than silently keeping a stale number).
  * A pair is only admitted when BOTH sides are stored in the fixture. A fixture that stores
    only the R side (the Julia side is computed when the test runs) cannot be turned into a
    receipt by copying, so it has no entry here.
  * A pair that would not pass at the test's tolerance aborts the run (exit 1): nothing is
    written for it and the row must not be bound.

Usage
  python3 tools/true_parity_fixture_receipts.py           # write receipts
  python3 tools/true_parity_fixture_receipts.py --check   # exit 1 if committed receipts differ
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path("docs/dev-log/core070/true-parity-latest/receipts/fixture-twins")
P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"


class Fail(Exception):
    pass


def sha256(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def parse_toml_subset(text):
    """Tables `[a.b]` and `key = number | "string" | [numbers]`; the zi_p1.toml subset."""
    data, cur = {}, None
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[([A-Za-z0-9_.]+)\]", line)
        if m:
            cur = data
            for part in m.group(1).split("."):
                cur = cur.setdefault(part, {})
            continue
        m = re.fullmatch(r"([A-Za-z0-9_]+)\s*=\s*(.+)", line)
        if not m:
            raise Fail(f"toml line {n}: cannot parse {line!r}")
        key, val = m.groups()
        tgt = data if cur is None else cur
        if val.startswith('"') and val.endswith('"'):
            tgt[key] = val[1:-1]
        elif val.startswith("["):
            inner = val[1:-1].strip()
            tgt[key] = [float(x) for x in inner.split(",")] if inner else []
        else:
            tgt[key] = int(val) if re.fullmatch(r"-?\d+", val) else float(val)
    return data


def test_tolerance(test_path, line_no, must_contain):
    """Read the numeric tolerance from the existing assertion at test_path:line_no."""
    lines = (ROOT / test_path).read_text().splitlines()
    text = lines[line_no - 1]
    if must_contain not in text:
        raise Fail(f"{test_path}:{line_no} is {text.strip()!r}, expected it to contain {must_contain!r}")
    m = re.search(r"<\s*([0-9.]+e-?[0-9]+)", text)
    if not m:
        raise Fail(f"{test_path}:{line_no}: no tolerance literal in {text.strip()!r}")
    return float(m.group(1)), text.strip()


# ---- ZI (itchyshin/GLLVModels.jl#557): test/fixtures/zi_p1.toml, test/test_zi_twin.jl -------
ZI_FIXTURE = "test/fixtures/zi_p1.toml"
ZI_TEST = "test/test_zi_twin.jl"
# (case suffix, quantity, R field, Julia field, test line, text that line must contain, note)
ZI_CASES = [
    ("LOGLIK-OPTIMUM", "logLik at the optimum", "r_reference.loglik", "r_at_julia.julia_loglik",
     195, "d_opt < 1e-6",
     "Test asserts |fit.loglik - R logLik|; the fixture's julia_loglik is the recorded Julia "
     "optimum (header of zi_p1.toml), and line 198 asserts a rerun reproduces it to 1e-6."),
    ("CROSS-OBJECTIVE-R-AT-JULIA-OPTIMUM", "R objective at the Julia optimum vs Julia logLik there",
     "r_at_julia.r_objective_at_julia_optimum", "r_at_julia.julia_loglik",
     197, "d_cross_r < 1e-6",
     "Both sides are stored in the fixture exactly as the test reads them."),
    ("BETA", "per-trait fixed intercept beta", "r_reference.beta", "r_at_julia.julia_beta",
     200, "d_beta < 1e-3",
     "Test asserts max|fit.beta - R beta|; the fixture's julia_beta is the recorded Julia optimum."),
]
ZI_ROWS = {  # row source_id -> (fixture table, receipt file stem, case id prefix)
    "zi/zi_poisson": ("zi_poisson", "zi_poisson", "P1-FIXTURE-ZI-POISSON"),
    "zi/zi_nbinom2": ("zi_nbinom2", "zi_nbinom2", "P1-FIXTURE-ZI-NBINOM2"),
    "zi/zi_binomial": ("zi_binomial", "zi_binomial", "P1-FIXTURE-ZI-BINOMIAL"),
}


def dig(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def max_abs_diff(r, j):
    if isinstance(r, list):
        if len(r) != len(j) or not r:
            raise Fail("vector length mismatch")
        return max(abs(a - b) for a, b in zip(r, j))
    return abs(r - j)


def build():
    fx = parse_toml_subset((ROOT / ZI_FIXTURE).read_text())
    if fx.get("gllvmTMB_commit") != P1_SHA:
        raise Fail("zi fixture is not pinned at P1")
    fx_sha, test_sha = sha256(ZI_FIXTURE), sha256(ZI_TEST)
    out = {}
    for source_id, (table, stem, prefix) in ZI_ROWS.items():
        cases = []
        for suffix, quantity, rf, jf, line, frag, note in ZI_CASES:
            tol, line_text = test_tolerance(ZI_TEST, line, frag)
            r, j = dig(fx[table], rf), dig(fx[table], jf)
            d = max_abs_diff(r, j)
            cid = f"{prefix}-{suffix}"
            if not d <= tol:
                raise Fail(f"{cid}: abs diff {d!r} > tolerance {tol!r}; do not bind {source_id}")
            cases.append({
                "case_id": cid,
                "quantity": quantity,
                "r_source": f"{ZI_FIXTURE} [{table}.{rf}]",
                "julia_source": f"{ZI_FIXTURE} [{table}.{jf}]",
                "r_value": r,
                "julia_value": j,
                "abs_diff": d,
                "tolerance": tol,
                "tolerance_source": f"{ZI_TEST}:{line}",
                "tolerance_source_line": line_text,
                "note": note,
            })
        out[f"zi/{stem}.json"] = {
            "schema": "true-parity-fixture-twin-receipt/v1",
            "source_ids": [source_id],
            "verdict": "PASS",
            "evidence_kind": "recorded_fixture_pair",
            "pin": "P1",
            "reference_commit": P1_SHA,
            "origin_pr": "itchyshin/GLLVModels.jl#557",
            "generator": "tools/true_parity_fixture_receipts.py",
            "source_fixture": {"path": ZI_FIXTURE, "sha256": fx_sha},
            "source_test": {"path": ZI_TEST, "sha256": test_sha},
            "what_this_is_not": ("Values are copied from the tracked fixture; the Julia side is the "
                                 "optimum recorded there, not a fresh fit made by this tool."),
            "comparison": {"pin": "P1", "cases": cases},
        }
    return out


def render(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    try:
        files = build()
    except Fail as e:
        print(f"FAIL {e}")
        return 1
    bad = []
    for rel, obj in files.items():
        p = ROOT / OUT_DIR / rel
        text = render(obj)
        if a.check:
            if not p.is_file() or p.read_text() != text:
                bad.append(str(OUT_DIR / rel))
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    if a.check:
        if bad:
            print("STALE " + ", ".join(bad))
            return 1
        print(f"OK {len(files)} receipts match")
        return 0
    print(f"wrote {len(files)} receipts under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

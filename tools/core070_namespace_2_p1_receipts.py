"""Write the tracked P1 receipt for the namespace-2 native-fit case (namespace/export/gllvmTMB).

Reads one retained run of tools/core070_namespace_2_batch.R at gllvmTMB pin P1
(GLLVM_PARITY_PIN=P1) and writes, under docs/dev-log/core070/true-parity-latest/receipts/namespace/:

  namespace-2-batch-p1/...   the run's artifacts copied verbatim, plus verify.txt (the
                             verifier's full output, --pin P1 --self-test, on the copy)
  cases/CORE070-NAMESPACE2-GLLVMTMB-NATIVE-FIT.json
                             case receipt with a `comparison` block pinned to P1

Only this one case is written: it is the executable case of the namespace/export/gllvmTMB
row. The other namespace-2 cases are left to their own rows.

The comparison uses the tolerance the harness itself applies (tools/core070_namespace_2_batch.jl,
tol["loglik_delta"] = tol["coef_delta"] = 1e-4), never a wider one. Both differences are
recomputed from the raw R and Julia values in julia-results.json and must agree with the
harness's own coef_delta / loglik_delta.

Degenerate-comparison gate: each comparison's `discriminating` flag comes from the shared rule
mark_degenerate / DEGENERATE_ABS = 1e-10 in tools/core070_postfit_p1_receipts.py (imported, as
tools/core070_data_p1_receipts.py does), never set by hand. The fixture's Y is row-centred, so
every R trait intercept is ~1e-14 and that comparison is discriminating: false. The row tier
follows the siblings' rule (tools/core070_data_p1_receipts.py build_rows): a row binds as
"numeric" only when its case passes, its batch verifier passes, and
all(e.get("discriminating", True) for e in comparison); with any degenerate entry it is
"numeric_non_discriminating" and its receipt is listed under evidence.non_binding_receipts.
The tool writes that row (namespace/export/gllvmTMB) in case-map-namespace.json.

The run directory must hold run-commit.json ({"glvmodels_commit": <HEAD at launch>, "dirty": []}),
written by whoever launched the batch from a clean checkout.

Usage:
  python3 tools/core070_namespace_2_p1_receipts.py --run <retained-run-dir>
  python3 tools/core070_namespace_2_p1_receipts.py --check

--check re-runs the verifier on the tracked copy, requires its acceptance marker, and
re-derives the case receipt and the case-map row from the tracked artifacts; it exits nonzero
on any difference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
# The shared degenerate-comparison rule (PR #569), as tools/core070_data_p1_receipts.py imports it.
from core070_postfit_p1_receipts import DEGENERATE_ABS, mark_degenerate  # noqa: E402,F401
OUT = ROOT / "docs/dev-log/core070/true-parity-latest/receipts/namespace"
BATCH = OUT / "namespace-2-batch-p1"
CASE_ID = "CORE070-NAMESPACE2-GLLVMTMB-NATIVE-FIT"
CASE_PATH = OUT / "cases" / f"{CASE_ID}.json"
CONTRACT = ROOT / "docs/dev-log/core070/true-parity-latest/namespace-2-batch-contract-p1.json"
CASEMAP = ROOT / "docs/dev-log/core070/true-parity-latest/case-map-namespace.json"
SOURCE_ID = "namespace/export/gllvmTMB"
P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
P0_SHA = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
HARNESS_TOL = 1e-4
FILES = ["receipt.json", "r-oracle.json", "julia-results.json", "results.tsv",
         "diagnostics.log", "julia-stdout.log", "julia-stderr.log", "run-commit.json"]
VERIFY_MARKER = "CORE070_NAMESPACE_2_BATCH_VERIFIED"


def rel(p):
    return str(Path(p).relative_to(ROOT))


def load(p):
    return json.loads(Path(p).read_text())


def run_verifier(state):
    argv = [sys.executable, "tools/core070_verify_namespace_2_batch.py", "--pin", "P1",
            "--self-test", "--state", rel(state)]
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    return proc.returncode, text


def max_abs(r, j):
    if len(r) != len(j):
        raise ValueError("length mismatch")
    return max(abs(a - b) for a, b in zip(r, j))


def case_receipt(verify_exit):
    batch = load(BATCH / "receipt.json")
    jres = load(BATCH / "julia-results.json")
    run = load(BATCH / "run-commit.json")
    contract = load(CONTRACT)
    case = jres["cases"][CASE_ID]
    if batch["reference_commit"] != P1_SHA or batch.get("pin") != "P1":
        raise ValueError("batch receipt is not a P1 run")
    if run.get("dirty") != []:
        raise ValueError("run was launched from a dirty checkout")
    r_coef, j_coef = case["r_coef"], case["julia_coef"]
    r_ll, j_ll = case["r_loglik"], case["julia_loglik"]
    coef_d = max_abs(r_coef, j_coef)
    ll_d = abs(r_ll - j_ll)
    if abs(coef_d - case["coef_delta"]) > 1e-12 * max(1.0, coef_d) or \
       abs(ll_d - case["loglik_delta"]) > 1e-9 * max(1.0, ll_d):
        raise ValueError("recomputed differences disagree with the harness figures")
    contract_case = next(c for c in contract["cases"] if c["case_id"] == CASE_ID)
    rule = ("harness tolerance tol[\"loglik_delta\"] = tol[\"coef_delta\"] = 1e-4 in "
            "tools/core070_namespace_2_batch.jl (the contract's prose says \"<=1e-6\"; the harness "
            "asserts 1e-4); max |R - Julia| elementwise")
    return {
        "schema": "core070-namespace-2-p1-case-receipt/v1",
        "case_id": CASE_ID,
        "verdict": "PASS" if case["pass"] else "FAIL",
        "evidence_kind": "numeric_r_vs_julia",
        "source_ids": [contract_case["source_id"]],
        "batch": "tools/core070_namespace_2_batch.R + .jl, GLLVM_PARITY_PIN=P1",
        "harness_pass": bool(case["pass"]),
        "batch_status": batch["status"],
        "batch_verifier": {
            "tool": "tools/core070_verify_namespace_2_batch.py",
            "argv": "tools/core070_verify_namespace_2_batch.py --pin P1 --self-test --state {state}",
            "status": "PASS" if verify_exit == 0 else "FAIL",
            "exit_code": verify_exit,
            "accept_marker": VERIFY_MARKER,
            "log": rel(BATCH / "verify.txt"),
        },
        "r_call": ("gllvmTMB(value ~ 0 + trait + latent(0 + trait | site, d = 2, unique = FALSE), "
                   "data = df_long_g, unit = \"site\", trait = \"trait\", family = gaussian(), "
                   "control = gllvmTMBcontrol(n_init = 1L, se = FALSE)); coef(fit), logLik(fit)"),
        "julia_call": "fit_gaussian_gllvm(Y; K = 2, X = trait-indicator X); coef(fit), fit.logLik",
        "fixture": "gaussian_fixture: set.seed(42), p = 5 traits, n = 80 sites, K = 2, Y row-centred",
        "raw": [rel(BATCH / "julia-results.json"), rel(BATCH / "r-oracle.json"),
                rel(BATCH / "receipt.json")],
        "pin": "P1",
        "reference_commit": P1_SHA,
        "p0_reference_commit": P0_SHA,
        "contract": rel(CONTRACT),
        "contract_sha256": batch["contract_sha256"],
        "gllvmtmb_version": batch["gllvmTMB_version"],
        "library_namespace_sha256": batch["library_pin"]["namespace_sha256"],
        "library_pin_note": ("the frozen library carries no CORE070_SOURCE_PIN.toml marker; the runner "
                             "checked its installed NAMESPACE sha256 and version against "
                             "tools/core070_oracle_pins.toml [P1]"),
        "r_version": batch["r_version"],
        "glvmodels_commit": run["glvmodels_commit"],
        "glvmodels_worktree_dirty": run["dirty"],
        "host": "local Mac (M1 Ultra), OPENBLAS/OMP threads 1, JULIA_NUM_THREADS=1",
        "comparison": {
            "pin": "P1",
            "cases": [
                mark_degenerate({
                    "case_id": CASE_ID,
                    "quantity": "logLik",
                    "max_abs_diff": ll_d,
                    "tolerance": HARNESS_TOL,
                    "tolerance_rule": rule,
                    "n_values": 1,
                    "diff_source": "recomputed from raw R and Julia values",
                    "r_value": [r_ll],
                    "julia_value": [j_ll],
                }, [r_ll]),
                mark_degenerate({
                    "case_id": CASE_ID,
                    "quantity": "coef (trait intercepts)",
                    "max_abs_diff": coef_d,
                    "tolerance": HARNESS_TOL,
                    "tolerance_rule": rule,
                    "n_values": len(r_coef),
                    "diff_source": "recomputed from raw R and Julia values",
                    "r_value": r_coef,
                    "julia_value": j_coef,
                    "discriminating_note": ("Y is row-centred in the fixture, so every R intercept is "
                                            "~1e-14 (below 1e-10): this comparison cannot tell a right "
                                            "implementation from a zero one. The logLik comparison "
                                            "above is the discriminating check."),
                }, r_coef),
            ],
        },
    }


def dump(value):
    return json.dumps(value, indent=2) + "\n"


def marker_line():
    lines = [ln for ln in (BATCH / "verify.txt").read_text().splitlines() if ln.startswith(VERIFY_MARKER)]
    return lines[-1] if lines else "verifier acceptance marker absent"


def casemap_row(base, rec):
    """The case-map row for SOURCE_ID, tiered by the siblings' rule (core070_data_p1_receipts.py build_rows)."""
    comp = rec["comparison"]["cases"]
    disc = all(e.get("discriminating", True) for e in comp)
    verdicts = {CASE_ID: rec["verdict"]}
    batch_ok = {CASE_ID: rec["batch_verifier"]["status"]}
    path = rel(CASE_PATH)
    if rec["evidence_kind"] != "numeric_r_vs_julia" or rec["verdict"] != "PASS":
        raise ValueError(f"{CASE_ID}: no tier rule here for {rec['evidence_kind']} / {rec['verdict']}")
    by_q = {e["quantity"]: e for e in comp}
    measured = ("R gllvmTMB(value ~ 0 + trait + latent(0 + trait | site, d = 2, unique = FALSE), "
                "family = gaussian()) on the batch's Gaussian fixture (p = 5, n = 80) against Julia "
                "fit_gaussian_gllvm(Y; K = 2, X) on the same Y, at the P1 pin (gllvmTMB "
                f"{rec['gllvmtmb_version']}): logLik |R - Julia| = {by_q['logLik']['max_abs_diff']:.3g} and "
                f"trait intercepts max |R - Julia| = {by_q['coef (trait intercepts)']['max_abs_diff']:.3g}, "
                f"both within the harness tolerance {HARNESS_TOL:g}.")
    row = dict(base)
    row["executable_case_ids"] = [CASE_ID]
    row["measured_against"] = P1_SHA
    row.pop("not_measured_reason", None)
    common_ev = {"batch": "namespace-2 at P1",
                 "verifier": ("python3 tools/core070_verify_namespace_2_batch.py --pin P1 --self-test --state "
                              f"{rel(BATCH)} ({marker_line()})")}
    if batch_ok[CASE_ID] != "PASS":
        row["evidence_tier"] = "numeric_held_batch_verifier_failed"
        row["evidence"] = {**common_ev, "non_binding_receipts": [path],
                           "tier": ("numeric comparison blocks pass, but the batch verifier rejected the "
                                    "run, so the row does not bind")}
        row["measured_result"] = {"case_verdicts": verdicts, "batch_verifier": batch_ok,
                                  "discriminating": {CASE_ID: disc}}
        row["reason"] = "Measured at P1, but the namespace-2 batch verifier rejected the run. " + measured
    elif not disc:
        degenerate = [e["quantity"] for e in comp if not e.get("discriminating", True)]
        row["evidence_tier"] = "numeric_non_discriminating"
        row["evidence"] = {**common_ev, "non_binding_receipts": [path],
                           "tier": ("numeric comparison blocks pass, but at least one is degenerate (the R "
                                    "values are one constant or all ~0), so the row does not bind. "
                                    f"Degenerate under the shared rule (DEGENERATE_ABS = {DEGENERATE_ABS:g}): "
                                    f"{', '.join(degenerate)}.")}
        row["measured_result"] = {"case_verdicts": verdicts, "batch_verifier": batch_ok,
                                  "discriminating": {CASE_ID: disc}}
        row["reason"] = ("Measured at P1, not bound: the case receipt carries a degenerate comparison, so "
                         "under the siblings' shared rule the row is numeric_non_discriminating. " + measured)
    else:
        row["evidence_tier"] = "numeric"
        row["evidence"] = {**common_ev, "receipt": [path],
                           "tier": ("numeric: every executable case receipt carries an R-vs-Julia comparison "
                                    "block pinned to P1, within the harness tolerance")}
        row["measured_result"] = {"case_verdicts": verdicts, "batch_verifier": batch_ok, "row_verdict": "PASS"}
        row["reason"] = "Numeric R-vs-Julia native fit at P1. " + measured
    return row


def casemap_derived():
    """(original text, re-derived text) of case-map-namespace.json with SOURCE_ID's row re-tiered."""
    text = CASEMAP.read_text()
    cm = json.loads(text)
    idx = [i for i, r in enumerate(cm["rows"]) if r["source_id"] == SOURCE_ID]
    if len(idx) != 1:
        raise ValueError(f"{SOURCE_ID}: expected one row in {rel(CASEMAP)}, found {len(idx)}")
    cm["rows"][idx[0]] = casemap_row(cm["rows"][idx[0]], json.loads(CASE_PATH.read_text()))
    return text, json.dumps(cm, indent=1) + "\n"


def write(run_dir):
    run_dir = Path(run_dir)
    for name in FILES:
        if not (run_dir / name).is_file():
            sys.exit(f"missing {name} in {run_dir}")
    BATCH.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(run_dir / name, BATCH / name)
    code, text = run_verifier(BATCH)
    (BATCH / "verify.txt").write_text(text)
    if code != 0 or VERIFY_MARKER not in text:
        sys.exit("verifier rejected the copied run; see verify.txt")
    CASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CASE_PATH.write_text(dump(case_receipt(code)))
    print("wrote", rel(CASE_PATH))
    CASEMAP.write_text(casemap_derived()[1])
    print("wrote", rel(CASEMAP), f"({SOURCE_ID})")


def check():
    code, text = run_verifier(BATCH)
    if code != 0 or VERIFY_MARKER not in text:
        sys.exit("verifier rejected the tracked run:\n" + text)
    if (BATCH / "verify.txt").read_text() != text:
        sys.exit("verify.txt differs from a fresh verifier run")
    if CASE_PATH.read_text() != dump(case_receipt(code)):
        sys.exit(f"{rel(CASE_PATH)} differs from its re-derivation")
    text, derived = casemap_derived()
    if text != derived:
        sys.exit(f"{rel(CASEMAP)} row {SOURCE_ID} differs from its re-derivation")
    print("CORE070_NAMESPACE_2_P1_RECEIPTS_CURRENT")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path)
    g.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
    else:
        write(args.run)

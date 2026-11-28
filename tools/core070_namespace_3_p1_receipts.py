"""Write the tracked P1 receipt for the extract_phylo_signal numeric case (namespace-3, D5).

Reads one retained run of tools/core070_namespace_3_d5_batch.R + .jl (launched by
tools/core070_namespace_3_d5_run.sh) and writes, under
docs/dev-log/core070/true-parity-latest/receipts/namespace/:

  namespace-3-d5-p1/...   the run's files copied verbatim, plus verify.txt (this tool's own
                          verification of the copy)
  cases/CORE070-NAMESPACE3-EXTRACT-PHYLO-SIGNAL-JOINT-PAIRED-A.json
                          case receipt with a `comparison` block pinned to P1

and re-tiers ONE row (namespace/export/extract_phylo_signal) in case-map-namespace.json: its numeric
case becomes the only executable case, and the old registration case and receipts stay cited under
registration_case_ids / registration_receipts (the shape of the coef,gllvmTMB_multi row). Nothing else
changes: not the frozen namespace-1 contract, not tools/true_parity_check.mjs or
tools/true_parity_assemble.py. Run `python3 tools/true_parity_assemble.py` afterwards to regenerate
the scoreboard and the assembled case map.

The contract is docs/dev-log/core070/true-parity-latest/namespace-3-d5-batch-contract-p1.json. The seed
(4401), the bars and the decision rule are also fixed in this file, so editing the contract alone cannot
change them; the tool refuses a run if the two disagree. The decision rule is "matched point decides":
the comparison block holds the four matched-point quantities (Julia evaluated at R's estimates) at 1e-6;
the independent Julia fit is recorded under `independent_fit` against its bars (logLik 1e-4, H2 1e-3) and
reported, but it is not in the comparison block.

Refusals (each a nonzero exit, tested in tools/test_core070_namespace_3_p1_receipts.py): a run whose seed
is not 4401, whose frozen data or tree bytes differ from the contract, whose design block no longer
hashes to the contract's design_sha256, whose R fit did not converge, whose oracle is not gllvmTMB 0.7.1,
whose run commit is not an ancestor of HEAD or whose Julia sources or harness changed since it, or whose
matched point is outside a bar.

Usage:
  python3 tools/core070_namespace_3_p1_receipts.py --run <retained-run-dir>
  python3 tools/core070_namespace_3_p1_receipts.py --check [--root DIR]

--check re-verifies the tracked copy and re-derives the case receipt and the case-map row; it exits
nonzero on any difference.
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
from core070_postfit_p1_receipts import DEGENERATE_ABS, mark_degenerate  # noqa: E402

LEDGER = "docs/dev-log/core070/true-parity-latest"
APPROVED_SEED = 4401
MATCHED_BAR = 1e-6
BARS = {"matched_point": {"logLik": 1e-6, "H2": 1e-6, "Psi": 1e-6, "V_eta": 1e-6},
        "independent_fit": {"logLik": 1e-4, "H2": 1e-3}}
CASE_ID = "CORE070-NAMESPACE3-EXTRACT-PHYLO-SIGNAL-JOINT-PAIRED-A"
SOURCE_ID = "namespace/export/extract_phylo_signal"
P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
GLLVMTMB_VERSION = "0.7.1"
FILES = ["frozen_A.csv", "tree.nwk", "R_params_A.csv", "R_signal_A.csv", "R_summary_A.csv", "R_session.csv",
         "julia-results.json", "run-commit.json"]
# Files whose bytes must be the same at the run commit and at HEAD (what the run executed).
RUN_BOUND = ["src", "Project.toml", "tools/core070_namespace_3_d5_batch.R", "tools/core070_namespace_3_d5_batch.jl",
             "tools/core070_namespace_3_d5_run.sh"]
VERIFY_MARKER = "CORE070_NAMESPACE_3_D5_RUN_VERIFIED"
SIGNAL_KEYS = [("H2", "H2"), ("Psi", "Psi"), ("V_eta", "V_eta")]


class Refuse(Exception):
    pass


def paths(root):
    out = root / LEDGER / "receipts/namespace"
    return {
        "out": out, "batch": out / "namespace-3-d5-p1", "case": out / "cases" / f"{CASE_ID}.json",
        "contract": root / LEDGER / "namespace-3-d5-batch-contract-p1.json",
        "casemap": root / LEDGER / "case-map-namespace.json",
    }


def rel(root, p):
    return str(Path(p).relative_to(root))


def load(p):
    return json.loads(Path(p).read_text())


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def design_hash(design):
    return hashlib.sha256(json.dumps(design, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def read_kv(p):
    lines = Path(p).read_text().splitlines()
    assert lines[0] == "key,value"
    return dict(ln.split(",", 1) for ln in lines[1:])


def read_table(p):
    lines = Path(p).read_text().splitlines()
    head = lines[0].split(",")
    return head, [ln.split(",") for ln in lines[1:]]


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def max_abs(r, j):
    if len(r) != len(j) or not r:
        raise Refuse("length mismatch between R and Julia values")
    return max(abs(a - b) for a, b in zip(r, j))


def check_contract(root):
    P = paths(root)
    c = load(P["contract"])
    if c.get("seed") != APPROVED_SEED or c["design"].get("seed") != APPROVED_SEED:
        raise Refuse(f"contract seed is not the approved {APPROVED_SEED}")
    if c.get("bars") != BARS:
        raise Refuse("contract bars differ from the approved bars")
    if c.get("case_id") != CASE_ID or c.get("source_id") != SOURCE_ID:
        raise Refuse("contract case or source id differs")
    if design_hash(c["design"]) != c.get("design_sha256"):
        raise Refuse("contract design block does not hash to its design_sha256")
    return c


def verify_run(root, state):
    """Verify one run directory against the contract; return the facts the receipt needs."""
    P = paths(root)
    c = check_contract(root)
    for name in FILES:
        if not (state / name).is_file():
            raise Refuse(f"missing {name}")
    summ = read_kv(state / "R_summary_A.csv")
    if int(float(summ["seed"])) != APPROVED_SEED:
        raise Refuse(f"seed {summ['seed']} is not the approved {APPROVED_SEED}")
    if int(float(summ["convergence"])) != 0:
        raise Refuse("R fit did not converge")
    if sha256_file(state / "frozen_A.csv") != c["design"]["frozen_data_sha256"]:
        raise Refuse("frozen data bytes differ from the contract (design hash mismatch)")
    if sha256_file(state / "tree.nwk") != c["design"]["tree_file_sha256"]:
        raise Refuse("tree bytes differ from the contract (design hash mismatch)")
    sess = read_kv(state / "R_session.csv")
    if sess["gllvmtmb_version"] != GLLVMTMB_VERSION:
        raise Refuse(f"oracle is gllvmTMB {sess['gllvmtmb_version']}, not {GLLVMTMB_VERSION}")
    run = load(state / "run-commit.json")
    if run.get("dirty") != [] or not isinstance(run.get("glvmodels_commit"), str):
        raise Refuse("run was not launched from a clean exported commit")
    rc = run["glvmodels_commit"]
    if git(root, "merge-base", "--is-ancestor", rc, "HEAD").returncode != 0:
        raise Refuse("run commit is not an ancestor of HEAD")
    if git(root, "diff", "--quiet", rc, "HEAD", "--", *RUN_BOUND).returncode != 0:
        raise Refuse("Julia sources or the harness changed between the run commit and HEAD")
    jr = load(state / "julia-results.json")
    # R values
    head, rows = read_table(state / "R_signal_A.csv")
    rsig = {k: [float(r[head.index(col)]) for r in rows] for k, col in SIGNAL_KEYS}
    r_ll = float(summ["logLik"])
    m = jr["matched"]
    diffs = {"logLik": abs(r_ll - m["logLik"])}
    for k, _ in SIGNAL_KEYS:
        diffs[k] = max_abs(rsig[k], m[k])
    bad = [k for k, d in diffs.items() if not d <= BARS["matched_point"][k]]
    if bad:
        raise Refuse("matched point outside its bar for " + ", ".join(f"{k} ({diffs[k]:.3g})" for k in bad))
    ind = jr["independent"]
    ind_diffs = {"logLik": abs(r_ll - ind["logLik"]), "H2": max_abs(rsig["H2"], ind["H2"])}
    return {"c": c, "summ": summ, "sess": sess, "run": run, "jr": jr, "rsig": rsig, "r_ll": r_ll,
            "diffs": diffs, "ind_diffs": ind_diffs, "state": state}


def case_receipt(root, v, verify_text):
    P = paths(root)
    c, jr, rsig, r_ll = v["c"], v["jr"], v["rsig"], v["r_ll"]
    m, ind = jr["matched"], jr["independent"]
    tol_rule = "contract bar at R's estimates, 1e-6 on every quantity; max |R - Julia| elementwise"
    cases = []
    for q, rv, jv, d in [("logLik (at R estimates)", [r_ll], [m["logLik"]], v["diffs"]["logLik"])] + [
            (f"{k} (at R estimates)", rsig[k], m[k], v["diffs"][k]) for k, _ in SIGNAL_KEYS]:
        key = q.split(" ")[0]
        cases.append(mark_degenerate({
            "case_id": CASE_ID, "quantity": q, "max_abs_diff": d, "tolerance": BARS["matched_point"][key],
            "tolerance_rule": tol_rule, "n_values": len(rv), "diff_source": "recomputed from raw R and Julia values",
            "r_value": rv, "julia_value": jv}, rv))
    ind_block = {
        "note": "recorded and reported; under the decision rule (matched point decides) this block is not part of the binding comparison",
        "logLik": {"r_value": r_ll, "julia_value": ind["logLik"], "abs_diff": v["ind_diffs"]["logLik"],
                   "bar": BARS["independent_fit"]["logLik"], "within_bar": v["ind_diffs"]["logLik"] <= BARS["independent_fit"]["logLik"]},
        "H2": {"r_value": rsig["H2"], "julia_value": ind["H2"], "max_abs_diff": v["ind_diffs"]["H2"],
               "bar": BARS["independent_fit"]["H2"], "within_bar": v["ind_diffs"]["H2"] <= BARS["independent_fit"]["H2"]},
        "converged": ind["converged"], "stopping_reason": ind["stopping_reason"], "iterations_cap": ind["iterations_cap"],
    }
    return {
        "schema": "core070-namespace-3-p1-case-receipt/v1",
        "case_id": CASE_ID,
        "verdict": "PASS",
        "evidence_kind": "numeric_r_vs_julia",
        "source_ids": [SOURCE_ID],
        "batch": "tools/core070_namespace_3_d5_batch.R + .jl via tools/core070_namespace_3_d5_run.sh, GLLVM_PARITY_PIN=P1",
        "harness_pass": True,
        "batch_status": "PASS",
        "batch_verifier": {"tool": "tools/core070_namespace_3_p1_receipts.py", "argv": "--check", "status": "PASS",
                           "accept_marker": VERIFY_MARKER, "log": rel(root, P["batch"] / "verify.txt")},
        "decision_rule": c["decision_rule"],
        "r_call": c["design"]["r_call"],
        "julia_call": c["design"]["julia_call"],
        "fixture": c["design"]["dimensions"] + "; fixed tree; seed " + str(APPROVED_SEED) + "; one draw, no reseeding",
        "seed": APPROVED_SEED,
        "design_sha256": c["design_sha256"],
        "frozen_data_sha256": c["design"]["frozen_data_sha256"],
        "raw": [rel(root, P["batch"] / n) for n in ("julia-results.json", "R_params_A.csv", "R_signal_A.csv", "R_summary_A.csv")],
        "pin": "P1",
        "reference_commit": P1_SHA,
        "contract": rel(root, P["contract"]),
        "contract_sha256": sha256_file(P["contract"]),
        "gllvmtmb_version": v["sess"]["gllvmtmb_version"],
        "gllvmtmb_remote_sha": v["sess"]["gllvmtmb_remote_sha"],
        "r_version": v["sess"]["r_version"],
        "glvmodels_commit": v["run"]["glvmodels_commit"],
        "glvmodels_worktree_dirty": v["run"]["dirty"],
        "julia_version": jr["julia_version"],
        "threads": {"julia": jr["julia_threads"], "openblas": v["sess"]["openblas_threads"], "omp": v["sess"]["omp_threads"]},
        "r_elapsed_s": float(v["summ"]["elapsed_s"]),
        "comparison": {"pin": "P1", "cases": cases},
        "independent_fit": ind_block,
        "caveats": c["caveats"],
    }


def dump(value):
    return json.dumps(value, indent=2) + "\n"


def casemap_row(root, base, rec):
    P = paths(root)
    comp = rec["comparison"]["cases"]
    if not all(e.get("discriminating", True) for e in comp):
        raise Refuse("a matched-point comparison is degenerate under the shared rule; the row would not bind")
    by_q = {e["quantity"]: e["max_abs_diff"] for e in comp}
    ind = rec["independent_fit"]
    row = dict(base)
    old_receipts = list(base["evidence"]["receipt"])
    old_cases = list(base["executable_case_ids"])
    row["executable_case_ids"] = [CASE_ID]
    row["evidence_tier"] = "numeric"
    row["evidence"] = {
        "receipt": [rel(root, P["case"])],
        "registration_receipts": old_receipts,
        "registration_case_ids": old_cases,
        "registration_tier": base["evidence"]["tier"],
        "batch": "namespace-3 (D5) at P1",
        "verifier": ("python3 tools/core070_namespace_3_p1_receipts.py --check (" + VERIFY_MARKER + ")"),
        "ruling": ("seed 4401, bars 1e-6 at the matched point (1e-4 logLik and 1e-3 H2 for the independent fit) and the rule "
                   "'matched point decides' approved by the maintainer on 13 Nov 2026; the numeric case lives in "
                   "namespace-3-d5-batch-contract-p1.json, the frozen namespace-1 contract is unchanged."),
        "tier": ("numeric: R gllvmTMB phylo_latent + indep species term against Julia fit_joint_phylo_grouped_gaussian "
                 "(residual_mode = :shared) on one frozen draw (8 species x 3 sites x 3 traits), compared at R's estimates: "
                 "logLik, H2, Psi, V_eta all within 1e-6. The old name-match registration receipts are kept under "
                 "registration_receipts."),
    }
    row["measured_result"] = {
        "case_verdicts": {CASE_ID: "PASS"}, "row_verdict": "PASS",
        "matched_point_max_abs_diff": by_q,
        "independent_fit_within_bars": bool(ind["logLik"]["within_bar"] and ind["H2"]["within_bar"]),
    }
    row["reason"] = ("Numeric R-vs-Julia comparison at R's estimates on one frozen draw (seed 4401), "
                     f"gllvmTMB {rec['gllvmtmb_version']} at the P1 pin. One draw with 8 species; trait 2 is on the "
                     "boundary; H2, Psi and V_eta at the matched point test the name mapping and the formula, logLik is "
                     "the only tree-sensitive number.")
    return row


def casemap_derived(root, rec):
    P = paths(root)
    text = P["casemap"].read_text()
    cm = json.loads(text)
    idx = [i for i, r in enumerate(cm["rows"]) if r["source_id"] == SOURCE_ID]
    if len(idx) != 1:
        raise Refuse(f"{SOURCE_ID}: expected one row, found {len(idx)}")
    base = cm["rows"][idx[0]]
    if base["evidence_tier"] == "numeric" and base["executable_case_ids"] == [CASE_ID]:
        # already re-tiered: rebuild the registration row from the fields the re-tiering preserved
        ev = base["evidence"]
        base = dict(base)
        base["evidence"] = {"receipt": ev["registration_receipts"], "tier": ev["registration_tier"]}
        base["executable_case_ids"] = ev["registration_case_ids"]
    cm["rows"][idx[0]] = casemap_row(root, base, rec)
    return text, json.dumps(cm, indent=1) + "\n"


def write(root, state):
    P = paths(root)
    if git(root, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise Refuse("tracked files are modified; run from a clean commit")
    v = verify_run(root, state)
    P["batch"].mkdir(parents=True, exist_ok=True)
    for n in FILES:
        shutil.copyfile(state / n, P["batch"] / n)
    verify_text = f"{VERIFY_MARKER} seed={APPROVED_SEED} matched_diffs={json.dumps(v['diffs'])}\n"
    (P["batch"] / "verify.txt").write_text(verify_text)
    rec = case_receipt(root, v, verify_text)
    P["case"].parent.mkdir(parents=True, exist_ok=True)
    P["case"].write_text(dump(rec))
    print("wrote", rel(root, P["case"]))
    P["casemap"].write_text(casemap_derived(root, rec)[1])
    print("wrote", rel(root, P["casemap"]), f"({SOURCE_ID})")


def check(root):
    P = paths(root)
    v = verify_run(root, P["batch"])
    verify_text = f"{VERIFY_MARKER} seed={APPROVED_SEED} matched_diffs={json.dumps(v['diffs'])}\n"
    if (P["batch"] / "verify.txt").read_text() != verify_text:
        raise Refuse("verify.txt differs from a fresh verification")
    rec = case_receipt(root, v, verify_text)
    if P["case"].read_text() != dump(rec):
        raise Refuse(f"{rel(root, P['case'])} differs from its re-derivation")
    text, derived = casemap_derived(root, rec)
    if text != derived:
        raise Refuse(f"{rel(root, P['casemap'])} row {SOURCE_ID} differs from its re-derivation")
    print("CORE070_NAMESPACE_3_P1_RECEIPTS_CURRENT")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path)
    g.add_argument("--check", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args()
    try:
        if a.check:
            check(a.root.resolve())
        else:
            write(a.root.resolve(), a.run.resolve())
    except Refuse as e:
        sys.exit(f"REFUSED: {e}")

"""Write tracked P1 receipts and case-map rows for the postfit and postfit-policy families.

Reads the raw outputs of the seven postfit batches run at gllvmTMB pin P1
(GLLVM_PARITY_PIN=P1) and writes, under docs/dev-log/core070/true-parity-latest/:

  receipts/postfit/<batch>-p1/...   batch artifacts copied verbatim (JSON/TSV only, no logs)
  receipts/postfit/cases/<id>.json  one receipt per executable case id
  case-map-postfit.json             the 36 postfit rows the P1 carry scan lists as
                                    DANGLING (34) or RETIRED (2), and the 16 DANGLING
                                    postfit-policy rows

A case gets a `comparison` block only when its harness compares an R number
with a Julia number against a declared tolerance > 0. The tolerance is the
one the harness itself applies (never a wider one), and abs_diff is the
maximum absolute elementwise difference the harness bounds. Where the raw
R and Julia vectors are both in the batch output, max_abs_diff is recomputed
here and must agree with the harness figure; otherwise the harness figure is
used and marked so.

Cases that are verdicts (both engines report a boolean), own-consistency
checks (each engine checked against itself), exact-integer equalities, empty
lengths, or signature/default-policy checks carry no comparison block, and
their rows cite evidence.non_binding_receipts. A row is evidence_tier
"numeric" only when every executable case id has a comparison block and the
harness verdict is PASS.

Usage (inputs are the raw run directories under local-scratch):
  python3 tools/core070_postfit_p1_receipts.py --runs DIR --runtimes JSON
where DIR holds surface-conversion-p1/, wave6-conversion-p1/, wave7-conversion-p1/,
wave8-conversion-p1/, estimand-rebind-p1/, postfit-policy-p1/, postfit-1-r-p1/,
postfit-1-julia-p1/, each with the verifier's output in verify.txt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/dev-log/core070/true-parity-latest"
REC = OUT / "receipts/postfit"
REC_REL = "docs/dev-log/core070/true-parity-latest/receipts/postfit"
P0_CASEMAP = ROOT / "docs/dev-log/core070/required-source-case-map.json"
PINS = tomllib.loads((ROOT / "tools/core070_oracle_pins.toml").read_text())
P1_SHA = PINS["P1"]["reference_commit"]
P0_SHA = PINS["P0"]["reference_commit"]
ORACLE_BUILD = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build.json"
ORACLE_SOURCE = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/source.json"
MAX_INLINE = 25  # vectors longer than this stay in the batch raw file only

# Rows in scope: P1 carry scan (branch claude/true-parity-p1-carry, carry-scan-p1.json).
POSTFIT_DANGLING = [
    "check_auto_residual", "coef.gllvmTMB_multi", "compare_loadings", "confint.gllvmTMB_multi",
    "deviance.gllvmTMB_multi", "extract_ICC_site", "extract_Omega", "extract_Sigma", "extract_Sigma_table",
    "extract_communality", "extract_correlations", "extract_cutpoints", "extract_loadings",
    "extract_ordination", "extract_proportions", "extract_repeatability", "extract_residual_cor",
    "extract_residual_cov", "extract_rotated_loadings_table", "fitted.gllvmTMB_multi", "getLV",
    "getLoadings", "getResidualCor", "getResidualCov", "logLik.gllvmTMB_multi", "nobs.gllvmTMB_multi",
    "predict.gllvmTMB_multi", "predict_missing", "residuals.gllvmTMB_multi", "rotate_loadings",
    "sanity_multi", "simulate_unit_trait", "summary.gllvmTMB_multi", "tidy.gllvmTMB_multi",
]
POSTFIT_RETIRED = [".proportions_bootstrap_ci", ".proportions_wald_ci"]
POLICY_DANGLING = [
    "POST-COEF-EMPTY", "POST-COEF-NAMED", "POST-CONFINT-METHODS", "POST-DEVIANCE", "POST-FITTED-DEFAULT",
    "POST-LOGLIK-DF", "POST-LOGLIK-NOBS", "POST-LOGLIK-VALUE", "POST-NOBS-COUNT", "POST-NOBS-FALLBACK",
    "POST-PREDICT-DEFAULT", "POST-RE-FORM-FULL", "POST-RESIDUAL-CONDITIONAL", "POST-RESIDUAL-SCALES",
    "POST-RESIDUAL-TYPES", "POST-SIMULATE-DEFAULT",
]

# estimand-rebind has no contract file; this is its verifier's CASE_META, verbatim.
ESTIMAND_SOURCE = {
    "CORE070-ESTIMAND-REBIND-EXTRACT-COMMUNALITY": "postfit/POSTFIT-SURFACE-extract_communality",
    "CORE070-ESTIMAND-REBIND-EXTRACT-CORRELATIONS": "postfit/POSTFIT-SURFACE-extract_correlations",
    "CORE070-ESTIMAND-REBIND-EXTRACT-PROPORTIONS": "postfit/POSTFIT-SURFACE-extract_proportions",
    "CORE070-ESTIMAND-REBIND-EXTRACT-OMEGA": "postfit/POSTFIT-SURFACE-extract_Omega",
}

WAVE6_NOTE = ("The wave6 batch receipt reads FAIL, and tools/core070_verify_wave6_conversion_batch.py rejects the "
              "state, because of one case, CORE070-WAVE6-POSTFIT-NOBS-MULTI (own_receipt_defect): its frozen "
              "expectation that Julia nobs returns n = 80 no longer holds (Julia now returns p*n = 400, as R does). "
              "The other nine cases pass; each case receipt carries its own verdict.")

# postfit-policy: case id -> contract tolerance key (numeric cases only).
POLICY_TOL_KEY = {
    "CORE070-POSTFIT-COEF-NAMED-NATIVE": "coefficient_delta",
    "CORE070-POSTFIT-CONFINT-METHODS-WALD-NATIVE": "wald_ci_bound_delta",
    "CORE070-POSTFIT-FITTED-DEFAULT-NATIVE": "link_response_residual_delta",
    "CORE070-POSTFIT-LOGLIK-VALUE-NATIVE": "loglik_delta",
    "CORE070-POSTFIT-RE-FORM-FULL-NATIVE": "link_response_residual_delta",
    "CORE070-POSTFIT-RESIDUAL-CONDITIONAL-NATIVE": "link_response_residual_delta",
    "CORE070-POSTFIT-RESIDUAL-SCALES-NATIVE": "link_response_residual_delta",
    "CORE070-POSTFIT-RESIDUAL-TYPES-NATIVE": "link_response_residual_delta",
}
# postfit-policy cases that carry a numeric leg but whose fact is a documented default divergence.
POLICY_PARTIAL_WITH_NUMERIC_LEG = {"CORE070-POSTFIT-PREDICT-DEFAULT-NATIVE": "link_response_residual_delta"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git_head():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def verifier_lines(run_dir):
    p = run_dir / "verify.txt"
    lines = [l for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    return lines[-3:]


def copy_batch(src_dir, dest_name, names):
    dest = REC / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for n in names:
        p = src_dir / n
        if p.is_file():
            shutil.copyfile(p, dest / n)
            out.append(f"{REC_REL}/{dest_name}/{n}")
    return out


def vec_entry(case_id, quantity, rv, jv, tol, harness_diff, rule):
    rv = rv if isinstance(rv, list) else [rv]
    jv = jv if isinstance(jv, list) else [jv]
    if len(rv) != len(jv):
        raise SystemExit(f"{case_id}: R length {len(rv)} != Julia length {len(jv)}")
    diff = max(abs(a - b) for a, b in zip(rv, jv))
    if harness_diff is not None and abs(diff - harness_diff) > 1e-12 * max(1.0, abs(harness_diff)):
        raise SystemExit(f"{case_id}: recomputed max_abs_diff {diff} != harness {harness_diff}")
    e = {"case_id": case_id, "quantity": quantity, "max_abs_diff": diff, "tolerance": tol,
         "tolerance_rule": rule, "n_values": len(rv), "diff_source": "recomputed from raw R and Julia values"}
    if len(rv) <= MAX_INLINE:
        e["r_value"], e["julia_value"] = rv, jv
    return e


def harness_entry(case_id, quantity, diff, tol, rule):
    return {"case_id": case_id, "quantity": quantity, "max_abs_diff": diff, "tolerance": tol,
            "tolerance_rule": rule,
            "diff_source": "harness-reported (the Julia results file records the max |R - Julia| it bounded, "
                           "not the Julia vector)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--runtimes", type=Path, required=True)
    args = ap.parse_args()
    runs = args.runs
    runtimes = load(args.runtimes)
    head = git_head()

    common = {"pin": "P1", "reference_commit": P1_SHA, "p0_reference_commit": P0_SHA, "gllvmtmb_version": "0.7.1",
              "oracle_build_receipt": ORACLE_BUILD, "oracle_source_receipt": ORACLE_SOURCE,
              "glvmodels_commit_at_receipt_write": head,
              "glvmodels_src_tree": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD:src"],
                                                   capture_output=True, text=True).stdout.strip(),
              "host": "local Mac (M1 Ultra), OPENBLAS/OMP threads 1, JULIA_NUM_THREADS=4"}
    receipts = {}  # case_id -> (path, kind, verdict)
    artifacts = {}

    def emit(cid, kind, verdict, body, comparison=None):
        rec = {"schema": "core070-postfit-p1-case-receipt/v1", "case_id": cid, "verdict": verdict,
               "evidence_kind": kind, **body, **common}
        if comparison is not None:
            rec["comparison"] = {"pin": "P1", "cases": comparison}
        path = REC / "cases" / f"{cid}.json"
        write_json(path, rec)
        receipts[cid] = (str(path.relative_to(ROOT)), kind, verdict)

    # ---- point-style batches: surface-conversion, estimand-rebind, wave6, wave7, wave8 ----
    point_batches = [
        ("surface-conversion-p1", "tools/core070_surface_conversion_batch.R + .jl, GLLVM_PARITY_PIN=P1",
         OUT / "surface-conversion-batch-contract-p1.json"),
        ("estimand-rebind-p1", "tools/core070_estimand_rebind_batch.R + .jl, GLLVM_PARITY_PIN=P1 (no contract file)", None),
        ("wave6-conversion-p1", "tools/core070_wave6_conversion_batch.R + .jl, GLLVM_PARITY_PIN=P1 (contract from PR #567)",
         OUT / "wave6-conversion-batch-contract-p1.json"),
        ("wave7-conversion-p1", "tools/core070_wave7_conversion_batch.R + .jl, GLLVM_PARITY_PIN=P1",
         OUT / "wave7-conversion-batch-contract-p1.json"),
        ("wave8-conversion-p1", "tools/core070_wave8_conversion_batch.R + .jl, GLLVM_PARITY_PIN=P1",
         OUT / "wave8-conversion-batch-contract-p1.json"),
    ]
    for d, batch, contract_path in point_batches:
        rd = runs / d
        artifacts[d] = copy_batch(rd, d, ["receipt.json", "results.tsv", "julia-results.json", "r-oracle.json"])
        julia, oracle, breceipt = load(rd / "julia-results.json"), load(rd / "r-oracle.json"), load(rd / "receipt.json")
        ccases = {c["case_id"]: c for c in load(contract_path)["cases"]} if contract_path else {}
        vlines = verifier_lines(rd)
        for cid, jc in julia["cases"].items():
            cc = ccases.get(cid, {})
            srcs = cc.get("source_ids") or ([cc["source_id"]] if cc.get("source_id") else [])
            if d == "estimand-rebind-p1":
                srcs = [ESTIMAND_SOURCE[cid]]
            if not any(s.startswith("postfit") for s in srcs):
                continue  # namespace / inference / covariance cases in the same batch are out of scope
            kind_h = jc.get("kind") or cc.get("kind") or "point"
            passed = bool(jc.get("pass"))
            body = {"source_ids": srcs, "batch": batch, "harness_kind": kind_h, "harness_pass": passed,
                    "batch_status": breceipt["status"], "verifier_output": vlines,
                    "batch_status_note": (WAVE6_NOTE if d == "wave6-conversion-p1" and breceipt["status"] != "PASS" else ""),
                    "r_call": cc.get("r_call"), "julia_call": cc.get("julia_call"),
                    "raw": [f"{REC_REL}/{d}/julia-results.json", f"{REC_REL}/{d}/r-oracle.json"]}
            if kind_h in ("point", "ci") and "max_abs_diff" in jc:
                tol = jc["tolerance"]
                rule = (f"{contract_path.name if contract_path else 'tools/core070_verify_estimand_rebind_batch.py TOLERANCE'}"
                        f" per-case tolerance (carried verbatim from P0); max |R - Julia| elementwise")
                rv = oracle["oracle_values"].get(cid)
                jv = jc.get("julia_values")
                entry = (vec_entry(cid, jc.get("quantity") or cc.get("quantity"), rv, jv, tol, jc["max_abs_diff"], rule)
                         if isinstance(rv, (list, float, int)) and jv is not None
                         else harness_entry(cid, jc.get("quantity") or cc.get("quantity"), jc["max_abs_diff"], tol, rule))
                ok = passed and entry["max_abs_diff"] <= tol
                emit(cid, "numeric_r_vs_julia", "PASS" if ok else "FAIL", body, [entry])
            elif kind_h == "own_receipt_defect":
                body.update(measured={k: jc.get(k) for k in ("r_nobs", "julia_nobs", "r_expected_p_times_n", "julia_expected_n")},
                            frozen_expectation={"r": cc.get("expected_r_value_formula"), "julia": cc.get("expected_julia_value_formula")},
                            why_failing=("The frozen contract case asserts each engine against its own formula: R == p*n and "
                                         "Julia == n (a known defect pending decision). At P1 R returns p*n = 400 as at P0, but "
                                         "Julia now also returns 400, so the Julia-side expectation (n = 80) no longer holds and "
                                         "the harness reports FAIL. R and Julia agree; the expectation, not the parity, is what "
                                         "failed. Not edited here; recorded as failing."))
                emit(cid, "own_receipt_defect_expectation", "FAIL" if not passed else "PASS", body)
            else:
                verdict_fields = {k: jc.get(k) for k in ("r_verdict", "julia_verdict", "r_frobenius", "julia_frobenius", "tolerance") if k in jc}
                body.update(measured=verdict_fields,
                            why_not_numeric={"verdict": "Both engines report a boolean verdict and the harness checks they agree; "
                                                        "there is no R-vs-Julia number with a tolerance.",
                                             "own_consistency": "Each engine is checked against itself (Frobenius self-consistency); "
                                                                "the two are not compared with each other."}.get(kind_h, f"harness kind {kind_h}"))
                emit(cid, f"paired_{kind_h}", "PASS" if passed else "FAIL", body)

    # ---- postfit-1 (coef readback) ----
    r1, j1 = runs / "postfit-1-r-p1", runs / "postfit-1-julia-p1"
    artifacts["postfit-1-p1"] = copy_batch(r1, "postfit-1-p1", ["receipt.json", "postfit-1-r-results.json"]) + \
        copy_batch(j1, "postfit-1-p1", ["postfit-1-julia-results.json"])
    rr, jr = load(r1 / "postfit-1-r-results.json"), load(j1 / "postfit-1-julia-results.json")
    c1 = load(OUT / "postfit-1-batch-contract-p1.json")["executable_batch"]
    tol = c1["tolerance"]["max_abs_diff"]
    cid = c1["case_id"]
    entry = vec_entry(cid, "coef", rr["coef"], jr["coef"], tol, None,
                      "postfit-1-batch-contract-p1.json executable_batch.tolerance.max_abs_diff (carried verbatim from P0)")
    emit(cid, "numeric_r_vs_julia", "PASS" if entry["max_abs_diff"] <= tol and rr["all_checks"] and jr["all_checks"] else "FAIL",
         {"source_ids": [c1["source_id"]], "batch": "tools/core070_postfit_1_batch.R then .jl, GLLVM_PARITY_PIN=P1",
          "verifier_output": verifier_lines(j1),
          "raw": [f"{REC_REL}/postfit-1-p1/postfit-1-r-results.json", f"{REC_REL}/postfit-1-p1/postfit-1-julia-results.json"]},
         [entry])

    # ---- postfit-policy ----
    rp = runs / "postfit-policy-p1"
    artifacts["postfit-policy-p1"] = copy_batch(rp, "postfit-policy-p1", ["receipt.json", "results.tsv", "julia-results.json", "r-oracle.json"])
    pj, po = load(rp / "julia-results.json"), load(rp / "r-oracle.json")
    pc = load(OUT / "postfit-policy-batch-contract-p1.json")
    ptol = pc["tolerances"]
    vlines = verifier_lines(rp)
    for c in pc["cases"]:
        cid = c["case_id"]
        jc = pj["cases"][cid]
        passed = bool(jc["pass"])
        body = {"source_ids": [c["source_id"]], "batch": "tools/core070_postfit_policy_batch.R + .jl, GLLVM_PARITY_PIN=P1",
                "harness_pass": passed, "verifier_output": vlines, "r_call": c.get("r_call"),
                "julia_surface": c.get("julia_surface"), "comparand": c.get("comparand"), "check": c.get("check"),
                "harness_fields": jc,
                "raw": [f"{REC_REL}/postfit-policy-p1/julia-results.json", f"{REC_REL}/postfit-policy-p1/r-oracle.json"]}
        if cid in POLICY_TOL_KEY:
            key = POLICY_TOL_KEY[cid]
            e = harness_entry(cid, c.get("comparand"), jc["delta"], ptol[key],
                              f"postfit-policy-batch-contract-p1.json tolerances.{key} (carried verbatim from P0)")
            emit(cid, "numeric_r_vs_julia", "PASS" if passed and jc["delta"] <= ptol[key] else "FAIL", body, [e])
        else:
            if cid in POLICY_PARTIAL_WITH_NUMERIC_LEG:
                why = ("The link-scale value leg is numeric (delta recorded), but the fact this case pays is a documented "
                       "default divergence (R predict type default 'link', Julia :response). A default that differs is not "
                       "numeric parity, so the case carries no comparison block.")
            elif "integer" in (c.get("comparand") or ""):
                why = ("Exact integer equality (contract tolerances.integer_exact = 0). R and Julia values are recorded; the "
                       "numeric-tier rule needs a tolerance > 0, and none is invented here.")
            elif "length" in (c.get("comparand") or ""):
                why = "Both sides return an empty coefficient vector; there is no number to compare."
            else:
                why = "Signature-level documented-divergence check (default and keyword reflection), not a numeric comparison."
            body["why_not_numeric"] = why
            emit(cid, "paired_policy_check", "PASS" if passed else "FAIL", body)

    # ---- case-map rows ----
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    carried_keys = ("uncertain", "reclassify_proposed", "original_classification")
    out_rows = []
    counts = {"numeric_pass": 0, "numeric_fail": 0, "partial_non_numeric_case": 0,
              "needs_surface_not_executed": 0, "retired_at_p1_not_measured": 0, "not_measured": 0}
    in_scope = [(f"postfit/POSTFIT-SURFACE-{s}", "DANGLING") for s in POSTFIT_DANGLING] + \
               [(f"postfit/POSTFIT-SURFACE-{s}", "RETIRED") for s in POSTFIT_RETIRED] + \
               [(f"postfit-policy/{s}", "DANGLING") for s in POLICY_DANGLING]
    for sid, carry in in_scope:
        base = p0[sid]
        ids = base["executable_case_ids"]
        row = {"source_id": sid, "classification": base["classification"], "arc": "A3", "carry_scan_status": carry,
               "executable_case_ids": ids, "disposition": base.get("disposition")}
        for k in carried_keys:
            if k in base:
                row[k] = base[k]
        have = [receipts.get(i) for i in ids]
        if carry == "RETIRED":
            row.update(evidence_tier="not_measured", measured_against=None, evidence={},
                       reason=("The R export was removed between P0 and P1 (NAMESPACE at P1 no longer exports it); the row has "
                               "no executable case ids and keeps its P0 classification and disposition. Retiring or "
                               "re-scoping the row is a maintainer decision."))
            counts["retired_at_p1_not_measured"] += 1
        elif not ids or any(h is None for h in have):
            missing = [i for i, h in zip(ids, have) if h is None]
            if sid == "postfit-policy/POST-DEVIANCE":
                row.update(evidence_tier="needs_surface_not_executed", measured_against=None, evidence={},
                           reason=("Its executable case CORE070-POSTFIT-DEVIANCE-NATIVE was never authored into the "
                                   "postfit-policy batch (the P0 contract lists POST-DEVIANCE under needs_new_julia_surface), "
                                   "so nothing ran at P1. deviance.gllvmTMB_multi is measured numerically at P1 under a "
                                   "different case id (CORE070-WAVE8-DEVIANCE-MULTI, row postfit/POSTFIT-SURFACE-"
                                   "deviance.gllvmTMB_multi); rebinding is a maintainer decision."))
                counts["needs_surface_not_executed"] += 1
            else:
                row.update(evidence_tier="not_measured", measured_against=None, evidence={},
                           reason=f"Not re-measured at P1 in this PR (missing: {', '.join(missing)}).")
                counts["not_measured"] += 1
        else:
            kinds = {h[1] for h in have}
            verdicts = {i: h[2] for i, h in zip(ids, have)}
            paths = [h[0] for h in have]
            all_pass = all(v == "PASS" for v in verdicts.values())
            if kinds == {"numeric_r_vs_julia"} and all_pass:
                row.update(evidence_tier="numeric", measured_against=P1_SHA,
                           evidence={"receipt": paths,
                                     "tier": "numeric: every executable case receipt carries an R-vs-Julia comparison "
                                             "block pinned to P1, within the harness tolerance"},
                           measured_result={"case_verdicts": verdicts, "row_verdict": "PASS"})
                counts["numeric_pass"] += 1
            elif not all_pass:
                row.update(evidence_tier="numeric_fail", measured_against=P1_SHA,
                           evidence={"non_binding_receipts": paths,
                                     "tier": "measured at P1; the harness verdict is FAIL, so the row does not bind"},
                           measured_result={"case_verdicts": verdicts, "row_verdict": "FAIL"})
                counts["numeric_fail"] += 1
            else:
                row.update(evidence_tier="partial_non_numeric_case", measured_against=P1_SHA,
                           evidence={"non_binding_receipts": paths,
                                     "tier": "measured at P1 and the harness passes, but at least one case is a verdict, "
                                             "own-consistency, exact-integer, empty-length or default-policy check with "
                                             "no R-vs-Julia number and tolerance, so the row does not bind"},
                           measured_result={"case_verdicts": verdicts, "case_kinds": {i: h[1] for i, h in zip(ids, have)}})
                counts["partial_non_numeric_case"] += 1
        out_rows.append(row)

    casemap = {
        "schema": 1, "reference_commit": P1_SHA,
        "scope": ("postfit family: the 36 rows the P1 carry scan lists as DANGLING (34) or RETIRED (2); postfit-policy "
                  "family: the 16 rows it lists as DANGLING. NOT_BOUND_AT_P0 rows (64 postfit, 5 postfit-policy) had no "
                  "P0 evidence and are out of scope here."),
        "note": ("Separate from case-map.json so none of its rows are touched; read by tools/true_parity_check.mjs with "
                 "PARITY_CASEMAP pointing at this file. Classification, disposition and the P0 `uncertain` / "
                 "`reclassify_proposed` fields are carried from docs/dev-log/core070/required-source-case-map.json "
                 "unchanged; nothing is signed by an agent. Only rows whose every executable case id carries a numeric "
                 "R-vs-Julia comparison block within tolerance cite evidence.receipt; every other measured row cites "
                 "evidence.non_binding_receipts, so it reads as not bound under both the current checker and the "
                 "numeric-tier rule proposed in PR #561."),
        "generator": "tools/core070_postfit_p1_receipts.py",
        "counts": counts, "runtimes_seconds": runtimes,
        "batch_artifacts": artifacts,
        "rows": out_rows,
    }
    write_json(OUT / "case-map-postfit.json", casemap)
    print(json.dumps(counts))
    print("case receipts", len(receipts), "artifacts", sum(len(v) for v in artifacts.values()))


if __name__ == "__main__":
    main()

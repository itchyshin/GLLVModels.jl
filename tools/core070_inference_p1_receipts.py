"""Write tracked P1 receipts and case-map rows for the inference family.

The 63 required inference rows the P1 carry scan lists as DANGLING are paid by
three batches, run at gllvmTMB pin P1 (GLLVM_PARITY_PIN=P1):

  inference-batch (wave2, 45 rows)   R: frozen-source route probe, re-run live on the P1
                                     source tree (tools/core070_inference_routes_p1.R).
                                     Julia: which CI solver ran (route tag). No fit on
                                     the R side, no interval compared: ROUTING ONLY.
  inference-remainder (wave4, 14)    R: installed P1 oracle, confint(method = <bad>) must
                                     raise with a named message. Julia: MethodError.
                                     ERROR-CLASS ONLY, no number.
  surface-conversion (wave5, 4)      CI-ROUTE-008..011, two-level ICC. Already run at P1 by
                                     PR #569 (receipts/postfit/surface-conversion-p1); this
                                     tool reads those tracked raw files, it does not re-run.
                                     008 and 010 compare R and Julia CI bounds with a
                                     tolerance; 009 is a refusal pair, 011 a structural
                                     bootstrap check.

Writes, under docs/dev-log/core070/true-parity-latest/:

  receipts/inference/<batch>-p1/...    batch artifacts copied verbatim (JSON/TSV only, no logs)
  receipts/inference/cases/<id>.json   one receipt per executable case id
  case-map-inference.json              the 63 rows, P0 classification and disposition kept

A case gets a `comparison` block only when its harness compares an R number
with a Julia number against a declared tolerance > 0; max_abs_diff is
recomputed here from the saved R and Julia vectors and must agree with the
harness figure. A row is evidence_tier "numeric" only when every executable
case id has such a block and passes. Routing and error-class rows cite
evidence.non_binding_receipts and stay free.

Usage:
  python3 tools/core070_inference_p1_receipts.py --runs DIR --runtimes JSON
where DIR holds inference-p1/{julia,r-crosscheck,verify.txt} and
inference-remainder-p1/ (+ inference-remainder-p1/verify.txt).
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/dev-log/core070/true-parity-latest"
REC = OUT / "receipts/inference"
REC_REL = "docs/dev-log/core070/true-parity-latest/receipts/inference"
SURF_REL = "docs/dev-log/core070/true-parity-latest/receipts/postfit/surface-conversion-p1"
P0_CASEMAP = ROOT / "docs/dev-log/core070/required-source-case-map.json"
PINS = tomllib.loads((ROOT / "tools/core070_oracle_pins.toml").read_text())
P1_SHA = PINS["P1"]["reference_commit"]
P0_SHA = PINS["P0"]["reference_commit"]
ORACLE_BUILD = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build.json"
ORACLE_SOURCE = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/source.json"

TIER_TEXT = {
    "routing_control_flow": ("routing / control-flow only: R side is a frozen-source probe of which internal "
                             "function+method confint.gllvmTMB_multi dispatches to (no fit, no interval); Julia "
                             "side records which CI solver ran. No R-vs-Julia number, so the row does not bind"),
    "reject_error_class": ("error-class only: R (installed P1 oracle) must raise a named 'not supported / not "
                           "implemented' error and Julia must raise MethodError. No number, so the row does not bind"),
    "partial_non_numeric_case": ("measured at P1 and the harness passes, but the case is a refusal pair or a "
                                 "structural bootstrap check with no R-vs-Julia number and tolerance, so the row "
                                 "does not bind"),
}


def load(p):
    return json.loads(Path(p).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git(*argv):
    return subprocess.run(["git", "-C", str(ROOT), *argv], capture_output=True, text=True).stdout.strip()


def verifier_lines(path):
    return [l for l in Path(path).read_text().splitlines() if l.strip()][-3:] if Path(path).exists() else []


def copy(src_dir, dest_name, names):
    dest = REC / dest_name
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for n in names:
        p = src_dir / n
        if not p.is_file():
            raise SystemExit(f"missing batch artifact {p}")
        shutil.copyfile(p, dest / Path(n).name)
        out.append(f"{REC_REL}/{dest_name}/{Path(n).name}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--runtimes", type=Path, required=True)
    args = ap.parse_args()
    runs, runtimes = args.runs, load(args.runtimes)
    common = {"pin": "P1", "reference_commit": P1_SHA, "p0_reference_commit": P0_SHA,
              "oracle_build_receipt": ORACLE_BUILD, "oracle_source_receipt": ORACLE_SOURCE,
              "glvmodels_commit_at_receipt_write": git("rev-parse", "HEAD"),
              "glvmodels_src_tree": git("rev-parse", "HEAD:src"),
              "host": "local Mac (M1 Ultra), OPENBLAS/OMP threads 1, JULIA_NUM_THREADS=4"}
    receipts, artifacts = {}, {}

    def emit(cid, kind, verdict, body, comparison=None):
        rec = {"schema": "core070-inference-p1-case-receipt/v1", "case_id": cid, "verdict": verdict,
               "evidence_kind": kind, **body, **common}
        if comparison is not None:
            rec["comparison"] = {"pin": "P1", "cases": comparison}
        path = REC / "cases" / f"{cid}.json"
        write_json(path, rec)
        receipts[cid] = (str(path.relative_to(ROOT)), kind, verdict)

    # ---- wave2 inference-batch: routing ----
    ib = runs / "inference-p1"
    artifacts["inference-batch-p1"] = (
        copy(ib / "julia", "inference-batch-p1", ["receipt.json", "inference-batch-results.json", "raw.tsv"])
        + copy(ib / "r-crosscheck", "inference-batch-p1/r-crosscheck",
               ["receipt.json", "inference-batch-r-crosscheck.json", "r-comparand-crosscheck.tsv",
                "p1-route-probe-results.tsv"]))
    ic = load(OUT / "inference-batch-contract-p1.json")
    jres = {c["source_id"]: c for c in load(ib / "julia/inference-batch-results.json")["cases"]}
    with open(ib / "r-crosscheck/p1-route-probe-results.tsv") as fh:
        rprobe = {r["id"]: r for r in csv.DictReader(fh, delimiter="\t", escapechar="\\", doublequote=False)}
    vlines = verifier_lines(ib / "verify.txt")
    by_case = {}
    for row in ic["rows"]:
        if row["bucket"] == "EXECUTABLE_NOW":
            by_case.setdefault(row["case_id"], []).append(row)
    for cid, rows in sorted(by_case.items()):
        per_row, ok = [], True
        for row in rows:
            sid = row["source_id"].split("/")[-1]
            j, r = jres[sid], rprobe[sid]
            row_ok = bool(j["ok"]) and j["actual"] == row["expected_route_tag"] and r["pass"] == "TRUE"
            ok = ok and row_ok
            per_row.append({"source_id": row["source_id"], "r_route_actual": r["actual"], "r_route_pass": r["pass"] == "TRUE",
                            "julia_call": row["julia_call"], "julia_route_expected": row["expected_route_tag"],
                            "julia_route_actual": j["actual"], "julia_pass": bool(j["ok"])})
        emit(cid, "routing_control_flow", "PASS" if ok else "FAIL",
             {"source_ids": [r["source_id"] for r in rows],
              "batch": "tools/core070_inference_batch.R (P1 route probe) + tools/core070_inference_batch.jl, GLLVM_PARITY_PIN=P1",
              "verifier_output": vlines, "per_row": per_row,
              "why_not_numeric": ("Routing probe. The R side evaluates only function definitions from the P1 source "
                                  "files and intercepts every CI endpoint, so no model is fit and no interval is "
                                  "computed; the Julia side reports which solver ran. There is no R-vs-Julia number."),
              "raw": [f"{REC_REL}/inference-batch-p1/inference-batch-results.json",
                      f"{REC_REL}/inference-batch-p1/r-crosscheck/p1-route-probe-results.tsv"]})

    # ---- wave4 inference-remainder: error class ----
    rb = runs / "inference-remainder-p1"
    artifacts["inference-remainder-p1"] = copy(rb, "inference-remainder-p1",
                                               ["receipt.json", "results.tsv", "julia-results.json", "r-oracle.json"])
    rc = load(OUT / "inference-remainder-batch-contract-p1.json")
    rj, ro = load(rb / "julia-results.json"), load(rb / "r-oracle.json")
    oracle_key = {"ICC": "icc_reject", "PHYLO-SIGNAL": "phylo_reject", "COMMUNALITY": "communality_reject",
                  "RHO": "rho_reject", "PROPORTION": "proportion_reject"}
    rvl = verifier_lines(rb / "verify.txt")
    for c in rc["cases"]:
        cid = c["case_id"]
        key = oracle_key[cid.split("CORE070-INFERENCE-")[1].split("-CI-")[0]]
        jc = rj["cases"][cid]
        measured = {m: {"r_raised": ro[key][m]["raised"], "r_message_matches": ro[key][m]["matches"],
                        "r_message": ro[key][m]["message"], "julia_error_kind": jc["methods"][m]["julia_error_kind"],
                        "pass": jc["methods"][m]["pass"]} for m in jc["methods"]}
        emit(cid, "reject_error_class", "PASS" if jc["pass"] else "FAIL",
             {"source_ids": c["source_ids"],
              "batch": "tools/core070_inference_remainder_batch.R + .jl, GLLVM_PARITY_PIN=P1",
              "verifier_output": rvl, "r_call": c["r_call"], "julia_surface": c["julia_surface"], "check": c["check"],
              "measured": measured, "gllvmtmb_version": load(rb / "receipt.json")["gllvmTMB_version"],
              "why_not_numeric": "Both engines must refuse the method; the check is error class and message, not a number.",
              "raw": [f"{REC_REL}/inference-remainder-p1/julia-results.json",
                      f"{REC_REL}/inference-remainder-p1/r-oracle.json"]})

    # ---- wave5 surface-conversion ICC rows (tracked P1 run from PR #569, not re-run) ----
    sd = ROOT / SURF_REL
    sj, so, sr = load(sd / "julia-results.json"), load(sd / "r-oracle.json"), load(sd / "receipt.json")
    scases = {c["case_id"]: c for c in load(OUT / "surface-conversion-batch-contract-p1.json")["cases"]}
    if sr["reference_commit"] != P1_SHA:
        raise SystemExit("surface-conversion receipt is not pinned at P1")
    for cid, cc in scases.items():
        if not cc["source_id"].startswith("inference/"):
            continue
        jc = sj["cases"][cid]
        body = {"source_ids": [cc["source_id"]],
                "batch": "tools/core070_surface_conversion_batch.R + .jl, GLLVM_PARITY_PIN=P1 (run by PR #569; read, not re-run)",
                "batch_status": sr["status"], "harness_kind": cc["kind"], "harness_pass": bool(jc["pass"]),
                "r_call": cc["r_call"], "julia_call": cc["julia_call"], "gllvmtmb_version": "0.7.1",
                "raw": [f"{SURF_REL}/julia-results.json", f"{SURF_REL}/r-oracle.json"]}
        if cc["kind"] == "ci":
            rv, jv, tol = so["oracle_values"][cid], jc["julia_values"], cc["tolerance"]
            if len(rv) != len(jv):
                raise SystemExit(f"{cid}: length mismatch")
            diff = max(abs(a - b) for a, b in zip(rv, jv))
            if abs(diff - jc["max_abs_diff"]) > 1e-12 * max(1.0, jc["max_abs_diff"]):
                raise SystemExit(f"{cid}: recomputed {diff} != harness {jc['max_abs_diff']}")
            entry = {"case_id": cid, "quantity": cc["quantity"], "max_abs_diff": diff, "tolerance": tol,
                     "tolerance_rule": "surface-conversion-batch-contract-p1.json per-case tolerance (carried verbatim from P0); max |R - Julia| over CI bounds",
                     "n_values": len(rv), "diff_source": "recomputed from raw R and Julia values",
                     "r_value": rv, "julia_value": jv}
            emit(cid, "numeric_r_vs_julia", "PASS" if jc["pass"] and diff <= tol else "FAIL", body, [entry])
        else:
            if cc["kind"] == "refusal_pair":
                body["measured"] = {k: jc.get(k) for k in ("r_raised", "julia_raised", "julia_message")}
                body["why_not_numeric"] = "Both engines must refuse method = profile; there is no number to compare."
            else:
                body["measured"] = {"r_structural": jc["r_structural"], "julia_structural": jc["julia_structural"]}
                pt = max(abs(a - b) for a, b in zip(jc["r_structural"]["point"], jc["julia_structural"]["point"]))
                body["why_not_numeric"] = (
                    "Structural bootstrap check (finite, ordered, brackets the point), per the contract's "
                    "structural_justification: two independent stochastic bootstraps, so endpoints are not compared. "
                    f"The point legs agree to {pt:.3g} but the contract declares no tolerance for them, and none is invented.")
                body["point_leg_max_abs_diff_unbound"] = pt
            emit(cid, f"paired_{cc['kind']}", "PASS" if jc["pass"] else "FAIL", body)

    # ---- route-probe adaptation record: P0 probe on P0 source, P0 probe on P1 source, P1 probe on P1 source ----
    def probe(name):
        with open(runs / name) as fh:
            return list(csv.DictReader(fh, delimiter="\t", escapechar="\\", doublequote=False))
    p0r, p1u, p1a = probe("routes-p0.tsv"), probe("routes-p1-unadapted.tsv"), probe("routes-p1-adapted.tsv")
    cols = ("id", "pass", "actual_class", "actual", "messages")
    write_json(REC / "route-probe-adaptation.json", {
        "schema": "core070-inference-route-probe-adaptation/v1",
        "fixture": "test/parity/fixtures/core070_inference_routes.tsv",
        "runs": {
            "p0_probe_on_p0_source": {"script": "tools/core070_inference_routes.R", "source": f"git show {P0_SHA}:R/<file>",
                                      "pass": sum(r["pass"] == "TRUE" for r in p0r), "rows": len(p0r)},
            "p0_probe_on_p1_source": {"script": "tools/core070_inference_routes.R", "source": "CORE070_P1_R_SOURCE_ROOT",
                                      "pass": sum(r["pass"] == "TRUE" for r in p1u), "rows": len(p1u),
                                      "distinct_errors": sorted({r["actual"] for r in p1u if r["pass"] != "TRUE"})},
            "p1_probe_on_p1_source": {"script": "tools/core070_inference_routes_p1.R", "source": "CORE070_P1_R_SOURCE_ROOT",
                                      "pass": sum(r["pass"] == "TRUE" for r in p1a), "rows": len(p1a)},
        },
        "p0_and_adapted_p1_outputs_identical": [tuple(r[c] for c in cols) for r in p0r] ==
                                              [tuple(r[c] for c in cols) for r in p1a],
        "note": ("The only harness change at P1 is parsing R/temporal.R as well, because P1's confint.gllvmTMB_multi "
                 "calls .temporal_assert_no_iid_inference() first. With it, every route string, error class and "
                 "message is identical to the P0 probe on P0 source."),
    })

    # ---- case-map rows ----
    carry = load(runs / "carry-scan-p1.json")
    in_scope = [r["source_id"] for r in carry["rows"]
                if r["source_id"].startswith("inference/") and r["status"] == "DANGLING"]
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    counts = {"numeric_pass": 0, "numeric_fail": 0, "partial_non_numeric_case": 0, "routing_control_flow": 0,
              "reject_error_class": 0, "needs_surface_not_executed": 0, "retired_at_p1_not_measured": 0,
              "not_measured": 0}
    out_rows = []
    for sid in in_scope:
        base = p0[sid]
        ids = base["executable_case_ids"]
        row = {"source_id": sid, "classification": base["classification"], "arc": "A3",
               "carry_scan_status": "DANGLING", "executable_case_ids": ids,
               "disposition": base.get("disposition"), "p0_batch": (base.get("evidence") or {}).get("batch")}
        have = [receipts.get(i) for i in ids]
        if not ids or any(h is None for h in have):
            row.update(evidence_tier="not_measured", measured_against=None, evidence={},
                       reason="Not re-measured at P1 in this PR.")
            counts["not_measured"] += 1
            out_rows.append(row)
            continue
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
            row.update(evidence_tier="numeric_fail" if "numeric_r_vs_julia" in kinds else "measured_fail",
                       measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "measured at P1; the harness verdict is FAIL, so the row does not bind"},
                       measured_result={"case_verdicts": verdicts, "row_verdict": "FAIL"})
            counts["numeric_fail"] += 1
        else:
            tier = kinds.pop() if len(kinds) == 1 and next(iter(kinds)) in ("routing_control_flow", "reject_error_class") \
                else "partial_non_numeric_case"
            row.update(evidence_tier=tier, measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths, "tier": TIER_TEXT[tier]},
                       measured_result={"case_verdicts": verdicts, "case_kinds": {i: h[1] for i, h in zip(ids, have)}})
            counts[tier] += 1
        out_rows.append(row)

    casemap = {
        "schema": 1, "reference_commit": P1_SHA,
        "scope": ("inference family: the 63 required rows the P1 carry scan lists as DANGLING (45 wave2 "
                  "inference-batch, 14 wave4 inference-remainder, 4 wave5 surface-conversion). The one "
                  "NOT_BOUND_AT_P0 row (CI-ROUTE-005, BLOCKED_NEEDS_JULIA_SURFACE) and the 34 out-of-scope "
                  "rejected/excluded rows are not included."),
        "note": ("Separate from case-map.json so none of its rows are touched; read by tools/true_parity_check.mjs with "
                 "PARITY_CASEMAP pointing at this file. Classification and disposition are carried from "
                 "docs/dev-log/core070/required-source-case-map.json unchanged (all 63 are compatibility_adapter); "
                 "nothing is signed by an agent. Only rows whose every executable case id carries a numeric R-vs-Julia "
                 "comparison block within tolerance cite evidence.receipt. Routing (wave2) and error-class (wave4) rows "
                 "carry no number and cite evidence.non_binding_receipts."),
        "generator": "tools/core070_inference_p1_receipts.py",
        "counts": counts, "runtimes_seconds": runtimes,
        "batch_artifacts": {**artifacts, "surface-conversion-p1 (from PR #569, read only)":
                            [f"{SURF_REL}/receipt.json", f"{SURF_REL}/julia-results.json", f"{SURF_REL}/r-oracle.json"]},
        "rows": out_rows,
    }
    write_json(OUT / "case-map-inference.json", casemap)
    print(json.dumps(counts))
    print("rows", len(out_rows), "case receipts", len(receipts))


if __name__ == "__main__":
    main()

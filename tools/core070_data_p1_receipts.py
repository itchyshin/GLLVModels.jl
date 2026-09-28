"""Write tracked P1 receipts and case-map rows for the data and fit-input families.

In scope: the 28 required data rows and the 6 required fit-input rows the P1
carry scan lists as DANGLING. Two batches pay them, run at gllvmTMB pin P1
(GLLVM_PARITY_PIN=P1):

  data (wave1, 28 rows)          R: replays the pinned P1 R helpers (normalise_weights,
                                 gll_prepare_offset, .gllvmTMB_offset_vec,
                                 .gllvmTMB_offset_newdata, miss_control) from the P1 source
                                 tree; each case is an identical() expression that must be
                                 TRUE (FALSE for the 2 negative controls). No fit.
                                 Julia: runtime introspection that none of the planned Julia
                                 surfaces exists (SPEC_DEFECT). No R-vs-Julia number, so no
                                 data row can bind: R SIDE ONLY, JULIA SURFACE ABSENT.
  fit-input-2 (wave4, 6 rows)    R: installed P1 oracle fits seven small fixtures. Julia: native
                                 and formula-interface refits. Each case compares logLik and
                                 the trait-intercept coefficients with the contract tolerance
                                 (1e-4 each).

Writes, under docs/dev-log/core070/true-parity-latest/:

  receipts/<family>/<batch>-p1/...  batch artifacts copied verbatim (JSON/TSV only, no logs),
                                    run-commit.json, verify.txt
  receipts/<family>/cases/<id>.json one receipt per executable case id
  case-map-data.json                the 28 data rows, P0 classification and disposition kept
  case-map-fit-input.json           the 6 fit-input rows, likewise

A case gets a `comparison` block only when its harness compares an R number
with a Julia number against a declared tolerance > 0; max_abs_diff is
recomputed here from the saved R and Julia values and must agree with the
harness figure. A row is evidence_tier "numeric" only when every executable
case id has a comparison block, the harness verdict is PASS, every batch those
cases came from passed its own verifier, and no comparison is degenerate.

Shared gates (PR #567 / #569 / #571):

  * Batch verifier. This tool runs both batch verifiers at P1 (with --self-test)
    and keeps their full output as the tracked <batch>/verify.txt. Every case
    receipt carries a batch_verifier block. A numeric row whose batch verifier did
    not pass is held back (numeric_held_batch_verifier_failed, non-binding
    receipts). There is no exception path.
  * Degenerate comparison. A comparison whose R values are one constant or all ~0
    is flagged discriminating: false and its row is numeric_non_discriminating
    (tools/core070_postfit_p1_receipts.py's rule and code).
  * Provenance. Every receipt records glvmodels_commit = HEAD. The tool refuses to
    write when tracked files outside its own outputs are modified (unless
    --allow-dirty, which is then recorded), and refuses unless each run directory
    holds a run-commit.json naming this HEAD with an empty dirty list.
  * Read-file hashes. Every case receipt records `read_from`, the sha256 of each
    tracked file it was derived from. Receipt bodies and case-map rows are derived
    only from tracked files, so --check re-hashes every read_from file, re-derives
    every case receipt and every case-map row in memory, and exits nonzero on any
    difference.

Usage:
  python3 tools/core070_data_p1_receipts.py --runs DIR --runtimes JSON [--allow-dirty]
  python3 tools/core070_data_p1_receipts.py --check
where DIR holds data-p1/ (the R run plus data-batch-julia-introspection.json and
run-commit.json), fit-input-2-p1/ (with run-commit.json) and carry-scan-p1.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from core070_postfit_p1_receipts import mark_degenerate  # noqa: E402  (PR #569's degenerate-comparison rule)

OUT = ROOT / "docs/dev-log/core070/true-parity-latest"
OUT_REL = "docs/dev-log/core070/true-parity-latest"
P0_CASEMAP = ROOT / "docs/dev-log/core070/required-source-case-map.json"
PINS = tomllib.loads((ROOT / "tools/core070_oracle_pins.toml").read_text())
P1_SHA = PINS["P1"]["reference_commit"]
P0_SHA = PINS["P0"]["reference_commit"]
ORACLE_BUILD = f"{OUT_REL}/receipts/covariance/oracle/build.json"
ORACLE_SOURCE = f"{OUT_REL}/receipts/covariance/oracle/source.json"
HOST = "local Mac (M1 Ultra), OPENBLAS/OMP threads 1, JULIA_NUM_THREADS=4"

FAMILIES = {"data": "data", "fit-input": "fit-input"}
BATCHES = {
    # batch dir -> (family, verifier argv template, accept marker, artifacts copied)
    "data-p1": ("data",
                ["tools/core070_verify_data_batch.py", "--state", "{state}", "--julia-receipt",
                 "{state}/data-batch-julia-introspection.json", "--self-test"],
                "CORE070_DATA_BATCH_VERIFIED",
                ["receipt.json", "data-batch-results.json", "raw.tsv", "data-batch-julia-introspection.json",
                 "run-commit.json"]),
    "fit-input-2-p1": ("fit-input",
                       ["tools/core070_verify_fit_input_2_batch.py", "--results", "{state}", "--self-test"],
                       "CORE070_FIT_INPUT_2_BATCH_VERIFIED",
                       ["receipt.json", "results.tsv", "julia-results.json", "r-oracle.json", "run-commit.json"]),
}
CONTRACT = {"data-p1": f"{OUT_REL}/data-batch-contract-p1.json",
            "fit-input-2-p1": f"{OUT_REL}/fit-input-2-batch-contract-p1.json"}

# fit-input-2: case id -> key in r-oracle.json (the fixture the R side fit).
FIT_INPUT_ORACLE_KEY = {
    "GAUSS-DEFAULT": "gauss_default", "GAUSS-LOADINGS": "gauss_loadings", "BINOMIAL-DEFAULT": "binomial_default",
    "ANIMAL-LATENT": "animal_latent", "KERNEL-ONE": "kernel_one", "KERNEL-TWO": "kernel_two",
    "KERNEL-TWO-AUTO": "kernel_two_auto",
}
SAME_MEASUREMENT_NOTE = (
    "One measurement counted twice: KERNEL-TWO-AUTO reuses KERNEL-TWO's data, gllvmTMB drops unique = TRUE in "
    "the two-kernel combination (the contract's critical_finding_kernel_two_auto, re-checked by the R runner at "
    "P1), and the contract maps the Julia side to the same two-source model. The R logLik/coefficients and the "
    "Julia logLik/coefficients of this case equal those of {twin}. Distinct surface rows, one comparison; whether "
    "it counts once or twice is for the maintainer.")

TIER_TEXT = {
    "needs_surface_r_side_measured": (
        "R side measured at P1 (the pinned P1 R helper replays to the frozen expectation), but GLLVModels has no "
        "surface to compare it with (runtime introspection: every planned symbol and keyword absent). No "
        "R-vs-Julia number, so the row does not bind"),
}
COUNT_KEYS = ("numeric_pass", "numeric_fail", "numeric_held_batch_verifier_failed", "numeric_non_discriminating",
              "partial_non_numeric_case", "needs_surface_r_side_measured", "needs_surface_not_executed",
              "retired_at_p1_not_measured", "not_measured")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git(*argv, check=True):
    return subprocess.run(["git", "-C", str(ROOT), *argv], check=check, capture_output=True, text=True)


def rec_rel(family):
    return f"{OUT_REL}/receipts/{family}"


def casemap_rel(family):
    return f"{OUT_REL}/case-map-{family}.json"


def git_state():
    """HEAD and the tracked paths modified outside this tool's own outputs."""
    head = git("rev-parse", "HEAD").stdout.strip()
    own = tuple(rec_rel(f) + "/" for f in FAMILIES) + tuple(casemap_rel(f) for f in FAMILIES)
    dirty = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").stdout.splitlines()
             if not line[3:].startswith(own)]
    return head, dirty


def check_run_commit(run_dir, head):
    p = run_dir / "run-commit.json"
    if not p.is_file():
        raise SystemExit(f"{run_dir} has no run-commit.json; re-run the batch from a clean commit")
    rc = load(p)
    if rc.get("glvmodels_commit") != head or rc.get("dirty") != []:
        raise SystemExit(f"{run_dir}: run at {rc.get('glvmodels_commit')} dirty={rc.get('dirty')}, "
                         f"not at clean HEAD {head}; re-run at HEAD")


def run_verifier(batch, state):
    """Run a batch verifier at P1 on the raw run and keep its full output as the tracked verify.txt."""
    family, argv_t, marker, _ = BATCHES[batch]
    argv = ["python3"] + [a.format(state=str(state)) for a in argv_t]
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, GLLVM_PARITY_PIN="P1"))
    log = ROOT / rec_rel(family) / batch / "verify.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ GLLVM_PARITY_PIN=P1 {' '.join(argv_t)}\n# exit code {proc.returncode}\n"
                   + proc.stdout + proc.stderr)
    return {"exit_code": proc.returncode, "status": "PASS" if proc.returncode == 0 and marker in proc.stdout else "FAIL"}


def verifier_block(batch):
    """The batch_verifier block, derived from the tracked verify.txt (so --check can re-derive it)."""
    family, argv_t, marker, _ = BATCHES[batch]
    rel = f"{rec_rel(family)}/{batch}/verify.txt"
    text = (ROOT / rel).read_text()
    ok = "# exit code 0\n" in text and marker in text
    return {"tool": argv_t[0], "argv": " ".join(argv_t), "status": "PASS" if ok else "FAIL",
            "accept_marker": marker, "log": rel}


def copy_batch(batch, run_dir):
    family, _, _, names = BATCHES[batch]
    dest = ROOT / rec_rel(family) / batch
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        p = run_dir / n
        if not p.is_file():
            raise SystemExit(f"missing batch artifact {p}")
        shutil.copyfile(p, dest / n)
    return [f"{rec_rel(family)}/{batch}/{n}" for n in names]


def read_from(*rels):
    return {rel: sha(ROOT / rel) for rel in rels}


# ---------------------------------------------------------------------------
# Derivation: case receipts from tracked files only.
# ---------------------------------------------------------------------------
def data_cases():
    """case_id -> (evidence_kind, verdict, body, comparison-or-None) for the data batch."""
    d = f"{rec_rel('data')}/data-p1"
    receipt, results = load(ROOT / d / "receipt.json"), load(ROOT / d / "data-batch-results.json")
    intro = load(ROOT / d / "data-batch-julia-introspection.json")
    contract = load(ROOT / CONTRACT["data-p1"])
    if receipt["reference_commit"] != P1_SHA or intro["reference_commit"] != P1_SHA:
        raise SystemExit("data batch receipt or Julia introspection is not pinned at P1")
    if receipt["results_sha256"] != sha(ROOT / d / "data-batch-results.json"):
        raise SystemExit("data-batch-results.json does not match its receipt's results_sha256")
    if receipt["contract_sha256"] != sha(ROOT / CONTRACT["data-p1"]):
        raise SystemExit("data batch receipt does not name the tracked P1 twin's sha256")
    verifier = verifier_block("data-p1")
    reads = read_from(f"{d}/receipt.json", f"{d}/data-batch-results.json", f"{d}/raw.tsv",
                      f"{d}/data-batch-julia-introspection.json", f"{d}/run-commit.json", f"{d}/verify.txt",
                      CONTRACT["data-p1"])
    ccases = {c["manifest_case_id"]: c for c in contract["cases"]}
    out = {}
    for rc in results["cases"]:
        cid = rc["manifest_case_id"]
        cc = ccases[cid]
        group = cc["planned_surface_group"]
        surf = intro["surfaces"][group]
        body = {"source_ids": [cc["source_id"]],
                "batch": "tools/core070_data_batch.R (P1 source tree) + tools/core070_data_batch.jl, GLLVM_PARITY_PIN=P1",
                "batch_status": receipt["status"], "batch_verifier": verifier,
                "gllvmtmb_version": receipt["gllvmTMB_version"],
                "r_expression": cc["expression"], "r_expected": cc["expected"],
                "r_actual": rc["actual"], "r_is_error": rc["is_error"], "r_ok": rc["ok"],
                "negative_control": cc["negative_control"],
                "julia_planned_surface_group": group,
                "julia_surface_introspection": {k: surf[k] for k in ("candidate_exported_symbols", "candidate_kwargs",
                                                                     "found_symbols", "found_kwargs", "surface_absent")},
                "read_from": reads,
                "raw": [f"{d}/data-batch-results.json", f"{d}/data-batch-julia-introspection.json"]}
        if cc["negative_control"]:
            kind = "r_replay_negative_control"
            verdict = "PASS" if rc["ok"] and rc["actual"] is False else "FAIL"
            body["why_not_numeric"] = ("R-side negative control: the expression must evaluate to FALSE, proving the "
                                       "identical() comparison is live. No Julia side and no number.")
        else:
            kind = "r_replay_julia_surface_absent"
            body["julia_surface"] = cc["julia_surface"]
            body["julia_verdict"] = rc["julia_verdict"]
            verdict = "PASS" if rc["ok"] and rc["actual"] is True and surf["surface_absent"] else "FAIL"
            body["why_not_numeric"] = ("The R side replays a pinned P1 helper to an exact identical() expectation; "
                                       "GLLVModels has no surface for it (runtime introspection found none of the "
                                       "planned symbols or keywords), so there is no Julia value to compare.")
            if not surf["surface_absent"]:
                body["note"] = ("A planned Julia surface now exists (see julia_surface_introspection); the "
                                "contract's SPEC_DEFECT verdict is stale at this commit.")
        out[cid] = (kind, verdict, body, None)
    return out


def fit_input_cases():
    """case_id -> (evidence_kind, verdict, body, comparison) for the fit-input-2 batch."""
    d = f"{rec_rel('fit-input')}/fit-input-2-p1"
    receipt, oracle, jres = (load(ROOT / d / n) for n in ("receipt.json", "r-oracle.json", "julia-results.json"))
    contract = load(ROOT / CONTRACT["fit-input-2-p1"])
    if receipt["reference_commit"] != P1_SHA:
        raise SystemExit("fit-input-2 receipt is not pinned at P1")
    if receipt["julia_results_sha256"] != sha(ROOT / d / "julia-results.json"):
        raise SystemExit("fit-input-2 julia-results.json does not match its receipt's julia_results_sha256")
    if receipt["contract_sha256"] != sha(ROOT / CONTRACT["fit-input-2-p1"]):
        raise SystemExit("fit-input-2 receipt does not name the tracked P1 twin's sha256")
    tol_ll, tol_coef = contract["tolerances"]["loglik_delta"], contract["tolerances"]["coef_delta"]
    verifier = verifier_block("fit-input-2-p1")
    reads = read_from(f"{d}/receipt.json", f"{d}/r-oracle.json", f"{d}/julia-results.json", f"{d}/results.tsv",
                      f"{d}/run-commit.json", f"{d}/verify.txt", CONTRACT["fit-input-2-p1"])
    rule = "fit-input-2-batch-contract-p1.json tolerances.{q}_delta (carried verbatim from P0)"
    ccases = {c["case_id"]: (row["source_id"], c) for row in contract["rows"] for c in row["cases"]
              if c["status"] == "EXECUTABLE_NOW"}
    out = {}
    for cid, (sid, cc) in sorted(ccases.items()):
        stem = cid.removeprefix("CORE070-FIT-INPUT-").rsplit("-NATIVE-MODEL", 1)[0].rsplit("-FORMULA-INTERFACE", 1)[0]
        okey = FIT_INPUT_ORACLE_KEY[stem]
        jc = jres["cases"][cid]
        r_ll, r_coef = oracle[okey]["loglik"], oracle[okey]["coef"]
        j_ll, j_coef = jc["julia_loglik"], jc["julia_coef"]
        entries = []
        for q, rv, jv, tol, harness in (("loglik", [r_ll], [j_ll], tol_ll, jc["loglik_delta"]),
                                        ("coef", r_coef, j_coef, tol_coef, jc["coef_delta"])):
            if len(rv) != len(jv):
                raise SystemExit(f"{cid} {q}: R length {len(rv)} != Julia length {len(jv)}")
            diff = max(abs(a - b) for a, b in zip(rv, jv))
            if abs(diff - harness) > 1e-12 * max(1.0, abs(harness)):
                raise SystemExit(f"{cid} {q}: recomputed {diff} != harness {harness}")
            e = {"case_id": cid, "quantity": q, "max_abs_diff": diff, "tolerance": tol,
                 "tolerance_rule": rule.format(q=q), "n_values": len(rv),
                 "diff_source": "recomputed from raw R and Julia values"}
            if q == "loglik":
                e["r_value"], e["julia_value"] = rv[0], jv[0]
            else:
                e["r_value"], e["julia_value"] = rv, jv
            entries.append(mark_degenerate(e, rv))
        within = all(e["max_abs_diff"] <= e["tolerance"] for e in entries)
        verdict = "PASS" if jc["pass"] and within else "FAIL"
        body = {"source_ids": [sid],
                "batch": "tools/core070_fit_input_2_batch.R + .jl, GLLVM_PARITY_PIN=P1",
                "batch_status": receipt["status"], "batch_verifier": verifier, "harness_pass": bool(jc["pass"]),
                "r_call": cc.get("r_call") or next(c.get("r_call") for r in contract["rows"] for c in r["cases"]
                                                   if r["source_id"] == sid and c.get("r_call")),
                "julia_call": cc["julia_call"], "acceptance": cc["acceptance"],
                "r_oracle_fixture": okey, "gllvmtmb_version": receipt["gllvmTMB_version"],
                "read_from": reads, "raw": [f"{d}/r-oracle.json", f"{d}/julia-results.json"]}
        if stem == "KERNEL-TWO-AUTO":
            twin = cid.replace("KERNEL-TWO-AUTO", "KERNEL-TWO")
            tj = jres["cases"][twin]
            identical = (oracle["kernel_two_auto"]["loglik"] == oracle["kernel_two"]["loglik"]
                         and oracle["kernel_two_auto"]["coef"] == oracle["kernel_two"]["coef"]
                         and j_ll == tj["julia_loglik"] and j_coef == tj["julia_coef"])
            body["r_matches_kernel_two"] = jc.get("r_matches_kernel_two")
            body["same_measurement_as"] = {"case_id": twin, "identical_r_and_julia_values": identical}
            if identical:
                body["note"] = SAME_MEASUREMENT_NOTE.format(twin=twin)
        if not any(cid in r["executable_case_ids"] for r in load(P0_CASEMAP)["rows"]):
            body["unbound_case_note"] = ("No required row lists this case id in executable_case_ids "
                                         f"({sid} is NOT_BOUND_AT_P0 with an empty list), so it pays no row here.")
        out[cid] = ("numeric_r_vs_julia", verdict, body, entries)
    return out


# ---------------------------------------------------------------------------
# case-map rows
# ---------------------------------------------------------------------------
def receipt_info(path, rec):
    comp = (rec.get("comparison") or {}).get("cases") or []
    disc = all(e.get("discriminating", True) for e in comp)
    return (path, rec["evidence_kind"], rec["verdict"], rec["batch_verifier"]["status"], disc, rec.get("note"))


def p0_evidence(base):
    ev = base.get("evidence") or {}
    if not ev:
        return {"recorded": False}
    return {"recorded": True, "batch": ev.get("batch"), "receipts": ev.get("receipts"),
            "preservation_sha256": ev.get("preservation_sha256"),
            "raw_available_in_repo_or_on_this_host": False,
            "note": "P0 receipts live under .unlazy/ (untracked; absent on this host). Only the P0 case map's record "
                    "(batch name, preservation sha256, verifier line) remains."}


def build_rows(in_scope, receipts):
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    counts = {k: 0 for k in COUNT_KEYS}
    out_rows = []
    for sid in in_scope:
        base = p0[sid]
        ids = base["executable_case_ids"]
        row = {"source_id": sid, "classification": base["classification"], "arc": "A3",
               "carry_scan_status": "DANGLING", "executable_case_ids": ids,
               "disposition": base.get("disposition"), "p0_batch": (base.get("evidence") or {}).get("batch"),
               "p0_evidence": p0_evidence(base)}
        have = [receipts.get(i) for i in ids]
        if not ids or any(h is None for h in have):
            row.update(evidence_tier="not_measured", measured_against=None, evidence={},
                       reason="Not re-measured at P1 in this PR.")
            counts["not_measured"] += 1
            out_rows.append(row)
            continue
        kinds = {h[1] for h in have}
        verdicts = {i: h[2] for i, h in zip(ids, have)}
        batch_ok = {i: h[3] for i, h in zip(ids, have)}
        disc = {i: h[4] for i, h in zip(ids, have)}
        notes = [h[5] for h in have if h[5]]
        paths = [h[0] for h in have]
        all_pass = all(v == "PASS" for v in verdicts.values())
        numeric = kinds == {"numeric_r_vs_julia"} and all_pass
        if numeric and not all(v == "PASS" for v in batch_ok.values()):
            row.update(evidence_tier="numeric_held_batch_verifier_failed", measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "numeric comparison blocks pass, but the batch verifier rejected the "
                                         "run, so the row does not bind"},
                       measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok, "discriminating": disc})
            counts["numeric_held_batch_verifier_failed"] += 1
        elif numeric and not all(disc.values()):
            row.update(evidence_tier="numeric_non_discriminating", measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "numeric comparison blocks pass, but at least one is degenerate (the R "
                                         "values are one constant or all ~0), so the row does not bind"},
                       measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok, "discriminating": disc})
            counts["numeric_non_discriminating"] += 1
        elif numeric:
            row.update(evidence_tier="numeric", measured_against=P1_SHA,
                       evidence={"receipt": paths,
                                 "tier": "numeric: every executable case receipt carries an R-vs-Julia comparison "
                                         "block pinned to P1, within the harness tolerance"},
                       measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok, "row_verdict": "PASS"})
            counts["numeric_pass"] += 1
        elif not all_pass:
            row.update(evidence_tier="numeric_fail" if "numeric_r_vs_julia" in kinds else "measured_fail",
                       measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths,
                                 "tier": "measured at P1; the harness verdict is FAIL, so the row does not bind"},
                       measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok, "row_verdict": "FAIL"})
            counts["numeric_fail"] += 1
        elif kinds <= {"r_replay_julia_surface_absent", "r_replay_negative_control"} and \
                "r_replay_julia_surface_absent" in kinds:
            tier = "needs_surface_r_side_measured"
            row.update(evidence_tier=tier, measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths, "tier": TIER_TEXT[tier]},
                       measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok,
                                        "case_kinds": {i: h[1] for i, h in zip(ids, have)}})
            counts[tier] += 1
        else:
            raise SystemExit(f"{sid}: no tier rule for case kinds {sorted(kinds)}")
        if notes:
            row["note"] = " ".join(dict.fromkeys(notes))
        out_rows.append(row)
    return out_rows, counts


def family_of(cid):
    return "data" if cid.startswith("CORE070-DATA-") else "fit-input"


def derive_all():
    return {**data_cases(), **fit_input_cases()}


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
PROVENANCE_KEYS = {"pin", "reference_commit", "p0_reference_commit", "oracle_build_receipt", "oracle_source_receipt",
                   "glvmodels_commit", "glvmodels_worktree_dirty", "glvmodels_src_tree", "host", "schema", "case_id",
                   "verdict", "evidence_kind", "comparison"}


def check():
    problems = []
    tracked = {}
    for fam in FAMILIES:
        for p in sorted((ROOT / rec_rel(fam) / "cases").glob("*.json")):
            tracked[p.stem] = (str(p.relative_to(ROOT)), load(p))
    for cid, (path, rec) in tracked.items():
        reads = rec.get("read_from")
        if not reads:
            problems.append(f"{path}: no read_from")
            continue
        for rel, digest in reads.items():
            if not (ROOT / rel).is_file():
                problems.append(f"{path}: read file {rel} is gone")
            elif sha(ROOT / rel) != digest:
                problems.append(f"{path}: read file {rel} changed (sha256 {sha(ROOT / rel)[:12]} != {digest[:12]})")
    try:
        fresh = derive_all()
    except (SystemExit, KeyError) as e:
        problems.append(f"re-derivation refused: {e}")
        fresh = {}
    for cid in set(tracked) - set(fresh):
        problems.append(f"{cid}: tracked receipt with no re-derived case")
    for cid, (kind, verdict, body, comparison) in fresh.items():
        if cid not in tracked:
            problems.append(f"{cid}: no tracked receipt")
            continue
        rec = tracked[cid][1]
        if (rec["evidence_kind"], rec["verdict"]) != (kind, verdict):
            problems.append(f"{cid}: kind/verdict {rec['evidence_kind']}/{rec['verdict']} != re-derived {kind}/{verdict}")
        if {k: v for k, v in rec.items() if k not in PROVENANCE_KEYS} != body:
            problems.append(f"{cid}: receipt body differs from the re-derivation")
        if (rec.get("comparison") or {}).get("cases") != comparison:
            problems.append(f"{cid}: comparison block differs from the re-derivation")
        if rec.get("reference_commit") != P1_SHA or rec.get("pin") != "P1":
            problems.append(f"{cid}: receipt not pinned at P1")
    n_rows = 0
    receipts = {cid: receipt_info(path, rec) for cid, (path, rec) in tracked.items()}
    for fam in FAMILIES:
        cm = load(ROOT / casemap_rel(fam))
        try:
            rows, counts = build_rows([r["source_id"] for r in cm["rows"]], receipts)
        except (SystemExit, KeyError) as e:
            problems.append(f"{fam}: case-map re-derivation refused: {e}")
            continue
        n_rows += len(rows)
        if rows != cm["rows"]:
            bad = [a["source_id"] for a, b in zip(rows, cm["rows"]) if a != b] or ["row count"]
            problems.append(f"{fam}: case-map rows differ from the re-derivation: {', '.join(bad)}")
        if counts != cm["counts"]:
            problems.append(f"{fam}: case-map counts {cm['counts']} != re-derived {counts}")
    if problems:
        print("STALE\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("CORE070_DATA_P1_RECEIPTS_CURRENT", len(tracked), "case receipts,", n_rows, "rows")


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
SCOPE = {
    "data": ("data family: the 28 required rows the P1 carry scan lists as DANGLING, all paid by the wave1 data "
             "batch (R helper replay against the pinned P1 source plus Julia surface introspection). The 28 "
             "rejected data rows are out of scope."),
    "fit-input": ("fit-input family: the 6 required rows the P1 carry scan lists as DANGLING, all paid by the wave4 "
                  "fit-input-2 batch (paired R/Julia fits). The 5 NOT_BOUND_AT_P0 required rows (INPUT-GAUSS-COMMON, "
                  "INPUT-GAUSS-LOADINGS, INPUT-POISSON-DEFAULT: BLOCKED_NEEDS_JULIA_SURFACE; INPUT-MN-LATENT, "
                  "INPUT-MN-ANIMAL-LATENT: PARTIAL_PENDING_DECISION_OPEN_QUESTION) and the 3 rejected rows are out "
                  "of scope."),
}
NOTE = {
    "data": ("Separate from case-map.json so none of its rows are touched; read by tools/true_parity_check.mjs with "
             "PARITY_CASEMAP pointing at this file. Classification and disposition are carried from "
             "docs/dev-log/core070/required-source-case-map.json unchanged; nothing is signed by an agent. No data "
             "row has a Julia surface to compare, so every row cites evidence.non_binding_receipts and is free."),
    "fit-input": ("Separate from case-map.json so none of its rows are touched; read by tools/true_parity_check.mjs "
                  "with PARITY_CASEMAP pointing at this file. Classification and disposition are carried from "
                  "docs/dev-log/core070/required-source-case-map.json unchanged; nothing is signed by an agent. Only "
                  "rows whose every executable case id carries a numeric R-vs-Julia comparison block within tolerance, "
                  "from a batch whose verifier passed, with no degenerate comparison, cite evidence.receipt. "
                  "INPUT-KERNEL-TWO-AUTO and INPUT-KERNEL-TWO are one comparison counted on two rows (see the note)."),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--runtimes", type=Path)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="write receipts from a checkout with modified tracked files (recorded, not hidden)")
    ap.add_argument("--check", action="store_true",
                    help="verify the tracked receipts against the files they read; write nothing")
    args = ap.parse_args()
    if args.check:
        check()
        return
    if args.runs is None or args.runtimes is None:
        ap.error("--runs and --runtimes are required unless --check")
    runs, runtimes = args.runs, load(args.runtimes)
    head, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit("tracked files are modified outside this tool's outputs; commit first or pass "
                         "--allow-dirty: " + ", ".join(dirty))
    for b in BATCHES:
        check_run_commit(runs / b, head)
    for fam in FAMILIES:  # stale case receipts from an earlier write must not survive
        shutil.rmtree(ROOT / rec_rel(fam) / "cases", ignore_errors=True)
    artifacts, verifiers = {}, {}
    for b in BATCHES:
        artifacts[b] = copy_batch(b, runs / b)
        verifiers[b] = {**verifier_block_stub(b), **run_verifier(b, runs / b)}
        artifacts[b].append(verifiers[b]["log"])
    common = {"pin": "P1", "reference_commit": P1_SHA, "p0_reference_commit": P0_SHA,
              "oracle_build_receipt": ORACLE_BUILD, "oracle_source_receipt": ORACLE_SOURCE,
              "glvmodels_commit": head, "glvmodels_worktree_dirty": dirty,
              "glvmodels_src_tree": git("rev-parse", f"{head}:src").stdout.strip(), "host": HOST}
    receipts = {}
    for cid, (kind, verdict, body, comparison) in derive_all().items():
        fam = family_of(cid)
        rec = {"schema": f"core070-{fam}-p1-case-receipt/v1", "case_id": cid, "verdict": verdict,
               "evidence_kind": kind, **body, **common}
        if comparison is not None:
            rec["comparison"] = {"pin": "P1", "cases": comparison}
        path = ROOT / rec_rel(fam) / "cases" / f"{cid}.json"
        write_json(path, rec)
        receipts[cid] = receipt_info(str(path.relative_to(ROOT)), rec)
    carry = load(runs / "carry-scan-p1.json")
    for fam in FAMILIES:
        in_scope = [r["source_id"] for r in carry["rows"]
                    if r["source_id"].startswith(fam + "/") and r["status"] == "DANGLING"]
        rows, counts = build_rows(in_scope, receipts)
        batch = "data-p1" if fam == "data" else "fit-input-2-p1"
        write_json(ROOT / casemap_rel(fam), {
            "schema": 1, "reference_commit": P1_SHA, "scope": SCOPE[fam], "note": NOTE[fam],
            "generator": "tools/core070_data_p1_receipts.py", "glvmodels_commit": head,
            "batch_verifiers": {batch: verifiers[batch]}, "counts": counts,
            "p0_evidence_summary": {
                "rows_with_p0_evidence_record": sum(r["p0_evidence"]["recorded"] for r in rows),
                "rows_without_p0_evidence_record": sum(not r["p0_evidence"]["recorded"] for r in rows),
                "p0_raw_receipts_available_here": False},
            "runtimes_seconds": runtimes.get(batch), "batch_artifacts": {batch: artifacts[batch]}, "rows": rows})
        print(fam, json.dumps(counts))
    print("case receipts", len(receipts))


def verifier_block_stub(batch):
    family, argv_t, marker, _ = BATCHES[batch]
    return {"tool": argv_t[0], "argv": " ".join(argv_t), "accept_marker": marker,
            "log": f"{rec_rel(family)}/{batch}/verify.txt"}


if __name__ == "__main__":
    main()

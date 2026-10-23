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

  receipts/inference/<batch>-p1/...    batch artifacts copied verbatim (JSON/TSV only, no logs),
                                       run-commit.json, verify.txt
  receipts/inference/cases/<id>.json   one receipt per executable case id
  case-map-inference.json              the 63 rows, P0 classification and disposition kept

A case gets a `comparison` block only when its harness compares an R number
with a Julia number against a declared tolerance > 0; max_abs_diff is
recomputed here from the saved R and Julia vectors and must agree with the
harness figure. Routing and error-class rows cite
evidence.non_binding_receipts and stay free. A row is evidence_tier "numeric"
only when every executable case id has a comparison block, the harness
verdict is PASS, every batch those cases came from passed its own verifier,
and no comparison is degenerate.

Shared gates (PR #567 / #569, adopted here per the PR #571 review):

  * Batch verifier. This tool runs the wave2 and wave4 verifiers at P1 (with
    --self-test) and keeps their full output as the tracked <batch>/verify.txt.
    The wave5 verifier was run by PR #569; its tracked verify.txt is read and
    must carry the accept marker. Every case receipt carries a batch_verifier
    block. A numeric row whose batch verifier did not pass is held back
    (numeric_held_batch_verifier_failed, non-binding receipts). There is no
    exception path.
  * Degenerate comparison. A comparison whose R values are one constant or all
    ~0 is flagged discriminating: false and its row is
    numeric_non_discriminating (same rule and code as
    tools/core070_postfit_p1_receipts.py).
  * Provenance. Every receipt records glvmodels_commit = HEAD. The tool refuses
    to write when tracked files outside its own outputs are modified (unless
    --allow-dirty, which is then recorded), and refuses unless each run
    directory it reads holds a run-commit.json naming this HEAD with an empty
    dirty list. PR #569's tracked run-commit.json must name an ancestor of HEAD
    with an empty dirty list.

Read-file hashes (PR #571 review F1). Every case receipt records `read_from`,
the sha256 of each tracked file it was derived from; the wave5 receipts also
cite PR #569's julia_results_sha256 / raw_sha256 / contract_sha256. --check
re-hashes every read_from file, re-derives the wave5 receipts and all 63
case-map rows in memory from tracked files, and exits nonzero if anything
differs, so a regenerated or rebased #569 run cannot leave these receipts
stale silently.

Usage:
  python3 tools/core070_inference_p1_receipts.py --runs DIR --runtimes JSON [--allow-dirty]
  python3 tools/core070_inference_p1_receipts.py --check
  python3 tools/core070_inference_p1_receipts.py --apply-behaviour   # after tools/core070_behaviour_receipts.py --write
  python3 tools/core070_inference_p1_receipts.py --ingest-rulings RUN  # RUN/inference-rulings-p1 from
                                       # tools/core070_inference_rulings_batch.jl, with run-commit.json at HEAD

Signed rulings of 2026-10-05 (vault D-319; the lane kit's signed-rulings-2026-10-05.md). Ruling B (Julia
defaults and fallbacks against R, row by row) closes the ten rows in RULING_B_ROWS by a signed
disposition on the row. Each row's `reason` is derived here from tracked files only (the P1 route probe
and the rulings run receipts/inference/inference-rulings-p1), so --check re-derives it; the row keeps its
case receipts under evidence.non_binding_receipts and cites the rulings run under evidence.ruling_evidence.
Rulings C and 3 and the fisher_z alias bind rows behaviourally (tools/core070_behaviour_receipts.py).

Ruling N4 (Monte-Carlo tolerance rule) for CI-ROUTE-011: the rule receipts/inference/ci-route-011-mc/rule.json
was committed before any run; the run (tools/core070_ci_route_011_mc.R + .jl, 5 seeds per engine) is ingested
with --ingest-mc-011 RUN (RUN/ci-route-011-mc with run-commit.json at HEAD). Once it is tracked, the
CI-ROUTE-011 receipt is re-derived with one comparison entry per endpoint: r_value and julia_value are the
seed means, tolerance is the rule's 4.5 combined Monte-Carlo standard errors plus 0.005, and the per-seed
structural check must hold on every seed. --check re-derives all of it from the tracked raw files.
where DIR holds inference-p1/{julia,r-crosscheck,run-commit.json},
inference-remainder-p1/ (with run-commit.json), routes-p0.tsv,
routes-p1-unadapted.tsv, routes-p1-adapted.tsv and carry-scan-p1.json.
"""
import argparse
import csv
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
import core070_behaviour_receipts as behaviour  # noqa: E402  (itchyshin/GLLVModels.jl#684 item 2)

OUT = ROOT / "docs/dev-log/core070/true-parity-latest"
REC = OUT / "receipts/inference"
REC_REL = "docs/dev-log/core070/true-parity-latest/receipts/inference"
CASEMAP = OUT / "case-map-inference.json"
CASEMAP_REL = "docs/dev-log/core070/true-parity-latest/case-map-inference.json"
SURF_REL = "docs/dev-log/core070/true-parity-latest/receipts/postfit/surface-conversion-p1"
SURF_CONTRACT_REL = "docs/dev-log/core070/true-parity-latest/surface-conversion-batch-contract-p1.json"
P0_CASEMAP = ROOT / "docs/dev-log/core070/required-source-case-map.json"
PINS = tomllib.loads((ROOT / "tools/core070_oracle_pins.toml").read_text())
P1_SHA = PINS["P1"]["reference_commit"]
P0_SHA = PINS["P0"]["reference_commit"]
ORACLE_BUILD = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build.json"
ORACLE_SOURCE = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/source.json"
SURF_VERIFY_MARKER = "CORE070_SURFACE_CONVERSION_STATE_OK"

VERIFIERS = {
    "inference-batch-p1": (["tools/core070_verify_inference_batch.py", "--julia-state", "{julia_state}",
                            "--r-state", "{r_state}", "--self-test"], "CORE070_INFERENCE_BATCH_FULLY_VERIFIED"),
    "inference-remainder-p1": (["tools/core070_verify_inference_remainder_batch.py", "--state", "{state}",
                                "--self-test"], "CORE070_INFERENCE_REMAINDER_BATCH_VERIFIED"),
}

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

# PR #571 review F2: CI-ROUTE-008 (default method) and 010 (method = 'wald') pay with one comparison.
SAME_MEASUREMENT = {"CORE070-SURFCONV-INFERENCE-CI-ROUTE-008": "CORE070-SURFCONV-INFERENCE-CI-ROUTE-010",
                    "CORE070-SURFCONV-INFERENCE-CI-ROUTE-010": "CORE070-SURFCONV-INFERENCE-CI-ROUTE-008"}
SAME_MEASUREMENT_NOTE = (
    "One measurement counted twice: CI-ROUTE-008 (confint(parm = 'icc'), default method) and CI-ROUTE-010 "
    "(method = 'wald') have identical R vectors and identical Julia vectors, because both engines' default ICC "
    "interval is Wald. They are distinct surface rows (default routing versus explicit method), but the numeric "
    "evidence behind them is one R-vs-Julia comparison. Not reclassified here; whether it counts once or twice "
    "is for the maintainer.")

# PR #571 review F3: collapsed Julia bootstrap lower bounds on CI-ROUTE-011.
COLLAPSE_RATIO = 1e-3  # a Julia lower bound below this fraction of R's is called collapsed
ANOMALY_NOTE = (
    "Anomaly: the Julia bootstrap lower bounds for {n} ICC entries collapsed ({pairs}). The structural check "
    "passes (finite, ordered, brackets the point) and endpoints are not compared, so this PASS does not mean "
    "the bootstraps agree. Root cause found and fixed in draft PR #576: a Woodbury quadratic form went negative "
    "at extreme variance ratios, so bootstrap refits reported converged with an impossible log-likelihood. This "
    "receipt predates that fix (PR #569's run); re-measure after #576 lands.")


NOTE = ("Separate from case-map.json so none of its rows are touched; read by tools/true_parity_check.mjs with "
        "PARITY_CASEMAP pointing at this file. Classification and disposition are carried from "
        "docs/dev-log/core070/required-source-case-map.json unchanged (all 63 are compatibility_adapter); "
        "nothing is signed by an agent. Only rows whose every executable case id carries a numeric R-vs-Julia "
        "comparison block within tolerance, from a batch whose verifier passed, with no degenerate "
        "comparison, cite evidence.receipt as numeric rows. Routing (wave2) and error-class (wave4) rows carry no "
        "number. A routing or error-class row in the frozen scope of itchyshin/GLLVModels.jl#684 item 2 binds as "
        "evidence_tier behavioural when its receipt's behaviour block shows both engines giving the same route, refusal "
        "or error class through behaviour-equivalence.json (tools/core070_behaviour_receipts.py); it then cites "
        "evidence.receipt. Since #709 the Sigma bootstrap rows, the derived-quantity profile, bootstrap and default rows and "
        "the bad-method refusal rows are measured through Julia's public confint(fit, y; parm, method) "
        "(receipts/inference/inference-post709-p1), and a refusal binds only with a valid-method control on each side. A "
        "row whose raw record shows R and Julia doing different things (a DEFAULT whose R route is profile, the "
        "fixed-effect bootstrap fallback) has no behaviour entry and stays under evidence.non_binding_receipts, with "
        "the reason in the receipt's behaviour_not_bound. Under the signed rulings of 2026-10-05 (vault D-319): those "
        "ten default and fallback rows carry a signed ruling-B disposition whose reason is derived from the P1 route "
        "probe and receipts/inference/inference-rulings-p1; the withdrawn-profile rows (023, 030, 037) bind "
        "behaviourally now that Julia withdraws the same profiles (ruling C); CI-ROUTE-029 (ruling 3) and CI-ROUTE-034 "
        "(the fisher_z alias) bind behaviourally through the class rho:fisher-z. CI-ROUTE-009 carries a behaviour "
        "block as non-binding evidence only: signed ruling A extends the behavioural scope to it, and it flips to "
        "behavioural once the scope lists in tools/true_parity_assemble.py and tools/true_parity_check.mjs name it. "
        "CI-ROUTE-008 and CI-ROUTE-010 are one R-vs-Julia comparison counted on two surface rows "
        "(see their notes); the count is left to the maintainer.")


# ---------------------------------------------------------------------------
# Ruling B (vault D-319): signed dispositions, one per row, each derived from the P1 route probe and the rulings run.
# ---------------------------------------------------------------------------
RULINGS_REL = f"{REC_REL}/inference-rulings-p1"
RULINGS_VERIFY = (["tools/core070_verify_inference_rulings_batch.py", "--state", "{state}", "--self-test"],
                  "CORE070_INFERENCE_RULINGS_BATCH_VERIFIED")
SIGNATURE = {"signed_by": "Shinichi Nakagawa", "signed_on": "2026-10-05",
             "signature_ref": ("vault D-319 (shinichi-brain memory/DECISIONS.md), 2026-10-05: signed rulings page "
                               "LOOP/lanes/true-parity-latest/signed-rulings-2026-10-05.md, item B 'Julia defaults and "
                               "fallbacks against R', recommendation 'yes, row by row'")}
RULING_B_ROWS = ("inference/CI-ROUTE-015", "inference/CI-ROUTE-043", "inference/CI-ROUTE-046", "inference/CI-ROUTE-055",
                 "inference/CI-ROUTE-058", "inference/CI-ROUTE-061", "inference/CI-ROUTE-065", "inference/CI-ROUTE-067",
                 "inference/CI-ROUTE-068", "inference/CI-ROUTE-070")
PROBE_REL = f"{REC_REL}/inference-batch-p1/r-crosscheck/p1-route-probe-results.tsv"


def ruling_b_disposition(sid):
    """The signed-disposition fields for one ruling-B row, derived from tracked files; raises if the raw record does
    not show the difference the ruling signs (R and Julia must still differ the way the reason says)."""
    n = sid.split("/")[-1]
    with open(ROOT / PROBE_REL) as fh:
        r = {x["id"]: x for x in csv.DictReader(fh, delimiter="\t", escapechar="\\", doublequote=False)}[n]
    c = {x["source_id"]: x for x in load(ROOT / RULINGS_REL / "inference-rulings-results.json")["cases"]}[sid]
    if r["pass"] != "TRUE" or c["outcome"] != "result":
        raise SystemExit(f"{sid}: the probe or the Julia call failed; no disposition")
    src = f"{PROBE_REL}#{n}; {RULINGS_REL}/inference-rulings-results.json#{n}"
    if c["group"] == "default":
        ep = c["explicit_profile"]
        if not r["actual"].endswith(":profile") or c["route_tag"] not in ("wald_derived", "wald_packed") or \
                ep["outcome"] != "result" or ep["route_tag"] != "profile":
            raise SystemExit(f"{sid}: the raw record does not show R default profile against Julia default Wald")
        fin = "finite bounds" if ep["finite"] else "non-finite bounds on this fixture"
        reason = (f"Documented default difference, signed under ruling B (Julia defaults against R, row by row). "
                  f"R: confint() with no method for parm {c['target']} routes to {r['actual']!r}, a profile interval "
                  f"(P1 route probe). Julia: {c['julia_call']} with no method returns a {c['route_tag']} result, the "
                  f"Wald interval, because Julia's confint defaults to method = :wald for every parm. R's default route "
                  f"is reachable in Julia by asking for it: {ep['call']} returned a profile result ({fin}). Same "
                  f"quantity, different default method; not a defect in either engine. Evidence: {src}.")
    elif c["group"] == "fallback":
        if r["actual"] != "tidy:wald" or "falling back" not in r["messages"] or c["route_tag"] != "bootstrap":
            raise SystemExit(f"{sid}: the raw record does not show R falling back to Wald against a Julia bootstrap")
        reason = (f"Documented fallback difference, signed under ruling B (Julia fallbacks against R, row by row). R: "
                  f"confint(method = \"bootstrap\") for parm {c['target']} does not bootstrap; it routes to "
                  f"{r['actual']!r} with the message {r['messages']!r} (P1 route probe). Julia: {c['julia_call']} runs "
                  f"the bootstrap it is asked for (a {c['route_tag']} result reporting method "
                  f"{c['result_method']!r}). Julia does what the request names where R substitutes Wald; not a defect in "
                  f"either engine. Evidence: {src}.")
    else:
        raise SystemExit(f"{sid}: group {c['group']} is not a ruling-B row")
    return {"disposition": "DISPOSITION-SIGNED", **SIGNATURE, "reason": reason,
            "ruling_evidence": [PROBE_REL, f"{RULINGS_REL}/inference-rulings-results.json"]}


def apply_ruling_b(row):
    """Put the signed ruling-B disposition on `row` (in place). The tier and receipts stay as measured."""
    if row["source_id"] not in RULING_B_ROWS:
        return
    d = ruling_b_disposition(row["source_id"])
    row["evidence"] = dict(row.get("evidence") or {}, ruling_evidence=d.pop("ruling_evidence"))
    row.update(d)


MC_REL = f"{REC_REL}/ci-route-011-mc"
MC_CASE = "CORE070-SURFCONV-INFERENCE-CI-ROUTE-011"


def mc_011():
    """(body, comparison entries, verdict, read_from) for CI-ROUTE-011 under the N4 rule, or None when no run is
    tracked. Derived only from the tracked rule and raw files."""
    d = ROOT / MC_REL
    if not (d / "r-mc.json").is_file():
        return None
    rule, r, j = load(d / "rule.json"), load(d / "r-mc.json"), load(d / "julia-mc.json")
    rc = load(d / "run-commit.json")
    if rc.get("dirty") != [] or git("merge-base", "--is-ancestor", rc.get("glvmodels_commit", ""), "HEAD",
                                    check=False).returncode != 0:
        raise SystemExit(f"{MC_REL}/run-commit.json {rc} is not a clean ancestor of HEAD")
    # the rule must have been committed before the run commit
    first = git("log", "--diff-filter=A", "--format=%H", "--", f"{MC_REL}/rule.json").stdout.split()
    if not first or git("merge-base", "--is-ancestor", first[-1], rc["glvmodels_commit"], check=False).returncode != 0:
        raise SystemExit("the N4 rule file was not committed before the Monte-Carlo run commit")
    if r["reference_commit"] != P1_SHA or j["r_input_sha256"] != sha(d / "r-mc.json"):
        raise SystemExit("the Monte-Carlo run is not pinned at P1, or Julia did not read this R file")
    seeds = rule["seeds"]
    if r["seeds"] != seeds or j["seeds"] != seeds:
        raise SystemExit("the Monte-Carlo run used other seeds than the rule states")
    reps = {"r": r["replicates"], "julia": j["replicates"]}
    structural = all(x["ok"] and x["finite"] and x["ordered"] and x["brackets_point"]
                     for side in reps.values() for x in side)
    k, mult, floor = len(seeds), 4.5, 0.005  # rule.json test.pass_rule; checked against the rule text below
    if "4.5 * se_q + 0.005" not in rule["test"]["pass_rule"]:
        raise SystemExit("rule.json pass_rule is not the one this tool implements")
    entries = []
    for side in ("lower", "upper"):
        for t in range(len(r["point"])):
            rv = [x[side][t] for x in reps["r"]]
            jv = [x[side][t] for x in reps["julia"]]
            mr, mj = sum(rv) / k, sum(jv) / k
            var = lambda v, m: sum((a - m) ** 2 for a in v) / (k - 1)
            se = (var(rv, mr) / k + var(jv, mj) / k) ** 0.5
            entries.append(mark_degenerate(
                {"case_id": MC_CASE, "quantity": f"icc[{t + 1}] bootstrap {side} endpoint, mean over {k} seeds",
                 "max_abs_diff": abs(mr - mj), "tolerance": mult * se + floor,
                 "tolerance_rule": (f"N4 Monte-Carlo rule stated before any run ({MC_REL}/rule.json): "
                                    f"|mean_R - mean_J| <= 4.5 * sqrt(var_R/{k} + var_J/{k}) + 0.005"),
                 "mc_standard_error": se, "n_values": 1, "diff_source": "recomputed from the per-seed raw values",
                 "r_value": mr, "julia_value": mj, "r_seed_values": rv, "julia_seed_values": jv}, [mr]))
    ok = structural and all(e["max_abs_diff"] <= e["tolerance"] for e in entries)
    body = {"rule": f"{MC_REL}/rule.json", "run_commit": rc["glvmodels_commit"], "seeds": seeds,
            "nsim": 200, "per_seed_structural_check": structural,
            "r_point": r["point"], "julia_point": j["point"],
            "r_seconds_per_seed": [x["elapsed_seconds"] for x in reps["r"]],
            "julia_seconds_per_seed": [x["elapsed_seconds"] for x in reps["julia"]],
            "raw": [f"{MC_REL}/r-mc.json", f"{MC_REL}/julia-mc.json"],
            "verdict": "PASS" if ok else "FAIL"}
    reads = read_from(f"{MC_REL}/rule.json", f"{MC_REL}/r-mc.json", f"{MC_REL}/julia-mc.json",
                      f"{MC_REL}/run-commit.json")
    return body, entries, ("PASS" if ok else "FAIL"), reads


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git(*argv, check=True):
    return subprocess.run(["git", "-C", str(ROOT), *argv], check=check, capture_output=True, text=True)


def git_state():
    """HEAD and the tracked paths modified outside this tool's own outputs."""
    head = git("rev-parse", "HEAD").stdout.strip()
    own = (REC_REL + "/", CASEMAP_REL)
    dirty = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").stdout.splitlines()
             if not line[3:].startswith(own)]
    return head, dirty


def check_run_commit(run_dir, head):
    """The run directory's run-commit.json must name this HEAD and a clean tree."""
    p = run_dir / "run-commit.json"
    if not p.is_file():
        raise SystemExit(f"{run_dir} has no run-commit.json; re-run the batch from a clean commit")
    rc = load(p)
    if rc.get("glvmodels_commit") != head or rc.get("dirty") != []:
        raise SystemExit(f"{run_dir}: run at {rc.get('glvmodels_commit')} dirty={rc.get('dirty')}, "
                         f"not at clean HEAD {head}; re-run at HEAD")


def run_verifier(name, **states):
    """Run a batch verifier at P1 and keep its full output as the tracked <batch>/verify.txt."""
    argv_t, marker = VERIFIERS[name]
    argv = ["python3"] + [a.format(**{k: str(v) for k, v in states.items()}) for a in argv_t]
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                          env=dict(os.environ, GLLVM_PARITY_PIN="P1"))
    log = REC / name / "verify.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout + proc.stderr)
    ok = proc.returncode == 0 and marker in proc.stdout
    return {"tool": argv_t[0], "argv": " ".join(argv_t), "status": "PASS" if ok else "FAIL",
            "exit_code": proc.returncode, "accept_marker": marker, "log": f"{REC_REL}/{name}/verify.txt"}


def read_from(*rels):
    return {rel: sha(ROOT / rel) for rel in rels}


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


# ---------------------------------------------------------------------------
# wave5: derived only from tracked files (PR #569's run), so --check can re-derive it.
# ---------------------------------------------------------------------------
def wave5_cases():
    """case_id -> (evidence_kind, verdict, body, comparison-or-None), from PR #569's tracked run."""
    sd = ROOT / SURF_REL
    sj, so, sr = load(sd / "julia-results.json"), load(sd / "r-oracle.json"), load(sd / "receipt.json")
    rc = load(sd / "run-commit.json")
    if sr["reference_commit"] != P1_SHA:
        raise SystemExit("surface-conversion receipt is not pinned at P1")
    if sr["julia_results_sha256"] != sha(sd / "julia-results.json"):
        raise SystemExit("PR #569's julia-results.json does not match its receipt's julia_results_sha256")
    if rc.get("dirty") != [] or git("merge-base", "--is-ancestor", rc.get("glvmodels_commit", ""), "HEAD",
                                    check=False).returncode != 0:
        raise SystemExit(f"PR #569's run-commit.json {rc} is not a clean ancestor of HEAD")
    verify_text = (sd / "verify.txt").read_text()
    verifier = {"tool": "tools/core070_verify_surface_conversion_batch.py",
                "argv": "tools/core070_verify_surface_conversion_batch.py {state} --self-test (run by PR #569)",
                "status": "PASS" if SURF_VERIFY_MARKER in verify_text else "FAIL",
                "accept_marker": SURF_VERIFY_MARKER, "log": f"{SURF_REL}/verify.txt",
                "note": "run by PR #569 on its raw run; this tool reads the tracked output and checks the marker"}
    reads = read_from(f"{SURF_REL}/julia-results.json", f"{SURF_REL}/r-oracle.json", f"{SURF_REL}/receipt.json",
                      f"{SURF_REL}/verify.txt", f"{SURF_REL}/run-commit.json", SURF_CONTRACT_REL)
    pr569 = {"receipt": f"{SURF_REL}/receipt.json", "julia_results_sha256": sr["julia_results_sha256"],
             "raw_sha256": sr["raw_sha256"], "contract_sha256": sr["contract_sha256"],
             "run_commit": rc["glvmodels_commit"]}
    scases = {c["case_id"]: c for c in load(ROOT / SURF_CONTRACT_REL)["cases"]}
    out = {}
    for cid, cc in scases.items():
        if not cc["source_id"].startswith("inference/"):
            continue
        jc = sj["cases"][cid]
        body = {"source_ids": [cc["source_id"]],
                "batch": "tools/core070_surface_conversion_batch.R + .jl, GLLVM_PARITY_PIN=P1 (run by PR #569; read, not re-run)",
                "batch_status": sr["status"], "batch_verifier": verifier, "harness_kind": cc["kind"],
                "harness_pass": bool(jc["pass"]), "r_call": cc["r_call"], "julia_call": cc["julia_call"],
                "gllvmtmb_version": sr["gllvmTMB_version"], "read_from": reads, "pr569_batch_receipt": pr569,
                "raw": [f"{SURF_REL}/julia-results.json", f"{SURF_REL}/r-oracle.json"]}
        if cc["kind"] == "ci":
            rv, jv, tol = so["oracle_values"][cid], jc["julia_values"], cc["tolerance"]
            if len(rv) != len(jv):
                raise SystemExit(f"{cid}: length mismatch")
            diff = max(abs(a - b) for a, b in zip(rv, jv))
            if abs(diff - jc["max_abs_diff"]) > 1e-12 * max(1.0, jc["max_abs_diff"]):
                raise SystemExit(f"{cid}: recomputed {diff} != harness {jc['max_abs_diff']}")
            entry = mark_degenerate(
                {"case_id": cid, "quantity": cc["quantity"], "max_abs_diff": diff, "tolerance": tol,
                 "tolerance_rule": "surface-conversion-batch-contract-p1.json per-case tolerance (carried verbatim from P0); max |R - Julia| over CI bounds",
                 "n_values": len(rv), "diff_source": "recomputed from raw R and Julia values",
                 "r_value": rv, "julia_value": jv}, rv)
            twin = SAME_MEASUREMENT.get(cid)
            if twin is not None:
                identical = (so["oracle_values"][twin] == rv and sj["cases"][twin]["julia_values"] == jv)
                body["same_measurement_as"] = {"case_id": twin, "identical_r_and_julia_vectors": identical}
                if identical:
                    body["note"] = SAME_MEASUREMENT_NOTE
            out[cid] = ("numeric_r_vs_julia", "PASS" if jc["pass"] and diff <= tol else "FAIL", body, [entry])
        else:
            if cc["kind"] == "refusal_pair":
                body["measured"] = {k: jc.get(k) for k in ("r_raised", "julia_raised", "julia_message")}
                body["why_not_numeric"] = "Both engines must refuse method = profile; there is no number to compare."
            else:
                rs, js = jc["r_structural"], jc["julia_structural"]
                body["measured"] = {"r_structural": rs, "julia_structural": js}
                pt = max(abs(a - b) for a, b in zip(rs["point"], js["point"]))
                body["why_not_numeric"] = (
                    "Structural bootstrap check (finite, ordered, brackets the point), per the contract's "
                    "structural_justification: two independent stochastic bootstraps, so endpoints are not compared. "
                    f"The point legs agree to {pt:.3g} but the contract declares no tolerance for them, and none is invented.")
                body["point_leg_max_abs_diff_unbound"] = pt
                collapsed = [{"entry": i + 1, "julia_lower": jl, "r_lower": rl}
                             for i, (jl, rl) in enumerate(zip(js["lower"], rs["lower"])) if jl < COLLAPSE_RATIO * rl]
                if collapsed:
                    pairs = "; ".join(f"entry {c['entry']}: Julia {c['julia_lower']:.3g} vs R {c['r_lower']:.3g}"
                                      for c in collapsed)
                    body["anomaly"] = {"collapsed_julia_lower_bounds": collapsed,
                                       "rule": f"Julia lower bound < {COLLAPSE_RATIO:g} x R lower bound",
                                       "fix": "draft PR #576", "action": "re-measure after #576 lands",
                                       "note": ANOMALY_NOTE.format(n=len(collapsed), pairs=pairs)}
            mc = mc_011() if cid == MC_CASE else None
            if mc is not None:
                mbody, entries, verdict, mreads = mc
                body["read_from"] = {**reads, **mreads}
                body["monte_carlo"] = mbody
                body["why_numeric"] = (
                    "Signed ruling N4 (vault D-319): the two engines' bootstrap endpoints are compared as Monte-Carlo "
                    "means over 5 seeds each, against the tolerance rule stated before any run. The structural check "
                    "of PR #569's single run is kept under `measured`.")
                if "anomaly" in body:
                    clean = all(jl >= COLLAPSE_RATIO * rl for e in entries if "lower" in e["quantity"]
                                for jl, rl in zip(e["julia_seed_values"], e["r_seed_values"]))
                    sup = (f"Superseded for binding by the N4 Monte-Carlo run ({MC_REL}, at "
                           f"{mbody['run_commit'][:9]}, after #576 landed): " +
                           ("no Julia lower bound collapsed on any of the 5 seeds, and the row binds on the "
                            "pre-stated Monte-Carlo comparison." if clean else
                            "a Julia lower bound still collapses in that run."))
                    body["anomaly"]["superseded_by"] = sup
                    body["anomaly"]["note"] = body["anomaly"]["note"] + " " + sup
                out[cid] = ("numeric_r_vs_julia", verdict if jc["pass"] else "FAIL", body, entries)
                continue
            out[cid] = (f"paired_{cc['kind']}", "PASS" if jc["pass"] else "FAIL", body, None)
    return out


# ---------------------------------------------------------------------------
# case-map rows: derived from (path, kind, verdict, batch verifier status, discriminating, note) per case.
# ---------------------------------------------------------------------------
COUNT_KEYS = ("numeric_pass", "numeric_fail", "numeric_held_batch_verifier_failed", "numeric_non_discriminating",
              "partial_non_numeric_case", "routing_control_flow", "reject_error_class",
              "needs_surface_not_executed", "retired_at_p1_not_measured", "not_measured", "behavioural")


def receipt_info(path, rec):
    comp = (rec.get("comparison") or {}).get("cases") or []
    disc = all(e.get("discriminating", True) for e in comp)
    note = (rec.get("anomaly") or {}).get("note") or rec.get("note")
    return (path, rec["evidence_kind"], rec["verdict"], rec["batch_verifier"]["status"], disc, note)


def build_rows(in_scope, receipts):
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    counts = {k: 0 for k in COUNT_KEYS}
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
        else:
            only = next(iter(kinds))
            tier = only if len(kinds) == 1 and only in ("routing_control_flow", "reject_error_class") \
                else "partial_non_numeric_case"
            row.update(evidence_tier=tier, measured_against=P1_SHA,
                       evidence={"non_binding_receipts": paths, "tier": TIER_TEXT[tier]},
                       measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok,
                                        "case_kinds": {i: h[1] for i, h in zip(ids, have)}})
            counts[tier] += 1
        if notes:
            row["note"] = " ".join(dict.fromkeys(notes))
        behaviour.overlay_row(row, counts)  # a row whose receipt carries a matching behaviour block
        apply_ruling_b(row)
        out_rows.append(row)
    return out_rows, counts


# ---------------------------------------------------------------------------
# --check: re-hash read_from, re-derive wave5 and every case-map row from tracked files.
# ---------------------------------------------------------------------------
PROVENANCE_KEYS = {"pin", "reference_commit", "p0_reference_commit", "oracle_build_receipt", "oracle_source_receipt",
                   "glvmodels_commit", "glvmodels_worktree_dirty", "glvmodels_src_tree", "host", "schema", "case_id",
                   "verdict", "evidence_kind", "comparison", "behaviour", "behaviour_not_bound"}


def check():
    problems = []
    tracked = {p.stem: (str(p.relative_to(ROOT)), load(p)) for p in sorted((REC / "cases").glob("*.json"))}
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
        fresh = wave5_cases()
    except SystemExit as e:
        problems.append(f"wave5 re-derivation refused: {e}")
        fresh = {}
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
    problems += behaviour.check_problems()
    cm = load(CASEMAP)
    try:
        receipts = {cid: receipt_info(path, rec) for cid, (path, rec) in tracked.items()}
        rows, counts = build_rows([r["source_id"] for r in cm["rows"]], receipts)
        if rows != cm["rows"]:
            bad = [a["source_id"] for a, b in zip(rows, cm["rows"]) if a != b] or ["row count"]
            problems.append(f"case-map rows differ from the re-derivation: {', '.join(bad)}")
        if counts != cm["counts"]:
            problems.append(f"case-map counts {cm['counts']} != re-derived {counts}")
    except KeyError as e:
        problems.append(f"a tracked receipt lacks {e} (written before the shared gates)")
        rows = []
    if problems:
        print("STALE\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("CORE070_INFERENCE_P1_RECEIPTS_CURRENT", len(tracked), "case receipts,", len(rows), "rows")


def apply_behaviour():
    """Re-derive rows, counts and note of case-map-inference.json from the tracked receipts, so rows whose receipts
    carry a matching behaviour block read as evidence_tier behavioural. Nothing else in the file changes."""
    tracked = {p.stem: (str(p.relative_to(ROOT)), load(p)) for p in sorted((REC / "cases").glob("*.json"))}
    receipts = {cid: receipt_info(path, rec) for cid, (path, rec) in tracked.items()}
    cm = load(CASEMAP)
    rows, counts = build_rows([r["source_id"] for r in cm["rows"]], receipts)
    cm["rows"], cm["counts"], cm["note"] = rows, counts, NOTE
    post = f"{REC_REL}/inference-post709-p1"
    cm["batch_verifiers"]["inference-post709-p1"] = {
        "tool": "tools/core070_verify_inference_post709_batch.py",
        "argv": "tools/core070_verify_inference_post709_batch.py --julia-state <batch dir> --r-state <batch dir>/r-lambda-reject --self-test",
        "status": "PASS" if "CORE070_INFERENCE_POST709_BATCH_VERIFIED" in (REC / "inference-post709-p1/verify.txt").read_text() else "FAIL",
        "exit_code": 0, "accept_marker": "CORE070_INFERENCE_POST709_BATCH_VERIFIED", "log": f"{post}/verify.txt"}
    cm["batch_artifacts"]["inference-post709-p1"] = [
        f"{post}/inference-post709-results.json", f"{post}/receipt.json", f"{post}/run-commit.json",
        f"{post}/verify.txt", f"{post}/r-lambda-reject/r-oracle.json", f"{post}/r-lambda-reject/receipt.json"]
    if (REC / "inference-rulings-p1/verify.txt").is_file():
        cm["batch_verifiers"]["inference-rulings-p1"] = {
            "tool": RULINGS_VERIFY[0][0], "argv": " ".join(RULINGS_VERIFY[0]),
            "status": "PASS" if RULINGS_VERIFY[1] in (REC / "inference-rulings-p1/verify.txt").read_text() else "FAIL",
            "exit_code": 0, "accept_marker": RULINGS_VERIFY[1], "log": f"{RULINGS_REL}/verify.txt"}
        cm["batch_artifacts"]["inference-rulings-p1"] = [
            f"{RULINGS_REL}/inference-rulings-results.json", f"{RULINGS_REL}/receipt.json",
            f"{RULINGS_REL}/run-commit.json", f"{RULINGS_REL}/verify.txt"]
    write_json(CASEMAP, cm)
    print(json.dumps(counts))


def ingest_mc_011(run):
    """Copy a CI-ROUTE-011 Monte-Carlo run (RUN/ci-route-011-mc, run at clean HEAD) next to the committed rule and
    re-derive the CI-ROUTE-011 receipt from it (provenance fields of the tracked receipt are kept)."""
    src = run / "ci-route-011-mc"
    head, _dirty = git_state()
    check_run_commit(src, head)
    copy(src, "ci-route-011-mc", ["r-mc.json", "julia-mc.json", "run-commit.json"])
    kind, verdict, body, comparison = wave5_cases()[MC_CASE]
    path = REC / "cases" / f"{MC_CASE}.json"
    old = load(path)
    rec = {"schema": old["schema"], "case_id": MC_CASE, "verdict": verdict, "evidence_kind": kind, **body,
           **{k: old[k] for k in PROVENANCE_KEYS if k in old and k not in
              ("schema", "case_id", "verdict", "evidence_kind", "comparison", "behaviour", "behaviour_not_bound")},
           "comparison": {"pin": "P1", "cases": comparison}}
    write_json(path, rec)
    print("ingested", src, "CI-ROUTE-011 verdict", verdict)


def ingest_rulings(run):
    """Copy a rulings run (RUN/inference-rulings-p1, run at clean HEAD) into the tracked receipts and run its verifier."""
    src = run / "inference-rulings-p1"
    head, _dirty = git_state()
    check_run_commit(src, head)
    copy(src, "inference-rulings-p1", ["inference-rulings-results.json", "receipt.json", "run-commit.json"])
    argv = ["python3"] + [a.format(state=str(REC / "inference-rulings-p1")) for a in RULINGS_VERIFY[0]]
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    (REC / "inference-rulings-p1/verify.txt").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or RULINGS_VERIFY[1] not in proc.stdout:
        raise SystemExit("the rulings verifier rejected the run:\n" + proc.stdout + proc.stderr)
    print("ingested", src, "verifier PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--runtimes", type=Path)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="write receipts from a checkout with modified tracked files (recorded, not hidden)")
    ap.add_argument("--check", action="store_true",
                    help="verify the tracked receipts against the files they read; write nothing")
    ap.add_argument("--apply-behaviour", action="store_true",
                    help="re-derive the case-map rows from the tracked receipts and their behaviour blocks")
    ap.add_argument("--ingest-rulings", type=Path, metavar="RUN",
                    help="copy RUN/inference-rulings-p1 into the tracked receipts and run its verifier")
    ap.add_argument("--ingest-mc-011", type=Path, metavar="RUN",
                    help="copy RUN/ci-route-011-mc into the tracked receipts and re-derive the CI-ROUTE-011 receipt")
    args = ap.parse_args()
    if args.ingest_mc_011:
        ingest_mc_011(args.ingest_mc_011)
        return
    if args.ingest_rulings:
        ingest_rulings(args.ingest_rulings)
        return
    if args.check:
        check()
        return
    if args.apply_behaviour:
        apply_behaviour()
        return
    if args.runs is None or args.runtimes is None:
        ap.error("--runs and --runtimes are required unless --check")
    runs, runtimes = args.runs, load(args.runtimes)
    head, dirty = git_state()
    if dirty and not args.allow_dirty:
        raise SystemExit("tracked files are modified outside this tool's outputs; commit first or pass "
                         "--allow-dirty: " + ", ".join(dirty))
    ib, rb = runs / "inference-p1", runs / "inference-remainder-p1"
    for d in (ib, rb):
        check_run_commit(d, head)
    common = {"pin": "P1", "reference_commit": P1_SHA, "p0_reference_commit": P0_SHA,
              "oracle_build_receipt": ORACLE_BUILD, "oracle_source_receipt": ORACLE_SOURCE,
              "glvmodels_commit": head, "glvmodels_worktree_dirty": dirty,
              "glvmodels_src_tree": git("rev-parse", f"{head}:src").stdout.strip(),
              "host": "local Mac (M1 Ultra), OPENBLAS/OMP threads 1, JULIA_NUM_THREADS=4"}
    receipts, artifacts, verifiers = {}, {}, {}

    def emit(cid, kind, verdict, body, comparison=None):
        rec = {"schema": "core070-inference-p1-case-receipt/v2", "case_id": cid, "verdict": verdict,
               "evidence_kind": kind, **body, **common}
        if comparison is not None:
            rec["comparison"] = {"pin": "P1", "cases": comparison}
        path = REC / "cases" / f"{cid}.json"
        write_json(path, rec)
        receipts[cid] = receipt_info(str(path.relative_to(ROOT)), rec)

    # ---- wave2 inference-batch: routing ----
    artifacts["inference-batch-p1"] = (
        copy(ib / "julia", "inference-batch-p1", ["receipt.json", "inference-batch-results.json", "raw.tsv"])
        + copy(ib / "r-crosscheck", "inference-batch-p1/r-crosscheck",
               ["receipt.json", "inference-batch-r-crosscheck.json", "r-comparand-crosscheck.tsv",
                "p1-route-probe-results.tsv"])
        + copy(ib, "inference-batch-p1", ["run-commit.json"]))
    verifiers["inference-batch-p1"] = run_verifier("inference-batch-p1", julia_state=ib / "julia",
                                                   r_state=ib / "r-crosscheck")
    artifacts["inference-batch-p1"].append(verifiers["inference-batch-p1"]["log"])
    ic = load(OUT / "inference-batch-contract-p1.json")
    jres = {c["source_id"]: c for c in load(ib / "julia/inference-batch-results.json")["cases"]}
    with open(ib / "r-crosscheck/p1-route-probe-results.tsv") as fh:
        rprobe = {r["id"]: r for r in csv.DictReader(fh, delimiter="\t", escapechar="\\", doublequote=False)}
    w2_reads = read_from(f"{REC_REL}/inference-batch-p1/inference-batch-results.json",
                         f"{REC_REL}/inference-batch-p1/r-crosscheck/p1-route-probe-results.tsv",
                         f"{REC_REL}/inference-batch-p1/r-crosscheck/receipt.json",
                         f"{REC_REL}/inference-batch-p1/run-commit.json",
                         f"{REC_REL}/inference-batch-p1/verify.txt",
                         "docs/dev-log/core070/true-parity-latest/inference-batch-contract-p1.json")
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
              "batch_verifier": verifiers["inference-batch-p1"], "per_row": per_row,
              "why_not_numeric": ("Routing probe. The R side evaluates only function definitions from the P1 source "
                                  "files and intercepts every CI endpoint, so no model is fit and no interval is "
                                  "computed; the Julia side reports which solver ran. There is no R-vs-Julia number."),
              "read_from": w2_reads,
              "raw": [f"{REC_REL}/inference-batch-p1/inference-batch-results.json",
                      f"{REC_REL}/inference-batch-p1/r-crosscheck/p1-route-probe-results.tsv"]})

    # ---- wave4 inference-remainder: error class ----
    artifacts["inference-remainder-p1"] = copy(rb, "inference-remainder-p1",
                                               ["receipt.json", "results.tsv", "julia-results.json", "r-oracle.json",
                                                "run-commit.json"])
    verifiers["inference-remainder-p1"] = run_verifier("inference-remainder-p1", state=rb)
    artifacts["inference-remainder-p1"].append(verifiers["inference-remainder-p1"]["log"])
    rc = load(OUT / "inference-remainder-batch-contract-p1.json")
    rj, ro, rrec = load(rb / "julia-results.json"), load(rb / "r-oracle.json"), load(rb / "receipt.json")
    w4_reads = read_from(f"{REC_REL}/inference-remainder-p1/julia-results.json",
                         f"{REC_REL}/inference-remainder-p1/r-oracle.json",
                         f"{REC_REL}/inference-remainder-p1/receipt.json",
                         f"{REC_REL}/inference-remainder-p1/run-commit.json",
                         f"{REC_REL}/inference-remainder-p1/verify.txt",
                         "docs/dev-log/core070/true-parity-latest/inference-remainder-batch-contract-p1.json")
    oracle_key = {"ICC": "icc_reject", "PHYLO-SIGNAL": "phylo_reject", "COMMUNALITY": "communality_reject",
                  "RHO": "rho_reject", "PROPORTION": "proportion_reject"}
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
              "batch_verifier": verifiers["inference-remainder-p1"], "r_call": c["r_call"],
              "julia_surface": c["julia_surface"], "check": c["check"], "measured": measured,
              "gllvmtmb_version": rrec["gllvmTMB_version"],
              "why_not_numeric": "Both engines must refuse the method; the check is error class and message, not a number.",
              "read_from": w4_reads,
              "raw": [f"{REC_REL}/inference-remainder-p1/julia-results.json",
                      f"{REC_REL}/inference-remainder-p1/r-oracle.json"]})

    # ---- wave5 surface-conversion ICC rows (tracked P1 run from PR #569, not re-run) ----
    for cid, (kind, verdict, body, comparison) in wave5_cases().items():
        emit(cid, kind, verdict, body, comparison)

    # ---- route-probe adaptation record: P0 probe on P0 source, P0 probe on P1 source, P1 probe on P1 source ----
    def probe(name):
        with open(runs / name) as fh:
            return list(csv.DictReader(fh, delimiter="\t", escapechar="\\", doublequote=False))
    p0r, p1u, p1a = probe("routes-p0.tsv"), probe("routes-p1-unadapted.tsv"), probe("routes-p1-adapted.tsv")
    cols = ("id", "pass", "actual_class", "actual", "messages")
    write_json(REC / "route-probe-adaptation.json", {
        "schema": "core070-inference-route-probe-adaptation/v1",
        "fixture": "test/parity/fixtures/core070_inference_routes.tsv",
        "glvmodels_commit": head,
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

    behaviour.write()  # behaviour blocks and behaviour-equivalence.json, derived from the raw files copied above

    # ---- case-map rows ----
    carry = load(runs / "carry-scan-p1.json")
    in_scope = [r["source_id"] for r in carry["rows"]
                if r["source_id"].startswith("inference/") and r["status"] == "DANGLING"]
    out_rows, counts = build_rows(in_scope, receipts)
    casemap = {
        "schema": 1, "reference_commit": P1_SHA,
        "scope": ("inference family: the 63 required rows the P1 carry scan lists as DANGLING (45 wave2 "
                  "inference-batch, 14 wave4 inference-remainder, 4 wave5 surface-conversion). The one "
                  "NOT_BOUND_AT_P0 row (CI-ROUTE-005, BLOCKED_NEEDS_JULIA_SURFACE) and the 34 out-of-scope "
                  "rejected/excluded rows are not included."),
        "note": NOTE,
        "generator": "tools/core070_inference_p1_receipts.py",
        "glvmodels_commit": head,
        "batch_verifiers": verifiers,
        "counts": counts, "runtimes_seconds": runtimes,
        "batch_artifacts": {**artifacts, "surface-conversion-p1 (from PR #569, read only)":
                            [f"{SURF_REL}/receipt.json", f"{SURF_REL}/julia-results.json", f"{SURF_REL}/r-oracle.json",
                             f"{SURF_REL}/verify.txt", f"{SURF_REL}/run-commit.json"]},
        "rows": out_rows,
    }
    write_json(CASEMAP, casemap)
    print(json.dumps(counts))
    print("rows", len(out_rows), "case receipts", len(receipts))


if __name__ == "__main__":
    main()

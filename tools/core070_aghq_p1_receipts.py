"""Write tracked P1 receipts and case-map rows for the aghq family.

In scope: the 21 required aghq rows (19 required_core, plus the
compatibility_adapter rows AGHQ-CTRL-NULL and AGHQ-CTRL-TRUE, which
tools/true_parity_check.mjs counts as required) the P1 carry scan lists as
DANGLING (7) or PARTIAL_STALE_AT_P1 (14). The rejected AGHQ-INVALID-* rows and
the intentionally_excluded rows are out of scope. Two batches pay the rows, both
run at gllvmTMB pin P1 (GLLVM_PARITY_PIN=P1):

  aghq-control-p1   tools/core070_aghq_batch.R + .jl, contract
                    aghq-batch-contract-p1.json. 7 AGHQ-CTRL-* rows. R evaluates
                    the frozen .gllvmTMB_normalize_aghq(x) call and assertion; the
                    Julia child evaluates GLLVModels._aghq_request(x) and compares
                    its label ("off", "auto", "1", ...) with the contract's
                    expected label. A paired control on categorical labels: no
                    fit, no number.
  aghq-policy-p1    tools/core070_aghq_public_policy_bind.R, contract
                    aghq-public-policy-contract-p1.json. 14 AUTO-K, DEFAULT-OFF
                    and POLICY rows. Public gllvmTMB() fits (or formals of
                    gllvmTMBcontrol) read for fit$aghq and checked against the
                    runner's assertion. R only: no Julia call is part of any case.

Neither batch writes a `comparison` block, so a row paid by a batch alone cites its
receipts under evidence.non_binding_receipts and does not bind under the #561 numeric
rule, whatever its verdict. The tiers say what each row does measure
(paired_control_categorical_*, r_only_policy_*).

Julia twins (overlay). Eleven of the 14 policy rows have a same-model Julia surface and a
numeric twin: R-at-P1 fits recorded in test/fixtures/aghq_p1/aghq_p1.toml against Julia
fits of the same data (test/test_aghq_p1_twin.jl), receipts written by
tools/true_parity_julia_receipts.jl under receipts/julia-twins/aghq/. Where such a receipt
exists, build_rows adds the evidence fields it supports to that row: the twin receipt under
evidence.receipt, executable_case_ids set to the twin's case ids (the R-only case id moves
to evidence.r_only_case_ids; its receipt stays under evidence.non_binding_receipts), and
evidence_tier numeric. Classification, disposition and every other field are untouched.
`--apply-twins` re-derives rows, counts and numeric_rows from the tracked receipts and
rewrites those fields of case-map-aghq.json; `--check` verifies them.

Shared gates (PR #567 / #569 / #571 / #579 / #584):

  * Batch verifier. aghq-control-p1: tools/core070_verify_aghq_batch.py
    --self-test --state <run> (kept verbatim) plus this tool's checks of the
    tracked files (pins, contract hash, file hashes named in receipt.json).
    aghq-policy-p1: this tool's checks of the tracked receipt (status, pins,
    contract hash, runner hash at the run commit, row ids). Each keeps a tracked
    verify.txt. A row whose batch verifier did not pass is held back
    (*_held_batch_verifier_failed). There is no exception path.
  * Degenerate comparison: not applicable; no row carries a number to compare.
  * Provenance. Every receipt records glvmodels_commit = HEAD; the tool refuses a
    dirty tree (unless --allow-dirty, recorded) and refuses unless each run
    directory's run-commit.json names HEAD with an empty dirty list. --check
    verifies every receipt's glvmodels_commit against its batch's tracked
    run-commit.json.
  * Read-file hashes. Every case receipt records `read_from`; --check re-hashes
    them, re-derives every case receipt, every verify.txt it owns, and every
    case-map row, and exits nonzero on any difference.

Usage:
  python3 tools/core070_aghq_p1_receipts.py --runs DIR --runtimes JSON [--allow-dirty]
  python3 tools/core070_aghq_p1_receipts.py --check
  python3 tools/core070_aghq_p1_receipts.py --apply-twins
where DIR holds aghq-control-p1/ and aghq-policy-p1/ (each with run-commit.json)
and carry-scan-p1.json.
"""
import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import core070_source_pin_check  # noqa: E402

OUT_REL = "docs/dev-log/core070/true-parity-latest"
REC_REL = f"{OUT_REL}/receipts/aghq"
CASEMAP_REL = f"{OUT_REL}/case-map-aghq.json"
P0_CASEMAP = ROOT / "docs/dev-log/core070/required-source-case-map.json"
PINS = tomllib.loads((ROOT / "tools/core070_oracle_pins.toml").read_text())
P1_SHA = PINS["P1"]["reference_commit"]
P0_SHA = PINS["P0"]["reference_commit"]
CONTROL_CONTRACT = f"{OUT_REL}/aghq-batch-contract-p1.json"
POLICY_CONTRACT = f"{OUT_REL}/aghq-public-policy-contract-p1.json"
POLICY_RUNNER = "tools/core070_aghq_public_policy_bind.R"
POLICY_P0_RECEIPT = "docs/dev-log/core070/aghq-public-policy-bind-receipt-2026-09-04.json"
CARRY_SCAN_SOURCE = "origin/claude/true-parity-p1-carry:docs/dev-log/core070/true-parity-latest/carry-scan-p1.json"
HOST = "local Mac (M1 Ultra), OPENBLAS/OMP threads 1, JULIA_NUM_THREADS=4"

CONTROL = "aghq-control-p1"
POLICY = "aghq-policy-p1"
BATCHES = [CONTROL, POLICY]
IN_SCOPE_STATUS = ("DANGLING", "PARTIAL_STALE_AT_P1", "NO_R_PINS")
COUNT_KEYS = ("numeric_pass", "numeric_fail",
              "paired_control_categorical_pass", "paired_control_categorical_fail",
              "paired_control_categorical_held_batch_verifier_failed",
              "r_only_policy_pass", "r_only_policy_fail", "r_only_policy_held_batch_verifier_failed")

WHY_NOT_NUMERIC = {
    CONTROL: ("A paired control on categorical labels: R's .gllvmTMB_normalize_aghq and GLLVModels._aghq_request "
              "each normalize the same scalar aghq request, R's frozen assertion must hold and the Julia label must "
              "equal the contract's expected label. There is no fit and no number, so there is no R-vs-Julia "
              "comparison block; under the numeric rule this row cannot bind, whatever its verdict."),
    POLICY: ("R only: the case runs a public gllvmTMB() fit (or reads formals(gllvmTMBcontrol)) and checks R's own "
             "fit$aghq record against the runner's assertion. No Julia call is part of the case, so there is no "
             "R-vs-Julia comparison; a pass shows the R contract still holds at P1, not parity."),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def git(*argv, check=True, text=True):
    return subprocess.run(["git", "-C", str(ROOT), *argv], check=check, capture_output=True, text=text)


def batch_rel(batch):
    return f"{REC_REL}/{batch}"


def git_state():
    head = git("rev-parse", "HEAD").stdout.strip()
    own = (REC_REL + "/", CASEMAP_REL)
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


def run_commit(batch):
    return load(ROOT / batch_rel(batch) / "run-commit.json")["glvmodels_commit"]


def read_from(*rels):
    return {rel: sha(ROOT / rel) for rel in rels}


def blob_sha_at(commit, rel):
    return hashlib.sha256(git("show", f"{commit}:{rel}", text=False).stdout).hexdigest()


# ---------------------------------------------------------------------------
# Batch verifiers: derived checks from tracked files (re-derived by --check),
# plus the control batch's own verifier (external, kept verbatim).
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def control_checks():
    d = ROOT / batch_rel(CONTROL)
    rec = load(d / "receipt.json")
    problem = core070_source_pin_check.source_pin_problem(rec, "P1")
    contract = load(ROOT / CONTROL_CONTRACT)
    return [
        ("receipt status PASS, julia exit code 0", rec.get("status") == "PASS" and rec.get("julia_exit_code") == 0),
        ("pinned at P1", rec.get("reference_commit") == P1_SHA),
        ("contract is the tracked P1 twin", rec.get("contract_sha256") == sha(ROOT / CONTROL_CONTRACT)),
        ("source pins are the twin's P1 pins", rec.get("source_pins") == contract["source_pins"]),
        ("R source-pin record matches tools/core070_oracle_pins.toml [P1]"
         + ("" if problem is None else f" ({problem})"), problem is None),
        ("julia-results.json is the file the receipt hashed",
         rec.get("julia_results_sha256") == sha(d / "julia-results.json")),
        ("results.tsv is the file the receipt hashed", rec.get("raw_sha256") == sha(d / "results.tsv")),
        ("r-oracle.json records every case with its R value",
         sorted(load(d / "r-oracle.json")["cases"]) == sorted(c["case_id"] for c in contract["cases"])
         and all("r_value" in v for v in load(d / "r-oracle.json")["cases"].values())),
    ]


@functools.lru_cache(maxsize=None)
def policy_checks():
    d = ROOT / batch_rel(POLICY)
    rec = load(d / "receipt.json")
    contract = load(ROOT / POLICY_CONTRACT)
    problem = core070_source_pin_check.source_pin_problem(rec, "P1")
    rc = run_commit(POLICY)
    return [
        ("receipt status PASS, 14 of 14 rows", rec.get("status") == "PASS" and rec.get("bound_count") == 14
         and rec.get("expected_count") == 14),
        ("pinned at P1", rec.get("parity_pin") == "P1" and rec.get("reference_commit") == P1_SHA
         and rec.get("r_engine", {}).get("reference_commit") == P1_SHA),
        ("contract is the tracked P1 contract", rec.get("contract") == POLICY_CONTRACT
         and rec.get("contract_sha256") == sha(ROOT / POLICY_CONTRACT)),
        (f"runner at run commit {rc[:12]} is the one the contract names",
         blob_sha_at(rc, POLICY_RUNNER) == contract["runner_sha256"]),
        ("R source-pin record matches tools/core070_oracle_pins.toml [P1]"
         + ("" if problem is None else f" ({problem})"), problem is None),
        ("row ids are the contract's 14", sorted(rec.get("bound_row_ids", [])) == sorted(c["row_id"] for c in
                                                                                      contract["cases"])
         and sorted(rec.get("cases", {})) == sorted(c["row_id"] for c in contract["cases"])),
        ("every row recorded an observation", all("observed" in v for v in rec.get("cases", {}).values())),
    ]


def render_checks(title, checks):
    lines = [f"# {title}"] + [f"{'PASS' if ok else 'FAIL'}  {name}" for name, ok in checks]
    ok = all(ok for _, ok in checks)
    return "\n".join(lines + [f"# status {'PASS' if ok else 'FAIL'}"]) + "\n", ok


def verify_text(batch):
    if batch == CONTROL:
        return render_checks("aghq control tracked-file checks (tools/core070_aghq_p1_receipts.py)", control_checks())
    return render_checks("aghq policy batch verifier (tools/core070_aghq_p1_receipts.py)", policy_checks())


EXTERNAL = {
    CONTROL: (["python3", "tools/core070_verify_aghq_batch.py", "--state", "{state}", "--self-test"],
              "CORE070_AGHQ_CONTROL_BATCH_VERIFIED", {"GLLVM_PARITY_PIN": "P1"}),
}
SEPARATOR = "# ---- derived checks ----\n"


def run_external(batch, state):
    argv_t, marker, env = EXTERNAL[batch]
    argv = [a.format(state=str(state)) for a in argv_t]
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, **env))
    shown = " ".join(a if a != str(state) else "<raw run>" for a in argv)
    return (f"$ {' '.join(f'{k}={v}' for k, v in env.items())} {shown}\n# exit code {proc.returncode}\n"
            + proc.stdout + proc.stderr)


def verifier_block(batch):
    rel = f"{batch_rel(batch)}/verify.txt"
    text = (ROOT / rel).read_text()
    derived, derived_ok = verify_text(batch)
    external, _, tail = text.partition(SEPARATOR) if SEPARATOR in text else (text, "", "")
    if batch in EXTERNAL:
        marker = EXTERNAL[batch][1]
        ok = derived_ok and tail == derived and "# exit code 0\n" in external and marker in external
        tool = f"{EXTERNAL[batch][0][1]} --self-test --state, plus tools/core070_aghq_p1_receipts.py tracked-file checks"
    else:
        ok = derived_ok and text == derived
        tool = "tools/core070_aghq_p1_receipts.py (policy receipt checks)"
    return {"tool": tool, "status": "PASS" if ok else "FAIL", "log": rel}


# ---------------------------------------------------------------------------
# Derivation: case receipts from tracked files only.
# ---------------------------------------------------------------------------
def control_case(cid):
    d = batch_rel(CONTROL)
    rec, oracle, jres = (load(ROOT / d / n) for n in ("receipt.json", "r-oracle.json", "julia-results.json"))
    contract = load(ROOT / CONTROL_CONTRACT)
    cc = next(c for c in contract["cases"] if c["case_id"] == cid)
    r = oracle["cases"][cid]
    j = jres["cases"][cid]
    if j["r_assertion_pass"] != r["r_assertion_pass"]:
        raise SystemExit(f"{cid}: Julia child's copy of the R assertion differs from r-oracle.json")
    if j["expected"] != cc["expected"]:
        raise SystemExit(f"{cid}: Julia child's expected label differs from the P1 contract")
    julia_match = j["julia_label"] == cc["expected"]
    negatives = jres.get("negative_controls", {})
    verdict = "PASS" if (r["r_assertion_pass"] is True and julia_match and j["pass"] is True) else "FAIL"
    body = {
        "batch": "tools/core070_aghq_batch.R + .jl, GLLVM_PARITY_PIN=P1",
        "measures": "paired control on categorical labels: the same scalar aghq request normalized by R's "
                    ".gllvmTMB_normalize_aghq and by GLLVModels._aghq_request; no fit",
        "why_not_numeric": WHY_NOT_NUMERIC[CONTROL],
        "r_call": cc["r_call"], "r_assertion": cc["r_assertion"],
        "r_value": r["r_value"], "r_call_error": r["r_call_error"], "r_assertion_pass": r["r_assertion_pass"],
        "julia_call": cc["julia_call"], "julia_label": j["julia_label"], "expected_label": cc["expected"],
        "julia_label_matches_expected": julia_match, "harness_pass": j["pass"],
        "acceptance_rule": cc["acceptance_rule"],
        "negative_controls_behaved": {k: v.get("behaved") for k, v in sorted(negatives.items())},
        "discrimination_note": "the batch's three negative controls (a flipped expected label, a valid input "
                               "expected to error, an invalid input expected to return a value) must fail, and did "
                               "if every entry above is true",
        "batch_status": rec["status"], "gllvmtmb_version": rec["gllvmTMB_version"],
        "batch_verifier": verifier_block(CONTROL),
        "read_from": read_from(*(f"{d}/{n}" for n in ("receipt.json", "julia-results.json", "r-oracle.json",
                                                      "results.tsv", "run-commit.json", "verify.txt")),
                               CONTROL_CONTRACT),
        "raw": [f"{d}/r-oracle.json", f"{d}/julia-results.json"],
    }
    return verdict, body


def health_flags(obs):
    flags = []
    if isinstance(obs.get("convergence"), int) and obs["convergence"] != 0:
        flags.append(f"optimizer convergence code {obs['convergence']}")
    reason = obs.get("reason") or ""
    if "stalled" in reason:
        flags.append("AGHQ adaptation stalled (see reason)")
    return flags


def policy_case(cid):
    d = batch_rel(POLICY)
    rec = load(ROOT / d / "receipt.json")
    contract = load(ROOT / POLICY_CONTRACT)
    cc = next(c for c in contract["cases"] if c["case_id"] == cid)
    rid = cc["row_id"]
    c = rec["cases"][rid]
    obs, p0 = c["observed"], cc["p0_observed"]
    p0_diff = None
    if all(isinstance(x.get("objective"), (int, float)) and math.isfinite(x["objective"]) for x in (obs, p0)):
        p0_diff = abs(obs["objective"] - p0["objective"])
    verdict = "PASS" if c["pass"] is True else "FAIL"
    body = {
        "batch": "tools/core070_aghq_public_policy_bind.R, GLLVM_PARITY_PIN=P1",
        "measures": "R-side AGHQ policy contract on one toy public fit (or the gllvmTMBcontrol default); R only",
        "why_not_numeric": WHY_NOT_NUMERIC[POLICY],
        "julia_side": "none: the case defines no Julia call",
        "public_call": c.get("public_call", cc["public_call"]), "fixture": c.get("fixture"),
        "expected_k": c.get("expected_k"), "runner_detail": c["detail"], "assertion_pass": c["pass"],
        "observed_p1": obs,
        "health_flags": health_flags(obs),
        "p0_observed": p0,
        "p0_to_p1_objective_abs_diff": p0_diff,
        "p0_note": "P0 observation from the 2026-09-04 bind, which ran on a gllvmTMB twin branch that is neither pin "
                   "(see the contract's p0_r_engine_note); the objective difference is R against R across versions, "
                   "recorded for reference, not a comparison",
        "batch_status": rec["status"], "gllvmtmb_version": rec["gllvmTMB_version"],
        "batch_verifier": verifier_block(POLICY),
        "read_from": read_from(f"{d}/receipt.json", f"{d}/run-commit.json", f"{d}/verify.txt", POLICY_CONTRACT),
        "raw": [f"{d}/receipt.json"],
    }
    same = contract.get("same_observation", {})
    if rid in same:
        body["same_observation_as"] = f"aghq/{same[rid]}"
    elif rid in same.values():
        body["same_observation_as"] = ", ".join(f"aghq/{k}" for k, v in same.items() if v == rid)
    return verdict, body


def case_batch(cid):
    control_ids = {c["case_id"] for c in load(ROOT / CONTROL_CONTRACT)["cases"]}
    policy_ids = {c["case_id"] for c in load(ROOT / POLICY_CONTRACT)["cases"]}
    if cid in control_ids:
        return CONTROL
    if cid in policy_ids:
        return POLICY
    raise SystemExit(f"{cid}: no batch pays this case id")


def derive_case(cid):
    batch = case_batch(cid)
    if batch == CONTROL:
        verdict, body = control_case(cid)
        return "paired_control_categorical", verdict, body
    verdict, body = policy_case(cid)
    return "r_only_policy_observation", verdict, body


def derive_all(case_ids):
    return {cid: derive_case(cid) for cid in case_ids}


def in_scope_case_ids():
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    cm = load(ROOT / CASEMAP_REL)
    return sorted({cid for r in cm["rows"] for cid in p0[r["source_id"]]["executable_case_ids"]})


# ---------------------------------------------------------------------------
# case-map rows
# ---------------------------------------------------------------------------
def receipt_info(path, rec):
    bv = (rec.get("batch_verifier") or {}).get("status", "n/a")
    return (path, rec["evidence_kind"], rec["verdict"], bv)


def p0_evidence(base):
    ev = base.get("evidence") or {}
    if not ev:
        return {"recorded": False}
    if ev.get("receipt"):
        tracked = (ROOT / ev["receipt"]).is_file()
        return {"recorded": True, "batch": ev.get("bind"), "receipts": ev["receipt"],
                "preservation_sha256": ev.get("receipt_sha256"), "raw_available_in_repo_or_on_this_host": tracked,
                "note": "P0 receipt is tracked. It records an R-only observation made on a gllvmTMB twin branch "
                        f"({ev.get('r_engine_git_head')}) that is neither pin; no Julia call is part of the case."}
    return {"recorded": True, "batch": ev.get("batch"), "receipts": ev.get("receipts"),
            "preservation_sha256": ev.get("preservation_sha256"), "raw_available_in_repo_or_on_this_host": False,
            "note": "P0 receipts live under .unlazy/ (untracked; absent on this host). Only the P0 case map's record "
                    "remains."}


def p0_reading(base):
    """What #561's C8 rule reads for the unchanged P0 row (derived from the row, same rule as the checker)."""
    ev = base.get("evidence") or {}
    if ev.get("receipt") and base.get("executable_case_ids"):
        return ("REGISTRATION_ONLY_NOT_TWINNED: the P0 row cites evidence.receipt but carries no evidence_tier, "
                "and a missing tier reads as registration-only")
    return "NOT_TWINNED_NOT_SIGNED: the P0 row cites no evidence.receipt (its receipts sit under .unlazy/)"


TIER_TEXT = {
    "paired_control_categorical": "paired control on categorical labels, measured at P1; non-numeric, so it does "
                                  "not bind under the numeric rule",
    "r_only_policy": "R-only policy observation, measured at P1; no Julia side, so it does not bind",
}


TWIN_REL = f"{OUT_REL}/receipts/julia-twins/aghq"
TWIN_TIER = ("numeric: Julia values recomputed by tools/true_parity_julia_receipts.jl with the same calls and settings as "
             "test/test_aghq_p1_twin.jl, against R-at-P1 values copied from test/fixtures/aghq_p1/aghq_p1.toml "
             "(integration used and node count, logLik, intercepts, loadings and, for Gaussian, the residual SD, at each "
             "side's own optimum); both engines converged; each case within the tolerance asserted in the test")


def twin_overlay(sid, row, counts):
    """Add the numeric-twin evidence fields to `row` when a Julia twin receipt exists for it."""
    path = ROOT / TWIN_REL / f"{sid.split('/', 1)[1]}.json"
    if not path.is_file():
        return
    rec = load(path)
    rel = str(path.relative_to(ROOT))
    ok = (rec.get("schema") == "true-parity-julia-twin-receipt/v1" and rec.get("source_ids") == [sid]
          and rec.get("verdict") == "PASS" and rec.get("pin") == "P1" and rec.get("reference_commit") == P1_SHA
          and rec.get("evidence_kind") == "julia_recomputed_vs_recorded_r")
    if not ok:
        raise SystemExit(f"{rel}: not a passing P1 Julia twin receipt for {sid}")
    case_ids = [c["case_id"] for c in rec["comparison"]["cases"]]
    counts[row["evidence_tier"]] -= 1
    counts["numeric_pass"] += 1
    prior = row["executable_case_ids"]
    row["executable_case_ids"] = case_ids
    row["evidence_tier"] = "numeric"
    row["evidence"] = {"receipt": [rel], "non_binding_receipts": row["evidence"]["non_binding_receipts"],
                       "r_only_case_ids": prior, "tier": TWIN_TIER}
    row["measured_result"] = {**row["measured_result"], "twin_case_ids": case_ids, "twin_verdict": rec["verdict"]}


def build_rows(in_scope, carry_status, receipts):
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    counts = {k: 0 for k in COUNT_KEYS}
    out_rows = []
    for sid in in_scope:
        base = p0[sid]
        ids = base["executable_case_ids"]
        row = {"source_id": sid, "classification": base["classification"], "arc": "A3",
               "carry_scan_status": carry_status[sid], "executable_case_ids": ids,
               "disposition": base.get("disposition"),
               "p0_batch": (base.get("evidence") or {}).get("batch") or (base.get("evidence") or {}).get("bind"),
               "p0_evidence": p0_evidence(base), "p0_case_map_reading_561": p0_reading(base)}
        have = [receipts.get(i) for i in ids]
        if not ids or any(h is None for h in have):
            raise SystemExit(f"{sid}: a case id has no receipt")
        kinds = {h[1] for h in have}
        if len(kinds) != 1:
            raise SystemExit(f"{sid}: case ids of mixed evidence kinds {kinds}")
        kind = kinds.pop()
        prefix = "paired_control_categorical" if kind == "paired_control_categorical" else "r_only_policy"
        verdicts = {i: h[2] for i, h in zip(ids, have)}
        batch_ok = {i: h[3] for i, h in zip(ids, have)}
        paths = list(dict.fromkeys(h[0] for h in have))
        result = {"case_verdicts": verdicts, "batch_verifier": batch_ok}
        if not all(v == "PASS" for v in verdicts.values()):
            tier, verdict = f"{prefix}_fail", "FAIL"
        elif not all(v == "PASS" for v in batch_ok.values()):
            tier, verdict = f"{prefix}_held_batch_verifier_failed", None
        else:
            tier, verdict = f"{prefix}_pass", "PASS"
        row.update(evidence_tier=tier, measured_against=P1_SHA,
                   evidence={"non_binding_receipts": paths, "tier": TIER_TEXT[prefix]},
                   measured_result={**result, "row_verdict": verdict} if verdict else result)
        counts[tier] += 1
        twin_overlay(sid, row, counts)
        out_rows.append(row)
    return out_rows, counts


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
PROVENANCE_KEYS = {"pin", "reference_commit", "p0_reference_commit", "glvmodels_commit", "glvmodels_worktree_dirty",
                   "glvmodels_src_tree", "host", "schema", "case_id", "verdict", "evidence_kind"}


def check():
    problems = []
    tracked = {p.stem: (str(p.relative_to(ROOT)), load(p)) for p in sorted((ROOT / REC_REL / "cases").glob("*.json"))}
    for cid, (path, rec) in tracked.items():
        for rel, digest in (rec.get("read_from") or {}).items():
            if not (ROOT / rel).is_file():
                problems.append(f"{path}: read file {rel} is gone")
            elif sha(ROOT / rel) != digest:
                problems.append(f"{path}: read file {rel} changed (sha256 {sha(ROOT / rel)[:12]} != {digest[:12]})")
        if not rec.get("read_from"):
            problems.append(f"{path}: no read_from")
        if "comparison" in rec:
            problems.append(f"{path}: carries a comparison block, but no aghq case compares numbers")
        try:
            batch = case_batch(cid)
            if rec.get("glvmodels_commit") != run_commit(batch):
                problems.append(f"{path}: glvmodels_commit {rec.get('glvmodels_commit')} is not the run commit "
                                f"{run_commit(batch)} recorded in run-commit.json")
        except SystemExit as e:
            problems.append(f"{path}: {e}")
    for b in BATCHES:
        derived, _ = verify_text(b)
        text = (ROOT / batch_rel(b) / "verify.txt").read_text()
        if not text.endswith(derived):
            problems.append(f"{b}/verify.txt: derived checks differ from the re-derivation")
    try:
        ids = in_scope_case_ids()
        fresh = derive_all(ids)
    except (SystemExit, KeyError) as e:
        problems.append(f"re-derivation refused: {e}")
        fresh = {}
    for cid in set(tracked) - set(fresh):
        problems.append(f"{cid}: tracked receipt with no re-derived case")
    for cid, (kind, verdict, body) in fresh.items():
        if cid not in tracked:
            problems.append(f"{cid}: no tracked receipt")
            continue
        rec = tracked[cid][1]
        if (rec["evidence_kind"], rec["verdict"]) != (kind, verdict):
            problems.append(f"{cid}: kind/verdict {rec['evidence_kind']}/{rec['verdict']} != re-derived {kind}/{verdict}")
        if {k: v for k, v in rec.items() if k not in PROVENANCE_KEYS} != body:
            problems.append(f"{cid}: receipt body differs from the re-derivation")
        if rec.get("reference_commit") != P1_SHA or rec.get("pin") != "P1":
            problems.append(f"{cid}: receipt not pinned at P1")
    cm = load(ROOT / CASEMAP_REL)
    receipts = {cid: receipt_info(path, rec) for cid, (path, rec) in tracked.items()}
    rows = []
    try:
        rows, counts = build_rows([r["source_id"] for r in cm["rows"]],
                                  {r["source_id"]: r["carry_scan_status"] for r in cm["rows"]}, receipts)
        if rows != cm["rows"]:
            bad = [a["source_id"] for a, b in zip(rows, cm["rows"]) if a != b] or ["row count"]
            problems.append(f"case-map rows differ from the re-derivation: {', '.join(bad)}")
        if counts != cm["counts"]:
            problems.append(f"case-map counts {cm['counts']} != re-derived {counts}")
        if cm.get("numeric_rows") != [r["source_id"] for r in rows if r["evidence_tier"].startswith("numeric")]:
            problems.append("case-map numeric_rows differ from the re-derivation")
        verifiers = {b: verifier_block(b) for b in BATCHES}
        if verifiers != cm["batch_verifiers"]:
            problems.append("case-map batch_verifiers differ from the re-derivation")
        if cm.get("glvmodels_commit") not in {run_commit(b) for b in BATCHES}:
            problems.append("case-map glvmodels_commit is not a run commit")
    except (SystemExit, KeyError) as e:
        problems.append(f"case-map re-derivation refused: {e}")
    if problems:
        print("STALE\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("CORE070_AGHQ_P1_RECEIPTS_CURRENT", len(tracked), "case receipts,", len(rows), "rows")


def apply_twins():
    """Re-derive rows, counts, numeric_rows and note of case-map-aghq.json from the tracked receipts
    (batch receipts plus any Julia twin receipts). Nothing else in the file changes."""
    tracked = {p.stem: (str(p.relative_to(ROOT)), load(p)) for p in sorted((ROOT / REC_REL / "cases").glob("*.json"))}
    receipts = {cid: receipt_info(path, rec) for cid, (path, rec) in tracked.items()}
    cm = load(ROOT / CASEMAP_REL)
    rows, counts = build_rows([r["source_id"] for r in cm["rows"]],
                              {r["source_id"]: r["carry_scan_status"] for r in cm["rows"]}, receipts)
    cm["rows"], cm["counts"] = rows, counts
    cm["numeric_rows"] = [r["source_id"] for r in rows if r["evidence_tier"].startswith("numeric")]
    cm["note"] = NOTE
    write_json(ROOT / CASEMAP_REL, cm)
    print(json.dumps(counts))
    print("numeric rows", len(cm["numeric_rows"]))


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
COPY = {
    CONTROL: ["receipt.json", "results.tsv", "julia-results.json", "r-oracle.json", "run-commit.json"],
    POLICY: ["receipt.json", "run-commit.json"],
}
NOTE = ("Separate from case-map.json so none of its rows are touched; read by tools/true_parity_check.mjs with "
        "PARITY_CASEMAP pointing at this file. Classification and disposition are carried from "
        "docs/dev-log/core070/required-source-case-map.json unchanged; nothing is signed by an agent. The two batches measure no "
        "number against Julia: 7 rows are a paired control on categorical labels (no fit) and 14 are R-only policy "
        "observations (no Julia side), so a row paid by a batch alone cites its receipts under "
        "evidence.non_binding_receipts and does not bind under the numeric rule. Eleven of the 14 policy rows (Poisson, NB2, "
        "binomial and Gaussian fits, which expose aghq= in Julia) additionally carry a numeric Julia twin receipt "
        "(receipts/julia-twins/aghq/, test/test_aghq_p1_twin.jl) under evidence.receipt with evidence_tier numeric. The "
        "policy fixtures of the batch are toys (p of 5 to 20 traits, n of 30 to 40 sites, d = 1); the twins use simulated "
        "data with a real latent factor.")


def copy_batch(batch, run_dir):
    dest = ROOT / batch_rel(batch)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for n in COPY[batch]:
        p = run_dir / n
        if not p.is_file():
            raise SystemExit(f"missing batch artifact {p}")
        shutil.copyfile(p, dest / n)
        out.append(f"{batch_rel(batch)}/{n}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--runtimes", type=Path)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="write receipts from a checkout with modified tracked files (recorded, not hidden)")
    ap.add_argument("--check", action="store_true",
                    help="verify the tracked receipts against the files they read; write nothing")
    ap.add_argument("--apply-twins", action="store_true",
                    help="re-derive the case-map rows from the tracked receipts and Julia twin receipts")
    args = ap.parse_args()
    if args.check:
        check()
        return
    if args.apply_twins:
        apply_twins()
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
    shutil.rmtree(ROOT / REC_REL, ignore_errors=True)  # stale receipts from an earlier write must not survive
    artifacts = {}
    for b in BATCHES:
        artifacts[b] = copy_batch(b, runs / b)
        external = run_external(b, runs / b) if b in EXTERNAL else ""
        derived, _ = verify_text(b)
        (ROOT / batch_rel(b) / "verify.txt").write_text(external + (SEPARATOR if external else "") + derived)
        artifacts[b].append(f"{batch_rel(b)}/verify.txt")
    carry_path = runs / "carry-scan-p1.json"
    carry = load(carry_path)
    carry_status = {r["source_id"]: r["status"] for r in carry["rows"]
                    if r["source_id"].startswith("aghq/") and r["status"] in IN_SCOPE_STATUS}
    p0 = {r["source_id"]: r for r in load(P0_CASEMAP)["rows"]}
    in_scope = [s for s in carry_status if p0[s]["classification"] in ("required_core", "compatibility_adapter")]
    ids = sorted({cid for sid in in_scope for cid in p0[sid]["executable_case_ids"]})
    common = {"pin": "P1", "reference_commit": P1_SHA, "p0_reference_commit": P0_SHA,
              "glvmodels_commit": head, "glvmodels_worktree_dirty": dirty,
              "glvmodels_src_tree": git("rev-parse", f"{head}:src").stdout.strip(), "host": HOST}
    receipts = {}
    for cid, (kind, verdict, body) in derive_all(ids).items():
        rec = {"schema": "core070-aghq-p1-case-receipt/v1", "case_id": cid, "verdict": verdict,
               "evidence_kind": kind, **body, **common}
        path = ROOT / REC_REL / "cases" / f"{cid}.json"
        write_json(path, rec)
        receipts[cid] = receipt_info(str(path.relative_to(ROOT)), rec)
    rows, counts = build_rows(in_scope, carry_status, receipts)
    by_status = {s: sum(carry_status[r] == s for r in in_scope) for s in IN_SCOPE_STATUS}
    by_class = {c: sum(p0[r]["classification"] == c for r in in_scope) for c in ("required_core",
                                                                                 "compatibility_adapter")}
    write_json(ROOT / CASEMAP_REL, {
        "schema": 1, "reference_commit": P1_SHA,
        "scope": (f"aghq family: the {len(in_scope)} required rows ({by_class['required_core']} required_core, plus "
                  f"{by_class['compatibility_adapter']} compatibility_adapter rows, which tools/true_parity_check.mjs "
                  f"counts as required) the P1 carry scan lists as "
                  f"{' or '.join(f'{s} ({n})' for s, n in by_status.items() if n)}. The rejected AGHQ-INVALID-* rows "
                  f"and the intentionally_excluded rows are out of scope."),
        "note": NOTE, "generator": "tools/core070_aghq_p1_receipts.py", "glvmodels_commit": head,
        "carry_scan": {"source": CARRY_SCAN_SOURCE, "sha256": sha(carry_path)},
        "numeric_rows": [r["source_id"] for r in rows if r["evidence_tier"].startswith("numeric")],
        "why_no_numeric_row": {"paired_control_categorical": WHY_NOT_NUMERIC[CONTROL],
                               "r_only_policy": WHY_NOT_NUMERIC[POLICY]},
        "batch_verifiers": {b: verifier_block(b) for b in BATCHES}, "counts": counts,
        "p0_evidence_summary": {
            "rows_with_p0_evidence_record": sum(r["p0_evidence"]["recorded"] for r in rows),
            "rows_with_tracked_p0_receipt": sum(r["p0_evidence"].get("raw_available_in_repo_or_on_this_host", False)
                                                for r in rows),
            "p0_policy_receipt": POLICY_P0_RECEIPT},
        "runtimes_seconds": runtimes, "batch_artifacts": artifacts, "rows": rows})
    print(json.dumps(counts))
    print("case receipts", len(receipts))


if __name__ == "__main__":
    main()

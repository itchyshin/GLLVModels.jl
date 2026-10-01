"""Write tracked P1 receipts and case-map rows for the isdm family.

In scope: the 20 required isdm rows the P1 carry scan lists as DANGLING. One
batch pays all 20, run at gllvmTMB pin P1 (GLLVM_PARITY_PIN=P1):

  isdm (wave1, 20 rows)   R: tools/core070_isdm_batch.R loads nine pinned P1 R functions
                          (R/isdm-sources.R, R/fit-multi.R, R/offset.R) from the P1 source
                          tree and evaluates each case's admission predicate on the frozen
                          fixture (test/parity/fixtures/core070_isdm_admission.R); each
                          expression must be TRUE. Seven are rejection cases written as
                          !admitted(...). No fit, no number, no Julia call.
                          Julia: none in this batch. At the run commit GLLVModels' src/ has
                          no iSDM code (recorded from `git grep -il isdm <commit> -- src`).

So no isdm row can bind here: R SIDE ONLY, NOTHING NUMERIC TO COMPARE. Draft PRs
#546 / #558 build the iSDM surface and a Julia-native twin of 19 of these
predicates on another branch; whether they should pay these rows is a proposal
for the maintainer, recorded in the PR body, not a rebinding made here.

Writes, under docs/dev-log/core070/true-parity-latest/:

  receipts/isdm/isdm-p1/...        batch artifacts copied verbatim (JSON/TSV only, no logs),
                                   run-commit.json, verify.txt
  receipts/isdm/cases/<id>.json    one receipt per executable case id
  case-map-isdm.json               the 20 isdm rows, P0 classification and disposition kept

Shared gates (PR #567 / #569 / #571 / #579 / #584):

  * Batch verifier. This tool runs the batch verifier at P1 (with --self-test)
    and keeps its full output as the tracked isdm-p1/verify.txt. Every case
    receipt carries a batch_verifier block. A row whose batch verifier did not
    pass is held (measured_held_batch_verifier_failed). There is no exception
    path. The degenerate-comparison rule has nothing to act on: no case carries a
    comparison block.
  * Provenance. Every receipt records glvmodels_commit = HEAD. The tool refuses to
    write when tracked files outside its own outputs are modified (unless
    --allow-dirty, which is then recorded), and refuses unless the run directory
    holds a run-commit.json naming this HEAD with an empty dirty list. --check
    compares each receipt's glvmodels_commit, glvmodels_worktree_dirty and
    glvmodels_src_tree (and the case map's glvmodels_commit) with the tracked
    run-commit.json.
  * Read-file hashes. Every case receipt records `read_from`, the sha256 of each
    tracked file it was derived from. --check re-hashes every read_from file,
    re-derives every case receipt and every case-map row in memory, and exits
    nonzero on any difference.

Usage:
  python3 tools/core070_isdm_p1_receipts.py --runs DIR --runtimes JSON [--allow-dirty]
  python3 tools/core070_isdm_p1_receipts.py --check
where DIR holds isdm-p1/ (the R run plus run-commit.json) and carry-scan-p1.json.
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
OUT_REL = "docs/dev-log/core070/true-parity-latest"
P0_CASEMAP = ROOT / "docs/dev-log/core070/required-source-case-map.json"
PINS = tomllib.loads((ROOT / "tools/core070_oracle_pins.toml").read_text())
P1_SHA = PINS["P1"]["reference_commit"]
P0_SHA = PINS["P0"]["reference_commit"]
ORACLE_BUILD = f"{OUT_REL}/receipts/covariance/oracle/build.json"
ORACLE_SOURCE = f"{OUT_REL}/receipts/covariance/oracle/source.json"
HOST = "local Mac (M1 Ultra), OPENBLAS/OMP threads 1, JULIA_NUM_THREADS=4"
FAMILY = "isdm"
BATCH = "isdm-p1"
REC_REL = f"{OUT_REL}/receipts/{FAMILY}"
CASEMAP_REL = f"{OUT_REL}/case-map-{FAMILY}.json"
CONTRACT = f"{OUT_REL}/isdm-batch-contract-p1.json"
VERIFIER_ARGV = ["tools/core070_verify_isdm_batch.py", "--state", "{state}", "--self-test"]
VERIFIER_MARKER = "CORE070_ISDM_BATCH_VERIFIED"
ARTIFACTS = ["receipt.json", "isdm-batch-results.json", "raw.tsv", "run-commit.json"]

TIER_TEXT = {
    "needs_surface_r_side_measured": (
        "R side measured at P1: the pinned P1 R admission predicate replays to its frozen TRUE expectation on the "
        "frozen fixture. That is a boolean predicate replay, not a fit: it produces no number, and GLLVModels at the "
        "run commit has no iSDM surface, so there is no R-vs-Julia comparison and the row does not bind"),
    "measured_held_batch_verifier_failed": (
        "R side replayed at P1, but the batch verifier rejected the run, so the row does not bind"),
    "measured_fail": "measured at P1; the R predicate did not replay to its expectation, so the row does not bind",
}
COUNT_KEYS = ("numeric_pass", "numeric_fail", "numeric_held_batch_verifier_failed", "numeric_non_discriminating",
              "needs_surface_r_side_measured", "measured_held_batch_verifier_failed", "measured_fail",
              "not_measured")
LEGACY_NOTE = ("ISDM-LEGACY replays R's legacy two-source route (family_var = \"isdm_family\", fixed names gbif / "
               "survey_pa). Packet 1b item 2 (signed 2026-09-27, vault decision D-296) records it as an R-only "
               "backward-compatibility disposition that GLLVModels will not twin. That disposition is not written "
               "into this row (dispositions are the maintainer's to record in the case map); the row stays "
               "required and free.")


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
    p = run_dir / "run-commit.json"
    if not p.is_file():
        raise SystemExit(f"{run_dir} has no run-commit.json; re-run the batch from a clean commit")
    rc = load(p)
    if rc.get("glvmodels_commit") != head or rc.get("dirty") != []:
        raise SystemExit(f"{run_dir}: run at {rc.get('glvmodels_commit')} dirty={rc.get('dirty')}, "
                         f"not at clean HEAD {head}; re-run at HEAD")


def run_verifier(state):
    """Run the batch verifier at P1 on the raw run and keep its full output as the tracked verify.txt."""
    argv = ["python3"] + [a.format(state=str(state)) for a in VERIFIER_ARGV]
    proc = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, GLLVM_PARITY_PIN="P1"))
    log = ROOT / REC_REL / BATCH / "verify.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"$ GLLVM_PARITY_PIN=P1 {' '.join(VERIFIER_ARGV)}\n# exit code {proc.returncode}\n"
                   + proc.stdout + proc.stderr)
    return {"exit_code": proc.returncode,
            "status": "PASS" if proc.returncode == 0 and VERIFIER_MARKER in proc.stdout else "FAIL"}


def verifier_block():
    """The batch_verifier block, derived from the tracked verify.txt (so --check can re-derive it)."""
    rel = f"{REC_REL}/{BATCH}/verify.txt"
    text = (ROOT / rel).read_text()
    ok = "# exit code 0\n" in text and VERIFIER_MARKER in text
    return {"tool": VERIFIER_ARGV[0], "argv": " ".join(VERIFIER_ARGV), "status": "PASS" if ok else "FAIL",
            "accept_marker": VERIFIER_MARKER, "log": rel}


def copy_batch(run_dir):
    dest = ROOT / REC_REL / BATCH
    dest.mkdir(parents=True, exist_ok=True)
    for n in ARTIFACTS:
        p = run_dir / n
        if not p.is_file():
            raise SystemExit(f"missing batch artifact {p}")
        shutil.copyfile(p, dest / n)
    return [f"{REC_REL}/{BATCH}/{n}" for n in ARTIFACTS]


def read_from(*rels):
    return {rel: sha(ROOT / rel) for rel in rels}


def julia_surface_at(commit):
    """Files under src/ at `commit` that mention isdm (case-insensitive), from git; [] means no iSDM code."""
    out = git("grep", "-il", "isdm", commit, "--", "src", check=False)
    if out.returncode not in (0, 1):
        raise SystemExit(f"git grep failed at {commit}: {out.stderr.strip()}")
    return sorted(line.split(":", 1)[1] for line in out.stdout.splitlines())


# ---------------------------------------------------------------------------
# Derivation: case receipts from tracked files only.
# ---------------------------------------------------------------------------
def isdm_cases():
    """case_id -> (evidence_kind, verdict, body) for the isdm batch."""
    d = f"{REC_REL}/{BATCH}"
    receipt, results = load(ROOT / d / "receipt.json"), load(ROOT / d / "isdm-batch-results.json")
    contract = load(ROOT / CONTRACT)
    run_commit = load(ROOT / d / "run-commit.json")["glvmodels_commit"]
    if receipt["reference_commit"] != P1_SHA or results["reference_commit"] != P1_SHA:
        raise SystemExit("isdm batch receipt or results are not pinned at P1")
    if receipt["results_sha256"] != sha(ROOT / d / "isdm-batch-results.json"):
        raise SystemExit("isdm-batch-results.json does not match its receipt's results_sha256")
    if receipt["raw_sha256"] != sha(ROOT / d / "raw.tsv"):
        raise SystemExit("raw.tsv does not match its receipt's raw_sha256")
    if receipt["contract_sha256"] != sha(ROOT / CONTRACT):
        raise SystemExit("isdm batch receipt does not name the tracked P1 twin's sha256")
    if receipt["source_pins"] != contract["source_pins"]:
        raise SystemExit("isdm batch receipt source pins differ from the P1 twin")
    verifier = verifier_block()
    reads = read_from(f"{d}/receipt.json", f"{d}/isdm-batch-results.json", f"{d}/raw.tsv",
                      f"{d}/run-commit.json", f"{d}/verify.txt", CONTRACT)
    raw = dict(line.split("\t") for line in (ROOT / d / "raw.tsv").read_text().splitlines())
    src_hits = julia_surface_at(run_commit)
    ccases = {c["manifest_case_id"]: c for c in contract["cases"]}
    if [c["manifest_case_id"] for c in results["cases"]] != list(ccases):
        raise SystemExit("isdm results case list differs from the P1 twin")
    out = {}
    for rc in results["cases"]:
        cid = rc["manifest_case_id"]
        cc = ccases[cid]
        ok = rc["ok"] is True and rc["actual"] is True and raw.get(cid) == "PASS"
        body = {"source_ids": [cc["source_id"]],
                "batch": "tools/core070_isdm_batch.R (P1 source tree), GLLVM_PARITY_PIN=P1",
                "batch_status": receipt["status"], "batch_verifier": verifier,
                "gllvmtmb_version": receipt["gllvmTMB_version"],
                "admission_case_id": cc["admission_case_id"], "stage": cc["stage"],
                "r_expression": cc["expression"], "r_expected": cc["expected"],
                "r_actual": rc["actual"], "r_ok": rc["ok"],
                "rejection_case": cc["negative_control"],
                "julia_side": {
                    "executed": False,
                    "src_files_mentioning_isdm_at_run_commit": src_hits,
                    "contract_status": cc["julia_status"],
                },
                "why_not_numeric": (
                    "The R side evaluates a pinned P1 admission predicate to an exact TRUE expectation on the "
                    "frozen fixture. It produces a boolean, not a fit number, and this batch runs no Julia side: "
                    + ("GLLVModels' src/ at the run commit has no iSDM code. " if not src_hits else
                       "note that src/ at the run commit does mention isdm (" + ", ".join(src_hits) + "); the "
                       "contract's Julia status is then stale. ")
                    + "A Julia-native twin of this predicate exists only on draft PR #546's branch "
                    "(test/parity/isdm_cases.jl, P1-ISDM-ADMISSION-20)"
                    + (", and none for this case: ISDM-LEGACY is R-only (D-296)."
                       if cc["admission_case_id"] == "ISDM-LEGACY" else ".")),
                "read_from": reads,
                "raw": [f"{d}/isdm-batch-results.json", f"{d}/raw.tsv"]}
        if cc["negative_control"]:
            body["rejection_case_note"] = ("One of the contract's seven rejection cases: the expression is "
                                           "!admitted(...) (or equivalent), so TRUE means R refused the input.")
        if cc["admission_case_id"] == "ISDM-LEGACY":
            body["note"] = LEGACY_NOTE
        out[cid] = ("r_replay_julia_surface_absent", "PASS" if ok else "FAIL", body)
    return out


# ---------------------------------------------------------------------------
# case-map rows
# ---------------------------------------------------------------------------
def receipt_info(path, rec):
    return (path, rec["evidence_kind"], rec["verdict"], rec["batch_verifier"]["status"], rec.get("note"))


def p0_evidence(base):
    ev = base.get("evidence") or {}
    if not ev:
        return {"recorded": False}
    return {"recorded": True, "batch": ev.get("batch"), "receipts": ev.get("receipts"),
            "preservation_sha256": ev.get("preservation_sha256"),
            "raw_available_in_repo_or_on_this_host": False,
            "note": "P0 receipts live under .unlazy/ (untracked; absent on this host). Only the P0 case map's record "
                    "(batch name, preservation sha256) remains."}


WIRED_PREFIXES = (f"{OUT_REL}/receipts/julia-twins/", f"{OUT_REL}/receipts/fixture-twins/")


def wired_overlay(row, committed):
    """A row later rewired to a Julia-twin or fixture-twin receipt (a numeric R-vs-Julia comparison) keeps
    those evidence fields. Everything this tool derives stays derived and is still compared; the R predicate
    case ids and receipts move to r_predicate_case_ids / non_binding_receipts. Returns True if applied."""
    ev = (committed or {}).get("evidence") or {}
    paths = ev.get("receipt")
    if not isinstance(paths, list) or not paths or not all(isinstance(x, str) and x.startswith(WIRED_PREFIXES) for x in paths):
        return False
    if committed.get("evidence_tier") != "numeric" or not committed.get("executable_case_ids"):
        return False
    row["r_predicate_case_ids"] = row["executable_case_ids"]
    row["executable_case_ids"] = committed["executable_case_ids"]
    row["evidence_tier"] = "numeric"
    row["measured_against"] = committed.get("measured_against")
    row["evidence"] = {**row["evidence"], "receipt": paths, "tier": ev.get("tier")}
    return True


def build_rows(in_scope, receipts, committed_rows=None):
    committed_by_id = {r["source_id"]: r for r in (committed_rows or [])}
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
        if kinds != {"r_replay_julia_surface_absent"}:
            raise SystemExit(f"{sid}: no tier rule for case kinds {sorted(kinds)}")
        verdicts = {i: h[2] for i, h in zip(ids, have)}
        batch_ok = {i: h[3] for i, h in zip(ids, have)}
        paths = [h[0] for h in have]
        if not all(v == "PASS" for v in verdicts.values()):
            tier = "measured_fail"
        elif not all(v == "PASS" for v in batch_ok.values()):
            tier = "measured_held_batch_verifier_failed"
        else:
            tier = "needs_surface_r_side_measured"
        row.update(evidence_tier=tier, measured_against=P1_SHA,
                   evidence={"non_binding_receipts": paths, "tier": TIER_TEXT[tier]},
                   measured_result={"case_verdicts": verdicts, "batch_verifier": batch_ok,
                                    "case_kinds": {i: h[1] for i, h in zip(ids, have)}})
        counts[tier] += 1
        if wired_overlay(row, committed_by_id.get(sid)):
            counts[tier] -= 1
            counts["numeric_pass"] += 1
        notes = [h[4] for h in have if h[4]]
        if notes:
            row["note"] = " ".join(dict.fromkeys(notes))
        out_rows.append(row)
    return out_rows, counts


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
PROVENANCE_KEYS = {"pin", "reference_commit", "p0_reference_commit", "oracle_build_receipt", "oracle_source_receipt",
                   "glvmodels_commit", "glvmodels_worktree_dirty", "glvmodels_src_tree", "host", "schema", "case_id",
                   "verdict", "evidence_kind"}


def provenance_problems(tracked, run_commit):
    problems = []
    src_tree = git("rev-parse", f"{run_commit}:src", check=False).stdout.strip()
    for cid, (path, rec) in tracked.items():
        if rec.get("glvmodels_commit") != run_commit:
            problems.append(f"{path}: glvmodels_commit {rec.get('glvmodels_commit')} != run-commit.json {run_commit}")
            continue
        if rec.get("glvmodels_worktree_dirty") != []:
            problems.append(f"{path}: written from a dirty tree")
        if rec.get("glvmodels_src_tree") != src_tree:
            problems.append(f"{path}: glvmodels_src_tree {rec.get('glvmodels_src_tree')} != {run_commit}:src "
                            f"{src_tree or '(commit not found)'}")
    return problems


def check():
    problems = []
    tracked = {p.stem: (str(p.relative_to(ROOT)), load(p)) for p in sorted((ROOT / REC_REL / "cases").glob("*.json"))}
    rc = load(ROOT / REC_REL / BATCH / "run-commit.json")
    if rc.get("dirty") != []:
        problems.append(f"{REC_REL}/{BATCH}/run-commit.json: run from a dirty tree")
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
        if "comparison" in rec:
            problems.append(f"{path}: carries a comparison block, but this batch compares no numbers")
    try:
        fresh = isdm_cases()
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
    problems += provenance_problems(tracked, rc.get("glvmodels_commit"))
    receipts = {cid: receipt_info(path, rec) for cid, (path, rec) in tracked.items()}
    cm = load(ROOT / CASEMAP_REL)
    n_rows = 0
    try:
        rows, counts = build_rows([r["source_id"] for r in cm["rows"]], receipts, cm["rows"])
        n_rows = len(rows)
        if rows != cm["rows"]:
            bad = [a["source_id"] for a, b in zip(rows, cm["rows"]) if a != b] or ["row count"]
            problems.append(f"case-map rows differ from the re-derivation: {', '.join(bad)}")
        if counts != cm["counts"]:
            problems.append(f"case-map counts {cm['counts']} != re-derived {counts}")
    except (SystemExit, KeyError) as e:
        problems.append(f"case-map re-derivation refused: {e}")
    if cm.get("glvmodels_commit") != rc.get("glvmodels_commit"):
        problems.append(f"case-map glvmodels_commit {cm.get('glvmodels_commit')} != run-commit.json "
                        f"{rc.get('glvmodels_commit')}")
    if cm.get("batch_verifiers", {}).get(BATCH, {}).get("status") != verifier_block()["status"]:
        problems.append("case-map batch_verifiers status differs from the tracked verify.txt")
    if problems:
        print("STALE\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("CORE070_ISDM_P1_RECEIPTS_CURRENT", len(tracked), "case receipts,", n_rows, "rows")


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
SCOPE = ("isdm family: the 20 required rows the P1 carry scan lists as DANGLING, all paid by the wave1 isdm "
         "admission batch (R admission-predicate replay against the pinned P1 source; no fit, no Julia side). The 17 "
         "rejected isdm rows and the two NOT_BOUND_AT_P0 namespace exports (isdm_source, isdm_sources) are out of "
         "scope.")
NOTE = ("Separate from case-map.json so none of its rows are touched; read by tools/true_parity_check.mjs with "
        "PARITY_CASEMAP pointing at this file. Classification and disposition are carried from "
        "docs/dev-log/core070/required-source-case-map.json unchanged; nothing is signed by an agent. The R side of "
        "every isdm row is a boolean predicate replay with no fit number and there is no Julia side at the run "
        "commit, so every row cites evidence.non_binding_receipts and is free. Draft PR #546's P1 iSDM twins are "
        "not used here; mapping them onto these rows is a proposal for the maintainer (PR body).")


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
    check_run_commit(runs / BATCH, head)
    shutil.rmtree(ROOT / REC_REL / "cases", ignore_errors=True)  # stale case receipts must not survive
    artifacts = copy_batch(runs / BATCH)
    verifier = {"tool": VERIFIER_ARGV[0], "argv": " ".join(VERIFIER_ARGV), "accept_marker": VERIFIER_MARKER,
                "log": f"{REC_REL}/{BATCH}/verify.txt", **run_verifier(runs / BATCH)}
    artifacts.append(verifier["log"])
    common = {"pin": "P1", "reference_commit": P1_SHA, "p0_reference_commit": P0_SHA,
              "oracle_build_receipt": ORACLE_BUILD, "oracle_source_receipt": ORACLE_SOURCE,
              "glvmodels_commit": head, "glvmodels_worktree_dirty": dirty,
              "glvmodels_src_tree": git("rev-parse", f"{head}:src").stdout.strip(), "host": HOST}
    receipts = {}
    for cid, (kind, verdict, body) in isdm_cases().items():
        rec = {"schema": "core070-isdm-p1-case-receipt/v1", "case_id": cid, "verdict": verdict,
               "evidence_kind": kind, **body, **common}
        path = ROOT / REC_REL / "cases" / f"{cid}.json"
        write_json(path, rec)
        receipts[cid] = receipt_info(str(path.relative_to(ROOT)), rec)
    carry = load(runs / "carry-scan-p1.json")
    in_scope = [r["source_id"] for r in carry["rows"]
                if r["source_id"].startswith(FAMILY + "/") and r["status"] == "DANGLING"]
    old_map = ROOT / CASEMAP_REL
    rows, counts = build_rows(in_scope, receipts, load(old_map)["rows"] if old_map.is_file() else None)
    write_json(ROOT / CASEMAP_REL, {
        "schema": 1, "reference_commit": P1_SHA, "scope": SCOPE, "note": NOTE,
        "generator": "tools/core070_isdm_p1_receipts.py", "glvmodels_commit": head,
        "batch_verifiers": {BATCH: verifier}, "counts": counts,
        "p0_evidence_summary": {
            "rows_with_p0_evidence_record": sum(r["p0_evidence"]["recorded"] for r in rows),
            "rows_without_p0_evidence_record": sum(not r["p0_evidence"]["recorded"] for r in rows),
            "p0_raw_receipts_available_here": False},
        "runtimes_seconds": runtimes.get(BATCH), "batch_artifacts": {BATCH: artifacts}, "rows": rows})
    print(FAMILY, json.dumps(counts))
    print("case receipts", len(receipts))


if __name__ == "__main__":
    main()

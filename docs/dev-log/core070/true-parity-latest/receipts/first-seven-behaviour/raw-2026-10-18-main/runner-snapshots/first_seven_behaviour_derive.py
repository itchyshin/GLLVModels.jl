#!/usr/bin/env python3
"""Fail-closed derivation for the seven P1 behavioural checkpoint rows.

The script consumes raw R and Julia TSV outputs plus run.json written after
both processes finish. It never runs a fit or assigns a result from a missing
row. --self-test uses synthetic raw labels to test both a supported match and
the known EXTRA-SOURCE mismatch (R's family-length guard).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
# Preserve the source loaded for this process. --finalize must attest the
# deriver that actually ran, which may include an uncommitted candidate fix.
EXECUTED_DERIVER_SOURCE = Path(__file__).resolve().read_bytes()
BASE = ROOT / "docs/dev-log/core070/true-parity-latest/receipts/first-seven-behaviour"
PIN = "9539352f66f2db2cc26b1c393e67212a359b60c9"
R_SOURCE = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/source.json"
R_BUILD = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build-totoro.json"
CASES = {
    "CORE070-FIRST7-CHECK-AUTO-RESIDUAL": "postfit/POSTFIT-SURFACE-check_auto_residual",
    "CORE070-FIRST7-ISDM-COUNT": "isdm/ISDM-COUNT",
    "CORE070-FIRST7-ISDM-EXTRA-SOURCE": "isdm/ISDM-EXTRA-SOURCE",
    "CORE070-FIRST7-ISDM-MISSING-IN-TRAIT": "isdm/ISDM-MISSING-IN-TRAIT",
    "CORE070-FIRST7-ISDM-MISSING-SOURCE": "isdm/ISDM-MISSING-SOURCE",
    "CORE070-FIRST7-ISDM-WRAPPER-LAW": "isdm/ISDM-WRAPPER-LAW",
}
FROZEN_CASES = {
    "CORE070-FIRST7-CHECK-AUTO-RESIDUAL": "CORE070-WAVE7-CHECK-AUTO-RESIDUAL",
    "CORE070-FIRST7-ISDM-COUNT": "CORE070-ISDM-COUNT-PAIRED-CONTROL",
    "CORE070-FIRST7-ISDM-EXTRA-SOURCE": "CORE070-ISDM-EXTRA-SOURCE-PAIRED-CONTROL",
    "CORE070-FIRST7-ISDM-MISSING-IN-TRAIT": "CORE070-ISDM-MISSING-IN-TRAIT-PAIRED-CONTROL",
    "CORE070-FIRST7-ISDM-MISSING-SOURCE": "CORE070-ISDM-MISSING-SOURCE-PAIRED-CONTROL",
    "CORE070-FIRST7-ISDM-WRAPPER-LAW": "CORE070-ISDM-WRAPPER-LAW-PAIRED-CONTROL",
}
SIGNED_SCOPE = set(CASES.values())
EXPECTED = {
    "CORE070-FIRST7-CHECK-AUTO-RESIDUAL": "residual-check:coherent",
    "CORE070-FIRST7-ISDM-COUNT": "all-count:nonmixed-admitted",
    "CORE070-FIRST7-ISDM-EXTRA-SOURCE": "guard:family-length",
    "CORE070-FIRST7-ISDM-MISSING-IN-TRAIT": "guard:family-scale-per-trait",
    "CORE070-FIRST7-ISDM-MISSING-SOURCE": "guard:family-length",
    "CORE070-FIRST7-ISDM-WRAPPER-LAW": "guard:wrapper-law-refusal",
}
CALL_FRAGMENTS = {
    "CORE070-FIRST7-CHECK-AUTO-RESIDUAL": "check_auto_residual",
    "CORE070-FIRST7-ISDM-COUNT": "isdm_sources",
    "CORE070-FIRST7-ISDM-EXTRA-SOURCE": "unknown",
    "CORE070-FIRST7-ISDM-MISSING-IN-TRAIT": "missing_trait",
    "CORE070-FIRST7-ISDM-MISSING-SOURCE": "missing_source",
    "CORE070-FIRST7-ISDM-WRAPPER-LAW": "isdm_sources",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_artifact(meta, relative_path):
    raw_root = (ROOT / meta["raw_directory"]).resolve()
    path = (raw_root / relative_path).resolve()
    if raw_root not in path.parents:
        raise ValueError("raw provenance artifact escapes its receipt directory")
    return path


def retained_runner_hashes(meta):
    snapshots = meta.get("runner_snapshots", {})
    expected = {}
    for key in ("R", "Julia", "derive"):
        item = snapshots.get(key, {})
        path = raw_artifact(meta, item.get("path", ""))
        actual = digest(path)
        if actual != item.get("sha256") or actual != meta.get("runner_sha256", {}).get(key):
            raise ValueError(f"retained {key} runner snapshot is missing or has the wrong digest")
        expected[key] = actual
    return expected


def snapshot_runners(snapshots_dir, commit):
    """Retain exact runner bytes, including this loaded deriver source."""
    snapshots_dir = Path(snapshots_dir)
    snapshots_dir.mkdir(exist_ok=True)
    snapshots = {}
    for key, name in (("R", "first_seven_behaviour_R.R"),
                      ("Julia", "first_seven_behaviour_J.jl"),
                      ("derive", "first_seven_behaviour_derive.py")):
        if key == "derive":
            data = EXECUTED_DERIVER_SOURCE
        else:
            data = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{commit}:tools/{name}"],
                check=True, capture_output=True).stdout
        (snapshots_dir / name).write_bytes(data)
        snapshots[key] = {"path": f"runner-snapshots/{name}",
                          "sha256": hashlib.sha256(data).hexdigest()}
    return snapshots


def matching_fixture_hashes(rrows, jrows):
    if set(rrows) != set(CASES) or set(jrows) != set(CASES):
        return False
    hashes = set()
    for case_id in CASES:
        rh = rrows[case_id].get("fixture_sha256", "")
        jh = jrows[case_id].get("fixture_sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", rh) or rh != jh:
            return False
        hashes.add(rh)
    return len(hashes) == len(CASES)


def tsv(path, engine):
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != len(CASES) or {r.get("case_id") for r in rows} != set(CASES):
        raise ValueError(f"{path}: expected exactly the six frozen case IDs")
    for r in rows:
        if r.get("engine") not in (None, "", engine):
            raise ValueError(f"{path}: wrong engine for {r['case_id']}")
        if r.get("outcome") not in {"RETURN", "ERROR"} or not r.get("class") or not r.get("call"):
            raise ValueError(f"{path}: malformed raw label for {r['case_id']}")
        if CALL_FRAGMENTS[r["case_id"]] not in r["call"]:
            raise ValueError(f"{path}: wrong public call for {r['case_id']}")
    return {r["case_id"]: r for r in rows}


def label(case_id, row, engine):
    """Convert exact recorded public-call output to a deliberately small class label."""
    outcome, cls, actual, message = (row[k].strip() for k in ("outcome", "class", "actual", "message"))
    if case_id.endswith("CHECK-AUTO-RESIDUAL"):
        if outcome != "RETURN":
            return f"error:{cls}"
        if "ordinal-probit" not in message.lower() or not any(x in message.lower() for x in ("warn", "flagged")):
            raise ValueError("check_auto_residual ordinal-probit negative control is absent or non-discriminating")
        return "residual-check:coherent" if actual.lower() in {"true", "ok", "pass", "coherent"} else "residual-check:not-coherent"
    if case_id.endswith("ISDM-COUNT"):
        return "all-count:nonmixed-admitted" if outcome == "RETURN" and actual == "all-count-nonmixed" else "all-count:wrong-outcome"
    if case_id.endswith("WRAPPER-LAW"):
        if outcome != "ERROR" or "REFUSED:" not in message or "isdm_sources" not in message:
            return "wrapper-law:wrong-outcome"
        return "guard:wrapper-law-refusal"
    if outcome == "RETURN":
        return "public-fit:returned"
    first = message.splitlines()[0]
    # This guard order is intentionally engine-neutral and grounded in the raw
    # message. In particular, R's P1 length(family) guard stays distinguishable.
    if "Response family/link cannot currently vary across rows within a trait." in first:
        return "guard:family-scale-per-trait"
    if "length(family)" in first or "number of distinct levels" in first:
        return "guard:family-length"
    if "unknown" in first.lower() or "undeclared" in first.lower():
        return "guard:unknown-source"
    if "source" in first.lower() and ("missing" in first.lower() or "every trait" in first.lower()):
        return "guard:source-coverage"
    if "trait" in first.lower() and ("missing" in first.lower() or "every" in first.lower()):
        return "guard:trait-coverage"
    if "detection" in first.lower() or "count arm" in first.lower() or "admitted" in first.lower() or "cloglog" in first.lower():
        return "guard:integrated-family-contract"
    return f"error:{cls}:{first}"


def derive(rrows, jrows, meta):
    if meta.get("reference_commit") != PIN:
        raise ValueError("run metadata is not pinned to P1")
    if meta.get("r_output_sha256") != digest(meta["r_output_path"]):
        raise ValueError("R output digest does not match run metadata")
    if meta.get("julia_output_sha256") != digest(meta["julia_output_path"]):
        raise ValueError("Julia output digest does not match run metadata")
    runner_hashes = retained_runner_hashes(meta)
    if not meta.get("r_source_sha256") or not meta.get("r_oracle_build_sha256") or not meta.get("julia_src_tree"):
        raise ValueError("run metadata lacks required R source/build or Julia source hashes")
    if any(r.get("engine") != "R" or r.get("pin") != "P1" or r.get("package_version") != "0.7.1"
           or r.get("oracle_build") != "totoro" or r.get("openblas_threads") != "1" or r.get("omp_threads") != "1"
           for r in rrows.values()):
        raise ValueError("R raw output lacks the P1 package/build or thread-cap evidence")
    if any(r.get("r_version") != meta.get("r_version") or r.get("host") != meta.get("r_host") for r in rrows.values()):
        raise ValueError("R process version/host differs from finalized run metadata")
    if any(r.get("runner_sha256") != runner_hashes["R"] for r in rrows.values()):
        raise ValueError("R raw runner digest differs from the executed source")
    if any(r.get("engine") != "Julia" or r.get("pin") != "P1" or r.get("julia_threads") != "4"
           or r.get("openblas_threads") != "1" or r.get("omp_threads") != "1"
           for r in jrows.values()):
        raise ValueError("Julia raw output lacks P1 or thread-cap evidence")
    if any(r.get("julia_version") != meta.get("julia_version") or r.get("host") != meta.get("julia_host") for r in jrows.values()):
        raise ValueError("Julia process version/host differs from finalized run metadata")
    if meta.get("thread_caps") != {"r_openblas": "1", "r_omp": "1", "julia_threads": "4",
                                   "julia_openblas": "1", "julia_omp": "1"}:
        raise ValueError("run metadata does not record the required thread caps")
    source = json.loads((ROOT / R_SOURCE).read_text())
    build = json.loads((ROOT / R_BUILD).read_text())
    if any(r.get("source_marker_sha256") != build.get("marker_sha256")
           or r.get("namespace_sha256") != source.get("namespace_sha256")
           or r.get("installed_tree_sha256") != build.get("installed_tree_sha256")
           or r.get("source_tree_sha256") != source.get("source_tree_sha256") for r in rrows.values()):
        raise ValueError("R loaded marker/NAMESPACE/installed tree differs from pinned source/build")
    if source.get("reference_commit") != PIN or build.get("reference_commit") != PIN:
        raise ValueError("tracked R source/build receipt is not at P1")
    if meta["r_source_receipt_sha256"] != digest(ROOT / R_SOURCE) or meta["r_oracle_build_sha256"] != digest(ROOT / R_BUILD):
        raise ValueError("R source/build receipt hashes differ from the tracked receipts")
    if meta["r_source_sha256"] != source.get("source_tree_sha256") or \
            meta["r_installed_tree_sha256"] != build.get("installed_tree_sha256") or \
            source.get("source_tree_sha256") != build.get("source_tree_sha256"):
        raise ValueError("R source and installed-build hashes do not resolve to the same pinned source")
    if meta.get("julia_commit") != jrows[next(iter(jrows))].get("glvmodels_commit"):
        raise ValueError("run metadata Julia commit differs from the runner's launch commit")
    tree = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{meta['julia_commit']}:src"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if meta["julia_src_tree"] != tree:
        raise ValueError("Julia source tree hash differs from the recorded launch commit")
    patch = raw_artifact(meta, meta.get("julia_src_patch", "" )).read_bytes()
    patch_hash = hashlib.sha256(patch).hexdigest()
    # Raw captures may be re-derived from a later branch. In that case the
    # current worktree diff from the launch commit includes intervening commits
    # and is not evidence about the source that ran. The retained patch digest,
    # per-row digest, launch commit, and source-tree hash bind the original
    # provenance. Compare the live diff only when re-running in that exact
    # launch checkout, where it still measures the captured worktree delta.
    current_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
    live_diff_matches = True
    if current_head == meta["julia_commit"]:
        src_diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--binary", meta["julia_commit"], "--", "src"],
                                  check=True, capture_output=True).stdout
        live_diff_matches = patch == src_diff
    if patch_hash != meta.get("julia_src_patch_sha256") or not live_diff_matches or \
            any(r.get("src_diff_sha256") != patch_hash for r in jrows.values()):
        raise ValueError("Julia raw source diff differs from the executed checkout")
    if any(r.get("runner_sha256") != runner_hashes["Julia"] for r in jrows.values()):
        raise ValueError("Julia raw runner digest differs from the executed source")
    if any(r.get("package_source") != "src/GLLVModels.jl" for r in jrows.values()):
        raise ValueError("Julia did not load GLLVModels from the recorded checkout")
    if not matching_fixture_hashes(rrows, jrows):
        raise ValueError("R and Julia raw outputs do not identify the same actual fixture rows")
    expected_fixtures = {
        "R": {cid: rrows[cid]["fixture_sha256"] for cid in CASES},
        "Julia": {cid: jrows[cid]["fixture_sha256"] for cid in CASES},
    }
    if meta.get("fixture_sha256") != expected_fixtures:
        raise ValueError("run metadata does not record the per-case fixture hashes")
    if any(r.get("public_positive_control") != "mixed-source-accepted" for r in list(rrows.values()) + list(jrows.values())):
        raise ValueError("the valid mixed-source public positive control did not pass on both engines")
    receipts, classes = {}, {}
    for cid, sid in CASES.items():
        rr, jr = rrows[cid], jrows[cid]
        rl, jl = label(cid, rr, "R"), label(cid, jr, "Julia")
        matched = rl == jl and (cid not in EXPECTED or rl == EXPECTED[cid])
        block = {"source_id": sid, "case_id": cid, "r_label": rl, "julia_label": jl,
                 "r_raw": rr, "julia_raw": jr, "signed_scope": "D-319 N6/N10"}
        if matched:
            canonical = rl
            classes.setdefault(canonical, {"canonical": canonical, "labels": {"R": set(), "Julia": set()},
                "basis": "D-319 behavioural scope; labels derived from the exact recorded public calls."})
            classes[canonical]["labels"]["R"].add(rl)
            classes[canonical]["labels"]["Julia"].add(jl)
        receipts[sid] = {"schema": "core070-first-seven-behaviour/v1", "case_id": FROZEN_CASES[cid],
            "source_id": sid, "reference_commit": PIN, "verdict": "PASS" if matched else "MISMATCH",
            "evidence_kind": "public_door_behaviour", "r_observed": rl, "julia_observed": jl,
            "raw_observations": {"R": rr, "Julia": jr}, "signed_scope": block["signed_scope"],
            "provenance": meta}
    clean = []
    for c in classes.values():
        c["labels"] = {k: sorted(v) for k, v in c["labels"].items()}
        clean.append(c)
    return receipts, {"schema": "core070-first-seven-equivalence/v1", "ruling": "D-319 N6/N10",
                      "classes": clean}


def self_test():
    ok = {"outcome": "RETURN", "class": "fit", "actual": "true", "message": ""}
    r = {cid: dict(ok) for cid in CASES}
    j = {cid: dict(ok) for cid in CASES}
    extra = "CORE070-FIRST7-ISDM-EXTRA-SOURCE"
    r[extra] = {"outcome": "ERROR", "class": "simpleError", "actual": "", "message": "length(family) must match the number of distinct levels"}
    j[extra] = {"outcome": "ERROR", "class": "ArgumentError", "actual": "", "message": "Unknown source: unknown"}
    r["CORE070-FIRST7-CHECK-AUTO-RESIDUAL"]["actual"] = "ok"
    j["CORE070-FIRST7-CHECK-AUTO-RESIDUAL"]["actual"] = "true"
    r["CORE070-FIRST7-CHECK-AUTO-RESIDUAL"]["message"] = "ordinal-probit control status=warn"
    j["CORE070-FIRST7-CHECK-AUTO-RESIDUAL"]["message"] = "ordinal-probit control flagged"
    assert label("CORE070-FIRST7-CHECK-AUTO-RESIDUAL", r["CORE070-FIRST7-CHECK-AUTO-RESIDUAL"], "R") == label(
        "CORE070-FIRST7-CHECK-AUTO-RESIDUAL", j["CORE070-FIRST7-CHECK-AUTO-RESIDUAL"], "Julia")
    assert label(extra, r[extra], "R") == "guard:family-length"
    assert label(extra, j[extra], "Julia") == "guard:unknown-source"
    wrapper = "CORE070-FIRST7-ISDM-WRAPPER-LAW"
    r[wrapper] = {"outcome": "ERROR", "class": "error", "actual": "refused",
                  "message": "REFUSED: isdm_sources refuses logit law"}
    j[wrapper] = dict(r[wrapper])
    assert label(wrapper, r[wrapper], "R") == "guard:wrapper-law-refusal"
    j[wrapper] = {"outcome": "RETURN", "class": "IsdmSources", "actual": "returned", "message": ""}
    assert label(wrapper, r[wrapper], "R") != label(wrapper, j[wrapper], "Julia")
    print("CORE070_FIRST7_DERIVATION_SELFTEST_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--source-id", action="append", choices=sorted(CASES.values()),
                    help="write only these approved source rows for a staged checkpoint; all six raw cases are still validated")
    ap.add_argument("--finalize", action="store_true", help="validate raw outputs and write their provenance metadata")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.raw_dir is None:
        ap.error("--raw-dir is required")
    d = args.raw_dir
    rp, jp, mp = d / "r-public.tsv", d / "julia-public.tsv", d / "run.json"
    if args.finalize:
        rrows, jrows = tsv(rp, "R"), tsv(jp, "Julia")
        rvals = list(rrows.values()); jvals = list(jrows.values())
        source = json.loads((ROOT / R_SOURCE).read_text()); build = json.loads((ROOT / R_BUILD).read_text())
        if any(r.get("pin") != "P1" or r.get("package_version") != "0.7.1" for r in rvals):
            raise ValueError("R raw output does not establish gllvmTMB 0.7.1 at P1")
        if any(r.get("pin") != "P1" or r.get("engine") != "Julia" for r in jvals):
            raise ValueError("Julia raw output is missing its P1 engine marker")
        if any(r.get(k) != "1" for r in rvals for k in ("openblas_threads", "omp_threads")):
            raise ValueError("R raw output does not prove OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1")
        if any(r.get("oracle_build") != "totoro" for r in rvals):
            raise ValueError("R raw output does not identify the registered Totoro oracle build")
        if any(r.get("runner_sha256") != digest(ROOT / "tools/first_seven_behaviour_R.R") for r in rvals):
            raise ValueError("R runner digest differs from the executed source")
        if any(r.get("source_marker_sha256") != build.get("marker_sha256") or r.get("namespace_sha256") != source.get("namespace_sha256")
               or r.get("installed_tree_sha256") != build.get("installed_tree_sha256")
               or r.get("source_tree_sha256") != source.get("source_tree_sha256") for r in rvals):
            raise ValueError("R loaded marker, NAMESPACE, or installed tree differs from pinned source/build")
        if any(r.get(k) != v for r in jvals for k, v in (("julia_threads", "4"), ("openblas_threads", "1"), ("omp_threads", "1"))):
            raise ValueError("Julia raw output does not prove the required thread caps")
        commits = {r.get("glvmodels_commit") for r in jvals}
        if len(commits) != 1 or not next(iter(commits)):
            raise ValueError("Julia outputs do not share one recorded launch commit")
        if any(r.get("runner_sha256") != digest(ROOT / "tools/first_seven_behaviour_J.jl") for r in jvals):
            raise ValueError("Julia runner digest differs from the executed runner")
        if not matching_fixture_hashes(rrows, jrows):
            raise ValueError("R and Julia raw outputs do not identify the same fixture rows")
        rvals_by_case, jvals_by_case = rrows, jrows
        commit = next(iter(commits))
        current = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                 check=True, capture_output=True, text=True).stdout.strip()
        launch_tree = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{commit}:src"],
                                     check=True, capture_output=True, text=True).stdout.strip()
        launched_runner = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:tools/first_seven_behaviour_J.jl"],
                                         check=True, capture_output=True).stdout
        if hashlib.sha256(launched_runner).hexdigest() != jvals[0]["runner_sha256"]:
            raise ValueError("Julia launch runner is not the recorded commit's runner")
        src_diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--binary", commit, "--", "src"],
                                  check=True, capture_output=True).stdout
        if any(r.get("src_diff_sha256") != hashlib.sha256(src_diff).hexdigest()
               or r.get("package_source") != "src/GLLVModels.jl" for r in jvals):
            raise ValueError("Julia raw source identity does not match the executed checkout")
        tree = subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{commit}:src"],
                              check=True, capture_output=True, text=True).stdout.strip()
        raw_rel = str(d.relative_to(ROOT) if d.is_absolute() else d)
        patch_path = d / "julia-src.patch"
        patch_path.write_bytes(src_diff)
        snapshots = snapshot_runners(d / "runner-snapshots", commit)
        meta = {"raw_directory": raw_rel, "derivation_commit": current, "reference_commit": PIN, "r_source_receipt": R_SOURCE, "r_source_receipt_sha256": digest(ROOT / R_SOURCE),
            "r_source_sha256": source["source_tree_sha256"], "r_oracle_build_receipt": R_BUILD,
            "r_oracle_build_sha256": digest(ROOT / R_BUILD), "r_installed_tree_sha256": build["installed_tree_sha256"],
            "r_host": rvals[0]["host"], "r_version": rvals[0]["r_version"],
            "julia_commit": commit, "julia_src_tree": tree, "julia_src_patch": "julia-src.patch",
            "julia_src_patch_sha256": hashlib.sha256(src_diff).hexdigest(), "runner_snapshots": snapshots,
            "julia_version": jvals[0]["julia_version"],
            "julia_host": jvals[0]["host"], "r_output_sha256": digest(rp), "julia_output_sha256": digest(jp),
            "runner_sha256": {"R": rvals[0]["runner_sha256"], "Julia": jvals[0]["runner_sha256"],
                              "derive": snapshots["derive"]["sha256"]},
            "fixture_sha256": {
                "R": {cid: rvals_by_case[cid]["fixture_sha256"] for cid in CASES},
                "Julia": {cid: jvals_by_case[cid]["fixture_sha256"] for cid in CASES}},
            "thread_caps": {"r_openblas": "1", "r_omp": "1", "julia_threads": "4",
                            "julia_openblas": "1", "julia_omp": "1"}}
        mp.write_text(json.dumps(meta, indent=2) + "\n")
        print("CORE070_FIRST7_PROVENANCE_FINALIZED")
        return
    meta = json.loads(mp.read_text())
    meta["r_output_path"], meta["julia_output_path"] = str(rp), str(jp)
    receipts, eq = derive(tsv(rp, "R"), tsv(jp, "Julia"), meta)
    meta.pop("r_output_path", None); meta.pop("julia_output_path", None)
    if not args.write:
        print("CORE070_FIRST7_RAW_DERIVATION_VALID")
        return
    dest = BASE / "derived"
    dest.mkdir(parents=True, exist_ok=True)
    selected = set(args.source_id or receipts)
    for sid, rec in receipts.items():
        if sid not in selected:
            continue
        name = sid.split("/", 1)[1]
        (dest / f"{name}.json").write_text(json.dumps(rec, indent=2) + "\n")
    (dest / "behaviour-equivalence.json").write_text(json.dumps(eq, indent=2) + "\n")
    print("CORE070_FIRST7_BEHAVIOUR_DERIVED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Negative and positive controls for tools/true_parity_assemble.py.

Each control builds a small throwaway root (temp dir) with the ledger layout the assembler
reads, then runs it. Run: python3 tools/test_true_parity_assemble.py
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import true_parity_assemble as A  # noqa: E402

P1 = A.P1_SHA


def row(sid, cls="required_core", tier="registration", **kw):
    r = {"source_id": sid, "classification": cls, "executable_case_ids": [], "disposition": None,
         "evidence_tier": tier, "measured_against": P1, "evidence": {}}
    r.update(kw)
    return r


def make_root(tmp: Path, maps: dict | None = None) -> Path:
    led = tmp / A.LEDGER
    (led / "receipts/namespace/cases").mkdir(parents=True)
    (tmp / "test/fixtures").mkdir(parents=True)
    for name in A.EXPECTED_MAPS:
        fam = name[len("case-map-"):-len(".json")]
        rows = (maps or {}).get(name, [row(f"{fam}/ROW-1", tier="not_measured")])
        (led / name).write_text(json.dumps({"schema": 1, "rows": rows}))
    (led / "receipts/namespace/cases/C1.json").write_text(json.dumps(
        {"source_id": "namespace/export/r_fun", "julia_check": {"symbol": "julia_twin"}}))
    (led / A.IN_REVERSE_GAP).write_text(json.dumps({
        "glvmodels_commit": "0" * 40, "r_names": ["shared_name"],
        "julia_exports": [{"name": "shared_name", "kind": "Function"},
                          {"name": "julia_twin", "kind": "Function"},
                          {"name": "julia_only", "kind": "Type"}]}))
    return tmp


def run(root: Path, *args) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = A.main(["--root", str(root), *args])
    return code, buf.getvalue()


FAILS = []


def expect(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  {detail}"))
    if not cond:
        FAILS.append(name)


def with_root(maps=None):
    tmp = Path(tempfile.mkdtemp(prefix="tp-assemble-"))
    return make_root(tmp, maps), tmp


RP = str(A.LEDGER / "receipts/r.json")
GOOD_CMP = {"pin": "P1", "cases": [{"case_id": "C", "r_value": 1.0, "julia_value": 1.0 + 1e-9, "tolerance": 1e-6}]}


def numeric_root(receipt_extra=None, **row_kw):
    """A root with one numeric row family/N citing RP, whose receipt holds GOOD_CMP."""
    kw = dict(tier="numeric", executable_case_ids=["C"], evidence={"receipt": [RP]})
    kw.update(row_kw)
    root, tmp = with_root({"case-map-family.json": [row("family/N", **kw)]})
    (root / RP).write_text(json.dumps({"comparison": GOOD_CMP, **(receipt_extra or {})}))
    return root, tmp


def status_of(root, rid="family-N"):
    m = re.search(rf"^\| {re.escape(rid)} `[^`]*` \| [^|]* \| ([^|]+?) \|", (root / A.LEDGER / A.OUT_SCOREBOARD).read_text(), re.M)
    return m.group(1) if m else None


def main():
    # Positive control: a clean root writes, then --check reads current.
    root, tmp = with_root()
    c1, o1 = run(root)
    c2, o2 = run(root, "--check")
    expect("clean_root_writes_and_checks", c1 == 0 and c2 == 0 and "ASSEMBLE_OK" in o2, o1 + o2)
    rg = json.loads((root / A.LEDGER / A.OUT_REVERSE_GAP).read_text())
    expect("reverse_gap_excludes_name_and_receipt_counterparts",
           [x["name"] for x in rg] == ["julia_only"] and rg[0]["status"] == "unsigned" and rg[0]["decision"] is None, rg)
    # Stale output: a hand edit to the scoreboard fails --check.
    sb = root / A.LEDGER / A.OUT_SCOREBOARD
    sb.write_text(sb.read_text().replace("NOT-MEASURED", "EVIDENCED", 1))
    c3, o3 = run(root, "--check")
    expect("hand_edited_scoreboard_fails_check", c3 == 1 and "ASSEMBLE_STALE" in o3, o3)
    shutil.rmtree(tmp)

    # Negative control: the same row id in two maps with a conflicting class fails --check.
    dup_a = row("shared/ROW", cls="required_core")
    dup_b = row("shared/ROW", cls="outside_boundary")
    root, tmp = with_root({"case-map-aghq.json": [dup_a], "case-map-data.json": [dup_b]})
    c, o = run(root, "--check")
    expect("duplicate_id_conflicting_class_fails", c == 1 and "conflicting rows" in o and "classification" in o, o)
    shutil.rmtree(tmp)

    # An identical duplicate is collapsed, not a conflict (reported in the scoreboard).
    root, tmp = with_root({"case-map-aghq.json": [dup_a], "case-map-data.json": [dict(dup_a)]})
    c, o = run(root)
    expect("identical_duplicate_collapsed", c == 0 and "identical" in (root / A.LEDGER / A.OUT_SCOREBOARD).read_text(), o)
    shutil.rmtree(tmp)

    # Negative control: a missing input map fails.
    root, tmp = with_root()
    (root / A.LEDGER / "case-map-isdm.json").unlink()
    c, o = run(root, "--check")
    expect("missing_input_map_fails", c == 1 and "missing input map" in o and "case-map-isdm.json" in o, o)
    shutil.rmtree(tmp)

    # An evidence_tier with no bucket fails (a human decides the bucket, not the tool).
    root, tmp = with_root({"case-map-family.json": [row("family/X", tier="brand_new_tier",
                                                        evidence={"receipt": ["docs/x.json"]})]})
    c, o = run(root, "--check")
    expect("unknown_tier_fails", c == 1 and "brand_new_tier" in o, o)
    shutil.rmtree(tmp)

    # A "numeric" label whose receipt has no comparison block is not EVIDENCED.
    rp = str(A.LEDGER / "receipts/r.json")
    root, tmp = with_root({"case-map-family.json": [row("family/N", tier="numeric", executable_case_ids=["C"],
                                                        evidence={"receipt": [rp]})]})
    (root / rp).write_text(json.dumps({"verdict": "PASS"}))
    c, o = run(root)
    txt = (root / A.LEDGER / A.OUT_SCOREBOARD).read_text()
    expect("numeric_label_without_comparison_not_evidenced", c == 0 and "| NUMERIC-UNVERIFIED |" in txt and "| EVIDENCED |" not in txt, o)
    # ...and with a valid P1 comparison block it is.
    (root / rp).write_text(json.dumps({"verdict": "PASS", "comparison": {"pin": "P1", "cases": [
        {"case_id": "C", "r_value": 1.0, "julia_value": 1.0 + 1e-9, "tolerance": 1e-6}]}}))
    c, o = run(root)
    txt = (root / A.LEDGER / A.OUT_SCOREBOARD).read_text()
    expect("numeric_with_p1_comparison_evidenced", c == 0 and "| EVIDENCED |" in txt, o)
    # ...and a DISPOSITION-SIGNED label with an agent signer is not DISPOSITION-SIGNED.
    shutil.rmtree(tmp)
    root, tmp = with_root({"case-map-data.json": [row("data/S", disposition="DISPOSITION-SIGNED",
                                                      signed_by="Claude (agent)", signed_on="2026-09-27")]})
    c, o = run(root)
    txt = (root / A.LEDGER / A.OUT_SCOREBOARD).read_text()
    expect("agent_signed_disposition_not_signed", c == 0 and "| DISPOSITION-UNVERIFIED |" in txt, o)
    shutil.rmtree(tmp)

    # A source id that would sanitize into a C3 id convention fails rather than moving clauses.
    root, tmp = with_root({"case-map-family.json": [row("family/GAUSSIAN-RSZ", tier="not_measured")]})
    c, o = run(root, "--check")
    expect("rsz_suffix_id_fails", c == 1 and "C3/C4/C5" in o, o)
    shutil.rmtree(tmp)

    # Review of #589, finding 1: numeric 1 is not a pass value (the checker's isPassValue is
    # strict: "PASS", "pass" or the boolean true only; in Python 1 == True).
    for val in (1, 1.0):
        root, tmp = numeric_root({"harness_pass": val})
        c, o = run(root)
        expect(f"numeric_{val!r}_status_not_pass", c == 0 and status_of(root) == "NUMERIC-UNVERIFIED", status_of(root))
        shutil.rmtree(tmp)
    root, tmp = numeric_root({"harness_pass": True})
    c, o = run(root)
    expect("boolean_true_status_is_pass", c == 0 and status_of(root) == "EVIDENCED", status_of(root))
    shutil.rmtree(tmp)

    # Review of #589, finding 2: a valid signature does not outrank a dangling receipt or a stale
    # carry. The checker's C1 runs both checks first and does not count such a row as signed.
    sig = dict(disposition="DISPOSITION-SIGNED", signed_by="Shinichi Nakagawa", signed_on="2026-09-27")
    root, tmp = with_root({"case-map-data.json": [row("data/S", evidence={"receipt": ["docs/does-not-exist.json"]}, **sig)]})
    c, o = run(root)
    st = status_of(root, "data-S")
    expect("signed_with_dangling_receipt_not_signed", c == 0 and st == "DISPOSITION-UNVERIFIED"
           and "dangling docs/does-not-exist.json" in (root / A.LEDGER / A.OUT_SCOREBOARD).read_text(), st)
    shutil.rmtree(tmp)
    root, tmp = with_root({"case-map-data.json": [row("data/S", measured_against="P0", evidence={"receipt": [RP]}, **sig)]})
    (root / RP).write_text(json.dumps({"comparison": GOOD_CMP}))
    c, o = run(root)
    st = status_of(root, "data-S")
    expect("signed_with_p0_receipt_no_carry_not_signed", c == 0 and st == "DISPOSITION-UNVERIFIED"
           and "PARTIAL_STALE_AT_P1(no carry.source_pins)" in (root / A.LEDGER / A.OUT_SCOREBOARD).read_text(), st)
    shutil.rmtree(tmp)
    root, tmp = with_root({"case-map-data.json": [row("data/S", **sig)]})
    c, o = run(root)
    expect("signed_without_receipt_is_signed", c == 0 and status_of(root, "data-S") == "DISPOSITION-SIGNED", status_of(root, "data-S"))
    shutil.rmtree(tmp)

    # Review of #589, finding 3: a valid maintainer-signed receipt_status_exception waives a failed
    # status field (the checker counts the row in bound_signed=, never bound=), so the row reads
    # DISPOSITION-SIGNED, never EVIDENCED. It waives the status only, not the comparison.
    exc = {"reason": "batch FAIL is an unrelated case", "signed_by": "Shinichi Nakagawa", "signed_on": "2026-09-27"}
    root, tmp = numeric_root({"verdict": "FAIL"}, receipt_status_exception=exc)
    c, o = run(root)
    txt = (root / A.LEDGER / A.OUT_SCOREBOARD).read_text()
    expect("signed_status_exception_reads_signed", c == 0 and status_of(root) == "DISPOSITION-SIGNED"
           and "signed_by: Shinichi Nakagawa; signed_on: 2026-09-27" in txt and "receipt_status_exception" in txt, status_of(root))
    shutil.rmtree(tmp)
    root, tmp = numeric_root({"verdict": "FAIL"}, receipt_status_exception=dict(exc, signed_by="Claude (agent)"))
    c, o = run(root)
    expect("agent_signed_status_exception_not_signed", c == 0 and status_of(root) == "NUMERIC-UNVERIFIED", status_of(root))
    shutil.rmtree(tmp)
    bad_cmp = {"pin": "P1", "cases": [{"case_id": "C", "r_value": 1.0, "julia_value": 2.0, "tolerance": 1e-6}]}
    root, tmp = numeric_root({"verdict": "FAIL", "comparison": bad_cmp}, receipt_status_exception=exc)
    c, o = run(root)
    expect("status_exception_does_not_waive_tolerance", c == 0 and status_of(root) == "NUMERIC-UNVERIFIED", status_of(root))
    shutil.rmtree(tmp)

    # Real tree: outputs current, and EVIDENCED count equals the checker's own C1 bound=.
    c, o = run(A.ROOT, "--check")
    expect("real_tree_outputs_current", c == 0, o)
    if shutil.which("node"):
        env = dict(os.environ, PARITY_REF="FS", PARITY_FS_ROOT=str(A.ROOT),
                   PARITY_CASEMAP=str(A.LEDGER / A.OUT_CASEMAP))
        out = subprocess.run(["node", str(HERE / "true_parity_check.mjs"), "C1"], env=env,
                             capture_output=True, text=True).stdout
        m = re.search(r"\bbound=(\d+)", out)
        n_ev = len(re.findall(r"^\| \S+ `[^`]*` \| [^|]* \| EVIDENCED \|", (A.ROOT / A.LEDGER / A.OUT_SCOREBOARD).read_text(), re.M))
        expect("real_tree_evidenced_equals_checker_c1_bound", m is not None and int(m.group(1)) == n_ev,
               f"checker bound={m.group(1) if m else '?'} scoreboard EVIDENCED={n_ev}")

    print(f"{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

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


BR = str(A.LEDGER / "receipts/b.json")
EQ_PATH = A.LEDGER / A.IN_BEHAVIOUR_EQUIVALENCE
WALD = {"kind": "route", "canonical": "wald", "r": ["r_wald"], "julia": ["jl_wald"], "basis": "R and Julia Wald routes."}


def beh_case(**kw):
    d = {"case_id": "C", "kind": "route", "r_observed": "wald", "julia_observed": "wald"}
    d.update(kw)
    return d


def behavioural_root(cases=None, receipt_extra=None, equiv=None, block_extra=None, **row_kw):
    """A root with one behavioural row inference/N citing BR, whose receipt holds a behaviour block."""
    kw = dict(tier="behavioural", executable_case_ids=["C"], evidence={"receipt": [BR]})
    kw.update(row_kw)
    root, tmp = with_root({"case-map-family.json": [row(kw.pop("sid", "inference/N"), **kw)]})
    blk = {"pin": "P1", "cases": cases if cases is not None else [beh_case()]}
    blk.update(block_extra or {})
    (root / BR).write_text(json.dumps({"behaviour": blk, **(receipt_extra or {})}))
    if equiv is not None:
        (root / EQ_PATH).write_text(json.dumps(equiv))
    return root, tmp


def checker_c1(root):
    """The checker's own C1 line over this root's assembled case map (keeps the two in step)."""
    if not shutil.which("node"):
        return None
    env = dict(os.environ, PARITY_REF="FS", PARITY_FS_ROOT=str(root), PARITY_CASEMAP=str(A.LEDGER / A.OUT_CASEMAP))
    return subprocess.run(["node", str(HERE / "true_parity_check.mjs"), "C1"], env=env, capture_output=True, text=True).stdout


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

    # Review of #589, finding 4: a malformed carry (a string, or a string in source_pins) is an
    # ASSEMBLE_FAIL with a message, not an uncaught AttributeError.
    for label, carry in (("string_carry", "abc"), ("string_source_pin", {"source_pins": ["abc"]})):
        root, tmp = numeric_root(measured_against="P0", carry=carry)
        try:
            c, o = run(root, "--check")
            ok = c == 1 and "ASSEMBLE_FAIL" in o and "family/N" in o and "carry" in o
        except Exception as e:  # noqa: BLE001 -- the red state this control exists to catch
            ok, o = False, f"{type(e).__name__}: {e}"
        expect(f"malformed_carry_{label}_fails_cleanly", ok, o)
        shutil.rmtree(tmp)

    # Review of #589, finding 7: --check --extra-map without --out-dir compared the folded rows with
    # the tracked outputs and always printed a false ASSEMBLE_STALE. It is now a usage error (2);
    # with --out-dir the same fold checks clean.
    root, tmp = with_root()
    run(root)
    extra = tmp / "extra-map.json"
    extra.write_text(json.dumps({"rows": [row("extra/ROW-1", tier="not_measured")]}))
    try:
        c, o = run(root, "--check", "--extra-map", str(extra))
    except SystemExit as e:
        c, o = e.code, ""
    expect("check_extra_map_without_out_dir_is_usage_error", c == 2 and "ASSEMBLE_STALE" not in o, f"{c} {o}")
    c, o = run(root, "--check", "--extra-map", str(extra), "--out-dir", str(tmp / "out"))
    expect("check_extra_map_with_out_dir_ok", c == 0 and "ASSEMBLE_OK" in o, o)
    shutil.rmtree(tmp)

    # Ruling 2 (itchyshin/GLLVModels.jl#684 item 2): the behavioural tier.
    def behaves(name, status, root, tmp, want_bound=None, contains=None):
        c, o = run(root)
        txt = (root / A.LEDGER / A.OUT_SCOREBOARD).read_text()
        ok = c == 0 and status_of(root, "inference-N") == status and (contains is None or contains in txt)
        if ok and want_bound is not None:
            out = checker_c1(root)
            m = re.search(r"\bbound_behavioural=(\d+)", out or "")
            ok = out is None or (m is not None and int(m.group(1)) == want_bound)
        expect(name, ok, f"{status_of(root, 'inference-N')} {o}")
        shutil.rmtree(tmp)

    root, tmp = behavioural_root()
    behaves("behavioural_matching_labels_evidenced_behavioural", "EVIDENCED-BEHAVIOURAL", root, tmp, want_bound=1)
    root, tmp = behavioural_root()
    run(root)
    txt = (root / A.LEDGER / A.OUT_SCOREBOARD).read_text()
    expect("behavioural_row_cites_receipt_and_is_in_totals",
           f"| EVIDENCED-BEHAVIOURAL | {BR} |" in txt and "EVIDENCED-BEHAVIOURAL" in txt.split("## P1 twin fixtures")[0], txt[:600])
    c, o = run(root, "--check")
    expect("behavioural_scoreboard_check_current", c == 0 and "ASSEMBLE_OK" in o, o)
    shutil.rmtree(tmp)
    mism = [beh_case(r_observed="r_wald", julia_observed="jl_wald")]
    root, tmp = behavioural_root(cases=mism)
    behaves("behavioural_mismatch_no_class_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0, contains="differ after canonicalisation")
    root, tmp = behavioural_root(cases=mism, equiv={"schema": 1, "pin": "P1", "classes": [WALD]})
    behaves("behavioural_mismatch_rescued_by_class", "EVIDENCED-BEHAVIOURAL", root, tmp, want_bound=1)
    root, tmp = behavioural_root(cases=[beh_case(kind="refusal", r_observed="r_wald", julia_observed="jl_wald")],
                                 equiv={"schema": 1, "pin": "P1", "classes": [WALD]})
    behaves("behavioural_class_of_other_kind_does_not_rescue", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0)
    root, tmp = behavioural_root(cases=[beh_case(source_id="inference/OTHER")])
    behaves("behavioural_entry_scoped_to_other_row_does_not_cover", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0,
            contains="without an applicable behaviour entry")
    root, tmp = behavioural_root(cases=[beh_case(source_id="inference/N")])
    behaves("behavioural_entry_scoped_to_this_row_covers", "EVIDENCED-BEHAVIOURAL", root, tmp, want_bound=1)
    root, tmp = behavioural_root(executable_case_ids=["C", "C2"])
    behaves("behavioural_uncovered_case_id_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0, contains="C2")
    root, tmp = behavioural_root(receipt_extra={"verdict": "FAIL"})
    behaves("behavioural_receipt_verdict_fail_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0, contains="receipt did not pass")
    root, tmp = behavioural_root(block_extra={"harness_pass": False})
    behaves("behavioural_block_status_fail_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0)
    root, tmp = behavioural_root(block_extra={"pin": "P0"})
    behaves("behavioural_wrong_pin_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0, contains="not pinned to P1")
    for label, over in (("empty_label", {"r_observed": ""}), ("blank_label", {"julia_observed": "  "}),
                        ("array_length_mismatch", {"r_observed": ["a", "b"], "julia_observed": ["a"]}),
                        ("shape_mismatch", {"r_observed": ["a"], "julia_observed": "a"})):
        root, tmp = behavioural_root(cases=[beh_case(**over)])
        behaves(f"behavioural_{label}_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0)
    root, tmp = behavioural_root(cases=[beh_case(kind="printed_fields", r_observed=["a", "b"], julia_observed=["a", "b"])])
    behaves("behavioural_array_labels_elementwise_ok", "EVIDENCED-BEHAVIOURAL", root, tmp, want_bound=1)
    root, tmp = behavioural_root(measured_against="P0")
    behaves("behavioural_stale_carry_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0, contains="PARTIAL_STALE_AT_P1")
    root, tmp = behavioural_root(evidence={"receipt": ["docs/does-not-exist.json"]})
    behaves("behavioural_dangling_receipt_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, contains="dangling")
    root, tmp = behavioural_root(executable_case_ids=[])
    behaves("behavioural_no_case_ids_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, contains="no executable_case_ids")
    root, tmp = behavioural_root()
    (root / BR).write_text(json.dumps({"comparison": GOOD_CMP}))
    behaves("behavioural_label_with_only_numeric_block_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0,
            contains="no behaviour block")
    dup = dict(WALD, canonical="wald_other")
    root, tmp = behavioural_root(equiv={"schema": 1, "pin": "P1", "classes": [WALD, dup]})
    c, o = run(root, "--check")
    expect("behavioural_ambiguous_table_fails_run", c == 1 and "ASSEMBLE_FAIL" in o and "ambiguous table" in o, o)
    shutil.rmtree(tmp)
    root, tmp = behavioural_root(equiv={"schema": 1, "pin": "P1", "classes": [dict(WALD, basis="")]})
    c, o = run(root, "--check")
    expect("behavioural_empty_basis_fails_run", c == 1 and "empty basis" in o, o)
    shutil.rmtree(tmp)
    # Scope of the behavioural tier: inference/* rows and four named C1 rows only.
    for sid in ("isdm/X", "postfit/POSTFIT-SURFACE-nobs", "inference2/N", "data/RD-01"):
        root, tmp = behavioural_root(sid=sid)
        rid = A.scoreboard_id(sid)
        c, o = run(root)
        out = checker_c1(root)
        m = re.search(r"\bbound_behavioural=(\d+)", out or "")
        expect(f"behavioural_scope_{rid}_not_covered",
               c == 0 and status_of(root, rid) == "BEHAVIOURAL-UNVERIFIED" and "not covered by itchyshin/GLLVModels.jl#684 item 2" in (root / A.LEDGER / A.OUT_SCOREBOARD).read_text()
               and (out is None or (m is not None and int(m.group(1)) == 0)), f"{status_of(root, rid)} {o}")
        shutil.rmtree(tmp)
    for sid in A.BEHAVIOURAL_NAMED_SOURCE_IDS:
        root, tmp = behavioural_root(sid=sid)
        rid = A.scoreboard_id(sid)
        c, o = run(root)
        out = checker_c1(root)
        m = re.search(r"\bbound_behavioural=(\d+)", out or "")
        expect(f"behavioural_scope_named_row_{rid}_evidenced",
               c == 0 and status_of(root, rid) == "EVIDENCED-BEHAVIOURAL" and (out is None or (m is not None and int(m.group(1)) == 1)), f"{status_of(root, rid)} {o}")
        shutil.rmtree(tmp)
    bad_cmp = {"pin": "P1", "cases": [{"case_id": "C", "r_value": 1.0, "julia_value": 99.0, "tolerance": 1e-6}]}
    root, tmp = behavioural_root(receipt_extra={"comparison": bad_cmp})
    behaves("behavioural_cited_receipt_failing_comparison_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0,
            contains="cited receipt's own comparison fails")
    root, tmp = behavioural_root(receipt_extra={"comparison": GOOD_CMP})
    behaves("behavioural_cited_receipt_passing_comparison_ok", "EVIDENCED-BEHAVIOURAL", root, tmp, want_bound=1)
    for label, over in (("verdict", {"verdict": "FAIL"}), ("harness_pass", {"harness_pass": False})):
        root, tmp = behavioural_root(cases=[beh_case(**over)])
        behaves(f"behavioural_case_level_{label}_unverified", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0, contains="receipt did not pass")
    root, tmp = behavioural_root(cases=[beh_case(r_observed="jl_wald", julia_observed="r_wald")], equiv={"schema": 1, "pin": "P1", "classes": [WALD]})
    behaves("behavioural_canonicalisation_is_side_specific", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0)
    dupc = dict(WALD, r=["other_r"], julia=["other_j"])
    root, tmp = behavioural_root(equiv={"schema": 1, "pin": "P1", "classes": [WALD, dupc]})
    c, o = run(root, "--check")
    expect("behavioural_duplicate_canonical_fails_run", c == 1 and "duplicate canonical" in o, o)
    shutil.rmtree(tmp)
    root, tmp = behavioural_root(equiv={"schema": 1, "pin": "P1", "classes": [WALD, dict(WALD, kind="refusal")]})
    c, o = run(root)
    expect("behavioural_same_canonical_in_two_kinds_ok", c == 0 and status_of(root, "inference-N") == "EVIDENCED-BEHAVIOURAL", o)
    shutil.rmtree(tmp)
    # A numeric row is not read as behavioural, and a behavioural row never reads EVIDENCED.
    root, tmp = numeric_root()
    run(root)
    expect("numeric_row_still_evidenced_with_equivalence_absent", status_of(root) == "EVIDENCED", status_of(root))
    shutil.rmtree(tmp)

    # Ruling 1 (itchyshin/GLLVModels.jl#684 item 1): integer equality.
    def int_cmp(**kw):
        c = {"case_id": "C", "kind": "integer_equality", "r_value": 15, "julia_value": 15, "tolerance": 0.5}
        c.update(kw)
        return {"comparison": {"pin": "P1", "cases": [c]}}

    def int_root(name, status, want_bound, **kw):
        root, tmp = with_root({"case-map-family.json": [row("family/N", tier="numeric", executable_case_ids=["C"], evidence={"receipt": [RP]})]})
        (root / RP).write_text(json.dumps(int_cmp(**kw)))
        c, o = run(root)
        ok = c == 0 and status_of(root) == status
        if ok and shutil.which("node"):
            env = dict(os.environ, PARITY_REF="FS", PARITY_FS_ROOT=str(root), PARITY_CASEMAP=str(A.LEDGER / A.OUT_CASEMAP))
            out = subprocess.run(["node", str(HERE / "true_parity_check.mjs"), "C1"], env=env, capture_output=True, text=True).stdout
            m = re.search(r"\bbound_numeric=(\d+)", out)
            ok = m is not None and int(m.group(1)) == want_bound
        expect(name, ok, status_of(root))
        shutil.rmtree(tmp)

    int_root("integer_equality_15_vs_15_evidenced", "EVIDENCED", 1)
    int_root("integer_equality_float_15_0_is_integer", "EVIDENCED", 1, r_value=15.0, julia_value=15.0)
    int_root("integer_equality_vectors_evidenced", "EVIDENCED", 1, r_value=[15, 3], julia_value=[15, 3])
    int_root("integer_equality_off_by_one_unverified", "NUMERIC-UNVERIFIED", 0, julia_value=16)
    int_root("integer_equality_non_integer_unverified", "NUMERIC-UNVERIFIED", 0, r_value=15.2, julia_value=15.2)
    int_root("integer_equality_tolerance_1_unverified", "NUMERIC-UNVERIFIED", 0, tolerance=1)
    int_root("integer_equality_length_mismatch_unverified", "NUMERIC-UNVERIFIED", 0, r_value=[1, 2], julia_value=[1])
    int_root("integer_equality_bool_is_not_integer", "NUMERIC-UNVERIFIED", 0, r_value=True, julia_value=True)
    int_root("unknown_comparison_kind_unverified", "NUMERIC-UNVERIFIED", 0, kind="close_enough")
    int_root("integer_equality_unsafe_integer_unverified", "NUMERIC-UNVERIFIED", 0, r_value=9007199254740993, julia_value=9007199254740992)
    int_root("integer_equality_2_pow_53_unverified", "NUMERIC-UNVERIFIED", 0, r_value=9007199254740992, julia_value=9007199254740992)
    int_root("integer_equality_max_safe_integer_evidenced", "EVIDENCED", 1, r_value=9007199254740991, julia_value=9007199254740991)
    root, tmp = with_root({"case-map-family.json": [row("family/N", tier="numeric", executable_case_ids=["C"], evidence={"receipt": [RP]})]})
    c15 = int_cmp(r_value=15.2, julia_value=15.2)
    del c15["comparison"]["cases"][0]["kind"]
    (root / RP).write_text(json.dumps(c15))
    run(root)
    expect("no_kind_case_keeps_todays_rule", status_of(root) == "EVIDENCED", status_of(root))
    shutil.rmtree(tmp)

    # Ruling 3 (itchyshin/GLLVModels.jl#684 item 3): reverse-gap-decisions.json.
    RULING = {"ref": "itchyshin/GLLVModels.jl#684 item 3", "signed_by": "Shinichi Nakagawa", "signed_on": "2026-10-02"}

    def decisions(**over):
        d = {"schema": 1, "ruling": RULING, "criterion": "test", "generator": "tools/none.py",
             "decisions": {"julia_only": {"decision": "EXCLUDED_INTERNAL_HELPER", "basis": "no docstring"}}}
        d.update(over)
        return d

    DEC = A.LEDGER / A.IN_REVERSE_GAP_DECISIONS
    root, tmp = with_root()
    (root / DEC).write_text(json.dumps(decisions()))
    c, o = run(root)
    rg = json.loads((root / A.LEDGER / A.OUT_REVERSE_GAP).read_text())
    it = rg[0]
    expect("decisions_copied_onto_matching_item",
           c == 0 and it["name"] == "julia_only" and it["status"] == "decided" and it["decision"] == "EXCLUDED_INTERNAL_HELPER"
           and it["basis"] == "no docstring" and it["ruling"] == RULING and list(it["ruling"]) == ["ref", "signed_by", "signed_on"], it)
    c, o = run(root, "--check")
    expect("decisions_output_checks_current", c == 0 and "ASSEMBLE_OK" in o, o)
    if shutil.which("node"):
        env = dict(os.environ, PARITY_REF="FS", PARITY_FS_ROOT=str(root))
        out = subprocess.run(["node", str(HERE / "true_parity_check.mjs"), "C6"], env=env, capture_output=True, text=True).stdout
        expect("checker_c6_accepts_assembled_decision", "C6_MET" in out and "EXCLUDED_INTERNAL_HELPER:1" in out and "unsigned_decision=none" in out, out)
    shutil.rmtree(tmp)
    root, tmp = with_root()
    run(root)
    base_rg = (root / A.LEDGER / A.OUT_REVERSE_GAP).read_text()
    (root / DEC).write_text(json.dumps(decisions(decisions={})))
    run(root)
    expect("empty_decisions_leave_reverse_gap_unchanged", (root / A.LEDGER / A.OUT_REVERSE_GAP).read_text() == base_rg)
    shutil.rmtree(tmp)
    root, tmp = with_root()
    (root / DEC).write_text(json.dumps(decisions(decisions={"no_such_export": {"decision": "KEPT_AS_JULIA_EXTRA", "basis": "x"}})))
    c, o = run(root, "--check")
    expect("decision_for_non_item_is_stale_and_fails", c == 1 and "ASSEMBLE_FAIL" in o and "no_such_export" in o and "stale" in o, o)
    shutil.rmtree(tmp)
    root, tmp = with_root()
    (root / DEC).write_text(json.dumps(decisions(decisions={"shared_name": {"decision": "KEPT_AS_JULIA_EXTRA", "basis": "x"}})))
    c, o = run(root, "--check")
    expect("decision_for_name_with_r_counterpart_is_stale", c == 1 and "shared_name" in o and "stale" in o, o)
    shutil.rmtree(tmp)
    for label, bad in (("unknown_word", {"julia_only": {"decision": "EXCLUDED_HELPER", "basis": "x"}}),
                       ("empty_basis", {"julia_only": {"decision": "KEPT_AS_JULIA_EXTRA", "basis": " "}}),
                       ("null_decision", {"julia_only": {"decision": None, "basis": "x"}})):
        root, tmp = with_root()
        (root / DEC).write_text(json.dumps(decisions(decisions=bad)))
        c, o = run(root, "--check")
        expect(f"decisions_file_{label}_fails", c == 1 and "ASSEMBLE_FAIL" in o and "julia_only" in o, o)
        shutil.rmtree(tmp)
    # The tool refuses to copy a signature it does not recognise (review of the first cut).
    for label, over, why in (
            ("unrecognised_ref", {"ruling": dict(RULING, ref="anything")}, "not a recognised signed ruling"),
            ("wrong_date", {"ruling": dict(RULING, signed_on="2026-01-01")}, "is not the date of"),
            ("agent_signer", {"ruling": dict(RULING, signed_by="Claude Opus (agent)")}, "signature rejected"),
            ("uncovered_word_port", {"decisions": {"julia_only": {"decision": "PORT_TO_MATCH_R", "basis": "x"}}}, "is not covered by"),
            ("uncovered_word_deprecate", {"decisions": {"julia_only": {"decision": "DEPRECATE_AND_REMOVE", "basis": "x"}}}, "is not covered by"),
            ("uncovered_word_rename", {"decisions": {"julia_only": {"decision": "RENAME_TO_AVOID_COLLISION", "basis": "x"}}}, "is not covered by"),
            ("missing_criterion", {"criterion": " "}, "criterion must be a non-empty string"),
            ("missing_generator", {"generator": None}, "generator must be a non-empty string")):
        root, tmp = with_root()
        (root / DEC).write_text(json.dumps(decisions(**over)))
        c, o = run(root, "--check")
        expect(f"decisions_file_{label}_fails", c == 1 and "ASSEMBLE_FAIL" in o and why in o, o)
        shutil.rmtree(tmp)
    root, tmp = with_root()
    (root / DEC).write_text(json.dumps(decisions(decisions={"julia_only": {"decision": "KEPT_AS_JULIA_EXTRA", "basis": "documented"}})))
    c, o = run(root)
    rg = json.loads((root / A.LEDGER / A.OUT_REVERSE_GAP).read_text())
    expect("decisions_file_kept_as_julia_extra_ok", c == 0 and rg[0]["decision"] == "KEPT_AS_JULIA_EXTRA" and rg[0]["status"] == "decided", o)
    shutil.rmtree(tmp)
    for label, over in (("no_ruling", {"ruling": None}), ("ruling_missing_signed_on", {"ruling": {"ref": "r", "signed_by": "x"}}),
                        ("bad_schema", {"schema": 2})):
        root, tmp = with_root()
        (root / DEC).write_text(json.dumps(decisions(**over)))
        c, o = run(root, "--check")
        expect(f"decisions_file_{label}_fails", c == 1 and "ASSEMBLE_FAIL" in o, o)
        shutil.rmtree(tmp)

    # The two tools carry copies of the C6 vocabulary, the C6 ruling table and the behavioural scope; they must not drift.
    mjs = (HERE / "true_parity_check.mjs").read_text()
    vocab = re.search(r"const C6_DECISION_VOCAB = new Set\(\[(.*?)\]\);", mjs, re.S).group(1)
    expect("c6_vocab_matches_checker", re.findall(r"'([A-Z_]+)'", vocab) == list(A.C6_DECISION_VOCAB), vocab)
    rtable = re.search(r"const C6_RULINGS = \{(.*?)\n\};", mjs, re.S).group(1)
    ref = re.search(r"'([^']+#684 item 3)': \{ signed_on: '([0-9-]+)', words: new Set\(\[([^\]]*)\]\)", rtable)
    expect("c6_rulings_match_checker",
           ref is not None and list(A.C6_RULINGS) == [ref.group(1)] and A.C6_RULINGS[ref.group(1)]["signed_on"] == ref.group(2)
           and re.findall(r"'([A-Z_]+)'", ref.group(3)) == list(A.C6_RULINGS[ref.group(1)]["words"]), rtable)
    named = re.search(r"const BEHAVIOURAL_NAMED_SOURCE_IDS = new Set\(\[(.*?)\]\);", mjs, re.S).group(1)
    expect("behavioural_named_ids_match_checker", re.findall(r"'([^']+)'", named) == list(A.BEHAVIOURAL_NAMED_SOURCE_IDS), named)

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

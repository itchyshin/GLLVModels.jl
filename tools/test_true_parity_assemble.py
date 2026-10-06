#!/usr/bin/env python3
"""Negative and positive controls for tools/true_parity_assemble.py.

Each control builds a small throwaway root (temp dir) with the ledger layout the assembler
reads, then runs it. Run: python3 tools/test_true_parity_assemble.py
"""
from __future__ import annotations

import importlib.util
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
    (tmp / "docs/src").mkdir(parents=True)
    (tmp / "docs/src/page.md").write_text("# page\n")  # a docs/src file a KEPT_AS_JULIA_EXTRA basis can cite
    for rel in ("docs/src/assets/x.json", "docs/src/notes.txt", "docs/src/code.jl", "docs/src/cfg.toml", "docs/src/real.md.bak", "docs/src/sub/deep.md"):
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / rel).write_text("x\n")  # existing docs/src files that are not exact .md page citations, plus one nested page
    (tmp / "tools").mkdir(parents=True)
    (tmp / "tools/gen.py").write_text("# the committed generator of reverse-gap-decisions.json\n")
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
B_SID = "inference/CI-ROUTE-001"  # one of the 59 listed rows the behavioural tier covers
BID = "inference-CI-ROUTE-001"  # its scoreboard id
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
    root, tmp = with_root({"case-map-family.json": [row(kw.pop("sid", B_SID), **kw)]})
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

    # Campaign rows (itchyshin/GLLVModels.jl#684 item 4): a row that declares its clause may carry a C3/C4/C5 id,
    # but only the id its declared clause selects.
    root, tmp = with_root({"case-map-family.json": [row("family/GAUSSIAN-RSZ", tier="not_measured", clause="C3")]})
    c, o = run(root, "--check")
    expect("declared_clause_rsz_id_allowed", c == 1 and "ASSEMBLE_STALE" in o and "C3/C4/C5" not in o, o)
    c, o = run(root)
    expect("declared_clause_rsz_id_written", c == 0 and "| family-GAUSSIAN-RSZ" in (root / A.LEDGER / A.OUT_SCOREBOARD).read_text(), o)
    shutil.rmtree(tmp)
    root, tmp = with_root({"case-map-family.json": [row("family/GAUSSIAN-RSZ", tier="not_measured", clause="C4")]})
    c, o = run(root, "--check")
    expect("declared_clause_mismatch_fails", c == 1 and "declares clause C4" in o, o)
    shutil.rmtree(tmp)
    root, tmp = with_root({"case-map-data.json": [row("data/RD-X", tier="not_measured", clause="C4")]})
    c, o = run(root)
    expect("declared_clause_rd_id_selected_by_c4", c == 0 and "| data-RD-X" in (root / A.LEDGER / A.OUT_SCOREBOARD).read_text(), o)
    shutil.rmtree(tmp)
    root, tmp = with_root({"case-map-data.json": [row("data/PLAIN", tier="not_measured", clause="C9")]})
    c, o = run(root, "--check")
    expect("declared_clause_unknown_fails", c == 1 and "not one of C3, C4, C5" in o, o)
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

    # Campaign receipts (itchyshin/GLLVModels.jl#684 item 4): every comparison block re-derives from the committed raw files.
    wr = subprocess.run([sys.executable, str(HERE / "true_parity" / "campaign" / "write_receipts.py"), "--check"],
                        capture_output=True, text=True)
    expect("real_tree_campaign_receipts_rederive", wr.returncode == 0 and "CAMPAIGN_RECEIPTS_OK" in wr.stdout, wr.stdout + wr.stderr)
    # The urbanisation row has no committed raw outputs (unpublished data): --check must say so, never pass silently.
    expect("campaign_check_says_urbanisation_not_rederived",
           "raw outputs kept off the public repo; re-derive locally with URBMAP_ROOT set" in wr.stdout and "internal consistency only" in wr.stdout, wr.stdout)
    expect("urbanisation_raw_outputs_not_committed",
           not list((A.ROOT / A.LEDGER / "receipts/data/campaign/raw").glob("urban_*")), "urban raw outputs are in the tree")
    # Receipt wording: relative cases are named as such and say where the raw values are; cond(H) says how each engine computes it;
    # the C5 rule claims no R Hessian leg (the #593 receipts record none).
    nb2 = json.loads((A.ROOT / A.LEDGER / "receipts/family/campaign/NB2-LOG-RSZ.json").read_text())
    rel = [c for c in nb2["comparison"]["cases"] if "convention" in c]
    expect("relative_cases_named_and_located", len(rel) == 2 and all(c["quantity"].endswith("relative difference") and "r_raw" in c and "julia_raw" in c
           and "raw_values_location" in c for c in rel), json.dumps([c["quantity"] for c in rel]))
    expect("cond_H_method_stated", "exact = FALSE" in nb2["cond_H_statement"] and "exact 2-norm" in nb2["cond_H_statement"]
           and "kappa" in nb2["engines"]["R"]["cond_H_method"] and "2-norm" in nb2["engines"]["julia"]["cond_H_method"], nb2["cond_H_statement"])
    tmpl = json.loads((A.ROOT / A.LEDGER / "receipts/covariance/campaign/COV-TEMPORAL-RSZ.json").read_text())
    expect("temporal_cond_H_method_names_its_hessian", "ForwardDiff" in tmpl["engines"]["julia"]["cond_H_method"], tmpl["engines"]["julia"]["cond_H_method"])
    grp = json.loads((A.ROOT / A.LEDGER / "receipts/fit-input/campaign/GRP-UNIT.json").read_text())
    expect("c5_rule_claims_no_hessian_leg", "R_pdHess_true" not in grp["pass_rule"]["legs"] and "no R Hessian leg" in grp["pass_rule"]["rule"], grp["pass_rule"]["rule"])
    # Row tiers: a measured row that fails no number but lacks a required step reads PARTIAL, never FAIL; a row that also
    # misses a tolerance keeps numeric_fail; the phylo row does not bind (its R receipt is unqualified until the maintainer signs).
    def camp_row(mapname, sid):
        return next(r for r in json.loads((A.ROOT / A.LEDGER / mapname).read_text())["rows"] if r["source_id"] == sid)
    crabs = camp_row("case-map-data.json", "data/RD-CRABS-GAUSSIAN"); spider = camp_row("case-map-data.json", "data/RD-SPIDER-NB2")
    phylo = camp_row("case-map-covariance.json", "covariance/COV-PHYLO-LATENT-RSZ")
    expect("crabs_is_partial_not_fail", crabs["evidence_tier"] == "partial_case_not_executed" and "bridge route" in crabs["evidence"]["tier"], crabs["evidence"]["tier"])
    expect("row_with_failed_tolerance_stays_numeric_fail", spider["evidence_tier"] == "numeric_fail" and "bridge route" in spider["evidence"]["tier"] and "logLik" in spider["evidence"]["tier"], spider["evidence"]["tier"])
    expect("phylo_row_does_not_bind", phylo["evidence_tier"] == "partial_case_not_executed" and "receipt" not in phylo["evidence"] and "qualified = false" in phylo["evidence"]["tier"], json.dumps(phylo["evidence"]))

    # ... and a hand edit that keeps the receipt self-consistent must still fail: a changed value (with a matching
    # max_abs_diff and flag), a widened tolerance, and a changed large-vector difference.
    def tamper(label, edit, needle="DERIVATION DRIFT"):
        with tempfile.TemporaryDirectory() as td:
            led = Path(td) / A.LEDGER
            shutil.copytree(A.ROOT / A.LEDGER, led, ignore=shutil.ignore_patterns("*.md"))
            shutil.copytree(A.ROOT / "tools/true_parity/campaign/data", Path(td) / "tools/true_parity/campaign/data")
            shutil.copytree(A.ROOT / "docs/dev-log/core070/phylo-latent-p1", Path(td) / "docs/dev-log/core070/phylo-latent-p1")
            edit(led / "receipts")
            r = subprocess.run([sys.executable, str(HERE / "true_parity" / "campaign" / "write_receipts.py"), "--check", "--root", td],
                               capture_output=True, text=True)
            expect(label, r.returncode == 1 and "CAMPAIGN_RECEIPTS_BAD" in r.stdout and needle in r.stdout, r.stdout + r.stderr)

    def edit_case(rel, quantity, fn):
        def go(rec):
            f = rec / rel; d = json.loads(f.read_text())
            hit = [c for c in d["comparison"]["cases"] if c["quantity"] == quantity]
            assert hit, f"tamper target {quantity!r} not found in {rel}"   # a vacuous edit would pass for the wrong reason
            for c in hit: fn(c)
            f.write_text(json.dumps(d, indent=1) + "\n")
        return go
    def nudge(c):
        c["julia_value"] = [x + 1e-9 for x in c["r_value"]] if isinstance(c["r_value"], list) else c["r_value"] + 1e-9
        c["max_abs_diff"] = 1e-9
    tamper("campaign_check_catches_consistent_value_edit", edit_case("family/campaign/POISSON-LOG-RSZ.json", "LLt", nudge))
    tamper("campaign_check_catches_widened_tolerance", edit_case("family/campaign/POISSON-LOG-RSZ.json", "beta", lambda c: c.update(tolerance=1.0)))
    def edit_leg(rel, leg, value):
        def go(rec):
            f = rec / rel; d = json.loads(f.read_text()); d["pass_rule"]["legs"][leg] = value
            f.write_text(json.dumps(d, indent=1) + "\n")
        return go
    tamper("campaign_check_catches_phylo_qualification_forged", edit_leg("covariance/campaign/COV-PHYLO-LATENT-RSZ.json", "R_side_receipt_qualified_by_maintainer", True))
    # the urbanisation summary receipt is not re-derivable, but a self-contradicting edit must still fail
    def urban_flag(c): c["within_tolerance"] = not c["within_tolerance"]
    tamper("campaign_check_catches_urbanisation_summary_inconsistency",
           edit_case("data/campaign/RD-URBANISATION-BINOMIAL.json", "logLik", urban_flag), needle="SUMMARY INCONSISTENT")
    tamper("campaign_check_catches_large_vector_diff_edit", edit_case("data/campaign/RD-CRABS-GAUSSIAN.json", "linear predictor on the training data (link scale)", lambda c: c.update(max_abs_diff=1e-12)))
    # Ruling 2 (itchyshin/GLLVModels.jl#684 item 2): the behavioural tier.
    def behaves(name, status, root, tmp, want_bound=None, contains=None):
        c, o = run(root)
        txt = (root / A.LEDGER / A.OUT_SCOREBOARD).read_text()
        ok = c == 0 and status_of(root, BID) == status and (contains is None or contains in txt)
        if ok and want_bound is not None:
            out = checker_c1(root)
            m = re.search(r"\bbound_behavioural=(\d+)", out or "")
            ok = out is None or (m is not None and int(m.group(1)) == want_bound)
        expect(name, ok, f"{status_of(root, BID)} {o}")
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
    behaves("behavioural_class_of_other_kind_does_not_rescue", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0,
            contains='(refusal): R "r_wald" vs Julia "jl_wald" differ after canonicalisation (R: no class; Julia: no class)')
    root, tmp = behavioural_root(cases=[beh_case(source_id="inference/OTHER")])
    behaves("behavioural_entry_scoped_to_other_row_does_not_cover", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0,
            contains="without an applicable behaviour entry")
    root, tmp = behavioural_root(cases=[beh_case(source_id=B_SID)])
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
    for sid in ("isdm/X", "postfit/POSTFIT-SURFACE-nobs", "inference2/N", "data/RD-01", "inference/CI-ROUTE-008", "isdm/ISDM-WRONG-ID",
                "isdm/ISDM-LEGACY", "aghq/AGHQ-CTRL-THREE", "inference/CI-ROUTE-010", "inference/CI-ROUTE-011", "inference/", "inference/../isdm/X", "inference/CI-ROUTE-999",
                "Inference/CI-ROUTE-001"):
        root, tmp = behavioural_root(sid=sid)
        rid = A.scoreboard_id(sid)
        c, o = run(root)
        out = checker_c1(root)
        m = re.search(r"\bbound_behavioural=(\d+)", out or "")
        expect(f"behavioural_scope_{sid}_not_covered",
               c == 0 and status_of(root, rid) == "BEHAVIOURAL-UNVERIFIED" and "not covered by itchyshin/GLLVModels.jl#684 item 2" in (root / A.LEDGER / A.OUT_SCOREBOARD).read_text()
               and (out is None or (m is not None and int(m.group(1)) == 0)), f"{status_of(root, rid)} {o}")
        shutil.rmtree(tmp)
    for sid in A.BEHAVIOURAL_NAMED_SOURCE_IDS + A.BEHAVIOURAL_EXTENDED_SOURCE_IDS:
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
    behaves("behavioural_canonicalisation_is_side_specific", "BEHAVIOURAL-UNVERIFIED", root, tmp, want_bound=0,
            contains='R "jl_wald" vs Julia "r_wald" differ after canonicalisation (R: no class; Julia: no class)')
    dupc = dict(WALD, r=["other_r"], julia=["other_j"])
    root, tmp = behavioural_root(equiv={"schema": 1, "pin": "P1", "classes": [WALD, dupc]})
    c, o = run(root, "--check")
    expect("behavioural_duplicate_canonical_fails_run", c == 1 and "duplicate canonical" in o, o)
    shutil.rmtree(tmp)
    root, tmp = behavioural_root(equiv={"schema": 1, "pin": "P1", "classes": [WALD, dict(WALD, kind="refusal")]})
    c, o = run(root)
    expect("behavioural_same_canonical_in_two_kinds_ok", c == 0 and status_of(root, BID) == "EVIDENCED-BEHAVIOURAL", o)
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
    KEPT_BASIS = "documented in docs/src/page.md"

    def decisions(**over):
        d = {"schema": 1, "ruling": RULING, "criterion": "test", "generator": "tools/gen.py",
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
    (root / DEC).write_text(json.dumps(decisions(decisions={"no_such_export": {"decision": "KEPT_AS_JULIA_EXTRA", "basis": KEPT_BASIS}})))
    c, o = run(root, "--check")
    expect("decision_for_non_item_is_stale_and_fails", c == 1 and "ASSEMBLE_FAIL" in o and "no_such_export" in o and "stale" in o, o)
    shutil.rmtree(tmp)
    root, tmp = with_root()
    (root / DEC).write_text(json.dumps(decisions(decisions={"shared_name": {"decision": "KEPT_AS_JULIA_EXTRA", "basis": KEPT_BASIS}})))
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
    (root / DEC).write_text(json.dumps(decisions(decisions={"julia_only": {"decision": "KEPT_AS_JULIA_EXTRA", "basis": KEPT_BASIS}})))
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

    # ---- Fix round on PR #687 (three adversarial reviews). Each control fails on the head before the round
    # (6f546fb00) for the reason it names; a control that raises counts as FAIL, so this block also runs
    # against the old assembler (missing names then read as failures, not as a crash). ----
    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 -- a missing name or a crash is a failed control
            ok, detail = False, f"{type(e).__name__}: {e}"
        expect(name, ok, detail)

    def text_of(root):
        return (root / A.LEDGER / A.OUT_SCOREBOARD).read_text()

    def bound_beh(root):
        """bound_behavioural= from the checker's C1 over this root's assembled case map (None without node)."""
        out = checker_c1(root)
        if out is None:
            return None
        m = re.search(r"\bbound_behavioural=(\d+)", out)
        return int(m.group(1)) if m else -1

    def beh_agree(root, status, bound, contains=None, rid=BID):
        """Assembler status and checker C1 agree on one behavioural row."""
        c, o = run(root)
        st, txt = status_of(root, rid), text_of(root)
        b = bound_beh(root)
        ok = c == 0 and st == status and (contains is None or contains in txt) and (b is None or b == bound)
        return ok, f"code={c} status={st} checker_bound={b} {o}"

    # Item 1: the scope is a frozen explicit list, the same one in both tools.
    def scope_list_case(sid):
        def f():
            root, tmp = behavioural_root(sid=sid)
            try:
                return beh_agree(root, "BEHAVIOURAL-UNVERIFIED", 0, contains="not covered by itchyshin/GLLVModels.jl#684 item 2", rid=A.scoreboard_id(sid))
            finally:
                shutil.rmtree(tmp)
        return f
    for sid in ("inference/CI-ROUTE-008", "inference/CI-ROUTE-010", "inference/CI-ROUTE-011", "inference/",
                "inference/../isdm/X", "inference/CI-ROUTE-999", "inference/CI-ROUTE-005", "Inference/CI-ROUTE-001", "inference/CI-ROUTE-001 "):
        check(f"scope_frozen_list_rejects_{sid!r}", scope_list_case(sid))

    def all_63():
        # All 59 listed rows plus the three inference rows still excluded (CI-ROUTE-009 joined in the 2026-10-05
        # extension), every one tier behavioural with a scoped, matching entry.
        excluded = [f"inference/CI-ROUTE-{n:03d}" for n in (8, 10, 11)]
        sids = list(A.BEHAVIOURAL_INFERENCE_SOURCE_IDS) + excluded
        rows = [row(sid, tier="behavioural", executable_case_ids=[f"CASE-{i}"], evidence={"receipt": [BR]}) for i, sid in enumerate(sids)]
        root, tmp = with_root({"case-map-family.json": rows})
        try:
            cases = [beh_case(case_id=f"CASE-{i}", source_id=sid) for i, sid in enumerate(sids)]
            (root / BR).write_text(json.dumps({"behaviour": {"pin": "P1", "cases": cases}}))
            c, o = run(root)
            statuses = {sid: status_of(root, A.scoreboard_id(sid)) for sid in sids}
            n_ok = sum(v == "EVIDENCED-BEHAVIOURAL" for sid, v in statuses.items() if sid in A.BEHAVIOURAL_INFERENCE_SOURCE_IDS)
            n_bad = sum(statuses[sid] == "BEHAVIOURAL-UNVERIFIED" for sid in excluded)
            b = bound_beh(root)
            return c == 0 and n_ok == 59 and n_bad == 3 and (b is None or b == 59), f"code={c} evidenced={n_ok} unverified_excluded={n_bad} checker_bound={b} {o}"
        finally:
            shutil.rmtree(tmp)
    check("scope_all_59_listed_rows_bind_and_the_three_others_do_not", all_63)
    check("scope_extended_list_has_14_unique_ids_disjoint_from_the_other_lists", lambda: (
        len(A.BEHAVIOURAL_EXTENDED_SOURCE_IDS) == 14 and len(set(A.BEHAVIOURAL_EXTENDED_SOURCE_IDS)) == 14
        and not set(A.BEHAVIOURAL_EXTENDED_SOURCE_IDS) & (set(A.BEHAVIOURAL_INFERENCE_SOURCE_IDS) | set(A.BEHAVIOURAL_NAMED_SOURCE_IDS))
        and not {"inference/CI-ROUTE-008", "inference/CI-ROUTE-010", "inference/CI-ROUTE-011", "isdm/ISDM-LEGACY",
                 "isdm/ISDM-NO-TRAITS", "isdm/ISDM-WRONG-ID", "isdm/ISDM-WRONG-LINK"} & set(A.BEHAVIOURAL_EXTENDED_SOURCE_IDS),
        A.BEHAVIOURAL_EXTENDED_SOURCE_IDS))
    check("scope_list_has_59_unique_ids_none_of_008_to_011", lambda: (
        len(A.BEHAVIOURAL_INFERENCE_SOURCE_IDS) == 59 and len(set(A.BEHAVIOURAL_INFERENCE_SOURCE_IDS)) == 59
        and not {f"inference/CI-ROUTE-{n:03d}" for n in (8, 9, 10, 11)} & set(A.BEHAVIOURAL_INFERENCE_SOURCE_IDS), len(A.BEHAVIOURAL_INFERENCE_SOURCE_IDS)))

    def scope_ties_to_ledger():
        led = A.ROOT / A.LEDGER
        inf = {r["source_id"]: r.get("evidence_tier") for r in json.loads((led / "case-map-inference.json").read_text())["rows"]}
        listed = set(A.BEHAVIOURAL_INFERENCE_SOURCE_IDS)
        extended = set(A.BEHAVIOURAL_EXTENDED_SOURCE_IDS)  # CI-ROUTE-009 (2026-10-05): partial until its slice relabels it
        routing = {sid for sid, t in inf.items() if t in ("routing_control_flow", "reject_error_class")}
        missing = sorted(listed - set(inf))
        unlisted_routing = sorted(routing - listed)
        # A listed row is a routing or error-class row, or has already been relabelled behavioural by its slice;
        # no row outside the list may be behavioural; the four rows the ruling does not name are numeric or partial.
        bad_listed = sorted(sid for sid in listed if inf.get(sid) not in ("routing_control_flow", "reject_error_class", "behavioural"))
        stray = sorted(sid for sid, t in inf.items() if t == "behavioural" and sid not in listed | extended)
        bad_extended = sorted(sid for sid in extended & set(inf) if inf[sid] not in ("partial_non_numeric_case", "behavioural"))
        others = sorted(set(inf) - listed - extended)
        ok_others = all(inf[sid] in ("numeric", "partial_non_numeric_case") for sid in others) and not bad_extended
        named = set()
        for f in sorted(led.glob("case-map*.json")):
            named |= {r["source_id"] for r in json.loads(f.read_text())["rows"]}
        named_missing = sorted(set(A.BEHAVIOURAL_NAMED_SOURCE_IDS) - named)
        extended_missing = sorted(extended - named)
        ok = not (missing or unlisted_routing or bad_listed or stray or named_missing or extended_missing) and ok_others and len(others) == 3
        return ok, (f"missing={missing} unlisted_routing={unlisted_routing} bad_listed={bad_listed} stray={stray} others={others} "
                    f"named_missing={named_missing} extended_missing={extended_missing} bad_extended={bad_extended}")
    check("scope_list_ties_to_case_map_inference_json", scope_ties_to_ledger)

    # Item 2: labels match by class identity, not by canonical string (external review F8a, F8b, F8e, F8f).
    COLLIDE = {"kind": "route", "canonical": "wald", "r": ["r_wald"], "julia": ["jl_wald"], "basis": "R and Julia Wald routes."}
    STOP = {"kind": "refusal", "canonical": "stop", "r": ["stop_dup"], "julia": ["throw_X"], "basis": "R stop() and Julia throw."}

    def label_case(klass, kind, r, j, status, bound, contains=None):
        def f():
            root, tmp = behavioural_root(cases=[beh_case(kind=kind, r_observed=r, julia_observed=j)], equiv={"schema": 1, "pin": "P1", "classes": [klass]})
            try:
                return beh_agree(root, status, bound, contains)
            finally:
                shutil.rmtree(tmp)
        return f
    check("label_identity_F8a_julia_raw_wald_does_not_match_listed_r_wald",
          label_case(COLLIDE, "route", "r_wald", "wald", "BEHAVIOURAL-UNVERIFIED", 0, 'R "r_wald" vs Julia "wald" differ after canonicalisation (R: class "wald"; Julia: no class)'))
    check("label_identity_F8b_r_raw_wald_does_not_match_listed_jl_wald",
          label_case(COLLIDE, "route", "wald", "jl_wald", "BEHAVIOURAL-UNVERIFIED", 0, 'R "wald" vs Julia "jl_wald" differ after canonicalisation (R: no class; Julia: class "wald")'))
    check("label_identity_F8f_refusal_r_raw_stop_does_not_match_throw_X",
          label_case(STOP, "refusal", "stop", "throw_X", "BEHAVIOURAL-UNVERIFIED", 0, 'R "stop" vs Julia "throw_X" differ after canonicalisation (R: no class; Julia: class "stop")'))
    for r, j in (("r_wald", "r_wald"), ("jl_wald", "jl_wald"), ("wald", "wald")):
        check(f"label_identity_F8e_same_literal_{r}_on_both_sides_matches", label_case(COLLIDE, "route", r, j, "EVIDENCED-BEHAVIOURAL", 1))
    check("label_identity_two_members_of_one_class_match", label_case(COLLIDE, "route", "r_wald", "jl_wald", "EVIDENCED-BEHAVIOURAL", 1))

    def two_classes():
        other = dict(COLLIDE, canonical="profile", r=["r_prof"], julia=["jl_prof"])
        root, tmp = behavioural_root(cases=[beh_case(r_observed="r_wald", julia_observed="jl_prof")], equiv={"schema": 1, "pin": "P1", "classes": [COLLIDE, other]})
        try:
            return beh_agree(root, "BEHAVIOURAL-UNVERIFIED", 0, 'R: class "wald"; Julia: class "profile"')
        finally:
            shutil.rmtree(tmp)
    check("label_identity_members_of_two_classes_do_not_match", two_classes)

    # Item 3: one definition of a visible text in both ports.
    INVIS = [("FEFF", "﻿"), ("200B", "​"), ("0085", "\u0085"), ("00A0", " "), ("2060", "⁠"), ("blank", " \t\n")]

    def invisible_label(ch):
        def f():
            root, tmp = behavioural_root(cases=[beh_case(r_observed=ch, julia_observed=ch)])
            try:
                return beh_agree(root, "BEHAVIOURAL-UNVERIFIED", 0, "must be a non-empty string or an array of non-empty strings")
            finally:
                shutil.rmtree(tmp)
        return f
    for name, ch in INVIS:
        check(f"visible_label_U{name}_is_empty_in_assembler_and_checker", invisible_label(ch))

    def padded_label():
        root, tmp = behavioural_root(cases=[beh_case(r_observed="​wald﻿", julia_observed="​wald﻿")])
        try:
            return beh_agree(root, "EVIDENCED-BEHAVIOURAL", 1)
        finally:
            shutil.rmtree(tmp)
    check("visible_label_with_a_visible_character_is_visible", padded_label)

    def invisible_table(field, ch):
        def f():
            klass = dict(COLLIDE)
            if field in ("r", "julia"):
                klass[field] = [ch]
            else:
                klass[field] = ch
            root, tmp = behavioural_root(equiv={"schema": 1, "pin": "P1", "classes": [klass]})
            try:
                c, o = run(root, "--check")
                return c == 1 and "ASSEMBLE_FAIL" in o and "behaviour-equivalence.json" in o, o
            finally:
                shutil.rmtree(tmp)
        return f
    for field in ("canonical", "basis", "r", "julia"):
        for name, ch in INVIS:
            check(f"visible_table_{field}_U{name}_fails_the_run", invisible_table(field, ch))

    def probe_parity():
        # The two ports' definitions agree on probe characters assigned long ago (a full code-point sweep would depend on
        # each runtime's Unicode version, so it is not asserted).
        if not shutil.which("node"):
            return True, "node missing"
        mjs = (HERE / "true_parity_check.mjs").read_text()
        lit = re.search(r"const VISIBLE_RE = (/.*?/u);", mjs).group(1)
        probes = ["﻿", "​", "\u0085", " ", "⁠", " ", "　", "᠎", " ", "\t", "a", "Z", "7", ".", "-", "$", "+", "é", "日", "Ω",
                  "́", "\U0001f600", "​x", "﻿.", "\ud800", "­", "؜", ""]
        js = f"const R={lit}; console.log(JSON.stringify(JSON.parse(process.argv[1]).map((s)=>R.test(s))))"
        out = subprocess.run(["node", "-e", js, json.dumps(probes)], capture_output=True, text=True).stdout
        got, want = json.loads(out), [A.is_visible(p) for p in probes]
        return got == want, f"js={got} py={want}"
    check("visible_text_probe_characters_agree_with_the_checkers_regex", probe_parity)

    # Item 4: strict JSON types for schema fields.
    def schema_case(raw, ok_expected):
        def f():
            root, tmp = behavioural_root()
            try:
                (root / EQ_PATH).write_text('{"schema": ' + raw + ', "pin": "P1", "classes": []}')
                c, o = run(root)  # writes the outputs when the table is accepted
                a_ok = c == 0
                chk = None
                if shutil.which("node"):
                    env = dict(os.environ, PARITY_REF="FS", PARITY_FS_ROOT=str(root), PARITY_CASEMAP=str(A.LEDGER / A.OUT_CASEMAP))
                    chk = subprocess.run(["node", str(HERE / "true_parity_check.mjs"), "C1"], env=env, capture_output=True, text=True)
                c_ok = None if chk is None else chk.returncode == 0
                return a_ok == ok_expected and (c_ok is None or c_ok == ok_expected), f"assembler_ok={a_ok} checker_ok={c_ok} {o}"
            finally:
                shutil.rmtree(tmp)
        return f
    for raw in ("true", '"1"', "[1]", "null", "0", "2", "false"):
        check(f"schema_{raw}_rejected_by_assembler_and_checker", schema_case(raw, False))
    for raw in ("1", "1.0"):
        check(f"schema_{raw}_accepted_by_assembler_and_checker", schema_case(raw, True))

    def decisions_schema_true():
        root, tmp = with_root()
        try:
            (root / A.LEDGER / A.IN_REVERSE_GAP_DECISIONS).write_text(json.dumps(decisions(schema=True)))
            c, o = run(root, "--check")
            return c == 1 and "schema must be 1" in o, o
        finally:
            shutil.rmtree(tmp)
    check("schema_true_rejected_in_reverse_gap_decisions", decisions_schema_true)

    # Item 5: failure detection for a behavioural receipt (the numeric tier's list is unchanged).
    def receipt_case(name, receipt_extra=None, block_extra=None, cases=None, contains=None):
        def f():
            root, tmp = behavioural_root(receipt_extra=receipt_extra, block_extra=block_extra, cases=cases)
            try:
                return beh_agree(root, "BEHAVIOURAL-UNVERIFIED", 0, contains)
            finally:
                shutil.rmtree(tmp)
        check(name, f)
    for res in ("FAIL", None, False, 0, {"family": "gaussian"}, ["PASS"]):
        receipt_case(f"behavioural_receipt_result_{json.dumps(res)}_does_not_bind", receipt_extra={"result": res}, contains="receipt did not pass: result=")
    for res in ("PASS", "pass", True):
        def ok_result(res=res):
            root, tmp = behavioural_root(receipt_extra={"result": res})
            try:
                return beh_agree(root, "EVIDENCED-BEHAVIOURAL", 1)
            finally:
                shutil.rmtree(tmp)
        check(f"behavioural_receipt_result_{json.dumps(res)}_binds", ok_result)
    receipt_case("behavioural_receipt_behaviour_block_result_fail", block_extra={"result": "FAIL"}, contains="receipt did not pass: behaviour.result=")
    for bv in ({"status": "FAIL", "exit_code": 1}, {"status": None}, {"status": "PASSED"}, "FAIL"):
        receipt_case(f"behavioural_receipt_batch_verifier_{json.dumps(bv)}_does_not_bind", receipt_extra={"batch_verifier": bv}, contains="batch_verifier")
    cmp_ok = {"pin": "P1", "cases": [{"case_id": "C", "r_value": 1.0, "julia_value": 1.0, "tolerance": 1e-6}]}
    for key, val in (("status", "FAIL"), ("verdict", "FAIL"), ("batch_status", "FAIL"), ("harness_pass", False), ("result", "FAIL")):
        receipt_case(f"behavioural_receipt_comparison_{key}_{json.dumps(val)}_does_not_bind", receipt_extra={"comparison": dict(cmp_ok, **{key: val})}, contains=f"receipt did not pass: comparison.{key}=")
    for over in ({"status": "FAIL"}, {"result": "FAIL"}, {"match": False}, {"match": None}, {"match": "yes"}, {"match": 0}):
        receipt_case(f"behavioural_receipt_case_{json.dumps(over)}_does_not_bind", cases=[beh_case(**over)], contains="receipt did not pass: behaviour.cases[C].")
    for over in ({"status": "PASS"}, {"match": True}):
        def ok_case(over=over):
            root, tmp = behavioural_root(cases=[beh_case(**over)])
            try:
                return beh_agree(root, "EVIDENCED-BEHAVIOURAL", 1)
            finally:
                shutil.rmtree(tmp)
        check(f"behavioural_receipt_case_{json.dumps(over)}_binds", ok_case)

    def numeric_result_unread():
        root, tmp = numeric_root({"result": "FAIL"})
        try:
            c, o = run(root)
            return c == 0 and status_of(root) == "EVIDENCED", status_of(root)
        finally:
            shutil.rmtree(tmp)
    check("numeric_receipt_status_list_unchanged_result_fail_is_not_read", numeric_result_unread)

    # Item 6: an entry without source_id covers none of several rows citing the same case id.
    def two_rows(cases, status1, status2, bound, contains=None):
        def f():
            rows = [row("inference/CI-ROUTE-001", tier="behavioural", executable_case_ids=["C"], evidence={"receipt": [BR]}),
                    row("inference/CI-ROUTE-002", tier="behavioural", executable_case_ids=["C"], evidence={"receipt": [BR]})]
            root, tmp = with_root({"case-map-family.json": rows})
            try:
                (root / BR).write_text(json.dumps({"behaviour": {"pin": "P1", "cases": cases}}))
                c, o = run(root)
                s1, s2 = status_of(root, "inference-CI-ROUTE-001"), status_of(root, "inference-CI-ROUTE-002")
                b = bound_beh(root)
                ok = c == 0 and (s1, s2) == (status1, status2) and (b is None or b == bound) and (contains is None or contains in text_of(root))
                return ok, f"{s1} {s2} checker_bound={b} {o}"
            finally:
                shutil.rmtree(tmp)
        return f
    check("unscoped_entry_covers_none_of_two_rows_citing_one_case_id",
          two_rows([beh_case()], "BEHAVIOURAL-UNVERIFIED", "BEHAVIOURAL-UNVERIFIED", 0,
                   "cited by 2 rows, so an entry without source_id covers none of them; scope each entry with source_id"))
    check("unscoped_entry_each_row_scoped_both_bind",
          two_rows([beh_case(source_id="inference/CI-ROUTE-001"), beh_case(source_id="inference/CI-ROUTE-002")], "EVIDENCED-BEHAVIOURAL", "EVIDENCED-BEHAVIOURAL", 2))
    check("unscoped_entry_one_row_scoped_only_that_row_binds",
          two_rows([beh_case(source_id="inference/CI-ROUTE-001")], "EVIDENCED-BEHAVIOURAL", "BEHAVIOURAL-UNVERIFIED", 1))

    def unscoped_cross_tier():
        # a numeric row citing the same case id counts as a second citer
        rows = [row("inference/CI-ROUTE-001", tier="behavioural", executable_case_ids=["C"], evidence={"receipt": [BR]}),
                row("family/N", tier="numeric", executable_case_ids=["C"], evidence={"receipt": [RP]})]
        root, tmp = with_root({"case-map-family.json": rows})
        try:
            (root / BR).write_text(json.dumps({"behaviour": {"pin": "P1", "cases": [beh_case()]}}))
            (root / RP).write_text(json.dumps({"comparison": GOOD_CMP}))
            c, o = run(root)
            b = bound_beh(root)
            return c == 0 and status_of(root, BID) == "BEHAVIOURAL-UNVERIFIED" and (b is None or b == 0), f"{status_of(root, BID)} {b} {o}"
        finally:
            shutil.rmtree(tmp)
    check("unscoped_entry_a_numeric_row_citing_the_case_id_counts_as_a_citer", unscoped_cross_tier)

    # Item 7: integer equality tolerance is exactly 0.5 (the review's mutation `> 0.5` accepted 0.4).
    for tol in (0.4, 0.49, 0.5000001, 0.51):
        int_root(f"integer_equality_tolerance_{tol}_unverified", "NUMERIC-UNVERIFIED", 0, tolerance=tol)

    # Item 9: C6 basis rules, the generator is a file in the tree.
    def decision_fail(name, why, **over):
        def f():
            root, tmp = with_root()
            try:
                (root / A.LEDGER / A.IN_REVERSE_GAP_DECISIONS).write_text(json.dumps(decisions(**over)))
                c, o = run(root, "--check")
                return c == 1 and "ASSEMBLE_FAIL" in o and why in o, o
            finally:
                shutil.rmtree(tmp)
        check(name, f)
    kept = lambda basis: {"julia_only": {"decision": "KEPT_AS_JULIA_EXTRA", "basis": basis}}  # noqa: E731
    helper = lambda basis: {"julia_only": {"decision": "EXCLUDED_INTERNAL_HELPER", "basis": basis}}  # noqa: E731
    for ref in ("constructor", "__proto__", "toString", "hasOwnProperty", "valueOf", "isPrototypeOf"):
        decision_fail(f"c6_ruling_ref_{ref}_is_not_a_recognised_ruling", "is not a recognised signed ruling", ruling=dict(RULING, ref=ref))
    decision_fail("c6_generator_that_does_not_exist_fails", "is not a file in the tree", generator="tools/does_not_exist.py")
    decision_fail("c6_generator_that_is_a_directory_fails", "is not a file in the tree", generator="tools")
    decision_fail("c6_generator_outside_the_tree_fails", "is not a file in the tree", generator="../outside.py")
    decision_fail("c6_generator_absolute_path_fails", "is not a file in the tree", generator=str(Path(__file__).resolve()))
    # Review of #687 follow-up 3: GATES.md says a `..` path is refused; a `..` that resolves back into the tree was not.
    for label, gen in (("tools_dotdot_tools", "tools/../tools/gen.py"), ("docs_dotdot_tools", "docs/../tools/gen.py"),
                       ("trailing_dotdot", "tools/gen.py/.."), ("backslash_dotdot", "tools\\..\\tools\\gen.py"),
                       ("many_dotdot", "tools/a/../../tools/gen.py")):
        decision_fail(f"c6_generator_with_a_dotdot_segment_is_refused_{label}", "a path with a '..' segment is refused", generator=gen)
    for label, basis, why in (("no_path", "documented in the README", "must cite a docs/src/... file"), ("bare_dot", ".", "must cite a docs/src/... file"),
                              ("outside_docs_src", "see tools/gen.py", "must cite a docs/src/... file"), ("dot_dot", "docs/src/../x.md", "must cite a docs/src/... file"),
                              ("missing_file", "docs/src/missing.md", "docs/src/missing.md, which does not exist in the tree"),
                              ("one_missing_one_present", "docs/src/page.md and docs/src/gone.md", "docs/src/gone.md, which does not exist")):
        decision_fail(f"c6_kept_basis_{label}_fails", why, decisions=kept(basis))
    # Review of #687 follow-up 3: the cited page is matched exactly (an existing .md file written docs/src/<path>.md).
    NOT_EXACT = "not an exact docs/src/<path>.md page: "
    for label, basis, bad in (
            ("bak_suffix_on_an_existing_page", "see docs/src/page.md.bak", "docs/src/page.md.bak"),
            ("existing_bak_file", "docs/src/real.md.bak", "docs/src/real.md.bak"),
            ("mdx_suffix", "docs/src/page.mdx", "docs/src/page.mdx"),
            ("tilde_suffix", "docs/src/page.md~", "docs/src/page.md~"),
            ("trailing_slash", "docs/src/page.md/", "docs/src/page.md/"),
            ("existing_json", "docs/src/assets/x.json", "docs/src/assets/x.json"),
            ("existing_txt", "see docs/src/notes.txt", "docs/src/notes.txt"),
            ("existing_jl", "see docs/src/code.jl", "docs/src/code.jl"),
            ("existing_toml", "see docs/src/cfg.toml", "docs/src/cfg.toml"),
            ("dot_slash_prefix", "./docs/src/page.md", "./docs/src/page.md"),
            ("dot_dot_slash_prefix", "../docs/src/page.md", "../docs/src/page.md"),
            ("directory_prefix", "other/docs/src/page.md", "other/docs/src/page.md"),
            ("absolute_prefix", "/abs/docs/src/page.md", "/abs/docs/src/page.md"),
            ("letter_before_docs", "xdocs/src/page.md", "xdocs/src/page.md"),
            ("url", "https://example.org/docs/src/page.md", "//example.org/docs/src/page.md"),
            ("dot_dot_segment_resolving_to_a_page", "docs/src/../src/page.md", "docs/src/../src/page.md"),
            ("dot_dot_after_a_directory", "docs/src/sub/../page.md", "docs/src/sub/../page.md"),
            ("dot_segment", "docs/src/./page.md", "docs/src/./page.md"),
            ("doubled_slash", "docs/src//page.md", "docs/src//page.md"),
            ("directory_only", "see docs/src/", "docs/src/"),
            ("no_break_space_after_the_page", "docs/src/page.md" + chr(0xa0), None)):
        decision_fail(f"c6_kept_basis_exact_page_{label}_fails", "must cite a docs/src/... file; " + (NOT_EXACT + bad if bad else "not an exact"), decisions=kept(basis))
    for label, basis, bad in (("good_beside_a_bak", "docs/src/page.md and docs/src/gone.md.bak", "docs/src/gone.md.bak"),
                              ("good_beside_a_dot_slash_copy", "docs/src/page.md and ./docs/src/page.md", "./docs/src/page.md"),
                              ("good_beside_the_directory", "docs/src/ and docs/src/page.md", "docs/src/")):
        decision_fail(f"c6_kept_basis_exact_page_{label}_fails", f"cites {bad}, which is not an exact docs/src/<path>.md page", decisions=kept(basis))
    for name, ch in INVIS:
        decision_fail(f"c6_helper_basis_U{name}_fails", "basis must be a non-empty string", decisions=helper(ch))
        decision_fail(f"c6_kept_basis_U{name}_fails", "basis must be a non-empty string", decisions=kept(ch))
        decision_fail(f"c6_ruling_ref_U{name}_fails", "ruling without a ref", ruling=dict(RULING, ref=ch))

    def decisions_ok(name, dec, generator="tools/gen.py"):
        def f():
            root, tmp = with_root()
            try:
                (root / A.LEDGER / A.IN_REVERSE_GAP_DECISIONS).write_text(json.dumps(decisions(decisions=dec, generator=generator)))
                c, o = run(root)
                rg = json.loads((root / A.LEDGER / A.OUT_REVERSE_GAP).read_text())
                return c == 0 and rg[0]["status"] == "decided", o
            finally:
                shutil.rmtree(tmp)
        check(name, f)
    decisions_ok("c6_kept_basis_citing_an_existing_docs_src_file_ok", kept("see docs/src/page.md, section extras."))
    for i, basis in enumerate(("docs/src/page.md", "(docs/src/page.md)", "`docs/src/page.md`", "docs/src/page.md:12", "see docs/src/page.md.",
                               "[extras](docs/src/page.md#x)", "docs/src/page.md;docs/src/sub/deep.md", "docs/src/sub/deep.md",
                               "documented in\ndocs/src/page.md\nand elsewhere", "a \"docs/src/page.md\" b", "'docs/src/page.md'")):
        decisions_ok(f"c6_kept_basis_exact_page_accepted_{i}", kept(basis))
    decisions_ok("c6_helper_basis_with_a_visible_character_ok", helper("."))
    decisions_ok("c6_generator_with_a_plain_in_tree_path_still_ok", helper("."), generator="tools/gen.py")

    # Item 10: a Julia export has a gllvmTMB counterpart when the names match after tools/parity_ledger.py norm().
    def reverse_gap_norm():
        root, tmp = with_root()
        try:
            p = root / A.LEDGER / A.IN_REVERSE_GAP
            inp = json.loads(p.read_text())
            inp["r_names"] = ["zi_poisson", "lognormal", "print.foo", "AIC", "shared_name"]
            inp["julia_exports"] = [{"name": n, "kind": "Function"} for n in ("ZiPoisson", "Lognormal", "printfoo", "aic", "ZIPoissonX", "zi_poisson_extra", "julia_twin", "julia_only")]
            p.write_text(json.dumps(inp))
            c, o = run(root)
            rg = [x["name"] for x in json.loads((root / A.LEDGER / A.OUT_REVERSE_GAP).read_text())]
            return c == 0 and rg == ["ZIPoissonX", "zi_poisson_extra", "julia_only"], f"{rg} {o}"
        finally:
            shutil.rmtree(tmp)
    check("reverse_gap_names_match_after_parity_ledger_norm", reverse_gap_norm)

    def reverse_gap_norm_matches_ledger():
        spec = importlib.util.spec_from_file_location("pl", str(HERE / "parity_ledger.py"))
        pl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pl)
        names = ["ZiPoisson", "zi_poisson", "Lognormal", "lognormal", "print.foo", "printfoo", "AIC", "aic", "A_b.C", "abc", "x", "y"]
        return all(A.norm_name(n) == pl.norm(n) for n in names), "norm_name differs from tools/parity_ledger.py norm()"
    check("reverse_gap_norm_is_parity_ledgers_norm", reverse_gap_norm_matches_ledger)

    # Neighbour of item 3: the signature fields are trimmed with JS trim()'s character set in both tools, so a
    # signed_by or signed_on with a trailing or leading U+0085 or U+FEFF reads the same in the scoreboard and in C1.
    def signature_agrees(by, on, want=None):
        def f():
            root, tmp = with_root({"case-map-data.json": [row("data/S", disposition="DISPOSITION-SIGNED", signed_by=by, signed_on=on)]})
            try:
                c, o = run(root)
                st = status_of(root, "data-S")
                out = checker_c1(root)
                m = re.search(r"\bbound_signed=(\d+)", out or "")
                signed_checker = None if out is None else (m is not None and int(m.group(1)) == 1)
                signed_board = st == "DISPOSITION-SIGNED"
                ok = c == 0 and (signed_checker is None or signed_checker == signed_board) and (want is None or signed_board == want)
                return ok, f"board={st} checker_signed={signed_checker} {o}"
            finally:
                shutil.rmtree(tmp)
        return f
    check("signature_signed_on_ending_in_U0085_is_a_bad_date_in_both_tools", signature_agrees("Shinichi Nakagawa", "2026-09-27\u0085", want=False))
    check("signature_signer_starting_with_U0085_is_not_allowed_in_both_tools", signature_agrees("\u0085Shinichi Nakagawa", "2026-09-27", want=False))
    check("signature_clean_signer_and_date_are_signed_in_both_tools", signature_agrees("Shinichi Nakagawa", "2026-09-27", want=True))
    for name, ch in (("FEFF", "\ufeff"), ("00A0", "\u00a0"), ("2028", "\u2028"), ("200B", "\u200b"), ("001F", "\u001f")):
        check(f"signature_with_U{name}_padding_reads_the_same_in_both_tools", signature_agrees("Shinichi Nakagawa" + ch, "2026-09-27" + ch))
        check(f"signature_with_leading_U{name}_reads_the_same_in_both_tools", signature_agrees(ch + "Shinichi Nakagawa", ch + "2026-09-27"))

    # Review of #687 follow-up 1: a case-map `disposition` is free text copied into the scoreboard's Status column,
    # so it must never read there as a status the assembler (or the checker's DONE set) trusts. Each control fails
    # on the head before this change (a disposition of "EVIDENCED" became the status verbatim).
    def x2_counts(root):
        """(rows, done) from the checker's X2 over this root's assembled scoreboard; None without node."""
        if not shutil.which("node"):
            return None
        env = dict(os.environ, PARITY_REF="FS", PARITY_FS_ROOT=str(root))
        out = subprocess.run(["node", str(HERE / "true_parity_check.mjs"), "X2"], env=env, capture_output=True, text=True).stdout
        m = re.search(r"\brows=(\d+) done=(\d+)", out)
        return (int(m.group(1)), int(m.group(2))) if m else None

    def unbound_numeric_root(**row_kw):
        """A numeric row whose receipt exists but says FAIL: honestly NUMERIC-UNVERIFIED, so X2 must not count it."""
        return numeric_root({"verdict": "FAIL"}, **row_kw)

    def forged_disposition(disp, want_status="DISPOSITION-UNVERIFIED", why=None):
        def f():
            root, tmp = unbound_numeric_root(disposition=disp)
            try:
                c, o = run(root)
                st = status_of(root)
                x2 = x2_counts(root)
                ok = c == 0 and st == want_status and (x2 is None or x2[1] == 0) and (why is None or why in text_of(root))
                return ok, f"exit={c} status={st!r} x2={x2} {o}"
            finally:
                shutil.rmtree(tmp)
        return f

    def honest_baseline():
        root, tmp = unbound_numeric_root()
        try:
            c, o = run(root)
            x2 = x2_counts(root)
            return c == 0 and status_of(root) == "NUMERIC-UNVERIFIED" and (x2 is None or x2[1] == 0), f"{status_of(root)} {x2} {o}"
        finally:
            shutil.rmtree(tmp)
    check("disposition_status_baseline_failed_receipt_row_is_not_done", honest_baseline)

    DONE_WORDS = ("EVIDENCED", "EVIDENCED-BEHAVIOURAL", "DISPOSITION-SIGNED")
    check("disposition_status_word_is_unverified_and_not_counted_by_x2_EVIDENCED", forged_disposition("EVIDENCED", why="reserved status word"))
    check("disposition_status_word_is_unverified_and_not_counted_by_x2_EVIDENCED_BEHAVIOURAL", forged_disposition("EVIDENCED-BEHAVIOURAL", why="reserved status word"))
    check("disposition_status_word_is_unverified_and_not_counted_by_x2_DISPOSITION_SIGNED_no_signature", forged_disposition("DISPOSITION-SIGNED"))
    check("disposition_status_word_with_spaces_is_unverified_and_not_counted_by_x2", forged_disposition("  EVIDENCED  "))
    check("disposition_status_word_with_a_trailing_U_FEFF_is_unverified_and_not_counted_by_x2", forged_disposition("EVIDENCED\ufeff"))
    check("disposition_status_word_with_a_leading_U_00A0_is_unverified_and_not_counted_by_x2", forged_disposition("\u00a0EVIDENCED-BEHAVIOURAL"))
    check("disposition_status_word_lower_case_is_unverified", forged_disposition("evidenced"))
    check("disposition_with_an_inner_CR_is_unverified", forged_disposition("see\rnotes"))
    check("disposition_pipe_forged_status_column_is_unverified_and_not_counted_by_x2", forged_disposition(f"EVIDENCED | {RP} | forged |", why="table delimiter"))
    check("disposition_with_a_line_break_is_unverified_not_a_failed_run", forged_disposition("EVIDENCED\n| family-FAKE | r | EVIDENCED | " + RP + " | n |", why="table delimiter or line break"))
    check("disposition_with_an_inner_line_break_is_unverified_not_a_failed_run", forged_disposition("see\nnotes", why="table delimiter or line break"))

    def reserved_variants():
        def pads(w):
            return [w, w.lower(), w.title(), f" {w} ", f"{w}\r", f"{w}\n", f"\t{w}", f"{w}\ufeff", f"\u00a0{w}", f"\u2028{w}\u2029", f"\u3000{w}\u3000"]
        bad = []
        for w in sorted(A.RESERVED_STATUS_WORDS):
            for d in pads(w):
                st, _why = A.derive_status(row("family/N", disposition=d), A.ROOT, {})
                if st != "DISPOSITION-UNVERIFIED":
                    bad.append((d, st))
        return not bad and len(A.RESERVED_STATUS_WORDS) >= 15, f"{bad[:5]} words={len(A.RESERVED_STATUS_WORDS)}"
    check("every_reserved_status_word_in_every_padding_and_case_is_unverified_as_a_disposition", reserved_variants)

    def odd_dispositions():
        bad = []
        for d in (5, 0, True, False, 1.5, ["EVIDENCED"], {"a": 1}, "", "   ", "\ufeff", "\u0085"):
            st, _why = A.derive_status(row("family/N", disposition=d), A.ROOT, {})
            if st != "DISPOSITION-UNVERIFIED":
                bad.append((d, st))
        return not bad, str(bad)
    check("non_string_or_blank_disposition_is_unverified", odd_dispositions)

    def plain_dispositions_still_pass_through():
        want = {"BLOCKED_NEEDS_JULIA_SURFACE": "NEEDS-SURFACE", "outside_boundary": "outside_boundary",
                "see EVIDENCED notes": "see EVIDENCED notes", "EVIDENCED-BEHAVIOURALX": "EVIDENCED-BEHAVIOURALX",
                "NEEDS_JULIA_SURFACE": "NEEDS-SURFACE", "PARTIAL_X": "PARTIAL_X"}
        got = {d: A.derive_status(row("family/N", disposition=d), A.ROOT, {})[0] for d in want}
        return got == want, f"{got}"
    check("plain_dispositions_keep_their_old_status", plain_dispositions_still_pass_through)

    def disposition_signed_still_signs():
        root, tmp = with_root({"case-map-data.json": [row("data/S", **sig)]})
        try:
            c, o = run(root)
            x2 = x2_counts(root)
            return c == 0 and status_of(root, "data-S") == "DISPOSITION-SIGNED" and (x2 is None or x2[1] >= 1), f"{status_of(root, 'data-S')} {x2} {o}"
        finally:
            shutil.rmtree(tmp)
    check("a_valid_signature_still_reads_disposition_signed_after_the_reserved_word_guard", disposition_signed_still_signs)

    def reserved_words_cover_emitted_statuses():
        import inspect
        src = inspect.getsource(A.derive_status) + inspect.getsource(A.disposition_status)
        emitted = set(re.findall(r'"([A-Z]+(?:-[A-Z]+)*)"', src))
        missing = sorted(emitted - A.RESERVED_STATUS_WORDS)
        return bool(emitted) and not missing and set(A.TIER_BUCKET.values()) <= A.RESERVED_STATUS_WORDS and set(A.STATUS_ORDER) <= A.RESERVED_STATUS_WORDS, f"missing={missing} emitted={sorted(emitted)}"
    check("reserved_status_words_cover_every_status_derive_status_emits", reserved_words_cover_emitted_statuses)

    def checker_done_set_is_reserved():
        mjs_text = (HERE / "true_parity_check.mjs").read_text()
        done = re.findall(r"'([A-Z-]+)'", re.search(r"const DONE = new Set\(\[(.*?)\]\);", mjs_text).group(1))
        return done == list(DONE_WORDS) and set(done) <= A.RESERVED_STATUS_WORDS, str(done)
    check("checker_done_set_is_a_subset_of_the_reserved_status_words", checker_done_set_is_reserved)

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
    ext = re.search(r"const BEHAVIOURAL_EXTENDED_SOURCE_IDS = new Set\(\[(.*?)\]\);", mjs, re.S).group(1)
    expect("behavioural_extended_ids_match_checker", re.findall(r"'([^']+)'", ext) == list(A.BEHAVIOURAL_EXTENDED_SOURCE_IDS), ext)
    infer = re.search(r"const BEHAVIOURAL_INFERENCE_SOURCE_IDS = new Set\(\[(.*?)\]\);", mjs, re.S).group(1)
    expect("behavioural_inference_ids_match_checker", re.findall(r"'([^']+)'", infer) == list(A.BEHAVIOURAL_INFERENCE_SOURCE_IDS), infer)
    expect("receipt_status_fields_match_checker",
           re.findall(r"'([a-z_]+)'", re.search(r"const RECEIPT_STATUS_FIELDS = \[(.*?)\];", mjs).group(1)) == A.STATUS_FIELDS
           and "const BEHAVIOURAL_STATUS_FIELDS = [...RECEIPT_STATUS_FIELDS, 'result'];" in mjs
           and A.BEHAVIOURAL_STATUS_FIELDS == A.STATUS_FIELDS + ["result"], "RECEIPT_STATUS_FIELDS or BEHAVIOURAL_STATUS_FIELDS drifted")
    js_split = re.search(r"const DOCS_SRC_SPLIT_RE = /(.*)/;", mjs).group(1).replace("\\/", "/")
    js_page = re.search(r"const DOCS_SRC_PAGE_RE = /(.*)/;", mjs).group(1).replace("\\/", "/")
    expect("docs_src_split_regex_matches_checker", js_split == A.DOCS_SRC_SPLIT_RE.pattern, f"{js_split} != {A.DOCS_SRC_SPLIT_RE.pattern}")
    expect("docs_src_page_regex_matches_checker", js_page == A.DOCS_SRC_PAGE_RE.pattern, f"{js_page} != {A.DOCS_SRC_PAGE_RE.pattern}")

    # The two tools must accept and refuse the same bases: run the checker's C6 on the corpus and compare, item by item,
    # with the assembler's kept_basis_problem over the same tree.
    def basis_agreement():
        if not shutil.which("node"):
            return True, "node not available"
        corpus = ["docs/src/page.md", "see docs/src/page.md, section extras.", "x docs/src/page.md#extras", "(docs/src/page.md)", "docs/src/page.md.",
                  "docs/src/sub/deep.md", "docs/src/page.md and docs/src/sub/deep.md", "docs/src/page.md.bak", "docs/src/real.md.bak", "docs/src/page.mdx",
                  "docs/src/page.md~", "docs/src/page.md/", "docs/src/assets/x.json", "docs/src/notes.txt", "docs/src/code.jl", "docs/src/cfg.toml",
                  "./docs/src/page.md", "../docs/src/page.md", "other/docs/src/page.md", "/abs/docs/src/page.md", "xdocs/src/page.md",
                  "https://example.org/docs/src/page.md", "docs/src/../src/page.md", "docs/src/sub/../page.md", "docs/src/./page.md", "docs/src//page.md",
                  "docs/src/", "docs/src", "docs/src/missing.md", "docs/src/page.md and docs/src/gone.md", "docs/src/page.md and docs/src/gone.md.bak",
                  "docs/src/page.md" + chr(0xa0), chr(0xa0) + "docs/src/page.md", "docs/src/page.md" + chr(0xfeff), "docs/src/page.md\u2028", "DOCS/SRC/page.md",
                  "docs\\src\\page.md", "nothing here", ".", "docs/src/.hidden.md", "docs/src/a b.md", "docs/src/pa\tge.md", "docs/src/page.md\tdocs/src/sub/deep.md",
                  "a,docs/src/page.md,b", "a;docs/src/page.md;b", "a'docs/src/page.md'b", "<docs/src/page.md>", "docs/src/page.md!", "docs/src/page.md?", "docs/src/page.md..."]
        root, tmp = with_root()
        try:
            items = [{"source_id": f"julia-export/b{i}", "decision": "KEPT_AS_JULIA_EXTRA", "basis": b,
                      "ruling": {"ref": "itchyshin/GLLVModels.jl#684 item 3", "signed_by": "Shinichi Nakagawa", "signed_on": "2026-10-02"}}
                     for i, b in enumerate(corpus)]
            (root / A.LEDGER / "reverse-gap.json").write_text(json.dumps(items))
            env = dict(os.environ, PARITY_REF="FS", PARITY_FS_ROOT=str(root))
            out = subprocess.run(["node", str(HERE / "true_parity_check.mjs"), "C6"], env=env, capture_output=True, text=True).stdout
            refused_js = {int(m) for m in re.findall(r"julia-export/b(\d+)\(KEPT_AS_JULIA_EXTRA basis ", out)}
            refused_py = {i for i, b in enumerate(corpus) if A.kept_basis_problem(b, root) is not None}
            diff = sorted(refused_js ^ refused_py)
            return bool(refused_py) and len(refused_py) < len(corpus) and not diff, f"differ on {[(i, corpus[i]) for i in diff]}; js refused {len(refused_js)}, py refused {len(refused_py)}"
        finally:
            shutil.rmtree(tmp)
    check("kept_basis_accept_and_refuse_agree_between_checker_and_assembler", basis_agreement)

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

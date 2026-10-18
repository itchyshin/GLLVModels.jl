"""Focused tests for the P1 iSDM behavioural overlay."""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core070_isdm_p1_receipts as I  # noqa: E402
import core070_behaviour_receipts as B  # noqa: E402


def tracked_batch_receipts(source_ids):
    rows = {r["source_id"]: r for r in I.load(I.P0_CASEMAP)["rows"]}
    receipts = {}
    for sid in source_ids:
        for cid in rows[sid]["executable_case_ids"]:
            path = I.ROOT / I.REC_REL / "cases" / f"{cid}.json"
            receipts[cid] = I.receipt_info(str(path.relative_to(I.ROOT)), I.load(path))
    return rows, receipts


def positive_rows_bind_only_through_the_measured_behaviour_receipts():
    source_ids = sorted(I.BEHAVIOURAL_SOURCE_IDS)
    p0, receipts = tracked_batch_receipts(source_ids)
    rows, counts = I.build_rows(source_ids, receipts)
    by_id = {r["source_id"]: r for r in rows}
    assert set(by_id) == I.BEHAVIOURAL_SOURCE_IDS
    assert counts["behavioural"] == len(source_ids)
    assert counts["needs_surface_r_side_measured"] == 0
    for sid in source_ids:
        assert by_id[sid]["evidence_tier"] == "behavioural", sid
        assert by_id[sid]["classification"] == p0[sid]["classification"], sid
        assert by_id[sid]["executable_case_ids"] == p0[sid]["executable_case_ids"], sid
        expected = [receipts[cid][0] for cid in p0[sid]["executable_case_ids"]]
        assert by_id[sid]["evidence"]["receipt"] == expected, sid


def stale_or_mismatched_behaviour_does_not_promote():
    sid = "isdm/ISDM-COUNT"
    cid = "CORE070-ISDM-COUNT-PAIRED-CONTROL"
    rel = f"{I.REC_REL}/cases/{cid}.json"
    row = {"source_id": sid, "classification": "required_core", "executable_case_ids": [cid],
           "evidence_tier": "needs_surface_r_side_measured", "measured_against": I.P1_SHA,
           "evidence": {"non_binding_receipts": [rel]}}

    stale = copy.deepcopy(row)
    stale["measured_against"] = "old-reference"
    counts = {"needs_surface_r_side_measured": 1, "behavioural": 0}
    assert not I.behavioural_overlay(stale, counts)
    assert stale["evidence_tier"] == "needs_surface_r_side_measured"
    assert counts == {"needs_surface_r_side_measured": 1, "behavioural": 0}

    # Run the real receipt checker against an isolated copy with one raw-side
    # class label changed. This exercises failure closed at the equivalence bind.
    receipt_path = I.ROOT / rel
    receipt = I.load(receipt_path)
    receipt["behaviour"]["cases"][0]["julia_observed"] = "deliberately-wrong-class"
    old_root, old_cites, old_asm = B.ROOT, B._CITES, B._ASM
    asm = old_asm or B.assembler()
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            target = root / rel
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(receipt))
            equiv = root / I.OUT_REL / "behaviour-equivalence.json"
            equiv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(I.ROOT / I.OUT_REL / "behaviour-equivalence.json", equiv)
            B.ROOT = root
            B._CITES = {cid: {sid}}
            B._ASM = asm
            bad = copy.deepcopy(row)
            counts = {"needs_surface_r_side_measured": 1, "behavioural": 0}
            assert not I.behavioural_overlay(bad, counts)
            assert bad["evidence_tier"] == "needs_surface_r_side_measured"
            assert bad["executable_case_ids"] == [cid]
            assert counts == {"needs_surface_r_side_measured": 1, "behavioural": 0}
    finally:
        B.ROOT, B._CITES, B._ASM = old_root, old_cites, old_asm


def generator_check_ignores_only_the_separately_rederived_behaviour_block():
    body = {"read_from": {"fixture": "sha256"}, "r_expression": "admitted(fixture)"}
    augmented = {**body, "behaviour": {"pin": "P1", "cases": []}, "glvmodels_commit": "run-commit"}
    assert I.comparable_receipt_body(augmented) == body
    changed_owned = {**augmented, "r_expression": "changed"}
    assert I.comparable_receipt_body(changed_owned) != body


def historical_batch_reuse_is_explicit_and_requires_a_clean_absent_surface_run():
    select = getattr(I, "select_run_commit", None)
    assert callable(select), "writer must expose a separately gated historical-run path"
    run_dir = I.ROOT / I.REC_REL / I.BATCH
    current_head = I.git("rev-parse", "HEAD").stdout.strip()
    historical = select(run_dir, current_head, allow_historical=True)
    assert historical == I.load(run_dir / "run-commit.json")["glvmodels_commit"]
    assert historical != current_head
    assert I.julia_surface_at(historical) == []

    try:
        select(run_dir, current_head, allow_historical=False)
    except SystemExit as exc:
        assert "not at clean HEAD" in str(exc)
    else:
        raise AssertionError("historical run was accepted without explicit authorization")


def case_map_generation_binds_the_exact_tool_bytes_and_dirty_paths():
    current_head = I.git("rev-parse", "HEAD").stdout.strip()
    dirty = ["tools/core070_isdm_p1_receipts.py"]
    generation = I.record_case_map_generation(current_head, dirty, "361516c7d03bea1d49b7b5639c04b153c0e77f94")
    assert generation["tool_sha256"] == I.sha(Path(I.__file__))
    assert generation["worktree_dirty"] == dirty
    assert generation["measurement_commit"] == "361516c7d03bea1d49b7b5639c04b153c0e77f94"
    assert generation["mode"] == "historical-absent-surface"


def regenerating_r_receipts_preserves_only_the_separately_owned_behaviour_blocks():
    write = getattr(I, "write_case_receipt", None)
    assert callable(write), "the R receipt writer must preserve independent behavior evidence"
    fresh = {"case_id": "case-1", "verdict": "PASS", "r_expression": "original"}
    old = {**fresh, "behaviour": {"verdict": "PASS"}, "behaviour_not_bound": {"reason": "held"},
           "unowned": "discard"}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "case.json"
        path.write_text(json.dumps(old))
        result = write(path, fresh)
        reread = json.loads(path.read_text())
    assert reread["behaviour"] == old["behaviour"]
    assert reread["behaviour_not_bound"] == old["behaviour_not_bound"]
    assert "unowned" not in reread
    assert reread["r_expression"] == "original"
    assert result == reread


def main():
    tests = [regenerating_r_receipts_preserves_only_the_separately_owned_behaviour_blocks,
             positive_rows_bind_only_through_the_measured_behaviour_receipts,
             stale_or_mismatched_behaviour_does_not_promote,
             generator_check_ignores_only_the_separately_rederived_behaviour_block,
             historical_batch_reuse_is_explicit_and_requires_a_clean_absent_surface_run,
             case_map_generation_binds_the_exact_tool_bytes_and_dirty_paths]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"{len(tests)}/{len(tests)} controls passed")


if __name__ == "__main__":
    main()

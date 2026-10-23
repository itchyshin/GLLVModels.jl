"""Focused tests for the approved five-row P1 ISDM behaviour overlay."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core070_isdm_p1_receipts as I  # noqa: E402
import core070_behaviour_receipts as B  # noqa: E402


def source_rows():
    return {r["source_id"]: r for r in I.load(I.P0_CASEMAP)["rows"]}


def positive_rows_bind_from_the_five_measured_receipts():
    sources = sorted(I.FIRST_SEVEN_ISDM)
    p0 = source_rows()
    receipts = {}
    for sid in sources:
        for cid in p0[sid]["executable_case_ids"]:
            path = I.ROOT / I.REC_REL / "cases" / f"{cid}.json"
            receipts[cid] = I.receipt_info(str(path.relative_to(I.ROOT)), I.load(path))

    rows, counts = I.build_rows(sources, receipts)
    by_source = {r["source_id"]: r for r in rows}
    assert set(by_source) == I.FIRST_SEVEN_ISDM
    assert counts["behavioural"] == 5
    assert counts["needs_surface_r_side_measured"] == 0
    for sid in sorted(I.FIRST_SEVEN_ISDM):
        row = by_source[sid]
        assert row["evidence_tier"] == "behavioural", sid
        assert row["measured_against"] == I.P1_SHA, sid
        assert row["classification"] == p0[sid]["classification"], sid
        assert row["executable_case_ids"] == p0[sid]["executable_case_ids"], sid
        expected = [receipts[cid][0] for cid in p0[sid]["executable_case_ids"]]
        assert row["evidence"]["receipt"] == expected, sid


def stale_reference_and_wrong_or_out_of_scope_rows_are_refused():
    sid = "isdm/ISDM-COUNT"
    expected = source_rows()[sid]["executable_case_ids"]
    base = {"source_id": sid, "classification": "required_core",
            "executable_case_ids": expected, "evidence_tier": "needs_surface_r_side_measured",
            "measured_against": I.P1_SHA,
            "evidence": {"non_binding_receipts": [
                f"{I.REC_REL}/cases/{expected[0]}.json"]}}
    counts = {"needs_surface_r_side_measured": 1, "behavioural": 0}

    stale = copy.deepcopy(base)
    stale["measured_against"] = "stale-reference"
    assert not I.behavioural_overlay(stale, counts)
    assert counts == {"needs_surface_r_side_measured": 1, "behavioural": 0}

    substituted = copy.deepcopy(base)
    substituted["executable_case_ids"] = ["CORE070-ISDM-WRAPPER-LAW-PAIRED-CONTROL"]
    with patch.object(B, "overlay_row", wraps=B.overlay_row) as dispatch:
        assert not I.behavioural_overlay(substituted, counts)
        dispatch.assert_not_called()

    out_of_scope = copy.deepcopy(base)
    out_of_scope["source_id"] = "isdm/ISDM-NO-TRAITS"
    with patch.object(B, "overlay_row", wraps=B.overlay_row) as dispatch:
        assert not I.behavioural_overlay(out_of_scope, counts)
        dispatch.assert_not_called()
    assert counts == {"needs_surface_r_side_measured": 1, "behavioural": 0}


def numeric_rows_are_not_sent_to_the_behaviour_overlay():
    sid = "isdm/ISDM-COUNT"
    cid = source_rows()[sid]["executable_case_ids"][0]
    row = {"source_id": sid, "classification": "required_core",
           "executable_case_ids": [cid], "evidence_tier": "numeric",
           "measured_against": I.P1_SHA,
           "evidence": {"receipt": [f"{I.REC_REL}/cases/{cid}.json"]}}
    counts = {"numeric": 1, "behavioural": 0}
    with patch.object(B, "overlay_row", wraps=B.overlay_row) as dispatch:
        assert not I.behavioural_overlay(row, counts)
        dispatch.assert_not_called()
    assert row["evidence_tier"] == "numeric"
    assert counts == {"numeric": 1, "behavioural": 0}


def run():
    tests = [positive_rows_bind_from_the_five_measured_receipts,
             stale_reference_and_wrong_or_out_of_scope_rows_are_refused,
             numeric_rows_are_not_sent_to_the_behaviour_overlay]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} focused ISDM receipt tests passed")


if __name__ == "__main__":
    run()

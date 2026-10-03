#!/usr/bin/env python3
"""Controls for tools/core070_behaviour_receipts.py (itchyshin/GLLVModels.jl#684 item 2).

Positive controls run on the tracked ledger (read only). Negative controls copy a receipt and the equivalence
table into a throwaway root and ask the assembler's port of the checker rule (the same rule the checker applies)
whether the row still binds. Run: python3 tools/test_core070_behaviour_receipts.py
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core070_behaviour_receipts as B  # noqa: E402

A = B.assembler()
ROOT = B.ROOT
INF_CASES = f"{B.INF}/cases"


def case_receipt(case_id):
    return B.load(ROOT / INF_CASES / f"{case_id}.json")


def row_for(source_id, case_id, tier="routing_control_flow"):
    return {"source_id": source_id, "classification": "compatibility_adapter", "executable_case_ids": [case_id],
            "disposition": None, "evidence_tier": tier, "measured_against": B.P1_SHA,
            "evidence": {"non_binding_receipts": [f"{INF_CASES}/{case_id}.json"], "tier": "x"},
            "measured_result": {}}


def make_root(tmp: Path, receipt: dict, case_id: str, equivalence=None) -> Path:
    led = tmp / B.LEDGER
    (tmp / INF_CASES).mkdir(parents=True)
    led.mkdir(parents=True, exist_ok=True)
    (tmp / INF_CASES / f"{case_id}.json").write_text(json.dumps(receipt))
    (led / "behaviour-equivalence.json").write_text(json.dumps(equivalence or B.equivalence_doc()))
    return tmp


def problem(root, source_id, case_id):
    row = dict(row_for(source_id, case_id), evidence_tier="behavioural",
               evidence={"receipt": [f"{INF_CASES}/{case_id}.json"]})
    return A.behavioural_receipt_problem(row, root, A.behaviour_equivalence(root))


results = []


def test(fn):
    results.append(fn)
    return fn


LAMBDA = "CORE070-INFERENCE-LAMBDA-CI-METHOD-ROUTE"
SIGMA_B = "CORE070-INFERENCE-SIGMA-B-CI-METHOD-ROUTE"


@test
def tracked_receipts_and_table_are_current():
    assert B.check_problems() == [], B.check_problems()


@test
def table_is_well_formed_and_cites_both_engines():
    A.behaviour_equivalence(ROOT)  # raises Fail on an ambiguous table, an empty basis, a bad kind
    for c in B.CLASSES:
        assert re.search(r"\.R:\d|R/[\w.-]+\.R:\d|R/gllvmTMB\.R:\d", c["basis"]), f"no P1 R line in {c['canonical']}"
        assert re.search(r"src/[\w/]+\.jl:\d", c["basis"]), f"no Julia src line in {c['canonical']}"
        assert "—" not in c["basis"] and "–" not in c["basis"], "no dashes in prose"


@test
def first_sentence_unwraps_r_line_breaks():
    msg = "A profile interval for canonical full-covariance repeatability is not\ncurrently available.\n\u2139 More."
    assert B.first_sentence(msg) == "A profile interval for canonical full-covariance repeatability is not currently available."


@test
def matching_row_binds():
    rec = case_receipt(LAMBDA)
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t), rec, LAMBDA)
        assert problem(root, "inference/CI-ROUTE-001", LAMBDA) is None


@test
def swapped_julia_label_does_not_bind():
    rec = case_receipt(LAMBDA)
    bad = copy.deepcopy(rec)
    for e in bad["behaviour"]["cases"]:
        if e["source_id"] == "inference/CI-ROUTE-003":
            e["julia_observed"] = "lambda:profile"  # R says wald
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t), bad, LAMBDA)
        assert "differ after canonicalisation" in problem(root, "inference/CI-ROUTE-003", LAMBDA)
        assert problem(root, "inference/CI-ROUTE-001", LAMBDA) is None  # the neighbour is untouched


@test
def other_targets_route_does_not_match():
    rec = case_receipt(SIGMA_B)
    bad = copy.deepcopy(rec)
    for e in bad["behaviour"]["cases"]:
        if e["source_id"] == "inference/CI-ROUTE-044":
            e["julia_observed"] = "communality:wald_derived"  # a real Julia label, wrong quantity
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t), bad, SIGMA_B)
        assert "differ after canonicalisation" in problem(root, "inference/CI-ROUTE-044", SIGMA_B)


@test
def row_without_its_own_entry_does_not_bind():
    rec = case_receipt(SIGMA_B)
    for sid in ("inference/CI-ROUTE-043", "inference/CI-ROUTE-055"):  # the DEFAULT rows get no entry
        assert not any(e.get("source_id") == sid for e in rec["behaviour"]["cases"])
        assert any(n["source_id"] == sid and n["reason"] == "default" for n in rec["behaviour_not_bound"])
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t), rec, SIGMA_B)
        assert "without an applicable behaviour entry" in problem(root, "inference/CI-ROUTE-043", SIGMA_B)


@test
def failed_receipt_verdict_does_not_bind():
    bad = copy.deepcopy(case_receipt(LAMBDA))
    bad["verdict"] = "FAIL"
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t), bad, LAMBDA)
        assert "did not pass" in problem(root, "inference/CI-ROUTE-001", LAMBDA)


@test
def overlay_flips_matching_rows_only():
    counts = {"routing_control_flow": 5, "behavioural": 0}
    ok = row_for("inference/CI-ROUTE-001", LAMBDA)
    assert B.overlay_row(ok, counts) is True
    assert ok["evidence_tier"] == "behavioural" and "receipt" in ok["evidence"] and "non_binding_receipts" not in ok["evidence"]
    assert counts == {"routing_control_flow": 4, "behavioural": 1}
    for sid, case in (("inference/CI-ROUTE-043", SIGMA_B), ("inference/CI-ROUTE-067", "CORE070-INFERENCE-BETA-BOOTSTRAP-FALLBACK-DIVERGENCE"),
                      ("inference/CI-ROUTE-023", "CORE070-INFERENCE-COMMUNALITY-CI-METHOD-ROUTE"),
                      ("inference/CI-ROUTE-065", "CORE070-INFERENCE-BETA-CI-METHOD-ROUTE")):
        r = row_for(sid, case)
        assert B.overlay_row(r, counts) is False and r["evidence_tier"] == "routing_control_flow", sid
    assert counts == {"routing_control_flow": 4, "behavioural": 1}


@test
def overlay_leaves_numeric_and_stale_rows_alone():
    counts = {"numeric": 1, "behavioural": 0}
    r = row_for("inference/CI-ROUTE-008", "CORE070-SURFCONV-INFERENCE-CI-ROUTE-008", tier="numeric")
    assert B.overlay_row(r, counts) is False
    r = row_for("inference/CI-ROUTE-001", LAMBDA)
    r["measured_against"] = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
    assert B.overlay_row(r, {"routing_control_flow": 1}) is False


@test
def derivation_refuses_when_per_row_disagrees_with_raw():
    rec = copy.deepcopy(case_receipt(LAMBDA))
    rec["per_row"][0]["julia_route_actual"] = "profile"
    try:
        B.wave2(LAMBDA, rec)
    except SystemExit as e:
        assert "per_row disagrees" in str(e)
    else:
        raise AssertionError("expected SystemExit")


@test
def fallback_withdrawn_and_default_rows_get_no_entry():
    want = {"CORE070-INFERENCE-BETA-BOOTSTRAP-FALLBACK-DIVERGENCE": ({"inference/CI-ROUTE-067"}, "fallback"),
            "CORE070-INFERENCE-SIGMA-EPS-BOOTSTRAP-FALLBACK-DIVERGENCE": ({"inference/CI-ROUTE-070"}, "fallback"),
            "CORE070-INFERENCE-COMMUNALITY-CI-METHOD-ROUTE": ({"inference/CI-ROUTE-023"}, "withdrawn"),
            "CORE070-INFERENCE-RHO-CI-METHOD-ROUTE": ({"inference/CI-ROUTE-030"}, "withdrawn"),
            "CORE070-INFERENCE-PROPORTION-CI-METHOD-ROUTE": ({"inference/CI-ROUTE-037"}, "withdrawn"),
            "CORE070-INFERENCE-PHYLO-SIGNAL-CI-METHOD-ROUTE": ({"inference/CI-ROUTE-015"}, "default")}
    for cid, (sids, why) in want.items():
        _, nb, _ = B.wave2(cid, case_receipt(cid))
        assert {n["source_id"] for n in nb} == sids and {n["reason"] for n in nb} == {why}, cid


def main():
    failed = 0
    for fn in results:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", fn.__name__, type(e).__name__, e)
    print(f"{len(results) - failed}/{len(results)} controls passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

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
def fallback_withdrawn_default_and_unconfirmed_route_rows_get_no_entry():
    """Rows that stay unbound after the post-#709 re-measurement, each with its reason."""
    want = {"CORE070-INFERENCE-BETA-BOOTSTRAP-FALLBACK-DIVERGENCE": {"inference/CI-ROUTE-067": "fallback"},
            "CORE070-INFERENCE-SIGMA-EPS-BOOTSTRAP-FALLBACK-DIVERGENCE": {"inference/CI-ROUTE-070": "fallback"},
            "CORE070-INFERENCE-COMMUNALITY-CI-METHOD-ROUTE": {"inference/CI-ROUTE-023": "withdrawn"},
            "CORE070-INFERENCE-RHO-CI-METHOD-ROUTE": {"inference/CI-ROUTE-030": "withdrawn",
                                                      "inference/CI-ROUTE-029": "default_class_unconfirmed",
                                                      "inference/CI-ROUTE-034": "public_route_differs"},
            "CORE070-INFERENCE-PROPORTION-CI-METHOD-ROUTE": {"inference/CI-ROUTE-037": "withdrawn"},
            "CORE070-INFERENCE-PHYLO-SIGNAL-CI-METHOD-ROUTE": {"inference/CI-ROUTE-015": "default"},
            "CORE070-INFERENCE-SIGMA-B-CI-METHOD-ROUTE": {"inference/CI-ROUTE-043": "default",
                                                          "inference/CI-ROUTE-055": "default"},
            "CORE070-INFERENCE-SIGMA-W-CI-METHOD-ROUTE": {"inference/CI-ROUTE-046": "default",
                                                          "inference/CI-ROUTE-058": "default"},
            "CORE070-INFERENCE-SIGMA-PHY-CI-METHOD-ROUTE": {"inference/CI-ROUTE-061": "default"}}
    for cid, expect in want.items():
        _, nb, _ = B.wave2(cid, case_receipt(cid))
        assert {n["source_id"]: n["reason"] for n in nb} == expect, cid


@test
def post709_sigma_and_derived_route_rows_bind_through_the_public_confint():
    """Since #709 the Sigma bootstrap rows and the derived profile, bootstrap and default rows are measured through
    Julia's public confint(fit, y; parm, method); the entry's Julia label names the solver the call reported."""
    want = {"045": "sigma_B:bootstrap", "057": "sigma_B:bootstrap", "048": "sigma_W:bootstrap",
            "060": "sigma_W:bootstrap", "063": "sigma_phy:bootstrap", "016": "phylo_signal:profile",
            "018": "phylo_signal:bootstrap", "022": "communality:wald_derived", "025": "communality:bootstrap",
            "032": "rho:bootstrap", "036": "proportion:wald_derived", "039": "proportion:bootstrap"}
    cm = B.load(ROOT / B.LEDGER / "case-map-inference.json")
    rows = {r["source_id"]: r for r in cm["rows"]}
    for n, label in want.items():
        r = rows[f"inference/CI-ROUTE-{n}"]
        assert r["evidence_tier"] == "behavioural" and "receipt" in r["evidence"], n
        rec = case_receipt(r["executable_case_ids"][0])
        es = [e for e in rec["behaviour"]["cases"] if e["source_id"] == r["source_id"]]
        assert len(es) == 1 and es[0]["julia_observed"] == label and es[0]["kind"] == "route", n
        assert "inference-post709-results.json" in es[0]["julia_source"], n
    for n in ("015", "023", "029", "030", "034", "037"):
        assert rows[f"inference/CI-ROUTE-{n}"]["evidence_tier"] == "routing_control_flow", n
    assert rows["inference/CI-ROUTE-034"]["note"] == B.NOT_BOUND_ROW_NOTES["public_route_differs"]
    for n in B.ROUTE_ONLY_ROWS:
        assert B.ROUTE_ONLY_NOTE in rows[f"inference/CI-ROUTE-{n}"]["note"], n
    assert rows["inference/CI-ROUTE-029"]["note"] == B.NOT_BOUND_ROW_NOTES["default_class_unconfirmed"]


@test
def refusal_rows_bind_only_with_a_valid_method_control_on_each_side():
    """All 16 bad-method rows (14 derived, 2 Lambda) carry an R refusal, a Julia ArgumentError that names the method,
    and a valid-method control for both engines."""
    cases = [f"CORE070-INFERENCE-{t}-CI-UNSUPPORTED-METHOD-REJECT"
             for t in ("ICC", "PHYLO-SIGNAL", "COMMUNALITY", "RHO", "PROPORTION", "LAMBDA")]
    total = 0
    for cid in cases:
        rec = case_receipt(cid)
        fn = B.wave2 if "LAMBDA" in cid else B.wave4
        entries, nb, _ = fn(cid, rec)
        assert nb == [], cid
        for e in entries:
            assert e["kind"] == "refusal" and e["julia_error_type"] == "ArgumentError", e["source_id"]
            assert f":{e['requested_method']}" in e["julia_observed"], e["source_id"]
            assert e["r_control"]["result"] and e["julia_control"]["result"].endswith("interval"), e["source_id"]
        assert [e["source_id"] for e in entries] == [e["source_id"] for e in rec["behaviour"]["cases"]], cid
        total += len(entries)
    assert total == 16, total
    lam = case_receipt("CORE070-INFERENCE-LAMBDA-CI-UNSUPPORTED-METHOD-REJECT")["behaviour"]["cases"]
    assert {e["r_control"]["kind"] for e in lam} == {"live_call"}


@test
def swapped_refusal_labels_do_not_bind():
    """The refusal classes are one (target, method) pair each: R's wald_asym refusal does not match Julia's bogus one."""
    cid = "CORE070-INFERENCE-ICC-CI-UNSUPPORTED-METHOD-REJECT"
    rec = copy.deepcopy(case_receipt(cid))
    es = {e["source_id"]: e for e in rec["behaviour"]["cases"]}
    a, b = es["inference/CI-ROUTE-012"], es["inference/CI-ROUTE-014"]
    a["julia_observed"], b["julia_observed"] = b["julia_observed"], a["julia_observed"]
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t), rec, cid)
        assert problem(root, "inference/CI-ROUTE-012", cid), "swapped refusal label still binds"
        assert problem(root, "inference/CI-ROUTE-013", cid) is None


@test
def writer_refuses_weak_refusal_evidence():
    """Surviving mutants of the review: R text that does not name the method, a Julia message for the wrong parm, a
    non-Wald Julia control, an empty R label."""
    import json as _json
    fx = {r["id"]: r for r in B.read_tsv(ROOT / B.FIXTURE)}
    rp = {r["id"]: r for r in B.read_tsv(ROOT / f"{B.INF}/inference-batch-p1/r-crosscheck/p1-route-probe-results.tsv")}
    orig, orig_r = B.post709(), B.r_refusal
    cid, sid = "CORE070-INFERENCE-ICC-CI-UNSUPPORTED-METHOD-REJECT", "inference/CI-ROUTE-012"

    def refused(mutate_case=None, r_label=None):
        base = _json.loads(_json.dumps({"cases": orig["cases"], "r_lambda": orig["r_lambda"]}))
        if mutate_case:
            mutate_case(base["cases"][sid])
        B._P709 = base
        if r_label is not None:
            B.r_refusal = lambda t, m: (r_label, "x")
        try:
            B.post709_entry(cid, sid, rp, fx)
        except SystemExit:
            return True
        finally:
            B._P709, B.r_refusal = orig, orig_r
        return False

    assert not refused(), "the unmutated row must bind"
    assert refused(r_label="Method not supported for `icc`."), "M15: R text without the method"
    assert refused(r_label=""), "M19: empty R label"
    assert refused(lambda c: c.update(error_message=c["error_message"].replace("icc[1]", "rho[1,2]"))), "M18: wrong parm"
    assert refused(lambda c: c.update(control_route_tag="bootstrap")), "M16: control not Wald"
    lam = "CORE070-INFERENCE-LAMBDA-CI-UNSUPPORTED-METHOD-REJECT"
    B.r_refusal = lambda t, m: ("some other error", "x")
    try:
        B.post709_entry(lam, "inference/CI-ROUTE-006", rp, fx)
    except SystemExit:
        pass
    else:
        raise AssertionError("M15: Lambda refusal that is not the match.arg error")
    finally:
        B.r_refusal = orig_r


@test
def public_fisher_z_refusal_unbinds_034():
    cid = "CORE070-INFERENCE-RHO-CI-METHOD-ROUTE"
    entries, nb, _ = B.wave2(cid, case_receipt(cid))
    assert "inference/CI-ROUTE-034" not in {e["source_id"] for e in entries}
    assert {n["source_id"]: n["reason"] for n in nb}["inference/CI-ROUTE-034"] == "public_route_differs"


@test
def unbound_rows_are_not_flipped_by_overlay():
    counts = {"reject_error_class": 3, "routing_control_flow": 3, "behavioural": 0}
    for sid, cid, tier in (("inference/CI-ROUTE-015", "CORE070-INFERENCE-PHYLO-SIGNAL-CI-METHOD-ROUTE", "routing_control_flow"),
                           ("inference/CI-ROUTE-029", "CORE070-INFERENCE-RHO-CI-METHOD-ROUTE", "routing_control_flow"),
                           ("inference/CI-ROUTE-023", "CORE070-INFERENCE-COMMUNALITY-CI-METHOD-ROUTE", "routing_control_flow"),
                           ("inference/CI-ROUTE-067", "CORE070-INFERENCE-BETA-BOOTSTRAP-FALLBACK-DIVERGENCE", "routing_control_flow")):
        r = row_for(sid, cid, tier=tier)
        assert B.overlay_row(r, counts) is False and r["evidence_tier"] == tier, sid
    assert counts["behavioural"] == 0


@test
def out_of_scope_rows_keep_their_block_but_do_not_bind():
    """The 7 aghq control rows and CI-ROUTE-009 are not in the frozen list of #684 item 2 (review of PR 690)."""
    out = [("aghq/AGHQ-CTRL-" + n, f"{B.AGHQ}/cases/CORE070-AGHQ-CTRL-{n}-PAIRED-CONTROL.json", "paired_control_categorical_pass")
           for n in ("AUTO", "FALSE", "NINE", "NULL", "ONE", "TRUE", "TWO")]
    out.append(("inference/CI-ROUTE-009", f"{INF_CASES}/CORE070-SURFCONV-INFERENCE-CI-ROUTE-009.json", "partial_non_numeric_case"))
    assert len(out) == 8
    for sid, path, tier in out:
        assert not A.behavioural_eligible_source_id(sid), sid
        rec = B.load(ROOT / path)
        assert any(e["source_id"] == sid for e in rec["behaviour"]["cases"]), f"{sid}: block dropped"
        cid = rec["case_id"]
        row = {"source_id": sid, "classification": "required_core", "executable_case_ids": [cid], "disposition": None,
               "evidence_tier": tier, "measured_against": B.P1_SHA, "evidence": {"non_binding_receipts": [path], "tier": "x"},
               "measured_result": {}}
        counts = {tier: 1, "behavioural": 0}
        assert B.overlay_row(row, counts) is False and row["evidence_tier"] == tier, sid
        assert row["note"] == B.OUT_OF_SCOPE_NOTE and counts == {tier: 1, "behavioural": 0}, sid
    for f in ("case-map-aghq", "case-map-inference"):
        for r in B.load(ROOT / B.LEDGER / f"{f}.json")["rows"]:
            if r["evidence_tier"] == "behavioural":
                assert A.behavioural_eligible_source_id(r["source_id"]), f"{r['source_id']} is behavioural outside the frozen list"


@test
def every_entry_is_scoped_and_both_labels_are_listed_in_one_class():
    """Class identity, not equality: no entry relies on two raw labels being the same string or on a raw
    label equal to a canonical, and every entry carries source_id (case ids are shared across rows)."""
    index = A.behaviour_equivalence(ROOT)
    seen = 0
    for cid, (path, rec) in B.tracked_receipts().items():
        for e in (rec.get("behaviour") or {}).get("cases", []):
            seen += 1
            assert e.get("source_id"), f"{path.name}: entry without source_id"
            for a, b in zip(A.as_list(e["r_observed"]), A.as_list(e["julia_observed"])):
                rc, jc = A._label_class(index, e["kind"], "r", a), A._label_class(index, e["kind"], "julia", b)
                assert rc is not None and jc is not None, f"{e['source_id']}: {a!r} / {b!r} not both listed in a class"
                assert rc[1] == jc[1], f"{e['source_id']}: {a!r} and {b!r} are in different classes"
    assert seen == 52, seen  # 44 bound inference rows + CI-ROUTE-009 + 7 aghq control rows


@test
def every_class_is_used_by_an_entry():
    """A class nothing uses is a class for a row that was unbound; the table lists only labels some entry carries."""
    used = set()
    index = A.behaviour_equivalence(ROOT)
    for cid, (path, rec) in B.tracked_receipts().items():
        for e in (rec.get("behaviour") or {}).get("cases", []):
            for a in A.as_list(e["r_observed"]):
                used.add((e["kind"], A._label_class(index, e["kind"], "r", a)[0]))
    assert used == {(c["kind"], c["canonical"]) for c in B.all_classes()}, sorted({(c["kind"], c["canonical"]) for c in B.all_classes()} ^ used)


@test
def an_unscoped_entry_covers_none_of_several_rows_sharing_a_case_id():
    rec = copy.deepcopy(case_receipt(LAMBDA))
    for e in rec["behaviour"]["cases"]:
        if e["source_id"] == "inference/CI-ROUTE-001":
            del e["source_id"]
    cites = {LAMBDA: {f"inference/CI-ROUTE-00{i}" for i in (1, 2, 3, 4)}}
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t), rec, LAMBDA)
        row = dict(row_for("inference/CI-ROUTE-001", LAMBDA), evidence_tier="behavioural",
                   evidence={"receipt": [f"{INF_CASES}/{LAMBDA}.json"]})
        idx = A.behaviour_equivalence(root)
        why = A.behavioural_receipt_problem(row, root, idx, cites)
        assert why and "covers none of them" in why, why
        assert A.behavioural_receipt_problem(row, root, idx, {LAMBDA: {"inference/CI-ROUTE-001"}}) is None  # sole citer


@test
def the_writers_citers_count_every_row_that_lists_a_case_id():
    c = B.citers()
    assert c[LAMBDA] == {f"inference/CI-ROUTE-00{i}" for i in (1, 2, 3, 4)}, c[LAMBDA]
    assert len(c["CORE070-INFERENCE-SIGMA-B-CI-METHOD-ROUTE"]) > 1


@test
def citation_note_claims_no_rerun():
    note = B.equivalence_doc()["citation_note"]
    assert "re-run" not in note and "rerun" not in note and "review" not in note, note


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

"""Verify the post-#709 inference re-measurement (W2-1).

Inputs: a Julia run of tools/core070_inference_post709_batch.jl and an R run of
tools/core070_inference_lambda_reject_p1.R. Checks:

  Julia  every expected row is present once; each route row returned a result whose shape and
         reported `method` agree with the method the row asked for; each refusal row raised an
         ArgumentError whose message names the requested method, AND its valid-method control on
         the same fit and parm returned a finite interval (so the refusal is about the method, not
         about the call); the negative controls behaved; the results hash matches the receipt.
  R      the receipt passes and carries the P1 source-pin record; fisher-z and bogus raised, wald
         (the control) returned a finite interval; the oracle hash matches the receipt.

  python3 tools/core070_verify_inference_post709_batch.py --julia-state DIR --r-state DIR [--self-test]

--self-test mutates a synthetic valid state in several independent ways and requires every
mutation to be rejected. It never substitutes for the real --julia-state and --r-state check.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import core070_source_pin_check  # noqa: E402

PIN = "P1"
RESULT_OK = "CORE070_INFERENCE_POST709_BATCH_VERIFIED"
EXPECTED_ROUTE = {"CI-ROUTE-015", "CI-ROUTE-016", "CI-ROUTE-018", "CI-ROUTE-022", "CI-ROUTE-023", "CI-ROUTE-025",
                  "CI-ROUTE-029", "CI-ROUTE-030", "CI-ROUTE-032", "CI-ROUTE-036", "CI-ROUTE-037", "CI-ROUTE-039",
                  "CI-ROUTE-045", "CI-ROUTE-048", "CI-ROUTE-057", "CI-ROUTE-060", "CI-ROUTE-063"}
EXPECTED_REFUSAL = {"CI-ROUTE-006", "CI-ROUTE-007", "CI-ROUTE-012", "CI-ROUTE-013", "CI-ROUTE-014", "CI-ROUTE-019",
                    "CI-ROUTE-020", "CI-ROUTE-021", "CI-ROUTE-026", "CI-ROUTE-027", "CI-ROUTE-028", "CI-ROUTE-033",
                    "CI-ROUTE-035", "CI-ROUTE-034", "CI-ROUTE-040", "CI-ROUTE-041", "CI-ROUTE-042"}
ROUTE_TAG = {"DEFAULT": "wald_derived", "profile": "profile", "bootstrap": "bootstrap"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def check_julia(results, receipt, results_sha):
    need(results.get("status") == "PASS" and receipt.get("status") == "PASS", "Julia run is not PASS")
    need(receipt.get("results_sha256") == results_sha, "receipt results_sha256 does not match the results file")
    need(results.get("negative_controls") and all(v.get("behaved") is True for v in results["negative_controls"].values()),
         "a negative control did not behave")
    cases = results.get("cases")
    need(isinstance(cases, list), "cases is not a list")
    ids = [c.get("source_id") for c in cases]
    need(len(ids) == len(set(ids)), "a source_id appears twice")
    need(set(ids) == {"inference/" + s for s in EXPECTED_ROUTE | EXPECTED_REFUSAL}, "row set differs from the expected rows")
    need(receipt.get("expected_case_source_ids") is not None and set(receipt["expected_case_source_ids"]) == set(ids),
         "receipt expected_case_source_ids differ from the results")
    for c in cases:
        sid = c["source_id"].split("/")[-1]
        meth = c["requested_method"]
        if sid in EXPECTED_ROUTE:
            need(c["kind"] == "route" and c["outcome"] == "result", f"{sid}: route row did not return a result")
            if sid in ("CI-ROUTE-023", "CI-ROUTE-030", "CI-ROUTE-037", "CI-ROUTE-016") or meth != "DEFAULT":
                need(c["route_tag"] == ROUTE_TAG[meth], f"{sid}: route tag {c['route_tag']} != {ROUTE_TAG[meth]}")
                need(c["result_method"] == meth, f"{sid}: result reports method {c['result_method']!r}, asked {meth!r}")
            else:
                need(c["route_tag"] == ROUTE_TAG[meth], f"{sid}: default route tag {c['route_tag']}")
        else:
            need(c["kind"] == "refusal" and c["outcome"] == "error", f"{sid}: bad method was not refused")
            need(c["error_type"] == "ArgumentError", f"{sid}: refusal is {c['error_type']}, not ArgumentError")
            need(f":{meth}" in c["error_message"], f"{sid}: refusal message does not name the method {meth!r}")
            need(c.get("control_outcome") == "result" and c.get("control_finite") is True,
                 f"{sid}: valid-method control did not return a finite interval")
            need(c.get("control_result_method") in ("wald", "") and c.get("control_route_tag") in ("wald_derived", "wald_packed"),
                 f"{sid}: control is not a Wald result")


def check_r(oracle, receipt, oracle_sha):
    need(receipt.get("status") == "PASS", "R run is not PASS")
    need(receipt.get("oracle_sha256") == oracle_sha, "receipt oracle_sha256 does not match r-oracle.json")
    core070_source_pin_check.check_source_pin(receipt, PIN, need)
    calls = oracle.get("calls") or {}
    for m in ("fisher-z", "bogus"):
        need(calls.get(m, {}).get("raised") is True and calls[m].get("message"), f"R did not refuse method {m!r}")
    need(calls.get("wald", {}).get("raised") is False and calls["wald"].get("finite") is True,
         "R valid-method control (wald) did not return a finite interval")


def verify_state(julia_dir, r_dir):
    julia_dir, r_dir = Path(julia_dir), Path(r_dir)
    jr = julia_dir / "inference-post709-results.json"
    check_julia(json.loads(jr.read_text()), json.loads((julia_dir / "receipt.json").read_text()), sha(jr))
    ro = r_dir / "r-oracle.json"
    check_r(json.loads(ro.read_text()), json.loads((r_dir / "receipt.json").read_text()), sha(ro))


def synthetic():
    pins = core070_source_pin_check.PINS[PIN]
    cases = []
    for s in sorted(EXPECTED_ROUTE):
        meth = {"CI-ROUTE-015": "DEFAULT", "CI-ROUTE-022": "DEFAULT", "CI-ROUTE-029": "DEFAULT",
                "CI-ROUTE-036": "DEFAULT"}.get(s)
        meth = meth or ("profile" if s in ("CI-ROUTE-016", "CI-ROUTE-023", "CI-ROUTE-030", "CI-ROUTE-037") else "bootstrap")
        cases.append({"source_id": "inference/" + s, "kind": "route", "outcome": "result", "requested_method": meth,
                      "route_tag": ROUTE_TAG[meth], "result_method": "" if meth == "DEFAULT" else meth})
    for s in sorted(EXPECTED_REFUSAL):
        cases.append({"source_id": "inference/" + s, "kind": "refusal", "outcome": "error", "requested_method": "bogus",
                      "error_type": "ArgumentError", "error_message": "confint: method = :bogus is not available",
                      "control_outcome": "result", "control_finite": True, "control_result_method": "wald",
                      "control_route_tag": "wald_derived"})
    results = {"status": "PASS", "negative_controls": {"N": {"behaved": True}}, "cases": cases}
    receipt = {"status": "PASS", "results_sha256": "x", "expected_case_source_ids": [c["source_id"] for c in cases]}
    oracle = {"calls": {"fisher-z": {"raised": True, "message": "m"}, "bogus": {"raised": True, "message": "m"},
                        "wald": {"raised": False, "finite": True}}}
    rreceipt = {"status": "PASS", "oracle_sha256": "y", "gllvmTMB_version": pins["version"],
                "source_pin": {**{k: pins[k] for k in core070_source_pin_check.SOURCE_PIN_KEYS}, "version": pins["version"]}}
    return results, receipt, oracle, rreceipt


def self_test():
    results, receipt, oracle, rreceipt = synthetic()
    check_julia(results, receipt, "x")
    check_r(oracle, rreceipt, "y")

    def case(r, sid):
        return next(c for c in r["cases"] if c["source_id"].endswith(sid))

    jm = {
        "refusal returned a result": lambda r, _: case(r, "CI-ROUTE-012").update(outcome="result"),
        "refusal is a MethodError": lambda r, _: case(r, "CI-ROUTE-012").update(error_type="MethodError"),
        "refusal does not name the method": lambda r, _: case(r, "CI-ROUTE-012").update(error_message="boom"),
        "control missing": lambda r, _: case(r, "CI-ROUTE-012").pop("control_outcome"),
        "control not finite": lambda r, _: case(r, "CI-ROUTE-012").update(control_finite=False),
        "route returned an error": lambda r, _: case(r, "CI-ROUTE-016").update(outcome="error"),
        "profile row ran bootstrap": lambda r, _: case(r, "CI-ROUTE-016").update(route_tag="bootstrap"),
        "bootstrap row reports wald": lambda r, _: case(r, "CI-ROUTE-025").update(result_method="wald"),
        "row missing": lambda r, _: r["cases"].pop(),
        "duplicate row": lambda r, _: r["cases"].append(deepcopy(r["cases"][0])),
        "negative control broke": lambda r, _: r["negative_controls"]["N"].update(behaved=False),
        "status FAIL": lambda r, _: r.update(status="FAIL"),
    }
    for label, mutate in jm.items():
        r = deepcopy(results)
        mutate(r, None)
        try:
            check_julia(r, receipt, "x")
        except ValueError:
            continue
        raise AssertionError(f"Julia mutation was NOT rejected: {label}")
    rm = {
        "R did not refuse fisher-z": lambda o, _: o["calls"]["fisher-z"].update(raised=False),
        "R refused the control": lambda o, _: o["calls"]["wald"].update(raised=True),
        "R refusal has no message": lambda o, _: o["calls"]["bogus"].update(message=""),
    }
    for label, mutate in rm.items():
        o = deepcopy(oracle)
        mutate(o, None)
        try:
            check_r(o, rreceipt, "y")
        except ValueError:
            continue
        raise AssertionError(f"R mutation was NOT rejected: {label}")
    for label, bad in {"R status FAIL": dict(status="FAIL"), "oracle hash": dict(oracle_sha256="z"),
                       "wrong version": dict(gllvmTMB_version="0.0.0")}.items():
        rr = deepcopy(rreceipt)
        rr.update(bad)
        try:
            check_r(oracle, rr, "y")
        except ValueError:
            continue
        raise AssertionError(f"R receipt mutation was NOT rejected: {label}")
    n_pin = core070_source_pin_check.self_test(PIN)
    print(f"self-test: {len(jm)} Julia, {len(rm) + 3} R and {n_pin} source-pin mutations rejected")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--julia-state", type=Path, required=True)
    ap.add_argument("--r-state", type=Path, required=True)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    try:
        verify_state(a.julia_state, a.r_state)
    except (ValueError, KeyError, OSError) as e:
        print("VERIFY FAILED:", e)
        sys.exit(1)
    if a.self_test:
        self_test()
    print(RESULT_OK)


if __name__ == "__main__":
    main()

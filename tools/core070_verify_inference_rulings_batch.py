"""Verify the inference rulings batch (tools/core070_inference_rulings_batch.jl, vault D-319).

Checks a Julia run directory holding inference-rulings-results.json and receipt.json:

  every expected row is present once and the receipt names the same rows and the results hash;
  the negative controls behaved;
  withdrawn rows (023, 030, 037): ArgumentError whose message says the profile is withdrawn and is
      not the bad-method message, AND a :wald control on the same fit and parm returned a finite
      Wald interval;
  default rows (015, 029, 043, 046, 055, 058, 061, 065, 068): the call with no method returned a Wald
      result (wald_derived or wald_packed); for the eight ruling-B rows the explicit :profile call
      returned a profile result;
  fallback rows (067, 070): method = :bootstrap returned a bootstrap result;
  CI-ROUTE-034: method = :fisher_z returned a Wald-derived result identical to :wald.

  python3 tools/core070_verify_inference_rulings_batch.py --state DIR [--self-test]

--self-test mutates a synthetic valid state in independent ways and requires each mutation to be
rejected. It never substitutes for the real --state check.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

RESULT_OK = "CORE070_INFERENCE_RULINGS_BATCH_VERIFIED"
WITHDRAWN = {"CI-ROUTE-023", "CI-ROUTE-030", "CI-ROUTE-037"}
DEFAULT_B = {"CI-ROUTE-015", "CI-ROUTE-043", "CI-ROUTE-046", "CI-ROUTE-055", "CI-ROUTE-058", "CI-ROUTE-061",
             "CI-ROUTE-065", "CI-ROUTE-068"}
FALLBACK = {"CI-ROUTE-067", "CI-ROUTE-070"}
EXPECTED = WITHDRAWN | DEFAULT_B | FALLBACK | {"CI-ROUTE-029", "CI-ROUTE-034"}
WALD_TAGS = ("wald_derived", "wald_packed")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def check(results, receipt, results_sha):
    need(results.get("status") == "PASS" and receipt.get("status") == "PASS", "Julia run is not PASS")
    need(receipt.get("results_sha256") == results_sha, "receipt results_sha256 does not match the results file")
    nc = results.get("negative_controls")
    need(nc and all(v.get("behaved") is True for v in nc.values()), "a negative control did not behave")
    cases = results.get("cases")
    need(isinstance(cases, list), "cases is not a list")
    ids = [c.get("source_id") for c in cases]
    need(len(ids) == len(set(ids)), "a source_id appears twice")
    need(set(ids) == {"inference/" + s for s in EXPECTED}, "row set differs from the expected rows")
    need(set(receipt.get("expected_case_source_ids") or []) == set(ids),
         "receipt expected_case_source_ids differ from the results")
    for c in cases:
        sid = c["source_id"].split("/")[-1]
        if sid in WITHDRAWN:
            need(c["outcome"] == "error" and c.get("error_type") == "ArgumentError", f"{sid}: profile was not refused")
            msg = c.get("error_message", "")
            need("withdrawn" in msg and "is not available for" not in msg,
                 f"{sid}: refusal is not the withdrawn-profile message")
            ctl = c.get("control") or {}
            need(ctl.get("outcome") == "result" and ctl.get("finite") is True and ctl.get("route_tag") in WALD_TAGS,
                 f"{sid}: :wald control did not return a finite Wald interval")
        elif sid in DEFAULT_B or sid == "CI-ROUTE-029":
            need(c["outcome"] == "result" and c.get("route_tag") in WALD_TAGS and c.get("result_method") in ("wald", ""),
                 f"{sid}: Julia's default is not a Wald result")
            if sid in DEFAULT_B:
                ep = c.get("explicit_profile") or {}
                need(ep.get("outcome") == "result" and ep.get("route_tag") == "profile" and ep.get("result_method") == "profile",
                     f"{sid}: the explicit :profile request did not return a profile result")
        elif sid in FALLBACK:
            need(c["outcome"] == "result" and c.get("route_tag") == "bootstrap" and c.get("result_method") == "bootstrap",
                 f"{sid}: method = :bootstrap did not run a bootstrap")
        else:  # CI-ROUTE-034
            need(c["outcome"] == "result" and c.get("route_tag") == "wald_derived" and c.get("identical_to_wald") is True,
                 f"{sid}: :fisher_z is not identical to :wald")


def verify_state(d):
    d = Path(d)
    rp = d / "inference-rulings-results.json"
    check(json.loads(rp.read_text()), json.loads((d / "receipt.json").read_text()), sha(rp))


def synthetic():
    cases = []
    for s in sorted(EXPECTED):
        c = {"source_id": "inference/" + s}
        if s in WITHDRAWN:
            c.update(outcome="error", error_type="ArgumentError", error_message="confint: profile is withdrawn",
                     control={"outcome": "result", "finite": True, "route_tag": "wald_derived"})
        elif s in FALLBACK:
            c.update(outcome="result", route_tag="bootstrap", result_method="bootstrap")
        elif s == "CI-ROUTE-034":
            c.update(outcome="result", route_tag="wald_derived", result_method="wald", identical_to_wald=True)
        else:
            c.update(outcome="result", route_tag="wald_packed", result_method="")
            if s in DEFAULT_B:
                c["explicit_profile"] = {"outcome": "result", "route_tag": "profile", "result_method": "profile"}
        cases.append(c)
    results = {"status": "PASS", "negative_controls": {"N": {"behaved": True}}, "cases": cases}
    receipt = {"status": "PASS", "results_sha256": "x", "expected_case_source_ids": [c["source_id"] for c in cases]}
    return results, receipt


def self_test():
    results, receipt = synthetic()
    check(results, receipt, "x")
    def case(r, sid):
        return next(c for c in r["cases"] if c["source_id"] == "inference/" + sid)
    mutations = {
        "status FAIL": lambda r, p: r.update(status="FAIL"),
        "hash mismatch": lambda r, p: p.update(results_sha256="z"),
        "negative control": lambda r, p: r["negative_controls"]["N"].update(behaved=False),
        "row dropped": lambda r, p: r["cases"].pop(),
        "withdrawn computes": lambda r, p: case(r, "CI-ROUTE-023").update(outcome="result"),
        "withdrawn is bad-method text": lambda r, p: case(r, "CI-ROUTE-030").update(
            error_message="confint: method = :profile is not available for X"),
        "withdrawn control fails": lambda r, p: case(r, "CI-ROUTE-037")["control"].update(finite=False),
        "default is profile": lambda r, p: case(r, "CI-ROUTE-043").update(route_tag="profile"),
        "explicit profile missing": lambda r, p: case(r, "CI-ROUTE-065").pop("explicit_profile"),
        "fallback is Wald": lambda r, p: case(r, "CI-ROUTE-067").update(route_tag="wald_packed"),
        "fisher_z differs": lambda r, p: case(r, "CI-ROUTE-034").update(identical_to_wald=False),
    }
    for name, mut in mutations.items():
        r, p = deepcopy(results), deepcopy(receipt)
        mut(r, p)
        try:
            check(r, p, "x")
        except ValueError:
            continue
        raise SystemExit(f"self-test: mutation {name!r} was not rejected")
    print(f"self-test: {len(mutations)} mutations rejected")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    try:
        verify_state(args.state)
    except (ValueError, KeyError, OSError) as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    print(RESULT_OK)


if __name__ == "__main__":
    main()

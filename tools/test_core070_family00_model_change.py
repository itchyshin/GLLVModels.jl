"""FAMILY-00's PUBLIC-R-BRIDGE case: R accepts the Gaussian family and fits a reduced model (reference_model_change).

A P1 probe record of that behaviour is the only thing that turns the case from NOT_EXECUTED into non-binding context.
Every record that does not show it (a refusal, one route, df other than 5, no warning, an edited record) must be
refused, and the row must not bind without the record. No R, JuliaCall or fit runs here.
Run: python3 tools/test_core070_family00_model_change.py
"""
import copy
import hashlib
import json
import struct
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import core070_family_p1_receipts as t

SID = "family/FAMILY-00-IDENTITY"
NATIVE = "CORE070-FAMILY-00-IDENTITY-NATIVE-MODEL"
FORMULA = "CORE070-FAMILY-00-IDENTITY-FORMULA-INTERFACE"
BRIDGE = t.FAMILY00_BRIDGE_CASE
WARNING = ("engine = 'julia' does not support the trait-specific Psi that ordinary `latent()` now carries by "
           "default.\nFitting the reduced-rank latent block only. Use `engine = \"tmb\"` for the decomposition.")


def good_record():
    pin = t.PINS["P1"]
    build = t.load(t.ROOT / t.ORACLE_BUILD_TOTORO_20261104)
    case = tomllib.loads((t.ROOT / t.FAMILY00_FIXTURE).read_text())["cases"][0]
    commit = t.load(t.ROOT / t.batch_rel(t.BRIDGE_BATCH) / "run-commit.json")["glvmodels_commit"]
    p, n, K = case["p"], case["n"], case["K"]
    Sigma = [[0.5 + (1.0 if i == j else 0.0) for j in range(p)] for i in range(p)]

    def fit():
        return {"class": ["gllvmTMB_julia"], "family": "gaussian", "loglik": -581.668831823926, "df": p * K + 1,
                "sigma_eps": 0.697282597486069, "converged": True, "d": K, "n_traits": p, "n_units": n,
                "max_abs_residual_deviation_from_common": 1e-15, "Sigma": [list(row) for row in Sigma]}

    def route(call, warnings=()):
        return {"call": call, "refused": False, "error_class": [], "message": "", "warnings": list(warnings),
                "fit": fit()}
    return {
        "schema": "core070-family00-r-model-change/v1", "case_id": BRIDGE, "pin": "P1",
        "reference_commit": t.P1_SHA, "gllvmtmb_version": pin["version"],
        "source_pin": {k: pin[k] for k in ("reference_commit", "source_tree_sha256", "archive_sha256",
                                           "namespace_sha256")}
                      | {"version": pin["version"], "installed_tree_sha256": build["installed_tree_sha256"],
                         "marker_sha256": build["marker_sha256"]},
        "namespace_sha256": pin["namespace_sha256"],
        "fixture": {"file": t.FAMILY00_FIXTURE, "file_sha256": t.sha(t.ROOT / t.FAMILY00_FIXTURE),
                    "case": case["id"], "seed": case["seed"], "p": p, "n": n, "K": K,
                    "data_sha256": hashlib.sha256(b"".join(struct.pack("<d", float(v))
                                                           for v in case["responses"])).hexdigest()},
        "required_unique_model_df": 2 * p,
        "routes": {"matrix": route("gllvm_julia_fit(Y, family = 'gaussian', num.lv = 1)"),
                   "formula": route("gllvmTMB(value ~ 0 + latent(0 + trait | site, d = 1), engine = 'julia')",
                                    [WARNING]),
                   "formula_explicit": route("gllvmTMB(... unique = FALSE ..., engine = 'julia')")},
        "capture": {"glvmodels_commit": commit, "glvmodels_src_tree": t.git("rev-parse", f"{commit}:src").stdout.strip(),
                    "r_version": "R 4.x", "julia_version": "1.10.12", "glvmodels_path": "/fixture/GLLVModels.jl"},
    }


class Validator(unittest.TestCase):
    def rejects(self, mutate):
        bad = copy.deepcopy(good_record())
        mutate(bad)
        with self.assertRaises(SystemExit):
            t.validate_family00_model_change(bad)

    def test_positive_record_is_accepted(self):
        fixture, capture, routes, commit = t.validate_family00_model_change(good_record())
        self.assertEqual((fixture["p"], fixture["K"], len(routes)), (4, 1, 3))

    def test_refusal_is_not_model_change(self):
        def refusal(r):
            r["routes"]["matrix"].update(refused=True, message="[GJL-GATE-FAMILY] unsupported family",
                                         error_class=["simpleError", "error", "condition"], fit=None)
        self.rejects(refusal)

    def test_formula_refusal_is_not_model_change(self):
        def refusal(r):
            r["routes"]["formula"].update(refused=True, message="unsupported", error_class=["error"], fit=None)
        self.rejects(refusal)

    def test_one_route_only(self):
        self.rejects(lambda r: r["routes"].pop("matrix"))
        self.rejects(lambda r: r["routes"].pop("formula"))
        self.rejects(lambda r: r["routes"].pop("formula_explicit"))

    def test_df_other_than_five(self):
        self.rejects(lambda r: r["routes"]["formula"]["fit"].update(df=8))
        self.rejects(lambda r: r["routes"]["matrix"]["fit"].update(df=6))
        self.rejects(lambda r: r["routes"]["formula_explicit"]["fit"].update(df=4))

    def test_no_warning(self):
        self.rejects(lambda r: r["routes"]["formula"].update(warnings=[]))
        self.rejects(lambda r: r["routes"]["formula"].update(warnings=["some other warning"]))

    def test_not_a_common_residual(self):
        self.rejects(lambda r: r["routes"]["formula"]["fit"].update(max_abs_residual_deviation_from_common=0.2))

    def test_not_converged(self):
        self.rejects(lambda r: r["routes"]["matrix"]["fit"].update(converged=False))

    def test_edited_record(self):
        # default request and explicit unique = FALSE must be the same reduced fit
        self.rejects(lambda r: r["routes"]["formula"]["fit"].update(loglik=-580.0))
        self.rejects(lambda r: r["routes"]["formula"]["fit"]["Sigma"][0].__setitem__(0, 99.0))
        # data, fixture, pin, build and provenance edits
        self.rejects(lambda r: r["fixture"].update(data_sha256="0" * 64))
        self.rejects(lambda r: r["fixture"].update(file_sha256="0" * 64))
        self.rejects(lambda r: r["fixture"].update(seed=1))
        self.rejects(lambda r: r.update(reference_commit="0" * 40))
        self.rejects(lambda r: r["source_pin"].update(installed_tree_sha256="0" * 64))
        self.rejects(lambda r: r["capture"].update(glvmodels_src_tree="0" * 40))
        self.rejects(lambda r: r["capture"].update(glvmodels_commit="0" * 40))
        self.rejects(lambda r: r.update(required_unique_model_df=5))
        self.rejects(lambda r: r.update(case_id="CORE070-FAMILY-11-LOG-PUBLIC-R-BRIDGE"))


class Derivation(unittest.TestCase):
    def derive(self, rec):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r-public-bridge-model-change.json"
            path.write_text(json.dumps(rec))
            with patch.object(t, "FAMILY00_MODEL_CHANGE_RAW", str(path)):
                return t.derive_case(BRIDGE)

    def test_good_record_gives_context_kind_with_accurate_body(self):
        kind, verdict, body, comparison = self.derive(good_record())
        self.assertEqual((kind, verdict), t.N1_CONTEXT_KIND)
        self.assertIsNone(comparison)
        self.assertEqual(body["disposition"], "reference_model_change")
        self.assertEqual((body["reduced_model_df"], body["required_unique_model_df"]), (5, 8))
        self.assertEqual(body["warnings_verbatim"], [WARNING])
        self.assertIn("not an R refusal", body["boundary_note"])
        self.assertNotIn("GJL-GATE-FAMILY", json.dumps(body))

    def test_bad_records_never_give_the_context_verdict(self):
        for mutate in (lambda r: r["routes"]["matrix"].update(refused=True, message="refused", error_class=["error"]),
                       lambda r: r["routes"].pop("formula"),
                       lambda r: r["routes"]["formula"]["fit"].update(df=8),
                       lambda r: r["routes"]["formula"].update(warnings=[]),
                       lambda r: r["fixture"].update(data_sha256="0" * 64)):
            bad = copy.deepcopy(good_record())
            mutate(bad)
            with self.assertRaises(SystemExit):
                self.derive(bad)

    def test_without_a_record_the_case_is_not_executed(self):
        with patch.object(t, "FAMILY00_MODEL_CHANGE_RAW", "/nonexistent/r-public-bridge-model-change.json"):
            kind, verdict, body, _ = t.derive_case(BRIDGE)
        self.assertEqual((kind, verdict), ("not_executed", "NOT_EXECUTED"))
        self.assertIn("why_not_executed", body)


class Row(unittest.TestCase):
    def build(self, bridge, native="PASS", formula="PASS"):
        info = lambda cid, kind, verdict: (f"receipts/{cid}.json", kind, verdict, "PASS", True)
        receipts = {NATIVE: info(NATIVE, "numeric_r_vs_julia", native),
                    FORMULA: info(FORMULA, "numeric_r_vs_julia", formula),
                    BRIDGE: info(BRIDGE, *bridge)}
        rows, counts = t.build_rows([SID], {SID: "DANGLING"}, receipts)
        return rows[0], counts

    def test_context_binds_with_family00_wording_not_n1(self):
        row, counts = self.build(t.N1_CONTEXT_KIND)
        self.assertEqual(row["evidence_tier"], "numeric")
        self.assertEqual(row["boundary_context_case_ids"], [BRIDGE])
        self.assertEqual(counts["numeric_pass"], 1)
        self.assertEqual(row["evidence"]["tier"], t.FAMILY00_TIER)
        self.assertNotEqual(row["evidence"]["tier"], t.N1_TIER)
        self.assertNotIn("R refuses", row["evidence"]["tier"])
        self.assertIn("reference_model_change", row["evidence"]["tier"])
        self.assertIn("not an R refusal", row["note"])
        self.assertNotIn("GJL-GATE-FAMILY", row["note"] + row["evidence"]["tier"])

    def test_without_the_record_the_row_is_partial_again(self):
        row, counts = self.build(("not_executed", "NOT_EXECUTED"))
        self.assertEqual(row["evidence_tier"], "partial_case_not_executed")
        self.assertEqual(counts["partial_case_not_executed"], 1)
        self.assertNotIn("boundary_context_case_ids", row)
        self.assertIn(t.FAMILY00_HELD_NOTE, row["note"])
        self.assertNotIn("refusal (GJL-GATE-FAMILY)", row["note"])

    def test_a_failing_native_case_fails_the_row(self):
        row, counts = self.build(t.N1_CONTEXT_KIND, native="FAIL")
        self.assertNotEqual(row["evidence_tier"], "numeric")
        self.assertNotIn("boundary_context_case_ids", row)
        self.assertEqual(counts["numeric_pass"], 0)
        row, _ = self.build(t.N1_CONTEXT_KIND, formula="FAIL")
        self.assertNotEqual(row["evidence_tier"], "numeric")

    def test_another_kind_is_not_context(self):
        row, _ = self.build(("numeric_r_vs_julia", "PASS"))
        self.assertNotIn("boundary_context_case_ids", row)
        row, _ = self.build(("r_public_bridge_boundary", "FAIL"))
        self.assertNotEqual(row["evidence_tier"], "numeric")


if __name__ == "__main__":
    unittest.main()

import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core070_verify_postfit_policy_batch as v

class CandidateContract(unittest.TestCase):
    def test_contract_checks_exact_three_fit_scope(self):
        contract = v.load_contract()
        v.verify_contract(contract)
        for field, value in (("length_tolerance", 1), ("expected_lengths", [0,0,0]),
                             ("accessor_controls", ["always_empty"])):
            bad = copy.deepcopy(contract)
            bad["cases"][0]["discriminating_comparison"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): v.verify_contract(bad)
        bad = copy.deepcopy(contract)
        bad["p0_contract_sha256"] = "0" * 64
        with self.assertRaises(ValueError): v.verify_contract(bad)

    def test_report_requires_real_discriminating_evidence(self):
        contract = v.load_contract()
        report = v.synthetic_report(contract)
        v.check_report(report, contract, "unused")
        case = report["cases"]["CORE070-POSTFIT-COEF-EMPTY-NATIVE"]
        for field, value in (("selected_accessor", "always_empty"),
                             ("julia_lengths", [0,0,0]), ("r_lengths", [0,0,0])):
            bad = copy.deepcopy(report)
            bad["cases"]["CORE070-POSTFIT-COEF-EMPTY-NATIVE"]["three_fit"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): v.check_report(bad,contract,"unused")
        self.assertIn("three_fit",case)

if __name__ == "__main__": unittest.main()

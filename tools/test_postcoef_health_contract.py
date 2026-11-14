"""Fail-closed controls for the follow-up three-fit health evidence."""
import copy
import math
import unittest
import core070_verify_postfit_policy_batch as verifier
import core070_postcoef_fit_health as health

# D2 (2026-11-09, "no certifier"): the POST-COEF verifier no longer requires FIT_HEALTH_PASS. The classifier is unchanged
# and still reports non-pass health; these tests now bind the classifier, and the verifier accepts held health.
PASS = 'FIT_HEALTH_PASS'


class HealthContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = verifier.load_contract()
        self.report = verifier.synthetic_report(self.contract)
        self.case = self.report['cases']['CORE070-POSTFIT-COEF-EMPTY-NATIVE']['three_fit']

    def test_synthetic_valid_health_is_held(self):
        # Numerical admission is held, so even nominal synthetic health is not a classifier pass.
        # Under D2 the verifier accepts the lengths and controls without it.
        self.assertEqual(len(self.case['fit_health']['R']), 3)
        self.assertEqual(len(self.case['fit_health']['Julia']), 3)
        self.assertNotEqual(health.classify(self.case['fit_health']), PASS)
        verifier.check_report(self.report, self.contract, 'unused')
        verifier.check_postcoef_three_fit_evidence(self.case)

    def test_float64_cholesky_counterexample_rejected_by_verifier(self):
        from test_postcoef_fit_health import with_indefinite_julia_hessian
        self.case['fit_health'] = with_indefinite_julia_hessian(self.case['fit_health'])
        self.assertNotEqual(health.classify(self.case['fit_health']), PASS)

    def test_each_failed_or_missing_diagnostic_rejected(self):
        for engine in ('R', 'Julia'):
            for index in range(3):
                for key, value in [('max_abs_gradient', 1e-5), ('pd_hessian', False),
                                   ('max_abs_gradient', math.nan), ('objective_verified', False),
                                   ('penalty_configuration_recorded', False), ('error', 'unavailable')]:
                    report = copy.deepcopy(self.report)
                    report['cases']['CORE070-POSTFIT-COEF-EMPTY-NATIVE']['three_fit']['fit_health'][engine][index][key] = value
                    mutated = report['cases']['CORE070-POSTFIT-COEF-EMPTY-NATIVE']['three_fit']['fit_health']
                    with self.subTest(engine=engine,index=index,key=key,value=value):
                        self.assertNotEqual(health.classify(mutated), PASS)

    def test_missing_health_is_not_qualified(self):
        del self.case['fit_health']
        self.assertNotEqual(health.classify(self.case.get('fit_health')), PASS)

    def test_r_health_is_bound_to_recorded_r_fits(self):
        fits = []
        for index in range(3):
            fits.append({'length': self.case['r_lengths'][index],
                         'fixed_columns': self.case['r_design_columns'][index],
                         'values': self.case['r_values'][index],
                         'names': self.case['r_names'][index],
                         'fit_health': copy.deepcopy(self.case['fit_health']['R'][index])})
        oracle = {'postcoef_three_fit': {'fits': fits}}
        verifier.check_postcoef_r_oracle_binding(self.case, oracle)
        fits[0]['fit_health']['max_abs_gradient'] = 0.1
        with self.assertRaises(ValueError):
            verifier.check_postcoef_r_oracle_binding(self.case, oracle)
        fits[0]['fit_health'] = copy.deepcopy(self.case['fit_health']['R'][0])
        fits[0]['fit_health']['pd_hessian'] = 1
        with self.assertRaises(ValueError):
            verifier.check_postcoef_r_oracle_binding(self.case, oracle)


if __name__ == '__main__': unittest.main()

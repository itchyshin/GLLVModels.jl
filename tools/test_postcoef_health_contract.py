"""Fail-closed controls for the follow-up three-fit health evidence."""
import copy
import math
import unittest
import core070_verify_postfit_policy_batch as verifier


class HealthContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = verifier.load_contract()
        self.report = verifier.synthetic_report(self.contract)
        self.case = self.report['cases']['CORE070-POSTFIT-COEF-EMPTY-NATIVE']['three_fit']

    def test_synthetic_valid_health_is_accepted(self):
        verifier.check_report(self.report, self.contract, 'unused')
        self.assertEqual(len(self.case['fit_health']['R']), 3)
        self.assertEqual(len(self.case['fit_health']['Julia']), 3)

    def test_each_failed_or_missing_diagnostic_rejected(self):
        for engine in ('R', 'Julia'):
            for index in range(3):
                for key, value in [('max_abs_gradient', 1e-5), ('pd_hessian', False),
                                   ('max_abs_gradient', math.nan), ('objective_verified', False),
                                   ('penalty_configuration_recorded', False), ('error', 'unavailable')]:
                    report = copy.deepcopy(self.report)
                    report['cases']['CORE070-POSTFIT-COEF-EMPTY-NATIVE']['three_fit']['fit_health'][engine][index][key] = value
                    with self.subTest(engine=engine,index=index,key=key,value=value), self.assertRaises(ValueError):
                        verifier.check_report(report, self.contract, 'unused')

    def test_missing_health_is_not_qualified(self):
        del self.case['fit_health']
        with self.assertRaises(ValueError):
            verifier.check_report(self.report, self.contract, 'unused')

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

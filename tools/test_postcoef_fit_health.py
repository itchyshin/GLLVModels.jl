"""Controls for a non-admitting POST-COEF fit-health review."""
import copy
import math
import sys
import unittest
import core070_postcoef_fit_health as health


def good():
    result = {}
    for engine in health.ENGINES:
        result[engine] = [dict(case=case, pd_hessian=True, max_abs_gradient=1e-7,
                               parameters=[0.0, 0.0], gradient=[1e-7, 0.0], free_indices=[1, 2],
                               optimizer_objective=3.0, recomputed_objective=3.0,
                               objective_verified=True, penalty_configuration_recorded=True,
                               error=None) for case in health.CASES]
        for row in result[engine]:
            if engine == 'R':
                row.update(fixed_prior_state='none', ridge_tau='Inf', ridge_penalised=False)
            else:
                row['reconstruction'] = dict(recovered_parameters=[0.0, 0.0], recovered_loglik=-3.0, parameters_match=True, loglik_matches=True, error=None)
                row['optimizer'] = dict(optimizer_parameters=[0.0, 0.0], optimizer_gradient=[1e-7, 0.0], optimizer_max_abs_gradient=1e-7, optimizer_minimum=3.0, optimizer_profile_value=3.0, optimizer_profile_verified=True, optimizer_error=None)
                row['objective_difference'] = 0.0
                row['objective_roundoff_limit'] = 8 * sys.float_info.epsilon * 3.0
                row['penalty_state'] = 'none'
                row['hessian'] = [[1.0, 0.0], [0.0, 1.0]]
                row['hessian_raw'] = [[1.0, 0.0], [0.0, 1.0]]
                row['hessian_asymmetry'] = 0.0
                row['hessian_asymmetry_limit'] = 64 * sys.float_info.epsilon
    return result


class FitHealthTests(unittest.TestCase):
    def test_all_six_valid(self):
        self.assertEqual(health.classify(good()), 'FIT_HEALTH_PASS')

    def test_any_missing_fit_or_engine_is_incomplete(self):
        for engine in health.ENGINES:
            data = good(); data[engine].pop()
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
            data = good(); del data[engine]
            self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_reconstruction_mismatches_cannot_qualify(self):
        for key, value in [('parameters_match', False), ('loglik_matches', False),
                           ('recovered_parameters', [0.0, 1.0]), ('recovered_loglik', -4.0),
                           ('error', 'unavailable')]:
            data = good()
            data['Julia'][0]['reconstruction'][key] = value
            with self.subTest(key=key):
                self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
        data = good(); del data['Julia'][0]['reconstruction']
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_julia_limit_uses_optimizer_coordinates(self):
        data = good()
        row = data['Julia'][0]
        row['gradient'][0] = 1e-3
        row['max_abs_gradient'] = 1e-3
        self.assertEqual(health.classify(data), 'FIT_HEALTH_PASS')
        row['optimizer']['optimizer_gradient'][0] = 1e-5
        row['optimizer']['optimizer_max_abs_gradient'] = 1e-5
        self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL')
        row['optimizer']['optimizer_gradient'][0] = 1e-7
        row['optimizer']['optimizer_max_abs_gradient'] = 1e-7
        row['gradient'][0] = float('nan')
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_finite_gradient_and_strict_signed_threshold(self):
        for value, expected in [(1e-5, 'FIT_HEALTH_FAIL'), (1e-4, 'FIT_HEALTH_FAIL'),
                                (math.nan, 'FIT_HEALTH_INCOMPLETE'),
                                (math.inf, 'FIT_HEALTH_INCOMPLETE'),
                                (-1.0, 'FIT_HEALTH_INCOMPLETE'),
                                (True, 'FIT_HEALTH_INCOMPLETE'),
                                (None, 'FIT_HEALTH_INCOMPLETE')]:
            data = good(); data['R'][0]['max_abs_gradient'] = value
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                data['R'][0]['gradient'][0] = value
            with self.subTest(value=value): self.assertEqual(health.classify(data), expected)

    def test_failed_hessian_is_failure_and_unknown_is_incomplete(self):
        for value, expected in [(False, 'FIT_HEALTH_FAIL'), (None, 'FIT_HEALTH_INCOMPLETE'),
                                (1, 'FIT_HEALTH_INCOMPLETE')]:
            data = good(); data['Julia'][1]['pd_hessian'] = value
            if value is False:
                data['Julia'][1]['hessian'][0][0] = -1.0
                data['Julia'][1]['hessian_raw'][0][0] = -1.0
            with self.subTest(value=value): self.assertEqual(health.classify(data), expected)

    def test_unverified_objective_penalty_or_error_never_passes(self):
        for key, value in [('objective_verified', False), ('penalty_configuration_recorded', False),
                           ('error', 'diagnostic failed')]:
            data = good(); data['Julia'][2][key] = value
            with self.subTest(key=key): self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_missing_error_field_is_incomplete(self):
        data = good(); del data['R'][0]['error']
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_duplicate_wrong_or_extra_fit_never_passes(self):
        for mutation in ('duplicate', 'wrong', 'extra'):
            data = good()
            if mutation == 'duplicate': data['R'][2]['case'] = data['R'][1]['case']
            if mutation == 'wrong': data['R'][2]['case'] = 'other'
            if mutation == 'extra': data['R'].append(copy.deepcopy(data['R'][0]))
            with self.subTest(mutation=mutation): self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_failure_is_not_hidden_by_incomplete_peer(self):
        data = good(); data['R'][0]['pd_hessian'] = False; data['Julia'][2]['error'] = 'missing'
        self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')

    def test_julia_objective_roundoff_bound_is_verified(self):
        data = good(); row = data['Julia'][0]
        row['recomputed_objective'] += 4 * math.ulp(3.0)
        row['objective_difference'] = abs(row['recomputed_objective'] - row['optimizer_objective'])
        row['objective_roundoff_limit'] = 8 * sys.float_info.epsilon * max(abs(row['recomputed_objective']), 3.0)
        self.assertEqual(health.classify(data), 'FIT_HEALTH_PASS')
        row['objective_roundoff_limit'] *= 2
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
        data = good(); data['Julia'][0]['recomputed_objective'] += 1e-7
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_julia_route_binds_source_tree_and_case(self):
        tree = 'df63c52066625049087a321bb9b010f8d834ddcf'
        meta = dict(source_tree=tree, case='trait_intercepts', route='gaussian_profile_default',
                    rank=1, integration='exact', prior='none')
        self.assertTrue(health.julia_prior_bound(meta, tree, 'trait_intercepts'))
        for key in meta:
            bad = dict(meta); bad[key] = None
            with self.subTest(key=key): self.assertFalse(health.julia_prior_bound(bad, tree, 'trait_intercepts'))
        self.assertFalse(health.julia_prior_bound(meta, 'a'*40, 'trait_intercepts'))
        self.assertFalse(health.julia_prior_bound(meta, tree, 'no_fixed_effects'))
        bad = dict(meta); bad['rank'] = True
        self.assertFalse(health.julia_prior_bound(bad, tree, 'trait_intercepts'))

    def test_prior_provenance_binds_to_receipt(self):
        provenance = dict(pin='P1', reference_commit='9539352f66f2db2cc26b1c393e67212a359b60c9',
                          source_tree_sha256='86fa00f05ee2ccb8ffef65a82cd59c47b1b2a256a5bad86957a92461a6dbe1db',
                          version='0.7.1', oracle_build_sha256='a'*64, call_verified=True, call_text='gllvmTMB(formula = f, data = df_long, trait = "trait", unit = "site", family = gaussian(), control = gllvmTMBcontrol(n_init = 1L, se = TRUE))')
        self.assertTrue(health.historical_r_prior_bound(provenance, 'a'*64))
        for key in provenance:
            bad = dict(provenance); bad[key] = None
            with self.subTest(key=key): self.assertFalse(health.historical_r_prior_bound(bad, 'a'*64))
        self.assertFalse(health.historical_r_prior_bound(provenance, 'b'*64))
        bad = dict(provenance); bad['call_verified'] = 1
        self.assertFalse(health.historical_r_prior_bound(bad, 'a'*64))

    def test_optimizer_evidence_required_and_threshold_enforced(self):
        data = good(); del data['Julia'][0]['optimizer']
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
        data = good(); data['Julia'][0]['optimizer']['optimizer_gradient'][0] = 1e-3
        self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')
        data = good(); data['Julia'][0]['optimizer']['optimizer_profile_value'] = 4.0
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_active_prior_requires_authenticated_objective(self):
        for engine, field in [('R', 'fixed_prior_state'), ('Julia', 'penalty_state')]:
            data = good(); data[engine][0][field] = 'active'
            with self.subTest(engine=engine):
                self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_active_ridge_hessian_not_assumed_from_sdreport(self):
        data = good(); data['R'][0]['ridge_penalised'] = True
        data['R'][0]['ridge_tau'] = 2.0
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_derived_numeric_fields_reject_boolean_values(self):
        for field in ('objective_difference', 'hessian_asymmetry'):
            data = good(); data['Julia'][0][field] = False
            with self.subTest(field=field):
                self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_hessian_record_detects_tampering(self):
        for field in ('hessian_raw', 'hessian'):
            data = good(); data['Julia'][0][field][0][1] = 0.1
            with self.subTest(field=field):
                self.assertNotEqual(health.classify(data), 'FIT_HEALTH_PASS')
        data = good(); data['Julia'][0]['hessian_asymmetry_limit'] *= 2
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_indefinite_hessian_cannot_be_hidden(self):
        data = good(); data['Julia'][0]['hessian'][0][0] = -1.0
        self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')

    def test_raw_failing_gradient_cannot_be_hidden(self):
        data = good(); data['R'][0]['gradient'][0] = 1e-3
        self.assertEqual(health.classify(data), 'FIT_HEALTH_FAIL_WITH_INCOMPLETE')

    def test_malformed_case_and_large_gradient_are_incomplete(self):
        data = good(); data['R'][0]['case'] = []
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')

    def test_reported_summary_cannot_hide_gradient_or_objective(self):
        for mutation in ('gradient', 'objective', 'missing_vector', 'indices', 'prior', 'parameters'):
            data = good(); row = data['R'][0]
            if mutation == 'gradient': row['gradient'][0] = 5e-7
            if mutation == 'objective': row['recomputed_objective'] = 4.0
            if mutation == 'missing_vector': del row['gradient']
            if mutation == 'indices': row['free_indices'] = [0, 2]
            if mutation == 'prior': row['fixed_prior_state'] = None
            if mutation == 'parameters': row['parameters'] = [0.0]
            with self.subTest(mutation=mutation):
                self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')
        data = good(); data['R'][0]['max_abs_gradient'] = 10 ** 1000
        self.assertEqual(health.classify(data), 'FIT_HEALTH_INCOMPLETE')


if __name__ == '__main__': unittest.main()

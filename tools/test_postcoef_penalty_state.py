"""penalty_state must be derived from the fit call arguments, never assumed.

The Julia batch writer builds its three POST-COEF fits from one keyword
NamedTuple and records penalty_state from those keywords through
postcoef_julia_penalty_state (tools/core070_postcoef_fit_health.jl).  A
penalised or prior-bearing record must stay INCOMPLETE; the certifier's
refusal is not changed here, only pinned.
"""
import pathlib
import re
import unittest

import core070_postcoef_fit_health as health
from test_postcoef_writer_fields import writer_record

TOOLS = pathlib.Path(__file__).resolve().parent
BATCH = (TOOLS / 'core070_postfit_policy_batch.jl').read_text()
HEALTH_JL = (TOOLS / 'core070_postcoef_fit_health.jl').read_text()
PD = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]


class BatchWriterSourceTests(unittest.TestCase):
    def test_writer_never_passes_an_unrecorded_penalty_state(self):
        self.assertNotIn('penalty_state=nothing', BATCH)
        self.assertIn('penalty_state=postcoef_julia_penalty_state(postcoef_fit_kwargs)', BATCH)

    def test_all_three_fits_use_the_recorded_call_arguments(self):
        calls = re.findall(r'gllvm\(@formula\([^)]*\), Y, site_data;([^)]*)\)', BATCH)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(c.strip() == 'postcoef_fit_kwargs...' for c in calls), calls)
        self.assertIn('postcoef_fit_kwargs = (K = 1,)', BATCH)

    def test_provenance_never_overwrites_the_derived_state(self):
        self.assertNotIn('h["penalty_state"] = "none"', BATCH)
        self.assertIn('h["penalty_state"] == "none" &&', BATCH)

    def test_julia_helper_is_defined(self):
        self.assertIn('function postcoef_julia_penalty_state(', HEALTH_JL)


class CertifierRefusalPinned(unittest.TestCase):
    def test_none_record_is_not_refused_for_penalty(self):
        row = writer_record(PD, [1, 2, 3])
        self.assertEqual(row['penalty_state'], 'none')
        self.assertNotEqual(health.certify_fit(row)['reason'],
                            'penalised or prior model: the no-prior diagnostic does not certify it')

    def test_active_or_unrecorded_penalty_stays_incomplete(self):
        for state in ('active', None, 'unknown', ''):
            with self.subTest(state=state):
                row = writer_record(PD, [1, 2, 3])
                row['penalty_state'] = state
                r = health.certify_fit(row)
                self.assertEqual(r['verdict'], health.CERT_INCOMPLETE)
                self.assertEqual(r['reason'], 'penalised or prior model: the no-prior diagnostic does not certify it')


if __name__ == '__main__':
    unittest.main()

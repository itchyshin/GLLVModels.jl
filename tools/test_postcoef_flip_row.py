#!/usr/bin/env python3
"""The --flip-postcoef-row step moves exactly the POST-COEF-EMPTY row of case-map-postfit.json and nothing else.

Runs on a copy of the tracked map in a temporary directory; no Julia, R, capture or network. The expected row is
written from the transition guard's own constants, not from the tool under test.
"""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

T = Path(__file__).resolve().parent
sys.path.insert(0, str(T))
import core070_postcoef_transition as guard  # noqa: E402
import core070_postfit_p1_receipts as receipts  # noqa: E402

REAL = T.parent
MAP = 'docs/dev-log/core070/true-parity-latest/case-map-postfit.json'
OLD_TIER = ('numeric comparison blocks pass, but at least one is degenerate (the R values are one constant or all ~0), '
            'so a constant or zero implementation would pass too; the row does not bind')
OLD_NOTE = ('integer_equality on the empty coef length (R 0, Julia 0; maintainer ruling 2026-10-05, D-319, item N2). '
            'The shared degenerate-comparison rule flags it: every |R value| < 1e-10, so a constant-zero implementation '
            'would pass too. Binding it would need the shared rule to accept a 0 = 0 length, which it does not.')


def pre_flip_map_bytes():
    """The tracked map with the POST-COEF-EMPTY row put back to its pre-flip state (the tracked map is already flipped)."""
    doc = json.loads((REAL / MAP).read_text())
    row = next(r for r in doc['rows'] if r['source_id'] == guard.TARGET)
    row.update(evidence_tier='numeric_non_discriminating', note=OLD_NOTE,
               evidence={'non_binding_receipts': [guard.OLD], 'tier': OLD_TIER},
               measured_result={'case_verdicts': {guard.CID: 'PASS'}, 'batch_verifier': {guard.CID: 'PASS'},
                                'discriminating': {guard.CID: False}})
    doc['counts']['numeric_pass'] -= 1
    doc['counts']['numeric_non_discriminating'] += 1
    return (json.dumps(doc, indent=2) + '\n').encode()


class FlipRow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='postcoef-flip-')
        self.root = Path(self.tmp.name)
        (self.root / MAP).parent.mkdir(parents=True)
        (self.root / MAP).write_bytes(pre_flip_map_bytes())
        self.before = json.loads((self.root / MAP).read_text())
        self.case = self.root / guard.NEW
        self.case.parent.mkdir(parents=True)
        self.write_case({'case_id': receipts.COEF_THREE_FIT_CID, 'harness_case_id': receipts.COEF_EMPTY_CID,
                         'verdict': 'PASS', 'batch_verifier': {'status': 'PASS'}})
        out = self.root / 'docs/dev-log/core070/true-parity-latest'
        patches = [patch.object(receipts, 'ROOT', self.root), patch.object(receipts, 'OUT', out),
                   patch.object(receipts, 'REC', out / 'receipts/postfit')]
        for p in patches:
            p.start(); self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def write_case(self, case):
        self.case.write_text(json.dumps(case))

    def flip(self):
        with redirect_stdout(StringIO()):
            receipts.flip_postcoef_row()

    def after(self):
        return json.loads((self.root / MAP).read_text())

    def test_only_the_target_row_and_counts_change_and_match_the_guard(self):
        self.flip()
        after = self.after()
        old = {r['source_id']: r for r in self.before['rows']}
        new = {r['source_id']: r for r in after['rows']}
        self.assertEqual(list(old), list(new))
        self.assertEqual([k for k in old if old[k] != new[k]], [guard.TARGET])
        row = new[guard.TARGET]
        self.assertEqual((row['evidence_tier'], row['measured_against'], row['note']), ('numeric', guard.PIN, guard.NOTE))
        self.assertEqual(row['evidence'], {'receipt': [guard.NEW], 'tier': guard.TIER})
        self.assertEqual(row['measured_result'], {'case_verdicts': {guard.CID: 'PASS'},
                                                  'batch_verifier': {guard.CID: 'PASS'}, 'row_verdict': 'PASS'})
        for key in ('disposition', 'signed_by', 'signed_on', 'signature_ref'):
            self.assertEqual(row.get(key), old[guard.TARGET].get(key))
        counts = copy.deepcopy(self.before['counts'])
        counts['numeric_pass'] += 1; counts['numeric_non_discriminating'] -= 1
        self.assertEqual(after['counts'], counts)

    def test_refuses_without_a_passing_case_receipt(self):
        self.case.unlink()
        with self.assertRaises(SystemExit):
            self.flip()
        self.write_case({'case_id': receipts.COEF_THREE_FIT_CID, 'harness_case_id': receipts.COEF_EMPTY_CID,
                         'verdict': 'UNQUALIFIED', 'batch_verifier': {'status': 'PASS'}})
        with self.assertRaises(SystemExit):
            self.flip()
        self.assertEqual(self.after(), self.before)

    def test_refuses_a_row_that_is_not_in_the_pre_flip_state_and_is_not_repeatable(self):
        self.flip()
        done = (self.root / MAP).read_bytes()
        with self.assertRaises(SystemExit):
            self.flip()
        self.assertEqual((self.root / MAP).read_bytes(), done)

    def test_the_reconstructed_pre_flip_row_is_the_one_the_guard_knows(self):
        row = next(r for r in self.before['rows'] if r['source_id'] == guard.TARGET)
        canonical = json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), guard.OLD_TARGET_SHA256)
        self.assertEqual(row['evidence']['non_binding_receipts'], [guard.OLD])
        self.assertIsNone(row['disposition'])

    def test_the_real_tracked_row_is_flipped_and_unsigned(self):
        real = json.loads((REAL / MAP).read_text())
        row = next(r for r in real['rows'] if r['source_id'] == guard.TARGET)
        self.assertEqual(row['evidence']['receipt'], [guard.NEW])
        self.assertIsNone(row['disposition'])
        self.assertNotIn('signed_by', row)

if __name__ == '__main__':
    unittest.main()

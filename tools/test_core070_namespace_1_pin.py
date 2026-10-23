"""Pin switch for the namespace-1 Tier 0 verifier (no R, no Julia needed).

Runs tools/core070_verify_namespace_1_batch.py --self-test under the default
pin (P0, unchanged behaviour) and under GLLVM_PARITY_PIN=P1, and checks an
unknown pin is refused rather than silently falling back.
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "tools/core070_verify_namespace_1_batch.py"


def run(pin=None):
    env = dict(os.environ)
    env.pop("GLLVM_PARITY_PIN", None)
    if pin is not None:
        env["GLLVM_PARITY_PIN"] = pin
    return subprocess.run([sys.executable, str(VERIFY), "--self-test"], cwd=ROOT, env=env,
                          capture_output=True, text=True)


class NamespaceOnePinTest(unittest.TestCase):
    def test_default_is_p1(self):
        # Default pin flipped P0 -> P1 (tools/parity_oracle.py _DEFAULT_PIN).
        r = run()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("pin=P1", r.stdout)
        self.assertIn("exec=50 needs=2 retired=2", r.stdout)

    def test_p0_selectable_original_counts(self):
        r = run("P0")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("pin=P0", r.stdout)
        self.assertIn("exec=48 needs=6 retired=0", r.stdout)

    def test_p1_counts_and_retired_mutation(self):
        r = run(" p1 ")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("pin=P1", r.stdout)
        self.assertIn("exec=50 needs=2 retired=2", r.stdout)
        self.assertIn("rejected_mutations=8", r.stdout)

    def test_unknown_pin_refused(self):
        r = run("P9")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not a recognized pin", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()

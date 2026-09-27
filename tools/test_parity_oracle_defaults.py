#!/usr/bin/env python3
"""Smoke: parity tools default to frozen gllvmTMB 0.7.0 oracle (P13), plus the
additive P1 pin and its opt-in GLLVM_PARITY_PIN switch (D-294/D-295)."""
import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import parity_oracle  # noqa: E402
import parity_ledger  # noqa: E402


class ParityOracleDefaults(unittest.TestCase):
    def test_frozen_oracle_constant(self):
        self.assertEqual(
            parity_oracle.FROZEN_GLLVMTMB_ORACLE,
            "b4d5fee64def88bc768dda1f1f77c29b295edd86",
        )
        self.assertEqual(parity_oracle.DEFAULT_R_REF, parity_oracle.FROZEN_GLLVMTMB_ORACLE)
        self.assertEqual(parity_ledger.DEFAULT_REF, parity_oracle.DEFAULT_R_REF)

    def test_argparse_defaults(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--ref", default=parity_ledger.DEFAULT_REF)
        ap.add_argument("--r-ref", default=None)
        ns = ap.parse_args([])
        self.assertEqual(ns.ref, parity_oracle.FROZEN_GLLVMTMB_ORACLE)
        self.assertIsNone(ns.r_ref)

    def test_self_test_passes(self):
        out = subprocess.run(
            [sys.executable, str(TOOLS / "parity_ledger.py"), "--self-test"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(out.returncode, 0, out.stderr or out.stdout)
        self.assertIn("SELFTEST_OK", out.stdout)

    def test_p1_oracle_constant(self):
        # Additive pin (D-294/D-295): full P1 SHA, distinct from P0, and
        # present in the named-pin lookup table alongside P0.
        self.assertEqual(
            parity_oracle.P1_GLLVMTMB_ORACLE,
            "9539352f66f2db2cc26b1c393e67212a359b60c9",
        )
        self.assertNotEqual(parity_oracle.P1_GLLVMTMB_ORACLE, parity_oracle.FROZEN_GLLVMTMB_ORACLE)
        self.assertEqual(parity_oracle.R_REF_PINS["P0"], parity_oracle.FROZEN_GLLVMTMB_ORACLE)
        self.assertEqual(parity_oracle.R_REF_PINS["P1"], parity_oracle.P1_GLLVMTMB_ORACLE)

    def test_default_stays_p0_without_the_switch(self):
        # The re-pin is additive: with GLLVM_PARITY_PIN unset, DEFAULT_R_REF
        # (and everything derived from it, e.g. parity_ledger.DEFAULT_REF)
        # must stay exactly what it is today. Flipping the default is a
        # separate, later PR (see tools/parity_oracle.py docstring).
        env = {k: v for k, v in os.environ.items() if k != "GLLVM_PARITY_PIN"}
        out = subprocess.run(
            [sys.executable, "-c", "import parity_oracle as po; print(po.DEFAULT_R_REF)"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=str(TOOLS),
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), parity_oracle.FROZEN_GLLVMTMB_ORACLE)

    def test_pin_switch_selects_p1_when_set(self):
        env = {k: v for k, v in os.environ.items() if k != "GLLVM_PARITY_PIN"}
        env["GLLVM_PARITY_PIN"] = "P1"
        out = subprocess.run(
            [sys.executable, "-c", "import parity_oracle as po; print(po.DEFAULT_R_REF)"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=str(TOOLS),
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), parity_oracle.P1_GLLVMTMB_ORACLE)

    def test_pin_switch_unknown_value_falls_back_to_p0(self):
        env = {k: v for k, v in os.environ.items() if k != "GLLVM_PARITY_PIN"}
        env["GLLVM_PARITY_PIN"] = "not-a-real-pin"
        out = subprocess.run(
            [sys.executable, "-c", "import parity_oracle as po; print(po.DEFAULT_R_REF)"],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=str(TOOLS),
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), parity_oracle.FROZEN_GLLVMTMB_ORACLE)


if __name__ == "__main__":
    raise SystemExit(unittest.main())

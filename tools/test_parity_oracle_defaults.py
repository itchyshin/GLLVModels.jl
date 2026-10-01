#!/usr/bin/env python3
"""Smoke: parity tools default to the P1 gllvmTMB pin (default flipped from the
frozen 0.7.0 oracle P0), with P0 still selectable via GLLVM_PARITY_PIN=P0
(D-294/D-295)."""
import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

# Scrub the switch before importing, so this test file's own module-level
# import of parity_oracle/parity_ledger (and every test below that reads
# their already-imported attributes rather than spawning a clean subprocess)
# reflects the default pin regardless of whatever the ambient shell exports.
os.environ.pop("GLLVM_PARITY_PIN", None)

import parity_oracle  # noqa: E402
import parity_ledger  # noqa: E402


def _clean_env(**overrides):
    env = {k: v for k, v in os.environ.items() if k != "GLLVM_PARITY_PIN"}
    env.update(overrides)
    return env


def _default_r_ref_in_subprocess(pin_value=None):
    env = _clean_env() if pin_value is None else _clean_env(GLLVM_PARITY_PIN=pin_value)
    return subprocess.run(
        [sys.executable, "-c", "import parity_oracle as po; print(po.DEFAULT_R_REF)"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(TOOLS),
    )


class ParityOracleDefaults(unittest.TestCase):
    def test_frozen_oracle_constant(self):
        self.assertEqual(
            parity_oracle.FROZEN_GLLVMTMB_ORACLE,
            "b4d5fee64def88bc768dda1f1f77c29b295edd86",
        )
        self.assertEqual(
            parity_oracle.DEFAULT_R_REF,
            parity_oracle.P1_GLLVMTMB_ORACLE,
            "DEFAULT_R_REF is not the P1 default -- if GLLVM_PARITY_PIN "
            "is exported in this shell, unset it before running this test",
        )
        self.assertEqual(parity_ledger.DEFAULT_REF, parity_oracle.DEFAULT_R_REF)

    def test_argparse_defaults(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--ref", default=parity_ledger.DEFAULT_REF)
        ap.add_argument("--r-ref", default=None)
        ns = ap.parse_args([])
        self.assertEqual(
            ns.ref,
            parity_oracle.P1_GLLVMTMB_ORACLE,
            "unset GLLVM_PARITY_PIN before running this test if it is exported",
        )
        self.assertIsNone(ns.r_ref)

    def test_self_test_passes(self):
        out = subprocess.run(
            [sys.executable, str(TOOLS / "parity_ledger.py"), "--self-test"],
            capture_output=True,
            text=True,
            check=False,
            env=_clean_env(),
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

    def test_default_is_p1_without_the_switch(self):
        # With GLLVM_PARITY_PIN unset, DEFAULT_R_REF (and everything derived
        # from it, e.g. parity_ledger.DEFAULT_REF) is the P1 pin.
        out = _default_r_ref_in_subprocess()
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), parity_oracle.P1_GLLVMTMB_ORACLE)

    def test_pin_switch_selects_p0_when_set(self):
        # P0 stays selectable (the frozen-R CI job relies on this).
        out = _default_r_ref_in_subprocess("P0")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), parity_oracle.FROZEN_GLLVMTMB_ORACLE)

    def test_pin_switch_selects_p1_when_set(self):
        out = _default_r_ref_in_subprocess("P1")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), parity_oracle.P1_GLLVMTMB_ORACLE)

    def test_pin_switch_is_case_and_whitespace_insensitive(self):
        for raw in ("p1", " P1 ", "P1\n"):
            with self.subTest(raw=raw):
                out = _default_r_ref_in_subprocess(raw)
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertEqual(out.stdout.strip(), parity_oracle.P1_GLLVMTMB_ORACLE)

    def test_pin_switch_rejects_unrecognized_value(self):
        # A mistyped pin must fail loudly, not silently fall back to P0.
        for raw in ("not-a-real-pin", "P2", "1P"):
            with self.subTest(raw=raw):
                env = _clean_env(GLLVM_PARITY_PIN=raw)
                out = subprocess.run(
                    [sys.executable, "-c", "import parity_oracle"],
                    capture_output=True,
                    text=True,
                    check=False,
                    env=env,
                    cwd=str(TOOLS),
                )
                self.assertNotEqual(out.returncode, 0, out.stdout)
                self.assertIn("GLLVM_PARITY_PIN", out.stderr)


if __name__ == "__main__":
    raise SystemExit(unittest.main())

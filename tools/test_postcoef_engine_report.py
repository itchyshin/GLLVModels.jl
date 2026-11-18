#!/usr/bin/env python3
"""The POST-COEF engine setting is the ruled one: a changed engine tree is reported, not refused."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core070_postcoef_live_ref as live  # noqa: E402


class EngineSetting(unittest.TestCase):
    def test_engine_is_reported_not_refused(self):
        self.assertIs(live.ENGINE_BOUND_STRICT, False)


if __name__ == '__main__':
    unittest.main()

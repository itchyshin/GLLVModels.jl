"""Fixture regression: a docstring mentioning the token in prose must not be
read as the real assignment. Flipping the default is a one-token change: set
``_DEFAULT_PIN = "P1"`` below.
"""

import os

FROZEN_GLLVMTMB_ORACLE = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
P1_GLLVMTMB_ORACLE = "9539352f66f2db2cc26b1c393e67212a359b60c9"
R_REF_PINS = {"P0": FROZEN_GLLVMTMB_ORACLE, "P1": P1_GLLVMTMB_ORACLE}
_PIN_ENV_VAR = "GLLVM_PARITY_PIN"

# The real assignment: still P0, further down the file than the docstring's prose mention.
_DEFAULT_PIN = "P0"

_raw_pin = os.environ.get(_PIN_ENV_VAR)
_selected_pin = _raw_pin.strip().upper() if _raw_pin else _DEFAULT_PIN
DEFAULT_R_REF = R_REF_PINS[_selected_pin]
CAPABILITY_LEDGER_REF = "origin/main"

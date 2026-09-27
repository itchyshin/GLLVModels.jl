"""Fixture copy of parity_oracle.py pin constants, for true_parity_check.mjs tests only.
Mirrors the real tools/parity_oracle.py shape added by PR #524 (P1 pin, R_REF_PINS,
GLLVM_PARITY_PIN switch, CAPABILITY_LEDGER_REF).
"""

import os

FROZEN_GLLVMTMB_ORACLE = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
FROZEN_GLLVMTMB_VERSION = "0.7.0"

P1_GLLVMTMB_ORACLE = "9539352f66f2db2cc26b1c393e67212a359b60c9"
P1_GLLVMTMB_VERSION = "0.7.1-candidate"

R_REF_PINS = {
    "P0": FROZEN_GLLVMTMB_ORACLE,
    "P1": P1_GLLVMTMB_ORACLE,
}

_PIN_ENV_VAR = "GLLVM_PARITY_PIN"
_selected_pin = os.environ.get(_PIN_ENV_VAR, "P0")

DEFAULT_R_REF = R_REF_PINS.get(_selected_pin, FROZEN_GLLVMTMB_ORACLE)

CAPABILITY_LEDGER_REF = "origin/main"

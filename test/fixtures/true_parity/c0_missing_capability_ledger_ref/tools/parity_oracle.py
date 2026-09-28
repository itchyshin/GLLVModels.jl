"""Fixture copy of parity_oracle.py pin constants, for true_parity_check.mjs tests only.
Mirrors the real tools/parity_oracle.py shape added by PR #524 (P1 pin, R_REF_PINS,
GLLVM_PARITY_PIN switch, CAPABILITY_LEDGER_REF, the _DEFAULT_PIN token).
"""

import os

FROZEN_GLLVMTMB_ORACLE = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
FROZEN_GLLVMTMB_VERSION = "0.7.0"

P1_GLLVMTMB_ORACLE = "9539352f66f2db2cc26b1c393e67212a359b60c9"
P1_GLLVMTMB_VERSION = "0.7.1"

R_REF_PINS = {
    "P0": FROZEN_GLLVMTMB_ORACLE,
    "P1": P1_GLLVMTMB_ORACLE,
}

_PIN_ENV_VAR = "GLLVM_PARITY_PIN"

# The only line that changes when the programme re-points the default pin: set this to "P1".
_DEFAULT_PIN = "P1"

_raw_pin = os.environ.get(_PIN_ENV_VAR)
if _raw_pin is None:
    _selected_pin = _DEFAULT_PIN
else:
    _selected_pin = _raw_pin.strip().upper()
    if _selected_pin not in R_REF_PINS:
        raise SystemExit(f"{_PIN_ENV_VAR}={_raw_pin!r} is not a recognized pin.")

DEFAULT_R_REF = R_REF_PINS[_selected_pin]


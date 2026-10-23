"""Frozen gllvmTMB oracle pins for GLLVModels.jl parity tools (P13 / D-220).

Export-surface parity reads R's ``NAMESPACE`` at ``DEFAULT_R_REF`` (frozen
gllvmTMB 0.7.0). Capability-status CLOSURE uses ``gllvmTMB/tools/parity_ledger.R``
with ``CAPABILITY_LEDGER_REF`` on both sides — ``docs/design/capability-status.md``
post-dates the oracle commit and is absent at ``DEFAULT_R_REF``.

P1 re-pin (D-294/D-295, 2026-09-27): the maintainer re-targeted the true-parity
programme at gllvmTMB ``main`` ``9539352f6`` (0.7.1, untagged as of 2026-09-27).
``FROZEN_GLLVMTMB_ORACLE`` (P0) is unchanged and every P0 receipt and pin string
stays valid. ``P1_GLLVMTMB_ORACLE`` is now the DEFAULT (``_DEFAULT_PIN = "P1"``);
set the ``GLLVM_PARITY_PIN`` environment variable to ``"P0"`` (case/whitespace
insensitive) to have ``DEFAULT_R_REF`` resolve to the frozen oracle instead. An unrecognized
value raises ``SystemExit`` at import time rather than silently falling back
to P0.

The default is the single token ``_DEFAULT_PIN`` below -- see
``docs/dev-log/core070/true-parity-latest/GATES.md`` clause C0, which checks
for exactly that token.
"""

import os

# frozen gllvmTMB 0.7.0 export oracle (CI + programme qualification pin, P0)
FROZEN_GLLVMTMB_ORACLE = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
FROZEN_GLLVMTMB_VERSION = "0.7.0"
FROZEN_GLLVMTMB_SHORT = FROZEN_GLLVMTMB_ORACLE[:8]

# gllvmTMB main re-pin target (P1, D-294/D-295, 2026-09-27 maintainer retarget;
# 0.7.1, untagged as of 2026-09-27).
P1_GLLVMTMB_ORACLE = "9539352f66f2db2cc26b1c393e67212a359b60c9"
P1_GLLVMTMB_VERSION = "0.7.1"
P1_GLLVMTMB_SHORT = P1_GLLVMTMB_ORACLE[:8]

# Named pins, keyed by short label, so a caller can select one explicitly
# instead of hardcoding a SHA (see GLLVM_PARITY_PIN below).
R_REF_PINS = {
    "P0": FROZEN_GLLVMTMB_ORACLE,
    "P1": P1_GLLVMTMB_ORACLE,
}

# Explicit, documented pin switch. Set the GLLVM_PARITY_PIN environment
# variable (any case, surrounding whitespace ignored: "p1", " P1 " and "P1"
# all select P1) to make DEFAULT_R_REF resolve to a pin other than the
# default below. The default is P1; set GLLVM_PARITY_PIN=P0 to select the
# frozen 0.7.0 oracle (the CI "Frozen R 0.7.0 family smoke" job does so
# explicitly).
_PIN_ENV_VAR = "GLLVM_PARITY_PIN"

# The only line that changes when the programme re-points the default pin.
# Flipped P0 -> P1 (maintainer-approved); C0 in GATES.md reads this token.
_DEFAULT_PIN = "P1"

_raw_pin = os.environ.get(_PIN_ENV_VAR)
if _raw_pin is None:
    _selected_pin = _DEFAULT_PIN
else:
    _selected_pin = _raw_pin.strip().upper()
    if _selected_pin not in R_REF_PINS:
        raise SystemExit(
            f"{_PIN_ENV_VAR}={_raw_pin!r} is not a recognized pin. "
            f"Set {_PIN_ENV_VAR} to one of {sorted(R_REF_PINS)} (case/whitespace "
            f"insensitive), or leave it unset to use the default ({_DEFAULT_PIN})."
        )

# Default R-side git ref for NAMESPACE / export parity (not working tree, not live main)
DEFAULT_R_REF = R_REF_PINS[_selected_pin]

# Public name of the pin GLLVM_PARITY_PIN selected ("P0" unless overridden).
# Other harness entry points (tools/core070_build_oracle.py,
# test/parity/parity_helpers.jl) key their own per-pin data off this name
# instead of re-deriving the switch themselves, so this module stays the
# single place that reads GLLVM_PARITY_PIN.
SELECTED_PIN = _selected_pin

# Capability ledger join: file landed after the frozen oracle (see r-ref-closure-receipt)
CAPABILITY_LEDGER_REF = "origin/main"

"""Frozen gllvmTMB oracle pins for GLLVModels.jl parity tools (P13 / D-220).

Export-surface parity reads R's ``NAMESPACE`` at ``DEFAULT_R_REF`` (frozen
gllvmTMB 0.7.0). Capability-status CLOSURE uses ``gllvmTMB/tools/parity_ledger.R``
with ``CAPABILITY_LEDGER_REF`` on both sides — ``docs/design/capability-status.md``
post-dates the oracle commit and is absent at ``DEFAULT_R_REF``.

P1 re-pin (D-294/D-295, 2026-09-27): the maintainer re-targeted the true-parity
programme at gllvmTMB ``main`` ``9539352f6`` (0.7.1 candidate). This is
ADDITIVE -- ``FROZEN_GLLVMTMB_ORACLE`` / ``DEFAULT_R_REF`` are unchanged and
every P0 receipt, pin string, and test that asserts P0 stays valid.
``P1_GLLVMTMB_ORACLE`` is a second, independently selectable pin; set the
``GLLVM_PARITY_PIN`` environment variable to ``"P1"`` to have ``DEFAULT_R_REF``
resolve to it instead. Flipping the *default* (unsetting the need for that
env var) is a separate, later change once P1 twin tests exist -- see
``docs/dev-log/core070/true-parity-latest/GATES.md`` clause C0.
"""

import os

# frozen gllvmTMB 0.7.0 export oracle (CI + programme qualification pin, P0)
FROZEN_GLLVMTMB_ORACLE = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
FROZEN_GLLVMTMB_VERSION = "0.7.0"
FROZEN_GLLVMTMB_SHORT = FROZEN_GLLVMTMB_ORACLE[:8]

# gllvmTMB main re-pin target (P1, D-294/D-295, 2026-09-27 maintainer retarget;
# 0.7.1 candidate, untagged as of 2026-09-27).
P1_GLLVMTMB_ORACLE = "9539352f66f2db2cc26b1c393e67212a359b60c9"
P1_GLLVMTMB_VERSION = "0.7.1-candidate"
P1_GLLVMTMB_SHORT = P1_GLLVMTMB_ORACLE[:8]

# Named pins, keyed by short label, so a caller can select one explicitly
# instead of hardcoding a SHA (see GLLVM_PARITY_PIN below).
R_REF_PINS = {
    "P0": FROZEN_GLLVMTMB_ORACLE,
    "P1": P1_GLLVMTMB_ORACLE,
}

# Explicit, documented pin switch. Set the GLLVM_PARITY_PIN environment
# variable to "P1" to make DEFAULT_R_REF resolve to the P1 oracle instead of
# the frozen P0 one. Unset (or any value other than "P1") keeps P0 as the
# default -- this PR does NOT flip the default itself; every existing P0
# test and tool (tools/parity_ledger.py's --self-test,
# tools/test_parity_oracle_defaults.py) keeps behaving exactly as today
# unless a caller opts in via this variable.
_PIN_ENV_VAR = "GLLVM_PARITY_PIN"
_selected_pin = os.environ.get(_PIN_ENV_VAR, "P0")

# Default R-side git ref for NAMESPACE / export parity (not working tree, not live main)
DEFAULT_R_REF = R_REF_PINS.get(_selected_pin, FROZEN_GLLVMTMB_ORACLE)

# Capability ledger join: file landed after the frozen oracle (see r-ref-closure-receipt)
CAPABILITY_LEDGER_REF = "origin/main"

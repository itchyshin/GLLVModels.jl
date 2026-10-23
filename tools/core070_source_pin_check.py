"""Oracle source-pin check shared by the postfit batch verifiers (PR #569 review finding 2).

Same rule as tools/core070_verify_wave6_conversion_batch.py (PR #567 review
finding 6), plus the package version: the batch R script records the library's
CORE070_SOURCE_PIN.toml marker (tools/core070_source_pin.R) in receipt.json as
`source_pin`, and its installed version as `gllvmTMB_version`. Both must match
the selected pin in tools/core070_oracle_pins.toml. A P0 receipt written
before the record existed carries none and is accepted only at P0.
"""
from copy import deepcopy
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
PINS = tomllib.loads((ROOT / "tools/core070_oracle_pins.toml").read_text())
SOURCE_PIN_KEYS = ("reference_commit", "source_tree_sha256", "archive_sha256", "namespace_sha256")


def source_pin_problem(receipt, pin):
    """None when the receipt's source-pin record and version match `pin`, else the reason."""
    record = receipt.get("source_pin")
    if record is None and pin == "P0":
        return None
    if not isinstance(record, dict):
        return "receipt carries no CORE070_SOURCE_PIN marker record"
    expected = PINS[pin]
    for key in SOURCE_PIN_KEYS:
        if record.get(key) != expected[key]:
            return f"source pin {key} does not match tools/core070_oracle_pins.toml [{pin}]"
    if receipt.get("gllvmTMB_version") != expected["version"]:
        return (f"receipt gllvmTMB_version {receipt.get('gllvmTMB_version')!r} != "
                f"tools/core070_oracle_pins.toml [{pin}] version {expected['version']!r}")
    if record.get("version") != expected["version"]:
        return f"source pin record version does not match tools/core070_oracle_pins.toml [{pin}]"
    return None


def check_source_pin(receipt, pin, need):
    problem = source_pin_problem(receipt, pin)
    need(problem is None, problem or "")


def self_test(pin):
    """Mutation battery for the check; returns the number of mutations rejected."""
    good = {"gllvmTMB_version": PINS[pin]["version"],
            "source_pin": {**{k: PINS[pin][k] for k in SOURCE_PIN_KEYS}, "version": PINS[pin]["version"]}}
    assert source_pin_problem(good, pin) is None, "synthetic source-pin receipt should pass"
    other = "P0" if pin == "P1" else "P1"
    mutations = {
        "tree sha from another build": lambda r: r["source_pin"].update(source_tree_sha256="0" * 64),
        "marker from the other pin": lambda r: r["source_pin"].update(
            {k: PINS[other][k] for k in SOURCE_PIN_KEYS}),
        "receipt version from the other pin": lambda r: r.update(gllvmTMB_version=PINS[other]["version"]),
        "marker record not a table": lambda r: r.update(source_pin="absent"),
    }
    if pin != "P0":
        mutations["marker record missing"] = lambda r: r.pop("source_pin")
    for label, mutate in mutations.items():
        r = deepcopy(good)
        mutate(r)
        if source_pin_problem(r, pin) is None:
            raise AssertionError(f"source-pin mutation was NOT rejected: {label}")
    return len(mutations)

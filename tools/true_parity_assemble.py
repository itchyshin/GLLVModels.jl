#!/usr/bin/env python3
"""Assemble the P1 true-parity ledger files that tools/true_parity_check.mjs reads.

Derives and collates only. Every classification, disposition, evidence tier and case id
comes from the per-family case maps already tracked under
docs/dev-log/core070/true-parity-latest/ (case-map-<family>.json). This tool never assigns,
edits or resolves any of them, and nothing it writes is a signature.

Outputs (all generated; regenerate, never hand-edit):

  scoreboard.md           one row per case-map row, in the table format the checker parses
                          (id | requires | status | receipt/disposition | notes), plus a totals
                          table by family that the checker's parser skips by construction.
  case-map-assembled.json the per-family rows folded into one file, for
                          PARITY_CASEMAP=... runs. Deliberately NOT named case-map.json: that
                          name belongs to PR #533, whose rows the maintainer signs.
  reverse-gap.json        exported GLLVModels symbols with no gllvmTMB counterpart (a JSON
                          array, the checker's C6 schema), every item status "unsigned" and
                          decision null. Built from reverse-gap-inputs.json.

Status of a scoreboard row (first rule that applies):

  DISPOSITION-SIGNED  disposition "DISPOSITION-SIGNED" with an allowed signer and a real past
                      date on the row, and (if it cites a receipt) no dangling receipt and no
                      stale carry (the checker's C1 rule and order). Also a numeric row whose
                      only problem is a failed receipt status field, waived by a valid
                      maintainer-signed receipt_status_exception (C1 bound_signed=).
  EVIDENCED           the row binds under the checker's own C1 numeric rule: evidence_tier
                      "numeric", non-empty executable_case_ids, every evidence.receipt a file,
                      carry fresh at P1, and every receipt's comparison block pinned to P1,
                      within tolerance, covering every case id, with no failed status field.
  NUMERIC-UNVERIFIED  evidence_tier "numeric" but the rule above does not hold (reason given).
  EVIDENCED-BEHAVIOURAL
                      evidence_tier "behavioural" and the row binds under the checker's behavioural
                      rule (itchyshin/GLLVModels.jl#684 item 2): non-empty executable_case_ids,
                      every receipt a file, carry fresh at P1, a `behaviour` block pinned to P1
                      whose applicable entries cover every case id and match after canonicalising
                      through behaviour-equivalence.json by class identity, and no failed receipt
                      status field (behavioural_receipt_problem lists them), and the row's source_id
                      is one the ruling covers (the 59 listed inference rows,
                      BEHAVIOURAL_INFERENCE_SOURCE_IDS, or one of four named C1 rows,
                      BEHAVIOURAL_NAMED_SOURCE_IDS, or one of the 14 rows of the 2026-10-05 extension,
                      BEHAVIOURAL_EXTENDED_SOURCE_IDS) and any comparison block in a cited receipt holds.
                      Never emitted for a data, grouping or realistic-size row.
  BEHAVIOURAL-UNVERIFIED
                      evidence_tier "behavioural" but the rule above does not hold (reason given).
  DISPOSITION-UNVERIFIED / NEEDS-SURFACE / <disposition>
                      a non-null disposition that is not a valid signature. It never becomes a status
                      word the assembler itself writes (RESERVED_STATUS_WORDS): after trimming and
                      upper-casing, a disposition equal to `EVIDENCED`, `EVIDENCED-BEHAVIOURAL`,
                      `DISPOSITION-SIGNED` without a signature, or any other such word is
                      DISPOSITION-UNVERIFIED, and so is one that is not a string, is blank, or carries a
                      `|` or a line break (disposition_status). Text containing NEEDS_JULIA_SURFACE reads
                      NEEDS-SURFACE; any other plain string is copied and is not done.
  NOT-MEASURED        no receipt and no non-binding receipt cited at all.
  otherwise           the row's evidence_tier, collated into a bucket by TIER_BUCKET below
                      (the tier string itself is copied verbatim into the notes column).

An evidence_tier missing from TIER_BUCKET fails the run: a new tier needs a human to decide
which bucket it reads as, rather than this tool guessing.

Rulings of 2026-10-05 (maintainer ruling 2026-10-05, D-319; GATES.md): ported from the checker are the 14-row
extension of the behavioural list (BEHAVIOURAL_EXTENDED_SOURCE_IDS), the bridge readback split
(BRIDGE_READBACK_ROW_PREFIX), boundary-context cases (boundary_context) and convergence parity
(convergence_parity_problem). The checker's C4 direct-engine test reads the scoreboard only and has no port here.

Integer equality (itchyshin/GLLVModels.jl#684 item 1): a numeric comparison case with
"kind": "integer_equality" needs safe-integer r_value and julia_value (magnitude below 2^53) and
tolerance exactly 0.5.

Visible text: a behaviour label, an equivalence class's canonical, labels and basis, a C6 basis and a C6
ruling ref must contain at least one character in Unicode category L, N, P or S (is_visible, the
checker's isVisible). Strings of only whitespace or format characters (U+FEFF, U+200B, U+0085) are empty.

Reverse-gap decisions (itchyshin/GLLVModels.jl#684 item 3): an optional reverse-gap-decisions.json
(schema in GATES.md) is read by build_reverse_gap. Each decision is copied, with its basis and the
ruling, onto the matching item, whose status becomes "decided". A decision naming no reverse-gap
item fails the run (stale). Nothing here is a signature: the file carries the ruling. The tool refuses
to copy a ruling it does not recognise (C6_RULINGS: only #684 item 3, dated 2026-10-02, covering
KEPT_AS_JULIA_EXTRA and EXCLUDED_INTERNAL_HELPER), a signer outside the allow-list, a decision word
the ruling does not cover, an empty criterion, a generator that is not a file in the tree (an absolute path,
a path with a ".." segment anywhere, a directory), a basis with
no visible character, or a KEPT_AS_JULIA_EXTRA basis that does not cite, exactly as docs/src/<path>.md, an
existing .md file under docs/src (every token of the basis that contains "docs/src" must be such a path;
kept_basis_problem has the rule); the checker's C6 judges the same.

Reverse gap: a Julia export has a gllvmTMB counterpart when its name equals an R export or S3 generic
name after removing "_" and "." and lower-casing (tools/parity_ledger.py norm()), or is the Julia side
of a tracked namespace receipt. ZiPoisson and zi_poisson are the same name, as are Lognormal and lognormal.

Usage:
  python3 tools/true_parity_assemble.py            # write the outputs
  python3 tools/true_parity_assemble.py --check    # regenerate in memory, fail if stale/conflicting
  python3 tools/true_parity_assemble.py --extra-map PATH --out-dir DIR   # also fold PATH (e.g.
        #533's case-map.json fetched with git show) and write to DIR; add --check for a conflict
        scan that writes nothing. --extra-map without --out-dir is a usage error.
  GLLVMTMB_DIR=/path/to/gllvmTMB python3 tools/true_parity_assemble.py \\
        --refresh-reverse-gap-inputs --julia-names-tsv names.tsv
        where names.tsv is `name<TAB>kind` for names(GLLVModels) at HEAD, e.g.
        julia --project=. -e 'using GLLVModels; for n in sort(string.(names(GLLVModels)));
          v = getfield(GLLVModels, Symbol(n)); println(n, "\\t", v isa Function ? "Function" :
          v isa Type ? "Type" : "Other"); end'

Exit codes: 0 ok; 1 --check found stale outputs, a conflict or a missing input; 2 usage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter, OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = Path("docs/dev-log/core070/true-parity-latest")
P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
NAMESPACE_SHA256_P1 = "c1e91cd75bf29966932bcaec4a2b15234b23c7f60e1e39b3ce0917c6537f0f0c"

# The per-family maps this assembly expects. A missing one fails the run (never a silent
# partial ledger). Adding a family is a one-line change here.
EXPECTED_MAPS = [
    "case-map-aghq.json",
    "case-map-covariance.json",
    "case-map-data.json",
    "case-map-family.json",
    "case-map-fit-input.json",
    "case-map-inference.json",
    "case-map-isdm.json",
    "case-map-namespace.json",
    "case-map-postfit.json",
]

OUT_SCOREBOARD = "scoreboard.md"
OUT_CASEMAP = "case-map-assembled.json"
OUT_REVERSE_GAP = "reverse-gap.json"
IN_REVERSE_GAP = "reverse-gap-inputs.json"
IN_REVERSE_GAP_DECISIONS = "reverse-gap-decisions.json"
IN_BEHAVIOUR_EQUIVALENCE = "behaviour-equivalence.json"
BEHAVIOUR_KINDS = ("route", "refusal", "error_class", "printed_fields")
# The checker's C6_DECISION_VOCAB (keep in step).
C6_DECISION_VOCAB = ("KEPT_AS_JULIA_EXTRA", "PORT_TO_MATCH_R", "DEPRECATE_AND_REMOVE",
                     "RENAME_TO_AVOID_COLLISION", "EXCLUDED_INTERNAL_HELPER")
# The checker's C6_RULINGS (keep in step): the only signature accepted on a reverse-gap decision is
# itchyshin/GLLVModels.jl#684 item 3, signed 2026-10-02, and it covers exactly these two words.
C6_RULINGS = {
    "itchyshin/GLLVModels.jl#684 item 3": {
        "signed_on": "2026-10-02",
        "words": ("KEPT_AS_JULIA_EXTRA", "EXCLUDED_INTERNAL_HELPER"),
    },
}
# Scope of the behavioural tier (the checker's BEHAVIOURAL_INFERENCE_SOURCE_IDS and
# BEHAVIOURAL_NAMED_SOURCE_IDS; keep in step, a test fails on drift): exactly the 59 inference rows whose
# evidence_tier on origin/main (5b186bf32, case-map-inference.json) is routing_control_flow (45) or
# reject_error_class (14), plus four named C1 rows (itchyshin/GLLVModels.jl#684 item 2). A prefix rule
# would also admit CI-ROUTE-008..011 (two numeric, two partial) and any new inference/... row.
BEHAVIOURAL_INFERENCE_SOURCE_IDS = (
    "inference/CI-ROUTE-001", "inference/CI-ROUTE-002", "inference/CI-ROUTE-003",
    "inference/CI-ROUTE-004", "inference/CI-ROUTE-006", "inference/CI-ROUTE-007",
    "inference/CI-ROUTE-012", "inference/CI-ROUTE-013", "inference/CI-ROUTE-014",
    "inference/CI-ROUTE-015", "inference/CI-ROUTE-016", "inference/CI-ROUTE-017",
    "inference/CI-ROUTE-018", "inference/CI-ROUTE-019", "inference/CI-ROUTE-020",
    "inference/CI-ROUTE-021", "inference/CI-ROUTE-022", "inference/CI-ROUTE-023",
    "inference/CI-ROUTE-024", "inference/CI-ROUTE-025", "inference/CI-ROUTE-026",
    "inference/CI-ROUTE-027", "inference/CI-ROUTE-028", "inference/CI-ROUTE-029",
    "inference/CI-ROUTE-030", "inference/CI-ROUTE-032", "inference/CI-ROUTE-033",
    "inference/CI-ROUTE-034", "inference/CI-ROUTE-035", "inference/CI-ROUTE-036",
    "inference/CI-ROUTE-037", "inference/CI-ROUTE-038", "inference/CI-ROUTE-039",
    "inference/CI-ROUTE-040", "inference/CI-ROUTE-041", "inference/CI-ROUTE-042",
    "inference/CI-ROUTE-043", "inference/CI-ROUTE-044", "inference/CI-ROUTE-045",
    "inference/CI-ROUTE-046", "inference/CI-ROUTE-047", "inference/CI-ROUTE-048",
    "inference/CI-ROUTE-055", "inference/CI-ROUTE-056", "inference/CI-ROUTE-057",
    "inference/CI-ROUTE-058", "inference/CI-ROUTE-059", "inference/CI-ROUTE-060",
    "inference/CI-ROUTE-061", "inference/CI-ROUTE-062", "inference/CI-ROUTE-063",
    "inference/CI-ROUTE-065", "inference/CI-ROUTE-066", "inference/CI-ROUTE-067",
    "inference/CI-ROUTE-068", "inference/CI-ROUTE-069", "inference/CI-ROUTE-070",
    "inference/CI-ROUTE-081", "inference/CI-ROUTE-084",
)
BEHAVIOURAL_NAMED_SOURCE_IDS = (
    "latent-scores/extract_latent_scores.default",
    "select-lv/print.gllvmTMB_select_lv",
    "model-comparison/print.anova.gllvmTMB_multi",
    "model-comparison/update.gllvmTMB_multi",
)
# Extension signed 2026-10-05 (maintainer ruling 2026-10-05 (D-319), GATES.md "Rulings of 2026-10-05"): item A (7 aghq
# control rows, inference/CI-ROUTE-009), N6 (the 5 iSDM rows reachable through R's public door) and N10
# (check_auto_residual). The checker's BEHAVIOURAL_EXTENDED_SOURCE_IDS (keep in step, a test fails on drift).
BEHAVIOURAL_EXTENDED_SOURCE_IDS = (
    "aghq/AGHQ-CTRL-AUTO", "aghq/AGHQ-CTRL-FALSE", "aghq/AGHQ-CTRL-NINE", "aghq/AGHQ-CTRL-NULL",
    "aghq/AGHQ-CTRL-ONE", "aghq/AGHQ-CTRL-TRUE", "aghq/AGHQ-CTRL-TWO",
    "inference/CI-ROUTE-009",
    "isdm/ISDM-COUNT", "isdm/ISDM-EXTRA-SOURCE", "isdm/ISDM-MISSING-IN-TRAIT", "isdm/ISDM-MISSING-SOURCE",
    "isdm/ISDM-WRAPPER-LAW",
    "postfit/POSTFIT-SURFACE-check_auto_residual",
)

# evidence_tier (verbatim from the maps) -> scoreboard status bucket. Collation only: each
# tier string was written by the PR that measured the row; this table only groups them.
TIER_BUCKET = {
    "registration": "REGISTRATION-ONLY",
    "numeric_held_batch_verifier_failed": "HELD",
    "partial_numeric_bridge_boundary": "PARTIAL",
    "partial_case_not_executed": "PARTIAL",
    "partial_non_numeric_case": "PARTIAL",
    "needs_surface_r_side_measured": "NEEDS-SURFACE",
    "needs_surface_not_executed": "NEEDS-SURFACE",
    "not_measured": "NOT-MEASURED",
    "r_only": "NON-NUMERIC",
    "r_only_policy_pass": "NON-NUMERIC",
    "routing_control_flow": "NON-NUMERIC",
    "reject_error_class": "NON-NUMERIC",
    "paired_control_categorical_pass": "NON-NUMERIC",
    "numeric_non_discriminating": "NON-DISCRIMINATING",
    "numeric_fail": "FAIL",
    None: "NO-TIER",
}
STATUS_ORDER = [
    "EVIDENCED", "EVIDENCED-BEHAVIOURAL", "DISPOSITION-SIGNED", "NUMERIC-UNVERIFIED",
    "BEHAVIOURAL-UNVERIFIED", "REGISTRATION-ONLY", "HELD",
    "PARTIAL", "NEEDS-SURFACE", "NON-NUMERIC", "NON-DISCRIMINATING", "FAIL", "NOT-MEASURED",
    "NO-TIER", "DISPOSITION-UNVERIFIED",
]
# Every word this tool itself writes into the Status column: the statuses derive_status emits (STATUS_ORDER
# lists them all) and the buckets TIER_BUCKET maps a tier to. A row's `disposition` is free text from the case
# map; it is copied into the Status column only when it is none of these (disposition_status), because the
# checker reads three of them (DONE in tools/true_parity_check.mjs) as done and X2 counts them.
RESERVED_STATUS_WORDS = frozenset(STATUS_ORDER) | frozenset(TIER_BUCKET.values())

# --- checker rules, ported from tools/true_parity_check.mjs (keep in step) -------------------

SIGNER_ALLOW = {"Shinichi Nakagawa", "itchyshin"}
SIGNER_DENY_RE = re.compile(r"agent|claude|codex|cursor|fable", re.I)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STATUS_FIELDS = ["status", "verdict", "batch_status", "harness_pass"]
# A behavioural receipt is judged on one more spelling than a numeric one (the checker's
# BEHAVIOURAL_STATUS_FIELDS); the numeric tier's list is unchanged.
BEHAVIOURAL_STATUS_FIELDS = STATUS_FIELDS + ["result"]
RECORDED_DIFF_REL_TOL = 1e-12


def is_pass_value(v):
    # The checker's isPassValue: v === 'PASS' || v === 'pass' || v === true. Strict on type:
    # in Python 1 == True and 1.0 == True, so a plain `in ("PASS", "pass", True)` would pass them.
    return v is True or (isinstance(v, str) and v in ("PASS", "pass"))


def is_visible(x):
    """The checker's isVisible: a string with at least one character in Unicode category L, N, P or S.
    A string of only whitespace or format characters (U+FEFF, U+200B, U+0085, ...) is empty. Neither
    str.strip() nor JS trim() is used: they disagree on those characters."""
    return isinstance(x, str) and any(unicodedata.category(ch)[0] in "LNPS" for ch in x)


def is_json_one(x):
    """The checker's `schema !== 1`: the JSON number 1, never true. JS cannot tell 1 from 1.0, so 1.0 is
    accepted here too; a bool (Python: True == 1) and a string are not."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == 1


class Fail(Exception):
    pass


# JS String.prototype.trim() strips exactly these characters; Python str.strip() strips a different set
# (it keeps U+FEFF and strips U+0085, U+001C..U+001F). The checker is the authority, so the signature fields
# are trimmed with the checker's set (a signed_on that ends in U+0085 is a bad date in both tools).
JS_TRIM_CHARS = "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def js_trim(s):
    return s.strip(JS_TRIM_CHARS)


def signature_problem(by, on):
    if not isinstance(by, str) or not js_trim(by) or not isinstance(on, str) or not js_trim(on):
        return "DISPOSITION-SIGNED-UNVERIFIED"
    who = js_trim(by)
    if SIGNER_DENY_RE.search(who) or who not in SIGNER_ALLOW:
        return "DISPOSITION-SIGNER-NOT-ALLOWED"
    s = js_trim(on)
    try:
        d = datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return "DISPOSITION-SIGNED-BAD-DATE"
    if d.isoformat() != s or d > (datetime.now(timezone.utc) + timedelta(hours=14)).date():
        return "DISPOSITION-SIGNED-BAD-DATE"
    return None


def status_exception_problem(row):
    # The checker's receiptStatusExceptionProblem.
    e = row.get("receipt_status_exception")
    if e is None:
        return "no receipt_status_exception"
    if not isinstance(e, dict) or not isinstance(e.get("reason"), str) or not js_trim(e["reason"]):
        return "receipt_status_exception without a reason"
    return signature_problem(e.get("signed_by"), e.get("signed_on"))


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def receipt_paths(row):
    return as_list((row.get("evidence") or {}).get("receipt"))


def nonbinding_paths(row):
    return as_list((row.get("evidence") or {}).get("non_binding_receipts"))


def carry_problem(row):
    ma = row.get("measured_against")
    if ma in ("P1", P1_SHA):
        return None
    if ma is None:
        return "PARTIAL_STALE_AT_P1(missing measured_against)"
    carry = row.get("carry")
    if carry is not None and not isinstance(carry, dict):
        # The checker reads this as stale; a malformed map field is a data error, so fail loudly.
        raise Fail(f"{row.get('source_id')}: carry is {type(carry).__name__}, not an object")
    pins = (carry or {}).get("source_pins")
    if not isinstance(pins, list) or not pins:
        return "PARTIAL_STALE_AT_P1(no carry.source_pins)"
    for sp in pins:
        if not isinstance(sp, dict):
            raise Fail(f"{row.get('source_id')}: carry.source_pins entry is {type(sp).__name__}, not an object")
        a, b = sp.get("sha256_at_p0"), sp.get("sha256_at_p1")
        if not (isinstance(a, str) and SHA256_RE.match(a) and isinstance(b, str) and SHA256_RE.match(b) and a == b):
            return "PARTIAL_STALE_AT_P1(hash mismatch or not 64-hex sha256)"
    return None


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def case_diff(c):
    computed = None
    r, j = c.get("r_value"), c.get("julia_value")
    if _fin(r) and _fin(j):
        computed = abs(r - j)
    elif isinstance(r, list) and isinstance(j, list) and r and len(r) == len(j) and all(map(_fin, r)) and all(map(_fin, j)):
        computed = max(abs(a - b) for a, b in zip(r, j))
    recorded = []
    for k in ("abs_diff", "max_abs_diff"):
        if k in c:
            if not _fin(c[k]) or c[k] < 0:
                return None, None
            recorded.append((k, c[k]))
    if computed is None:
        return (recorded[0][1] if recorded else None), None
    for k, v in recorded:
        if abs(v - computed) > RECORDED_DIFF_REL_TOL * max(abs(v), abs(computed)):
            return None, f"recorded {k} {v} != recomputed {computed}"
    return computed, None


MAX_SAFE_INTEGER = 2 ** 53 - 1


def behavioural_eligible_source_id(sid):
    return isinstance(sid, str) and (sid in BEHAVIOURAL_INFERENCE_SOURCE_IDS or sid in BEHAVIOURAL_NAMED_SOURCE_IDS
                                     or sid in BEHAVIOURAL_EXTENDED_SOURCE_IDS)


def _is_int(x):
    # The checker's Number.isSafeInteger: a finite number with no fractional part and magnitude at
    # most 2^53 - 1 (JSON 15 and 15.0 are the same number in JS; a larger integer is not exactly
    # representable there, so both tools refuse it), never a bool.
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return abs(x) <= MAX_SAFE_INTEGER
    return isinstance(x, float) and x == x and abs(x) != float("inf") and x.is_integer() and abs(x) <= MAX_SAFE_INTEGER


def integer_equality_problem(c):
    """The checker's integerEqualityProblem (ruling 1): None, or why the case is not an exact-integer
    comparison. A case without "kind" is judged as before; any other kind fails."""
    if "kind" not in c:
        return None
    if c["kind"] != "integer_equality":
        return f"unknown comparison kind {json.dumps(c['kind'])}"
    r, j = c.get("r_value"), c.get("julia_value")

    def ints(x):
        return _is_int(x) or (isinstance(x, list) and len(x) > 0 and all(map(_is_int, x)))

    if not ints(r) or not ints(j):
        return "integer_equality needs integer r_value and julia_value"
    if isinstance(r, list) != isinstance(j, list) or (isinstance(r, list) and len(r) != len(j)):
        return "integer_equality needs r_value and julia_value of the same shape and length"
    tol = c.get("tolerance")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol != 0.5:
        return "integer_equality needs tolerance exactly 0.5"
    return None


def behaviour_equivalence(root, ledger=None):
    """The checker's loadEquivalence: {kind: {side: {label: (canonical, class_index)}}}. A missing file
    is an empty table; a malformed or ambiguous one raises Fail."""
    p = root / (ledger or LEDGER) / IN_BEHAVIOUR_EQUIVALENCE
    index: dict = {}
    if not p.is_file():
        return index
    try:
        t = json.loads(p.read_text())
    except ValueError as e:
        raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE} is not valid JSON: {e}")
    if not isinstance(t, dict) or not is_json_one(t.get("schema")):
        raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: schema must be 1")
    if t.get("pin") not in ("P1", P1_SHA):
        raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: pin must be P1")
    if not isinstance(t.get("classes"), list):
        raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: classes must be an array")

    canonical_seen = set()
    for i, cls in enumerate(t["classes"]):
        if not isinstance(cls, dict) or cls.get("kind") not in BEHAVIOUR_KINDS:
            raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: class {i} has no valid kind")
        if not is_visible(cls.get("canonical")):
            raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: class {i} has no canonical label")
        if (cls["kind"], cls["canonical"]) in canonical_seen:
            raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: duplicate canonical {json.dumps(cls['canonical'])} for kind {cls['kind']} (add the labels to the existing class)")
        canonical_seen.add((cls["kind"], cls["canonical"]))
        if not is_visible(cls.get("basis")):
            raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: class {i} ({cls['canonical']}) has an empty basis")
        for side in ("r", "julia"):
            labels = cls.get(side)
            if not isinstance(labels, list) or not all(map(is_visible, labels)):
                raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: class {i} ({cls['canonical']}) {side} must be an array of non-empty labels")
            bucket = index.setdefault(cls["kind"], {}).setdefault(side, {})
            for label in labels:
                prior = bucket.get(label)
                if prior and prior[1] != i:
                    raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: ambiguous table, {cls['kind']} {side} label {json.dumps(label)} is in classes {prior[0]} and {cls['canonical']}")
                bucket[label] = (cls["canonical"], i)
    return index


def _label_class(index, kind, side, label):
    """(canonical, class_index) of the class a label is listed in, or None: an unlisted label stands only
    for itself."""
    return index.get(kind, {}).get(side, {}).get(label)


def _labels_match(index, kind, r, j):
    """The checker's labelsMatch: class identity, not canonical strings. Two raw labels match iff they
    are the same string, or both are listed in the same class."""
    if r == j:
        return True
    rc, jc = _label_class(index, kind, "r", r), _label_class(index, kind, "julia", j)
    return rc is not None and jc is not None and rc[1] == jc[1]


def _describe_class(c):
    return "no class" if c is None else f"class {json.dumps(c[0])}"


def behavioural_not_passed(j, p):
    """The checker's behaviouralNotPassed: why a behavioural receipt reads as failed, or None. A status
    field that is present must hold a pass value, at the top level, in the behaviour block and in a
    comparison block; a nested batch_verifier must be an object whose status, if present, passes."""
    for obj, pre in ((j, ""), (j.get("behaviour"), "behaviour."), (j.get("comparison"), "comparison.")):
        if not isinstance(obj, dict):
            continue
        for f in BEHAVIOURAL_STATUS_FIELDS:
            if f in obj and not is_pass_value(obj[f]):
                return f"{pre}{f}={json.dumps(obj[f])} in {p}"
    if "batch_verifier" in j:
        bv = j["batch_verifier"]
        if not isinstance(bv, dict):
            return f"batch_verifier={json.dumps(bv)} is not an object in {p}"
        if "status" in bv and not is_pass_value(bv["status"]):
            return f"batch_verifier.status={json.dumps(bv['status'])} in {p}"
    return None


def behaviour_case_not_passed(c, p):
    """The checker's behaviourCaseNotPassed: a case's status fields must pass and `match`, if present, is true."""
    for f in BEHAVIOURAL_STATUS_FIELDS:
        if f in c and not is_pass_value(c[f]):
            return f"behaviour.cases[{c['case_id']}].{f}={json.dumps(c[f])} in {p}"
    if "match" in c and c["match"] is not True:
        return f"behaviour.cases[{c['case_id']}].match={json.dumps(c['match'])} in {p}"
    return None


def behavioural_receipt_problem(row, root, index, cites=None):
    """None when the row binds behaviourally, else why not. A port of the checker's
    behaviouralReceiptStatus (ruling 2); the carry and dangling-receipt checks are done by the
    caller (derive_status), in the checker's C1 order. `cites` maps a case id to the set of source_ids
    of the rows in the case map that list it (an entry without source_id covers a case id only when one
    row cites it)."""
    paths = receipt_paths(row)
    if not paths:
        return "no receipt"
    if not behavioural_eligible_source_id(row.get("source_id")):
        return ("source_id not covered by itchyshin/GLLVModels.jl#684 item 2 (the 59 listed inference rows and four named C1 rows) "
                "or by its extension in maintainer ruling 2026-10-05 (D-319) (14 listed rows)")
    ids = as_list(row.get("executable_case_ids"))
    if not ids:
        return "no executable_case_ids"

    entries, blocks, not_passed = [], 0, None
    for p in paths:
        try:
            j = json.loads((root / p).read_text())
        except OSError:
            return f"unreadable {p}"
        except ValueError:
            continue
        if not isinstance(j, dict):
            continue
        if not_passed is None:
            not_passed = behavioural_not_passed(j, p)
        # A comparison block in a cited receipt must itself hold (the checker's checkComparisonBlock).
        if "comparison" in j:
            bad = comparison_block_problem(j["comparison"], p, set())
            if bad:
                return f"cited receipt's own comparison fails: {bad}"
        if "behaviour" not in j:
            continue
        b = j["behaviour"]
        if not isinstance(b, dict):
            return f"malformed behaviour block in {p}"
        if b.get("pin") not in ("P1", P1_SHA):
            return f"behaviour not pinned to P1 in {p}"
        cases = b.get("cases")
        if not isinstance(cases, list) or not cases:
            return f"behaviour has no cases in {p}"
        for c in cases:
            cid = c.get("case_id") if isinstance(c, dict) else None
            if not isinstance(cid, str) or not cid:
                return f"behaviour case without case_id in {p}"
            if "source_id" in c and (not isinstance(c["source_id"], str) or not c["source_id"]):
                return f"case {cid}: source_id must be a non-empty string"
            if c.get("kind") not in BEHAVIOUR_KINDS:
                return f"case {cid}: kind {json.dumps(c.get('kind'))} is not one of {'|'.join(BEHAVIOUR_KINDS)}"
            for k in ("r_observed", "julia_observed"):
                v = c.get(k)
                if not (is_visible(v) or (isinstance(v, list) and v and all(map(is_visible, v)))):
                    return f"case {cid}: {k} must be a non-empty string or an array of non-empty strings"
            if isinstance(c["r_observed"], list) != isinstance(c["julia_observed"], list):
                return f"case {cid}: r_observed and julia_observed must have the same shape"
            if isinstance(c["r_observed"], list) and len(c["r_observed"]) != len(c["julia_observed"]):
                return f"case {cid}: r_observed has {len(c['r_observed'])} labels, julia_observed {len(c['julia_observed'])}"
            if not_passed is None:
                not_passed = behaviour_case_not_passed(c, p)
            entries.append(c)
        blocks += 1
    if blocks == 0:
        return "no behaviour block in any receipt"
    uncovered = []
    sid = row.get("source_id")
    for cid in ids:
        own = [e for e in entries if e["case_id"] == cid and e.get("source_id") == sid and "source_id" in e]
        unscoped = [e for e in entries if e["case_id"] == cid and "source_id" not in e]
        cited_by = len((cites or {}).get(cid) or {sid})
        # An entry without source_id covers a case id only when this row is its only citer.
        app = own if cited_by > 1 else own + unscoped
        if not app:
            uncovered.append(f"{cid} (cited by {cited_by} rows, so an entry without source_id covers none of them; scope each entry with source_id)"
                             if cited_by > 1 and unscoped else cid)
            continue
        for e in app:
            rs, js = as_list(e["r_observed"]), as_list(e["julia_observed"])
            for a, b2 in zip(rs, js):
                if not _labels_match(index, e["kind"], a, b2):
                    return (f"case {cid} ({e['kind']}): R {json.dumps(a)} vs Julia {json.dumps(b2)} differ after canonicalisation "
                            f"(R: {_describe_class(_label_class(index, e['kind'], 'r', a))}; Julia: {_describe_class(_label_class(index, e['kind'], 'julia', b2))})")
    if uncovered:
        return "case ids without an applicable behaviour entry: " + ",".join(uncovered)
    return f"receipt did not pass: {not_passed}" if not_passed else None


def comparison_block_problem(cmp_, p, covered):
    """The checker's checkComparisonBlock: None when one `comparison` block holds, else why not. Adds
    each compared case id to `covered`."""
    if not isinstance(cmp_, dict):
        return f"malformed comparison in {p}"
    if cmp_.get("pin") not in ("P1", P1_SHA):
        return f"comparison not pinned to P1 in {p}"
    cases = cmp_.get("cases")
    if not isinstance(cases, list) or not cases:
        return f"comparison has no cases in {p}"
    for c in cases:
        cid = c.get("case_id") if isinstance(c, dict) else None
        if not isinstance(cid, str) or not cid:
            return f"comparison case without case_id in {p}"
        tol = c.get("tolerance")
        if not _fin(tol) or tol <= 0:
            return f"case {cid}: tolerance not a finite number > 0"
        int_problem = integer_equality_problem(c)
        if int_problem:
            return f"case {cid}: {int_problem}"
        d, mism = case_diff(c)
        if mism:
            return f"case {cid}: {mism} in {p}"
        if d is None:
            return f"case {cid}: no finite abs_diff or r_value/julia_value"
        if d > tol:
            return f"case {cid}: abs_diff {d} > tolerance {tol}" + (
                " (integer_equality: the integers differ)" if c.get("kind") == "integer_equality" else "")
        covered.add(cid)
    return None


# Maintainer ruling 2026-10-05 (D-319), ruling 1: the checker's BRIDGE_READBACK_ROW_PREFIX (keep in step, a test fails on
# drift). A live_bridge_readback receipt binds a numeric row only for these rows, and only on the row's own cases.
BRIDGE_READBACK_ROW_PREFIX = {
    "namespace/S3method/fitted,gllvmTMB_julia": "P1-BRIDGE-READBACK-FITTED-",
    "namespace/S3method/predict,gllvmTMB_julia": "P1-BRIDGE-READBACK-PREDICT-",
    "namespace/S3method/residuals,gllvmTMB_julia": "P1-BRIDGE-READBACK-RESIDUALS-",
    "namespace/export/gllvm_julia_fit": "P1-BRIDGE-READBACK-GJF-",
}


def bridge_readback_problem(row, p):
    """The checker's bridgeReadbackProblem: None, or why a live_bridge_readback receipt cannot bind this row."""
    sid = row.get("source_id")
    if sid not in BRIDGE_READBACK_ROW_PREFIX:
        return (f"bridge readback {p} binds only fitted, predict and residuals for gllvmTMB_julia (and gllvm_julia_fit); "
                "the other methods copy Julia's value and close by signed disposition (maintainer ruling 2026-10-05 (D-319), ruling 1)")
    prefix = BRIDGE_READBACK_ROW_PREFIX[sid]
    off = [i for i in as_list(row.get("executable_case_ids")) if not (isinstance(i, str) and i.startswith(prefix))]
    if off:
        return (f"bridge readback binds {sid} only on its own cases ({prefix}*), not {','.join(map(str, off))} "
                "(maintainer ruling 2026-10-05 (D-319), ruling 1)")
    return None


# Maintainer ruling 2026-10-05 (D-319), N1: the checker's BOUNDARY_CONTEXT_KINDS and boundaryContext (keep in step).
# evidence_kind -> (verdict, required case-id suffix or None).
BOUNDARY_CONTEXT_KINDS = {
    "r_public_bridge_boundary": ("R_BOUNDARY_UNCHANGED", "-PUBLIC-R-BRIDGE"),
    "not_executed": ("NOT_EXECUTED", "-PUBLIC-R-BRIDGE"),
    "r_only_formula_grammar": ("R_ONLY_PASS", None),
}


def boundary_context(row, root):
    """(ids, problem): the validated boundary-context case ids of a numeric row (N1), or why they are refused."""
    ctx = row.get("boundary_context_case_ids")
    if ctx is None:
        return set(), None
    exe = as_list(row.get("executable_case_ids"))
    if not isinstance(ctx, list) or not ctx or not all(isinstance(x, str) and x for x in ctx):
        return set(), "boundary_context_case_ids must be a non-empty array of case ids"
    if len(set(ctx)) != len(ctx):
        return set(), "boundary_context_case_ids lists a case id twice"
    if len(set(exe)) != len(exe):
        return set(), "executable_case_ids lists a case id twice"
    not_exec = [i for i in ctx if i not in exe]
    if not_exec:
        return set(), "boundary context case ids not in executable_case_ids: " + ",".join(not_exec)
    if len(ctx) >= len(exe):
        return set(), "every executable case is boundary context; at least one case must bind numerically"
    rps = (row.get("evidence") or {}).get("boundary_context_receipts") or []
    if not isinstance(rps, list) or not rps:
        return set(), "no evidence.boundary_context_receipts"
    by_case = {}
    for p in rps:
        if not isinstance(p, str) or not (root / p).is_file():
            return set(), f"boundary context receipt {p} does not resolve to a file"
        try:
            j = json.loads((root / p).read_text())
        except ValueError:
            return set(), f"boundary context receipt {p} is not JSON"
        if not isinstance(j, dict) or not isinstance(j.get("case_id"), str):
            return set(), f"boundary context receipt {p} has no case_id"
        by_case[j["case_id"]] = (p, j)
    for cid in ctx:
        if cid not in by_case:
            return set(), f"boundary context case {cid} has no receipt under evidence.boundary_context_receipts"
        p, j = by_case[cid]
        kind = j.get("evidence_kind")
        if not isinstance(kind, str) or kind not in BOUNDARY_CONTEXT_KINDS:
            return set(), f"boundary context case {cid}: evidence_kind {json.dumps(kind)} is not an admission-only or PUBLIC-R-BRIDGE boundary kind ({p})"
        verdict, suffix = BOUNDARY_CONTEXT_KINDS[kind]
        if j.get("verdict") != verdict:
            return set(), f"boundary context case {cid}: verdict {json.dumps(j.get('verdict'))} is not {verdict} for {kind} ({p})"
        if suffix and not cid.endswith(suffix):
            return set(), f"boundary context case {cid}: a {kind} case must be a {suffix} case"
        if "comparison" in j:
            return set(), f"boundary context case {cid}: its receipt carries a comparison block, so it is compared, not context ({p})"
    return set(ctx), None


# Maintainer ruling 2026-10-05 (D-319), N9: the checker's convergenceParityProblem (keep in step).
CONVERGENCE_GRADIENT_BOUND = 1e-5
CONVERGENCE_POINTS = ("returned", "newton_polished")


def convergence_parity_problem(cp, p):
    """None when a receipt's convergence_parity block holds (both engines at gradient max-abs <= 1e-5), else why not."""
    def why(m):
        return f"convergence parity (maintainer ruling 2026-10-05 (D-319), N9): {m} in {p}"
    if not isinstance(cp, dict):
        return why("convergence_parity is not an object")
    gb = cp.get("gradient_bound")
    if isinstance(gb, bool) or not isinstance(gb, (int, float)) or gb != CONVERGENCE_GRADIENT_BOUND:
        return why(f"gradient_bound {json.dumps(gb)} is not 1e-5")
    if cp.get("compared_point") not in CONVERGENCE_POINTS:
        return why(f"compared_point {json.dumps(cp.get('compared_point'))} is not returned or newton_polished")
    eng = cp.get("engines")
    if not isinstance(eng, dict):
        return why("no engines block")
    for side in ("R", "julia"):
        e = eng.get(side)
        g = e.get("max_abs_gradient") if isinstance(e, dict) else None
        if not _fin(g) or g < 0:
            return why(f"{side} max_abs_gradient {json.dumps(g)} is not a finite number >= 0")
        if g > CONVERGENCE_GRADIENT_BOUND:
            return why(f"{side} max_abs_gradient {g} > 1e-5")
    return None


def numeric_receipt_problem(row, root, waive_status=False):
    """None when the row binds numerically, else why not. waive_status=True skips only the
    receipt status fields (what a valid receipt_status_exception waives in the checker)."""
    paths = receipt_paths(row)
    if not paths:
        return "no receipt"
    covered, blocks, not_passed = set(), 0, None
    for p in paths:
        try:
            j = json.loads((root / p).read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(j, dict):
            continue
        if j.get("evidence_kind") == "live_bridge_readback":
            bp = bridge_readback_problem(row, p)
            if bp:
                return bp
        if "convergence_parity" in j:
            cp = convergence_parity_problem(j["convergence_parity"], p)
            if cp:
                return cp
        if not_passed is None:
            for obj, pre in ((j, ""), (j.get("comparison") if isinstance(j.get("comparison"), dict) else None, "comparison.")):
                if obj is None:
                    continue
                for f in STATUS_FIELDS:
                    if f in obj and not is_pass_value(obj[f]):
                        not_passed = f"{pre}{f}={json.dumps(obj[f])} in {p}"
                        break
                if not_passed:
                    break
        if "comparison" not in j:
            continue
        bad = comparison_block_problem(j["comparison"], p, covered)
        if bad:
            return bad
        blocks += 1
    if blocks == 0:
        return "no comparison block in any receipt"
    ctx, ctx_problem = boundary_context(row, root)
    if ctx_problem:
        return f"boundary context (maintainer ruling 2026-10-05 (D-319), N1): {ctx_problem}"
    exe = as_list(row.get("executable_case_ids"))
    missing = [i for i in exe if i not in covered and i not in ctx]
    if missing:
        return "case ids not compared: " + ",".join(missing)
    # N1: context never binds a row on its own (checker numericReceiptStatus, keep in step).
    if ctx and not any(i not in ctx and i in covered for i in exe):
        return ("boundary context (maintainer ruling 2026-10-05 (D-319), N1): no executable case outside "
                "boundary_context_case_ids is covered by a comparison")
    return None if waive_status else not_passed


# --- inputs ----------------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_maps(root: Path, ledger: Path, extra: list[Path]):
    """Returns (rows_with_family, inputs, conflicts, identical_duplicates)."""
    inputs, problems = [], []
    sources = []
    for name in EXPECTED_MAPS:
        p = root / ledger / name
        if not p.is_file():
            problems.append(f"missing input map {ledger / name}")
            continue
        sources.append((p, str(ledger / name), name[len("case-map-"):-len(".json")]))
    for p in extra:
        sources.append((p, str(p), "extra:" + p.stem))
    if problems:
        raise Fail("; ".join(problems))
    by_id: "OrderedDict[str, tuple[str, dict]]" = OrderedDict()
    conflicts, dups = [], []
    for p, label, family in sources:
        try:
            d = json.loads(p.read_text())
        except ValueError as e:
            raise Fail(f"{label} is not valid JSON: {e}")
        rows = d.get("rows")
        if not isinstance(rows, list):
            raise Fail(f"{label} has no rows array")
        inputs.append({"path": label, "sha256": sha256_file(p), "rows": len(rows)})
        for r in rows:
            sid = r.get("source_id")
            if not isinstance(sid, str) or not sid:
                raise Fail(f"{label}: row without source_id")
            if sid in by_id:
                other_family, other = by_id[sid]
                if other == r:
                    dups.append(f"{sid} ({other_family}, {family}; identical)")
                    continue
                fields = sorted(k for k in set(other) | set(r) if other.get(k) != r.get(k))
                conflicts.append(f"{sid}: {other_family} vs {family} differ in {','.join(fields)}")
                continue
            by_id[sid] = (family, r)
    return by_id, inputs, conflicts, dups


# --- derivation ------------------------------------------------------------------------------

# The checker's C3/C4/C5 id selectors (tools/true_parity_check.mjs isRSZ / isRD / isGRP), ported.
CLAUSE_ID = {
    "C3": re.compile(r"-RSZ$", re.I),
    "C4": re.compile(r"^(?:[a-z0-9_]+-)*RD-", re.I),
    "C5": re.compile(r"^(?:[a-z0-9_]+-)*GRP-", re.I),
}


def scoreboard_id(sid: str, clause=None) -> str:
    """A row may carry `"clause": "C3" | "C4" | "C5"` to say it is deliberately a C3/C4/C5 row (the true-parity
    campaign rows, itchyshin/GLLVModels.jl#684 item 4). Then its scoreboard id must be selected by exactly that
    clause's checker rule, or the run fails. Without the marker, an id that would silently move the row into
    C3/C4/C5 still fails, as before."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", sid).strip("-")
    if not re.match(r"^[A-Za-z0-9]", s):
        raise Fail(f"cannot form a scoreboard id from {sid!r}")
    if clause is not None:
        if clause not in CLAUSE_ID:
            raise Fail(f"{sid}: clause {clause!r} is not one of C3, C4, C5")
        picked = [c for c, rx in CLAUSE_ID.items() if rx.search(s)]
        if picked != [clause]:
            raise Fail(f"{sid}: declares clause {clause} but scoreboard id {s} is selected by {picked or 'no clause'} in the checker")
        return s
    if re.search(r"-RSZ$", s, re.I) or re.match(r"^(RD|GRP)-", s, re.I):
        # Would silently move the row into C3/C4/C5 (GATES.md id conventions).
        raise Fail(f"scoreboard id {s} collides with a C3/C4/C5 id convention")
    return s


def disposition_status(disp):
    """(status, reason) for a row whose disposition is not null and not DISPOSITION-SIGNED.

    The disposition is free text from the case map and becomes the Status column, so it must never read as
    a status this tool or the checker trusts. After trimming (JS trim() and Python strip(), the union of the
    two sets) and upper-casing, a disposition equal to any RESERVED_STATUS_WORDS entry is DISPOSITION-UNVERIFIED:
    `EVIDENCED`, `EVIDENCED-BEHAVIOURAL`, `DISPOSITION-SIGNED` without a signature, and the rest. So is a
    disposition that is not a string, is blank, or carries a table delimiter or line break (a `|` would shift
    the columns the checker splits the row into). Text containing NEEDS_JULIA_SURFACE reads NEEDS-SURFACE; any
    other plain string is copied as it is, and the checker does not count it done."""
    if not isinstance(disp, str):
        return "DISPOSITION-UNVERIFIED", f"disposition is not a string ({type(disp).__name__})"
    key = js_trim(cell(disp)).upper()
    if not key:
        return "DISPOSITION-UNVERIFIED", "disposition is blank"
    if key in RESERVED_STATUS_WORDS:
        return "DISPOSITION-UNVERIFIED", f"disposition {cell(disp)} is a reserved status word, not a signature"
    if "|" in disp or "\n" in disp or "\r" in disp:
        return "DISPOSITION-UNVERIFIED", "disposition contains a table delimiter or line break"
    return ("NEEDS-SURFACE" if "NEEDS_JULIA_SURFACE" in disp else cell(disp)), f"disposition {disp}"


def is_c4_row(row):
    """True when the row's scoreboard id is selected by the checker's C4 rule (isRD)."""
    sid = row.get("source_id")
    if not isinstance(sid, str):
        return False
    return bool(CLAUSE_ID["C4"].search(re.sub(r"[^A-Za-z0-9_-]+", "-", sid).strip("-")))


def direct_engine_receipt(p: Path) -> bool:
    """The checker's directEngineReceipt (keep in step). Maintainer ruling 2026-10-05 (D-319), C4: an EVIDENCED
    real-data row must cite a run of both engines on the data, a JSON receipt whose `engines` block holds a
    non-empty `R` and a non-empty `julia` object. Without this an EVIDENCED C4 row could reach the scoreboard while
    the checker counts it not done."""
    try:
        j = json.loads(p.read_text())
    except (OSError, ValueError):
        return False
    eng = j.get("engines") if isinstance(j, dict) else None
    return (isinstance(eng, dict) and isinstance(eng.get("R"), dict) and len(eng["R"]) > 0
            and isinstance(eng.get("julia"), dict) and len(eng["julia"]) > 0)


def derive_status(row, root, equiv=None, cites=None):
    """Returns (status, reason). Never writes to row.

    Order follows the checker's C1 (not C8): for a row that cites a receipt, a dangling receipt
    or a stale carry is reported before the disposition is read, so a valid signature cannot
    make such a row read DISPOSITION-SIGNED. (C8's dispositionSignedProperly short-circuits
    before its dangling check; the two clauses disagree there, see the PR body of #589.)
    """
    disp = row.get("disposition")
    tier = row.get("evidence_tier")
    paths = receipt_paths(row)
    pre = None
    if paths:
        dang = [p for p in paths if not (root / p).is_file()]
        pre = ("dangling " + ",".join(dang)) if dang else carry_problem(row)
    if disp == "DISPOSITION-SIGNED":
        if pre:
            return "DISPOSITION-UNVERIFIED", pre
        why = signature_problem(row.get("signed_by"), row.get("signed_on"))
        return ("DISPOSITION-SIGNED", "") if why is None else ("DISPOSITION-UNVERIFIED", why)
    if disp is None and tier == "numeric":
        if pre:
            return "NUMERIC-UNVERIFIED", pre
        if not as_list(row.get("executable_case_ids")):
            return "NUMERIC-UNVERIFIED", "no executable_case_ids"
        prob = numeric_receipt_problem(row, root)
        if prob is None and is_c4_row(row) and not any(direct_engine_receipt(root / p) for p in paths):
            prob = "C4_NOT_A_DIRECT_ENGINE_RUN"
        if prob is None:
            return "EVIDENCED", ""
        # A maintainer-signed receipt_status_exception waives a failed status field only (the
        # comparison must still hold). The checker counts such a row in bound_signed=, never in
        # bound= / bound_numeric=, so it reads DISPOSITION-SIGNED here, never EVIDENCED.
        if numeric_receipt_problem(row, root, waive_status=True) is None:
            why = status_exception_problem(row)
            if why is None:
                return "DISPOSITION-SIGNED", f"receipt_status_exception: {row['receipt_status_exception']['reason']}; waives {prob}"
            return "NUMERIC-UNVERIFIED", f"{prob}; {why}"
        return "NUMERIC-UNVERIFIED", prob
    if disp is None and tier == "behavioural":
        if pre:
            return "BEHAVIOURAL-UNVERIFIED", pre
        prob = behavioural_receipt_problem(row, root, equiv if equiv is not None else behaviour_equivalence(root), cites)
        return ("EVIDENCED-BEHAVIOURAL", "") if prob is None else ("BEHAVIOURAL-UNVERIFIED", prob)
    if disp is not None:
        return disposition_status(disp)
    if not paths and not nonbinding_paths(row):
        return "NOT-MEASURED", "no receipt cited"
    if tier not in TIER_BUCKET:
        raise Fail(f"{row['source_id']}: evidence_tier {tier!r} has no bucket in TIER_BUCKET")
    return TIER_BUCKET[tier], ""


def cell(s) -> str:
    return str(s).replace("|", "/").replace("\n", " ").strip()


def build(root: Path, ledger: Path, extra: list[Path]):
    by_id, inputs, conflicts, dups = load_maps(root, ledger, extra)
    equiv = behaviour_equivalence(root, ledger)
    if conflicts:
        raise Fail("conflicting rows across maps (not resolved here): " + "; ".join(conflicts))
    rows_out, table, ids_seen = [], [], {}
    counts: "dict[str, Counter]" = {}
    # case id -> source_ids of the rows (any tier) that list it, as the checker's caseCitations
    cites: "dict[str, set]" = {}
    for sid, (_family, r) in by_id.items():
        for cid in as_list(r.get("executable_case_ids")):
            if isinstance(cid, str):
                cites.setdefault(cid, set()).add(sid)
    for sid, (family, r) in by_id.items():
        bid = scoreboard_id(sid, r.get("clause"))
        if bid in ids_seen:
            raise Fail(f"scoreboard id {bid} formed from both {ids_seen[bid]} and {sid}")
        ids_seen[bid] = sid
        status, reason = derive_status(r, root, equiv, cites)
        counts.setdefault(family, Counter())[status] += 1
        cases = as_list(r.get("executable_case_ids"))
        requires = f"{r.get('classification')}; cases: {', '.join(cases) if cases else 'none'}"
        if status in ("EVIDENCED", "EVIDENCED-BEHAVIOURAL"):
            receipt = ", ".join(receipt_paths(r))
        elif status == "DISPOSITION-SIGNED" and r.get("disposition") != "DISPOSITION-SIGNED":
            e = r["receipt_status_exception"]
            receipt = f"{', '.join(receipt_paths(r))}; receipt_status_exception signed_by: {e.get('signed_by')}; signed_on: {e.get('signed_on')}"
        elif status == "DISPOSITION-SIGNED":
            receipt = f"Disposition: {r.get('disposition')}; signed_by: {r.get('signed_by')}; signed_on: {r.get('signed_on')}"
        else:
            cited = receipt_paths(r) + [f"non-binding {p}" for p in nonbinding_paths(r)]
            receipt = "not bound; cited: " + (", ".join(cited) if cited else "none")
        notes = f"family {family}; evidence_tier {r.get('evidence_tier')}; measured_against {r.get('measured_against')}"
        if reason:
            notes += f"; {reason}"
        table.append(f"| {bid} `{cell(sid)}` | {cell(requires)} | {status} | {cell(receipt)} | {cell(notes)} |")
        rows_out.append(r)
    return rows_out, table, counts, inputs, dups


def render_scoreboard(table, counts, inputs, dups, fixtures) -> str:
    total = Counter()
    for c in counts.values():
        total.update(c)
    present = [s for s in STATUS_ORDER if total[s]] + sorted(s for s in total if s not in STATUS_ORDER)
    out = [
        "# P1 true-parity scoreboard (generated)",
        "",
        "GENERATED by `tools/true_parity_assemble.py` from the per-family case maps in this",
        "directory; do not edit by hand (`--check` fails on any drift). It collates, it does not",
        "classify: every classification, disposition, evidence tier and case id is copied from the",
        "maps, and nothing here is a signature. A row reads `EVIDENCED` only when it binds under the",
        "checker's own C1 numeric rule; `EVIDENCED-BEHAVIOURAL` only when it binds under the checker's",
        "behavioural rule (itchyshin/GLLVModels.jl#684 item 2: a refusal, route, error class or printed",
        "summary, for the listed inference rows, four named C1 rows and the 14 rows of maintainer ruling",
        "2026-10-05 (D-319) only; it is not numeric evidence);",
        "and `DISPOSITION-SIGNED` only when the map row carries a valid maintainer signature. PR #533's",
        "`case-map.json` rows are not in this table (not tracked here).",
        "",
        f"Pin: gllvmTMB P1 `{P1_SHA}`.",
        "",
        "Inputs:",
        "",
    ]
    out += [f"- `{i['path']}` ({i['rows']} rows, sha256 `{i['sha256']}`)" for i in inputs]
    if dups:
        out += ["", "Identical duplicate rows collapsed: " + "; ".join(dups)]
    out += ["", "## Totals by family", "",
            "| `family` | " + " | ".join(present) + " | total |",
            "|---|" + "---|" * (len(present) + 1)]
    for fam in sorted(counts):
        c = counts[fam]
        out.append(f"| `{fam}` | " + " | ".join(str(c[s]) for s in present) + f" | {sum(c.values())} |")
    out.append("| `all` | " + " | ".join(str(total[s]) for s in present) + f" | {sum(total.values())} |")
    out += ["", "## P1 twin fixtures present at this head, not bound to any row", "",
            "Listed for the reader only. Binding a twin receipt to a case-map row is a mapping",
            "decision for the PR that measures it, not something this tool does.", ""]
    out += [f"- `{f}`" for f in fixtures] or ["- none"]
    out += ["", "## Rows", "",
            "| Row id | Requires (short) | Status | Receipt / disposition | Notes |",
            "|---|---|---|---|---|"]
    out += table
    return "\n".join(out) + "\n"


def checker_parse(md: str):
    """The checker's scoreboardRows() filter, ported, to assert the table parses as intended."""
    ids = []
    for line in md.split("\n"):
        if not re.match(r"^\|\s*[A-Za-z0-9][A-Za-z0-9_-]*\b", line) or re.match(r"^\|\s*-+\s*\|", line):
            continue
        if re.match(r"^\|\s*(Row id|Capability id)\b", line, re.I):
            continue
        c = [s.strip() for s in line.split("|")]
        if len(c) < 6:
            continue
        m = re.match(r"^[A-Za-z0-9][A-Za-z0-9_-]*", c[1])
        ids.append(m.group(0) if m else c[1])
    return ids


# --- reverse gap -----------------------------------------------------------------------------

def r_names_from_namespace(text: str):
    names = set()
    for line in text.splitlines():
        m = re.search(r"\bexport\(([^)]+)\)", line)
        if m:
            names.update(x.strip().strip('"`') for x in m.group(1).split(","))
        m = re.search(r"\bS3method\(([^,]+),", line)
        if m:
            names.add(m.group(1).split("::")[-1].strip().strip('"`'))
    return sorted(names)


def refresh_reverse_gap_inputs(root: Path, ledger: Path, tsv: Path):
    gdir = os.environ.get("GLLVMTMB_DIR")
    if not gdir:
        raise Fail("set GLLVMTMB_DIR to a local gllvmTMB clone (read-only; git show only)")
    ns = subprocess.run(["git", "-C", gdir, "show", f"{P1_SHA}:NAMESPACE"], check=True, capture_output=True).stdout
    h = hashlib.sha256(ns).hexdigest()
    if h != NAMESPACE_SHA256_P1:
        raise Fail(f"P1 NAMESPACE sha256 {h} != the namespace contract's {NAMESPACE_SHA256_P1}")
    julia = []
    for line in tsv.read_text().splitlines():
        if line.strip():
            n, k = line.split("\t")
            if n != "GLLVModels":
                julia.append({"name": n, "kind": k})
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    doc = OrderedDict([
        ("schema", "true-parity-reverse-gap-inputs/v1"),
        ("note", "Inputs for reverse-gap.json. julia_exports is names(GLLVModels) at glvmodels_commit "
                 "(module name dropped); r_names are the export() names and S3 generic names in gllvmTMB's "
                 "NAMESPACE at P1, read with git show and checked against the namespace contract's sha256."),
        ("glvmodels_commit", head),
        ("reference_commit", P1_SHA),
        ("r_namespace_sha256", h),
        ("r_names", r_names_from_namespace(ns.decode())),
        ("julia_exports", sorted(julia, key=lambda x: x["name"])),
    ])
    (root / ledger / IN_REVERSE_GAP).write_text(json.dumps(doc, indent=1) + "\n")


def namespace_receipt_counterparts(root: Path, ledger: Path):
    """Julia symbol -> [R export] from the tracked namespace case receipts (#561)."""
    out: "dict[str, set]" = {}
    d = root / ledger / "receipts/namespace/cases"
    if not d.is_dir():
        raise Fail(f"missing {ledger}/receipts/namespace/cases")
    for p in sorted(d.glob("*.json")):
        j = json.loads(p.read_text())
        sym = (j.get("julia_check") or {}).get("symbol")
        rexp = (j.get("source_id") or "").split("/", 2)[-1]
        if sym:
            out.setdefault(sym, set()).add(rexp)
    return out


# A documented Julia extra must be documented: the basis names at least one page under docs/src, and every page it
# names is an existing .md file written exactly as docs/src/<path>.md. The basis is read as tokens, as the checker
# reads it: it is split at ASCII whitespace and at ( ) [ ] { } < > " ' ` , ; : ! ? # (so "docs/src/a.md#sec" and
# "(docs/src/a.md)" name docs/src/a.md), trailing dots are dropped, and every token that contains "docs/src" must
# then BE a page path: it starts with docs/src/, has no "..", "." or dot-leading segment and no empty one, ends in
# .md, and has nothing after it. So "page.md.bak", "page.md~", "./docs/src/page.md", a URL, a directory and an
# existing .json, .txt, .jl or .toml file are not citations, and a malformed citation beside a good one fails too.
# The checker's DOCS_SRC_SPLIT_RE and DOCS_SRC_PAGE_RE, the same text (the delimiter set is explicit ASCII so both
# engines split the same way; \x22 \x27 \x60 are the quote, apostrophe and backtick).
DOCS_SRC_SPLIT_RE = re.compile(r"[ \t\n\r\f\v()\[\]{}<>\x22\x27\x60,;:!?#]+")
DOCS_SRC_PAGE_RE = re.compile(r"^docs/src/(?:[A-Za-z0-9_-][A-Za-z0-9._-]*/)*[A-Za-z0-9_-][A-Za-z0-9._-]*\.md$")


def docs_src_citations(basis):
    """(pages, malformed): the tokens of `basis` that contain "docs/src", split into exact page paths and the rest."""
    tokens = sorted({t.rstrip(".") for t in DOCS_SRC_SPLIT_RE.split(basis)} - {""})
    tokens = [t for t in tokens if "docs/src" in t]
    return ([t for t in tokens if DOCS_SRC_PAGE_RE.fullmatch(t)], [t for t in tokens if not DOCS_SRC_PAGE_RE.fullmatch(t)])


def kept_basis_problem(basis, root: Path):
    """The checker's keptBasisProblem, resolved against the tree under `root`."""
    pages, malformed = docs_src_citations(basis)
    not_exact = "not an exact docs/src/<path>.md page"
    if not pages:
        return "KEPT_AS_JULIA_EXTRA basis must cite a docs/src/... file" + (f"; {not_exact}: {','.join(malformed)}" if malformed else "")
    if malformed:
        return f"KEPT_AS_JULIA_EXTRA basis cites {','.join(malformed)}, which is {not_exact}"
    dangling = [q for q in pages if not (root / q).is_file()]
    if dangling:
        return f"KEPT_AS_JULIA_EXTRA basis cites {','.join(dangling)}, which does not exist in the tree"
    return None


def load_reverse_gap_decisions(root: Path, ledger: Path):
    """Returns (ruling, decisions) from the optional reverse-gap-decisions.json, or (None, {}).
    Copies only; the checker's C6 judges the signature and the basis."""
    p = root / ledger / IN_REVERSE_GAP_DECISIONS
    if not p.is_file():
        return None, {}
    try:
        d = json.loads(p.read_text())
    except ValueError as e:
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS} is not valid JSON: {e}")
    if not isinstance(d, dict) or not is_json_one(d.get("schema")):
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: schema must be 1")
    ruling, decisions = d.get("ruling"), d.get("decisions")
    if not isinstance(ruling, dict) or not all(k in ruling for k in ("ref", "signed_by", "signed_on")):
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: ruling must carry ref, signed_by and signed_on")
    # The tool refuses to copy a signature it does not recognise: the ruling must be a known one, signed
    # on its date, by an allowed signer, and every decision word must be one the ruling covers.
    if not is_visible(ruling["ref"]):
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: ruling without a ref")
    known = C6_RULINGS.get(ruling["ref"])
    if known is None:
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: ruling ref {json.dumps(ruling['ref'])} is not a recognised signed ruling ({'; '.join(C6_RULINGS)})")
    sig = signature_problem(ruling["signed_by"], ruling["signed_on"])
    if sig is not None:
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: ruling signature rejected ({sig})")
    if js_trim(ruling["signed_on"]) != known["signed_on"]:
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: ruling signed_on {json.dumps(ruling['signed_on'])} is not the date of {ruling['ref']} ({known['signed_on']})")
    if not isinstance(d.get("criterion"), str) or not d["criterion"].strip():
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: criterion must be a non-empty string")
    gen = d.get("generator")
    if not isinstance(gen, str) or not gen.strip():
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: generator must be a non-empty string")
    # The generator is the committed script that produced the file: it must be a file inside the tree, named
    # without a ".." segment anywhere (a "tools/../tools/gen.py" that resolves back into the tree is refused too).
    if ".." in re.split(r"[\\/]", gen):
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: generator {json.dumps(gen)} is not a file in the tree (a path with a '..' segment is refused)")
    gpath = (root / gen)
    try:
        inside = not Path(gen).is_absolute() and gpath.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        inside = False
    if not inside or not gpath.is_file():
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: generator {json.dumps(gen)} is not a file in the tree")
    if not isinstance(decisions, dict):
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: decisions must be an object")
    for name, v in decisions.items():
        if not isinstance(v, dict) or v.get("decision") not in C6_DECISION_VOCAB:
            raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: {name}: decision must be one of {'|'.join(C6_DECISION_VOCAB)}")
        if v["decision"] not in known["words"]:
            raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: {name}: decision {v['decision']} is not covered by {ruling['ref']} (covers {'|'.join(known['words'])})")
        if not is_visible(v.get("basis")):
            raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: {name}: basis must be a non-empty string")
        if v["decision"] == "KEPT_AS_JULIA_EXTRA":
            why = kept_basis_problem(v["basis"], root)
            if why:
                raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: {name}: {why}")
    return ruling, decisions


def norm_name(s: str) -> str:
    """tools/parity_ledger.py norm(): remove "_" and ".", lower-case. ZiPoisson and zi_poisson are one name."""
    return re.sub(r"[_.]", "", s).lower()


def build_reverse_gap(root: Path, ledger: Path):
    p = root / ledger / IN_REVERSE_GAP
    if not p.is_file():
        raise Fail(f"missing input {ledger / IN_REVERSE_GAP} (run --refresh-reverse-gap-inputs)")
    inp = json.loads(p.read_text())
    r_norm = {norm_name(n) for n in inp["r_names"]}
    named = namespace_receipt_counterparts(root, ledger)
    items = []
    for e in inp["julia_exports"]:
        n = e["name"]
        if norm_name(n) in r_norm or n in named:
            continue
        items.append(OrderedDict([
            ("source_id", f"julia-export/{n}"),
            ("name", n),
            ("kind", e["kind"]),
            ("status", "unsigned"),
            ("decision", None),
            ("derived_from", f"names(GLLVModels) at {inp['glvmodels_commit'][:12]}; no gllvmTMB export or S3 "
                             f"generic of the same name at P1 (compared with underscores and dots removed and "
                             f"lower-cased, as tools/parity_ledger.py norm()) and not named as the Julia side "
                             f"of any tracked namespace receipt"),
        ]))
    ruling, decisions = load_reverse_gap_decisions(root, ledger)
    stale = sorted(set(decisions) - {it["name"] for it in items})
    if stale:
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS} names {', '.join(stale)}, which is not a reverse-gap item (stale)")
    for it in items:
        dec = decisions.get(it["name"])
        if dec is None:
            continue
        it["status"] = "decided"
        it["decision"] = dec["decision"]
        it["basis"] = dec["basis"]
        it["ruling"] = OrderedDict((k, ruling[k]) for k in ("ref", "signed_by", "signed_on"))
    return items


def list_fixtures(root: Path):
    return sorted(str(p.relative_to(root)) for p in (root / "test/fixtures").rglob("*_p1*") if p.is_file())


# --- main ------------------------------------------------------------------------------------

def generate(root: Path, ledger: Path, extra: list[Path]):
    rows, table, counts, inputs, dups = build(root, ledger, extra)
    md = render_scoreboard(table, counts, inputs, dups, list_fixtures(root))
    parsed = checker_parse(md)
    if len(parsed) != len(rows) or len(set(parsed)) != len(parsed):
        raise Fail(f"scoreboard parses as {len(parsed)} rows ({len(set(parsed))} unique) for {len(rows)} case-map rows")
    cm = OrderedDict([
        ("schema", 1),
        ("reference_commit", P1_SHA),
        ("generator", "tools/true_parity_assemble.py"),
        ("note", "GENERATED fold of the per-family case maps listed in inputs; rows copied verbatim, "
                 "no field changed, nothing signed. Not case-map.json: PR #533 owns that file and the "
                 "maintainer's signature governs its rows."),
        ("inputs", inputs),
        ("rows", rows),
    ])
    return {
        OUT_SCOREBOARD: md,
        OUT_CASEMAP: json.dumps(cm, indent=1) + "\n",
        OUT_REVERSE_GAP: json.dumps(build_reverse_gap(root, ledger), indent=1) + "\n",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--extra-map", type=Path, action="append", default=[])
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--refresh-reverse-gap-inputs", action="store_true")
    ap.add_argument("--julia-names-tsv", type=Path)
    a = ap.parse_args(argv)
    root = a.root.resolve()
    try:
        if a.refresh_reverse_gap_inputs:
            if not a.julia_names_tsv:
                ap.error("--refresh-reverse-gap-inputs needs --julia-names-tsv")
            refresh_reverse_gap_inputs(root, LEDGER, a.julia_names_tsv)
        if a.extra_map and not a.out_dir:
            # Also with --check: the tracked outputs never contain extra rows, so comparing a fold
            # against them would always read stale.
            ap.error("--extra-map needs --out-dir (the tracked outputs use tracked maps only)")
        outs = generate(root, LEDGER, a.extra_map)
    except Fail as e:
        print(f"ASSEMBLE_FAIL {e}")
        return 1
    dest = a.out_dir if a.out_dir else root / LEDGER
    if a.check and not a.out_dir:
        stale = [n for n, t in outs.items() if not (dest / n).is_file() or (dest / n).read_text() != t]
        if stale:
            print("ASSEMBLE_STALE " + ", ".join(stale))
            return 1
        print(f"ASSEMBLE_OK {len(json.loads(outs[OUT_CASEMAP])['rows'])} rows current")
        return 0
    if a.check:
        print(f"ASSEMBLE_OK {len(json.loads(outs[OUT_CASEMAP])['rows'])} rows (extra maps folded, nothing written)")
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    for n, t in outs.items():
        (dest / n).write_text(t)
    print(f"ASSEMBLE_WROTE {len(json.loads(outs[OUT_CASEMAP])['rows'])} rows to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

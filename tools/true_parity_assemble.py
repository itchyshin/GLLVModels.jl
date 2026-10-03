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
                      through behaviour-equivalence.json, and no failed receipt status field,
                      and the row's source_id is one the ruling covers (inference/* or one of four
                      named C1 rows; BEHAVIOUR_NAMED_SOURCE_IDS) and any comparison block in a cited
                      receipt holds. Never emitted for a data, grouping or realistic-size row.
  BEHAVIOURAL-UNVERIFIED
                      evidence_tier "behavioural" but the rule above does not hold (reason given).
  DISPOSITION-UNVERIFIED / NEEDS-SURFACE / <disposition>
                      a non-null disposition that is not a valid signature.
  NOT-MEASURED        no receipt and no non-binding receipt cited at all.
  otherwise           the row's evidence_tier, collated into a bucket by TIER_BUCKET below
                      (the tier string itself is copied verbatim into the notes column).

An evidence_tier missing from TIER_BUCKET fails the run: a new tier needs a human to decide
which bucket it reads as, rather than this tool guessing.

Integer equality (itchyshin/GLLVModels.jl#684 item 1): a numeric comparison case with
"kind": "integer_equality" needs safe-integer r_value and julia_value (magnitude below 2^53) and
tolerance exactly 0.5.

Reverse-gap decisions (itchyshin/GLLVModels.jl#684 item 3): an optional reverse-gap-decisions.json
(schema in GATES.md) is read by build_reverse_gap. Each decision is copied, with its basis and the
ruling, onto the matching item, whose status becomes "decided". A decision naming no reverse-gap
item fails the run (stale). Nothing here is a signature: the file carries the ruling. The tool refuses
to copy a ruling it does not recognise (C6_RULINGS: only #684 item 3, dated 2026-10-02, covering
KEPT_AS_JULIA_EXTRA and EXCLUDED_INTERNAL_HELPER), a signer outside the allow-list, a decision word
the ruling does not cover, or an empty criterion or generator; the checker's C6 judges the same.

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
# Scope of the behavioural tier (the checker's behaviouralEligibleSourceId; keep in step): the
# inference routing and error-class rows and four named C1 rows (itchyshin/GLLVModels.jl#684 item 2).
BEHAVIOURAL_NAMED_SOURCE_IDS = (
    "latent-scores/extract_latent_scores.default",
    "select-lv/print.gllvmTMB_select_lv",
    "model-comparison/print.anova.gllvmTMB_multi",
    "model-comparison/update.gllvmTMB_multi",
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

# --- checker rules, ported from tools/true_parity_check.mjs (keep in step) -------------------

SIGNER_ALLOW = {"Shinichi Nakagawa", "itchyshin"}
SIGNER_DENY_RE = re.compile(r"agent|claude|codex|cursor|fable", re.I)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STATUS_FIELDS = ["status", "verdict", "batch_status", "harness_pass"]
RECORDED_DIFF_REL_TOL = 1e-12


def is_pass_value(v):
    # The checker's isPassValue: v === 'PASS' || v === 'pass' || v === true. Strict on type:
    # in Python 1 == True and 1.0 == True, so a plain `in ("PASS", "pass", True)` would pass them.
    return v is True or (isinstance(v, str) and v in ("PASS", "pass"))


class Fail(Exception):
    pass


def signature_problem(by, on):
    if not isinstance(by, str) or not by.strip() or not isinstance(on, str) or not on.strip():
        return "DISPOSITION-SIGNED-UNVERIFIED"
    who = by.strip()
    if SIGNER_DENY_RE.search(who) or who not in SIGNER_ALLOW:
        return "DISPOSITION-SIGNER-NOT-ALLOWED"
    s = on.strip()
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
    if not isinstance(e, dict) or not isinstance(e.get("reason"), str) or not e["reason"].strip():
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
    return isinstance(sid, str) and (sid.startswith("inference/") or sid in BEHAVIOURAL_NAMED_SOURCE_IDS)


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
    if not isinstance(t, dict) or t.get("schema") != 1:
        raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: schema must be 1")
    if t.get("pin") not in ("P1", P1_SHA):
        raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: pin must be P1")
    if not isinstance(t.get("classes"), list):
        raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: classes must be an array")

    def non_empty(x):
        return isinstance(x, str) and x.strip() != ""

    canonical_seen = set()
    for i, cls in enumerate(t["classes"]):
        if not isinstance(cls, dict) or cls.get("kind") not in BEHAVIOUR_KINDS:
            raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: class {i} has no valid kind")
        if not non_empty(cls.get("canonical")):
            raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: class {i} has no canonical label")
        if (cls["kind"], cls["canonical"]) in canonical_seen:
            raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: duplicate canonical {json.dumps(cls['canonical'])} for kind {cls['kind']} (add the labels to the existing class)")
        canonical_seen.add((cls["kind"], cls["canonical"]))
        if not non_empty(cls.get("basis")):
            raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: class {i} ({cls['canonical']}) has an empty basis")
        for side in ("r", "julia"):
            labels = cls.get(side)
            if not isinstance(labels, list) or not all(map(non_empty, labels)):
                raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: class {i} ({cls['canonical']}) {side} must be an array of non-empty labels")
            bucket = index.setdefault(cls["kind"], {}).setdefault(side, {})
            for label in labels:
                prior = bucket.get(label)
                if prior and prior[1] != i:
                    raise Fail(f"{IN_BEHAVIOUR_EQUIVALENCE}: ambiguous table, {cls['kind']} {side} label {json.dumps(label)} is in classes {prior[0]} and {cls['canonical']}")
                bucket[label] = (cls["canonical"], i)
    return index


def _canonical(index, kind, side, label):
    hit = index.get(kind, {}).get(side, {}).get(label)
    return hit[0] if hit else label


def behavioural_receipt_problem(row, root, index):
    """None when the row binds behaviourally, else why not. A port of the checker's
    behaviouralReceiptStatus (ruling 2); the carry and dangling-receipt checks are done by the
    caller (derive_status), in the checker's C1 order."""
    paths = receipt_paths(row)
    if not paths:
        return "no receipt"
    if not behavioural_eligible_source_id(row.get("source_id")):
        return "source_id not covered by itchyshin/GLLVModels.jl#684 item 2 (inference/* and four named C1 rows only)"
    ids = as_list(row.get("executable_case_ids"))
    if not ids:
        return "no executable_case_ids"

    def non_empty(x):
        return isinstance(x, str) and x.strip() != ""

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
            for obj, pre in ((j, ""), (j.get("behaviour") if isinstance(j.get("behaviour"), dict) else None, "behaviour.")):
                if obj is None:
                    continue
                for f in STATUS_FIELDS:
                    if f in obj and not is_pass_value(obj[f]):
                        not_passed = f"{pre}{f}={json.dumps(obj[f])} in {p}"
                        break
                if not_passed:
                    break
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
                if not (non_empty(v) or (isinstance(v, list) and v and all(map(non_empty, v)))):
                    return f"case {cid}: {k} must be a non-empty string or an array of non-empty strings"
            if isinstance(c["r_observed"], list) != isinstance(c["julia_observed"], list):
                return f"case {cid}: r_observed and julia_observed must have the same shape"
            if isinstance(c["r_observed"], list) and len(c["r_observed"]) != len(c["julia_observed"]):
                return f"case {cid}: r_observed has {len(c['r_observed'])} labels, julia_observed {len(c['julia_observed'])}"
            if not_passed is None:
                for f in STATUS_FIELDS:
                    if f in c and not is_pass_value(c[f]):
                        not_passed = f"behaviour.cases[{cid}].{f}={json.dumps(c[f])} in {p}"
                        break
            entries.append(c)
        blocks += 1
    if blocks == 0:
        return "no behaviour block in any receipt"
    uncovered = []
    for cid in ids:
        app = [e for e in entries if e["case_id"] == cid and ("source_id" not in e or e["source_id"] == row.get("source_id"))]
        if not app:
            uncovered.append(cid)
            continue
        for e in app:
            rs, js = as_list(e["r_observed"]), as_list(e["julia_observed"])
            for a, b2 in zip(rs, js):
                if _canonical(index, e["kind"], "r", a) != _canonical(index, e["kind"], "julia", b2):
                    return f"case {cid} ({e['kind']}): R {json.dumps(a)} vs Julia {json.dumps(b2)} differ after canonicalisation"
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
    missing = [i for i in as_list(row.get("executable_case_ids")) if i not in covered]
    if missing:
        return "case ids not compared: " + ",".join(missing)
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

def scoreboard_id(sid: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", sid).strip("-")
    if not re.match(r"^[A-Za-z0-9]", s):
        raise Fail(f"cannot form a scoreboard id from {sid!r}")
    if re.search(r"-RSZ$", s, re.I) or re.match(r"^(RD|GRP)-", s, re.I):
        # Would silently move the row into C3/C4/C5 (GATES.md id conventions).
        raise Fail(f"scoreboard id {s} collides with a C3/C4/C5 id convention")
    return s


def derive_status(row, root, equiv=None):
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
        prob = behavioural_receipt_problem(row, root, equiv if equiv is not None else behaviour_equivalence(root))
        return ("EVIDENCED-BEHAVIOURAL", "") if prob is None else ("BEHAVIOURAL-UNVERIFIED", prob)
    if disp is not None:
        return ("NEEDS-SURFACE" if "NEEDS_JULIA_SURFACE" in str(disp) else str(disp)), f"disposition {disp}"
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
    for sid, (family, r) in by_id.items():
        bid = scoreboard_id(sid)
        if bid in ids_seen:
            raise Fail(f"scoreboard id {bid} formed from both {ids_seen[bid]} and {sid}")
        ids_seen[bid] = sid
        status, reason = derive_status(r, root, equiv)
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
        "checker's own C1 numeric rule, and `DISPOSITION-SIGNED` only when the map row carries a valid",
        "maintainer signature. PR #533's `case-map.json` rows are not in this table (not tracked here).",
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
    if not isinstance(d, dict) or d.get("schema") != 1:
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: schema must be 1")
    ruling, decisions = d.get("ruling"), d.get("decisions")
    if not isinstance(ruling, dict) or not all(k in ruling for k in ("ref", "signed_by", "signed_on")):
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: ruling must carry ref, signed_by and signed_on")
    # The tool refuses to copy a signature it does not recognise: the ruling must be a known one, signed
    # on its date, by an allowed signer, and every decision word must be one the ruling covers.
    known = C6_RULINGS.get(ruling["ref"]) if isinstance(ruling["ref"], str) else None
    if known is None:
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: ruling ref {json.dumps(ruling['ref'])} is not a recognised signed ruling ({'; '.join(C6_RULINGS)})")
    sig = signature_problem(ruling["signed_by"], ruling["signed_on"])
    if sig is not None:
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: ruling signature rejected ({sig})")
    if ruling["signed_on"].strip() != known["signed_on"]:
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: ruling signed_on {json.dumps(ruling['signed_on'])} is not the date of {ruling['ref']} ({known['signed_on']})")
    for field in ("criterion", "generator"):
        if not isinstance(d.get(field), str) or not d[field].strip():
            raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: {field} must be a non-empty string")
    if not isinstance(decisions, dict):
        raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: decisions must be an object")
    for name, v in decisions.items():
        if not isinstance(v, dict) or v.get("decision") not in C6_DECISION_VOCAB:
            raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: {name}: decision must be one of {'|'.join(C6_DECISION_VOCAB)}")
        if v["decision"] not in known["words"]:
            raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: {name}: decision {v['decision']} is not covered by {ruling['ref']} (covers {'|'.join(known['words'])})")
        if not isinstance(v.get("basis"), str) or not v["basis"].strip():
            raise Fail(f"{IN_REVERSE_GAP_DECISIONS}: {name}: basis must be a non-empty string")
    return ruling, decisions


def build_reverse_gap(root: Path, ledger: Path):
    p = root / ledger / IN_REVERSE_GAP
    if not p.is_file():
        raise Fail(f"missing input {ledger / IN_REVERSE_GAP} (run --refresh-reverse-gap-inputs)")
    inp = json.loads(p.read_text())
    r_names = set(inp["r_names"])
    named = namespace_receipt_counterparts(root, ledger)
    items = []
    for e in inp["julia_exports"]:
        n = e["name"]
        if n in r_names or n in named:
            continue
        items.append(OrderedDict([
            ("source_id", f"julia-export/{n}"),
            ("name", n),
            ("kind", e["kind"]),
            ("status", "unsigned"),
            ("decision", None),
            ("derived_from", f"names(GLLVModels) at {inp['glvmodels_commit'][:12]}; no same-named gllvmTMB "
                             f"export or S3 generic at P1 and not named as the Julia side of any tracked "
                             f"namespace receipt"),
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

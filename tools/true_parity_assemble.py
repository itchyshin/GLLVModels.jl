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
  DISPOSITION-UNVERIFIED / NEEDS-SURFACE / <disposition>
                      a non-null disposition that is not a valid signature.
  NOT-MEASURED        no receipt and no non-binding receipt cited at all.
  otherwise           the row's evidence_tier, collated into a bucket by TIER_BUCKET below
                      (the tier string itself is copied verbatim into the notes column).

An evidence_tier missing from TIER_BUCKET fails the run: a new tier needs a human to decide
which bucket it reads as, rather than this tool guessing.

Usage:
  python3 tools/true_parity_assemble.py            # write the outputs
  python3 tools/true_parity_assemble.py --check    # regenerate in memory, fail if stale/conflicting
  python3 tools/true_parity_assemble.py --extra-map PATH ...   # also fold PATH (e.g. #533's
        case-map.json fetched with git show) for a conflict scan; with --out-dir, writes there
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
    "EVIDENCED", "DISPOSITION-SIGNED", "NUMERIC-UNVERIFIED", "REGISTRATION-ONLY", "HELD",
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
    pins = (row.get("carry") or {}).get("source_pins")
    if not isinstance(pins, list) or not pins:
        return "PARTIAL_STALE_AT_P1(no carry.source_pins)"
    for sp in pins:
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
        cmp_ = j["comparison"]
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
            d, mism = case_diff(c)
            if mism:
                return f"case {cid}: {mism} in {p}"
            if d is None:
                return f"case {cid}: no finite abs_diff or r_value/julia_value"
            if d > tol:
                return f"case {cid}: abs_diff {d} > tolerance {tol}"
            covered.add(cid)
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


def derive_status(row, root):
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
    if conflicts:
        raise Fail("conflicting rows across maps (not resolved here): " + "; ".join(conflicts))
    rows_out, table, ids_seen = [], [], {}
    counts: "dict[str, Counter]" = {}
    for sid, (family, r) in by_id.items():
        bid = scoreboard_id(sid)
        if bid in ids_seen:
            raise Fail(f"scoreboard id {bid} formed from both {ids_seen[bid]} and {sid}")
        ids_seen[bid] = sid
        status, reason = derive_status(r, root)
        counts.setdefault(family, Counter())[status] += 1
        cases = as_list(r.get("executable_case_ids"))
        requires = f"{r.get('classification')}; cases: {', '.join(cases) if cases else 'none'}"
        if status == "EVIDENCED":
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
        if a.extra_map and not (a.out_dir or a.check):
            ap.error("--extra-map writes only with --out-dir (the tracked outputs use tracked maps only)")
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

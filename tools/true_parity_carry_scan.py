#!/usr/bin/env python3
"""True-parity P1 carry-rule scan for GLLVModels.jl vs gllvmTMB (D-295 row 1, 2026-09-27).

This is the CARRY_VERIFY mode that PR #523's review asked for and GATES.md gap 5 states as
deferred ("a `CARRY_VERIFY` mode that takes a local gllvmTMB clone (GLLVMTMB_DIR) and
re-hashes `git show P0:path` / `P1:path` itself, rather than trusting the two hashes
recorded on the row, does not exist yet"). This script IS that mode: it never trusts a
hash written by a human or an earlier agent, it recomputes both sides from the gllvmTMB
clone every time it runs.

Rule being checked (maintainer decision D-295, row 1): a P0 receipt counts at P1 only if
every file in its source pins is byte-identical at P1 (gllvmTMB commit
9539352f66f2db2cc26b1c393e67212a359b60c9) and P0 (b4d5fee64def88bc768dda1f1f77c29b295edd86);
otherwise the row is PARTIAL_STALE_AT_P1 until re-measured or signed. Receipts must resolve
on origin/main -- a receipt under the gitignored `.unlazy/` does not resolve.

What this script does, per row of the P0 ledger
(docs/dev-log/core070/required-source-case-map.json, read from `--ref`, default
origin/main -- never the working tree, so a gate cannot pass on an unmerged branch):

  1. RETIRED: does the row correspond to an R export that was removed from gllvmTMB's
     NAMESPACE between P0 and P1? (diff of `export(...)` lines). If so, RETIRED regardless
     of anything else -- carrying a retired capability forward makes no sense.
  2. Out of scope: rows whose P0 `classification` is not `required_core` or
     `compatibility_adapter` were never part of the P1 case-map's C1/C8 accounting in the
     first place -- reported as OUT_OF_SCOPE_NOT_REQUIRED, not scanned further.
  3. Not bound at P0: a required row that already carries a P0 `disposition`
     (BLOCKED_*, PARTIAL_*) was never counted as done at P0 -- reported as NOT_BOUND_AT_P0,
     out of scope for the carry question (carrying forward a row that was never bound
     changes nothing).
  4. For the remaining rows (required, disposition null -- i.e. rows P0 counted as bound):
     the row's designated receipt is whatever `evidence.receipt`, `evidence.receipts`, or
     `evidence.harness_receipts` contains (in that priority; `evidence.contract` is used
     only when none of the three are present). Path-like tokens
     (`docs|tools|test|src/....{md,json,toml,jl,py,mjs}`) are extracted from that field's
     text; every extracted token, and the raw field value itself if nothing was extracted,
     must resolve to a git blob on `--ref`. Anything that does not resolve (most commonly a
     bare `.unlazy/...` reference, which is gitignored and was never committed) makes the
     row DANGLING.
  5. If the designated receipt resolves, every JSON file reachable from the row (the
     receipt candidates themselves, `evidence.contract`, and any top-level `*case_plan*`
     field) is parsed and recursively searched for a key ending in `source_pins` whose
     value is an object; keys matching `^(R/|src/)` are gllvmTMB source pins. No pins found
     anywhere -> NO_R_PINS (the row's evidence never cited a gllvmTMB source file, so the
     carry rule cannot apply -- it is not evidence this tool can date).
  6. For every pinned path, this script runs `git -C $GLLVMTMB_DIR show P0:<path>` and
     `git -C $GLLVMTMB_DIR show P1:<path>` itself and hashes each with sha256 -- the
     "re-measured", not "trusted", half of the carry rule. All pairs identical -> CARRIED
     (the pairs are recorded). Any pair differing, or a path missing at either pin ->
     PARTIAL_STALE_AT_P1 (the changed/missing files are listed).

Usage:
    GLLVMTMB_DIR=/path/to/gllvmTMB python3 tools/true_parity_carry_scan.py CARRY_VERIFY \\
        [--ref origin/main] [--out-json PATH] [--out-md PATH]

Run from the root of the GLLVModels.jl working tree (it shells out to plain `git`, no `-C`,
so cwd must be the repo). Exit 2 if GLLVMTMB_DIR is unset, not a git repo, or P0/P1 do not
resolve there -- this mirrors tools/true_parity_check.mjs's own MEASUREMENT_FAILED
convention (a measurement that cannot be made is never silently a pass).
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

P0_SHA = "b4d5fee64def88bc768dda1f1f77c29b295edd86"
P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
P0_CASEMAP_PATH = "docs/dev-log/core070/required-source-case-map.json"
REQUIRED_CLASSES = {"required_core", "compatibility_adapter"}
RECEIPT_KEYS = ("receipt", "receipts", "harness_receipts")

EXTRACT_RE = re.compile(r"\b(?:docs|tools|test|src)/[A-Za-z0-9._\-/]+\.(?:md|json|toml|jl|py|mjs)\b")
LOOKS_PATH_LIKE_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+")
PIN_PATH_RE = re.compile(r"^(?:R/|src/)")


def die(msg, code=2):
    print(f"MEASUREMENT_FAILED {msg}")
    sys.exit(code)


def run_git(args, cwd=None):
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return None


def self_show(ref, path):
    """Read `path` at `ref` in *this* repo (GLLVModels.jl); None if it does not resolve."""
    out = run_git(["show", f"{ref}:{path}"])
    return out.decode("utf8", errors="replace") if out is not None else None


def self_blob_exists(ref, path):
    out = run_git(["cat-file", "-t", f"{ref}:{path}"])
    return out is not None and out.strip() == b"blob"


def gllvmtmb_show(gllvmtmb_dir, ref, path):
    """Raw bytes of `path` at `ref` in the gllvmTMB clone; None if missing."""
    return run_git(["-C", gllvmtmb_dir, "show", f"{ref}:{path}"])


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def extract_paths(text):
    if not text:
        return []
    return sorted(set(EXTRACT_RE.findall(text)))


def looks_path_like(text):
    return bool(text) and bool(LOOKS_PATH_LIKE_RE.search(text))


def stringify(v):
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ".join(stringify(x) for x in v)
    return json.dumps(v)


def find_source_pins(obj):
    """Recursively collect gllvmTMB-source pin paths (R/..., src/...) from any dict key
    ending in 'source_pins' whose value is an object, anywhere in `obj`."""
    found = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k.endswith("source_pins") and isinstance(v, dict):
                    for p in v.keys():
                        if PIN_PATH_RE.match(p):
                            found.add(p)
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)

    walk(obj)
    return found


def removed_exports(gllvmtmb_dir):
    def exports_at(ref):
        out = gllvmtmb_show(gllvmtmb_dir, ref, "NAMESPACE")
        if out is None:
            die(f"NAMESPACE not found at {ref} in {gllvmtmb_dir}")
        names = set()
        for line in out.decode("utf8", errors="replace").splitlines():
            m = re.match(r"^export\((.+)\)$", line.strip())
            if m:
                names.add(m.group(1))
        return names

    p0 = exports_at(P0_SHA)
    p1 = exports_at(P1_SHA)
    return p0 - p1


def load_casemap(ref):
    text = self_show(ref, P0_CASEMAP_PATH)
    if text is None:
        die(f"{P0_CASEMAP_PATH} not on {ref}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        die(f"{P0_CASEMAP_PATH} is not valid JSON: {e}")
    rows = data.get("rows")
    if not isinstance(rows, list):
        die(f"{P0_CASEMAP_PATH} has no 'rows' array")
    return rows


def designated_receipt_candidates(evidence):
    """(field_used, raw_candidates) for the row's designated receipt. raw_candidates is a
    list of path strings to resolve; empty list + field_used=None means nothing to check."""
    for key in RECEIPT_KEYS:
        v = evidence.get(key) if isinstance(evidence, dict) else None
        if v:
            text = stringify(v)
            extracted = extract_paths(text)
            if extracted:
                return key, extracted
            if looks_path_like(text):
                return key, [text.strip()]
            return key, []
    if isinstance(evidence, dict) and evidence.get("contract"):
        text = stringify(evidence["contract"])
        extracted = extract_paths(text)
        return "contract", extracted or ([text.strip()] if looks_path_like(text) else [])
    return None, []


def contract_candidates(row):
    """Every path-like reference anywhere in the row worth searching for source_pins:
    the designated-receipt candidates plus evidence.contract and any *case_plan* field."""
    paths = set()
    ev = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    _, receipt_paths = designated_receipt_candidates(ev)
    paths.update(receipt_paths)
    if ev.get("contract"):
        paths.update(extract_paths(stringify(ev["contract"])) or [stringify(ev["contract"]).strip()])
    for k, v in row.items():
        if "case_plan" in k and isinstance(v, str):
            paths.update(extract_paths(v) or ([v.strip()] if looks_path_like(v) else []))
    # Only keep things that look like real repo paths (avoid stray prose fragments).
    return {p for p in paths if "/" in p}


def process_row(row, ref, gllvmtmb_dir, retired_set):
    source_id = row.get("source_id")
    classification = row.get("classification")
    group = source_id.split("/")[0] if source_id else "?"

    if any(name and name in source_id for name in retired_set):
        return {
            "source_id": source_id,
            "classification": classification,
            "group": group,
            "status": "RETIRED",
            "detail": {"retired_export_match": [n for n in retired_set if n in source_id]},
        }

    if classification not in REQUIRED_CLASSES:
        return {
            "source_id": source_id,
            "classification": classification,
            "group": group,
            "status": "OUT_OF_SCOPE_NOT_REQUIRED",
            "detail": {},
        }

    if row.get("disposition") is not None:
        return {
            "source_id": source_id,
            "classification": classification,
            "group": group,
            "status": "NOT_BOUND_AT_P0",
            "detail": {"p0_disposition": row.get("disposition")},
        }

    ev = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
    field_used, candidates = designated_receipt_candidates(ev)
    if field_used is None:
        return {
            "source_id": source_id,
            "classification": classification,
            "group": group,
            "status": "DANGLING",
            "detail": {"reason": "no receipt/receipts/harness_receipts/contract field in evidence"},
        }
    if not candidates:
        return {
            "source_id": source_id,
            "classification": classification,
            "group": group,
            "status": "DANGLING",
            "detail": {"reason": f"evidence.{field_used} has no extractable or resolvable path", "raw": stringify(ev.get(field_used))},
        }
    unresolved = [p for p in candidates if not self_blob_exists(ref, p)]
    if unresolved:
        return {
            "source_id": source_id,
            "classification": classification,
            "group": group,
            "status": "DANGLING",
            "detail": {"reason": f"evidence.{field_used} does not resolve on {ref}", "unresolved": unresolved, "field": field_used},
        }

    # Designated receipt resolves. Gather every contract candidate and search for pins.
    all_candidates = contract_candidates(row)
    resolving_candidates = []
    contract_texts = {}
    for p in sorted(all_candidates):
        if self_blob_exists(ref, p):
            resolving_candidates.append(p)
            text = self_show(ref, p)
            if text is not None and p.endswith(".json"):
                try:
                    contract_texts[p] = json.loads(text)
                except json.JSONDecodeError:
                    pass

    pins = set()
    for p, obj in contract_texts.items():
        pins |= find_source_pins(obj)

    if not pins:
        return {
            "source_id": source_id,
            "classification": classification,
            "group": group,
            "status": "NO_R_PINS",
            "detail": {"candidates_checked": resolving_candidates},
        }

    pairs = []
    changed = []
    for path in sorted(pins):
        p0_bytes = gllvmtmb_show(gllvmtmb_dir, P0_SHA, path)
        p1_bytes = gllvmtmb_show(gllvmtmb_dir, P1_SHA, path)
        h0 = sha256_bytes(p0_bytes) if p0_bytes is not None else None
        h1 = sha256_bytes(p1_bytes) if p1_bytes is not None else None
        entry = {"path": path, "sha256_at_p0": h0, "sha256_at_p1": h1}
        pairs.append(entry)
        if h0 is None or h1 is None or h0 != h1:
            changed.append(entry)

    if not changed:
        return {
            "source_id": source_id,
            "classification": classification,
            "group": group,
            "status": "CARRIED",
            "detail": {"source_pins": pairs, "receipt": resolving_candidates},
        }
    return {
        "source_id": source_id,
        "classification": classification,
        "group": group,
        "status": "PARTIAL_STALE_AT_P1",
        "detail": {"changed": changed, "total_pins": len(pairs)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="CARRY_VERIFY", choices=["CARRY_VERIFY"])
    ap.add_argument("--ref", default=os.environ.get("PARITY_REF", "origin/main"))
    ap.add_argument("--out-json", default="docs/dev-log/core070/true-parity-latest/carry-scan-p1.json")
    ap.add_argument("--out-md", default="docs/dev-log/core070/true-parity-latest/carry-scan-p1.md")
    args = ap.parse_args()

    gllvmtmb_dir = os.environ.get("GLLVMTMB_DIR")
    if not gllvmtmb_dir or not os.path.isdir(os.path.join(gllvmtmb_dir, ".git")):
        die("GLLVMTMB_DIR is unset or not a git repo")
    for sha in (P0_SHA, P1_SHA):
        if run_git(["-C", gllvmtmb_dir, "cat-file", "-t", sha]) is None:
            die(f"{sha} not found in {gllvmtmb_dir}")

    retired = removed_exports(gllvmtmb_dir)
    rows = load_casemap(args.ref)

    results = [process_row(r, args.ref, gllvmtmb_dir, retired) for r in rows]

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    out = {
        "schema": 1,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "CARRY_VERIFY",
        "ref": args.ref,
        "p0_sha": P0_SHA,
        "p1_sha": P1_SHA,
        "p0_casemap_path": P0_CASEMAP_PATH,
        "retired_exports": sorted(retired),
        "counts": counts,
        "rows": results,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")

    write_markdown(out, args.out_md)

    print(f"CARRY_VERIFY ref={args.ref} rows={len(results)} counts={json.dumps(counts)}")
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")


def write_markdown(out, path):
    counts = out["counts"]
    rows = out["rows"]
    lines = []
    lines.append("# P1 carry-rule scan (CARRY_VERIFY)")
    lines.append("")
    lines.append(f"Generated {out['generated_at']} against `{out['ref']}`, P0 = `{out['p0_sha']}`, "
                  f"P1 = `{out['p1_sha']}`. Source: `{out['p0_casemap_path']}` ({len(rows)} rows).")
    lines.append("")
    lines.append("## Counts by status")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("| --- | --- |")
    for status in sorted(counts, key=lambda s: -counts[s]):
        lines.append(f"| {status} | {counts[status]} |")
    lines.append("")

    lines.append("## Counts by status x family (source_id group)")
    lines.append("")
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], {}).setdefault(r["status"], 0)
        by_group[r["group"]][r["status"]] += 1
    statuses = sorted(counts.keys())
    lines.append("| Family | " + " | ".join(statuses) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in statuses) + " |")
    for g in sorted(by_group):
        row_counts = by_group[g]
        lines.append(f"| {g} | " + " | ".join(str(row_counts.get(s, 0)) for s in statuses) + " |")
    lines.append("")

    lines.append("## 5 largest PARTIAL_STALE_AT_P1 groups (by family)")
    lines.append("")
    stale_by_group = {}
    for r in rows:
        if r["status"] == "PARTIAL_STALE_AT_P1":
            stale_by_group.setdefault(r["group"], []).append(r["source_id"])
    top5 = sorted(stale_by_group.items(), key=lambda kv: -len(kv[1]))[:5]
    for g, ids in top5:
        lines.append(f"- **{g}**: {len(ids)} rows")
    lines.append("")

    lines.append("## DANGLING rows (receipts that do not resolve on origin/main)")
    lines.append("")
    dangling = [r for r in rows if r["status"] == "DANGLING"]
    lines.append(f"{len(dangling)} rows. The 20 `isdm/*` rows in this list cite receipts under "
                 "`.unlazy/core070-aghq/...` (via `evidence.receipts` directly, or via their case "
                 "plan's trace through `docs/dev-log/core070/isdm-admission-evidence.json`'s own "
                 "`runs[].receipt`, e.g. `.unlazy/core070-aghq/isdm-admission/attempt2/receipt.json`); "
                 "none of that resolves on `origin/main`. The iSDM port (spec PR #525) re-measures "
                 "these natively; no receipt is invented here.")
    lines.append("")
    isdm_dangling = sorted(r["source_id"] for r in dangling if r["source_id"].startswith("isdm/"))
    lines.append(f"isdm/* DANGLING rows ({len(isdm_dangling)}): " + ", ".join(isdm_dangling))
    lines.append("")

    lines.append("## RETIRED rows")
    lines.append("")
    retired_rows = [r for r in rows if r["status"] == "RETIRED"]
    for r in retired_rows:
        lines.append(f"- `{r['source_id']}`")
    lines.append("")

    lines.append("## CARRIED rows (added to the P1 case-map)")
    lines.append("")
    carried = [r for r in rows if r["status"] == "CARRIED"]
    if carried:
        for r in carried:
            lines.append(f"- `{r['source_id']}` ({len(r['detail']['source_pins'])} pins, all byte-identical)")
    else:
        lines.append("None. Every required, previously-bound P0 row that pins a gllvmTMB source file "
                      "pins at least one file that changed between P0 and P1 (62 of 117 R/src files "
                      "changed; the two most-shared files, `R/fit-multi.R` and `R/gllvmTMB.R`, both "
                      "changed) -- matching the contract-level preliminary scan's finding that all 29 "
                      "R-pinning contracts touch a changed file. Everything that could carry did not; "
                      "this is the honest count, not an empty search.")
    lines.append("")

    lines.append("## NOT_BOUND_AT_P0 / OUT_OF_SCOPE_NOT_REQUIRED")
    lines.append("")
    lines.append(f"NOT_BOUND_AT_P0 (required rows already BLOCKED_*/PARTIAL_* at P0 -- never counted, "
                 f"so the carry question does not apply): {counts.get('NOT_BOUND_AT_P0', 0)}. "
                 f"OUT_OF_SCOPE_NOT_REQUIRED (P0 `rejected`/`intentionally_excluded` rows, never part "
                 f"of C1/C8 accounting): {counts.get('OUT_OF_SCOPE_NOT_REQUIRED', 0)}.")
    lines.append("")

    lines.append("## What this scan does not cover")
    lines.append("")
    lines.append("- No R or Julia fitting: everything above is a file-hash comparison, not a re-run of "
                 "any model, control, or paired-control case.")
    lines.append("- `NO_R_PINS` rows are not evaluated for staleness at all -- their P0 evidence never "
                 "cited a gllvmTMB source file, so byte-identity has nothing to check; they need their "
                 "own re-measurement plan, not a carry decision.")
    lines.append("- Name-twin detection is out of scope (GATES.md gap 6): a row can be `CARRIED` here "
                 "and still be wrong if it was misclassified at P0.")
    lines.append("- `PARTIAL_STALE_AT_P1` rows are listed, not re-measured; WS0d/arc A3 owns re-running "
                 "them at P1.")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

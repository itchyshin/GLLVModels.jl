"""Regenerate the Core070 covariance contracts at gllvmTMB pin P1.

Three P0 files are read and left untouched as history:

  * docs/dev-log/core070/frozen-r070-contract.toml -- the manifest the
    test/parity runner (runparity.jl) validates before any required case runs;
  * docs/dev-log/core070/covariance-batch-contract.json -- the R-only
    formula-grammar batch (tools/core070_covariance_batch.R);
  * docs/dev-log/core070/wave6-conversion-batch-contract.json -- the paired
    R/Julia structured-term batch whose two kernel_latent cases pay
    covariance/COV-KERNEL-FOLDED-UNIQUE and covariance/COV-KERNEL-LATENT.

and three P1 files are written under docs/dev-log/core070/true-parity-latest/:

  * frozen-r070-contract-p1.toml
  * covariance-batch-contract-p1.json
  * wave6-conversion-batch-contract-p1.json

R side: every gllvmTMB byte is read with `git -C $GLLVMTMB_DIR show <P1>:<path>`
(GIT_OPTIONAL_LOCKS=0; the clone is never checked out or edited). The P1 commit
and its archive / NAMESPACE / source-tree hashes come from the shared pin source,
tools/core070_oracle_pins.toml, not from a literal in this file.

What changes, all recorded in each output's regeneration log:

  frozen-r070-contract-p1.toml
    * the provenance header (reference_commit, NAMESPACE blob + sha256,
      R/fit-multi.R and R/families.R blobs, archive and source-tree sha256,
      source inventory path, `source` line) is recomputed at P1;
    * every `source = "..."` citation that names the P0 commit or the P0
      NAMESPACE blob is re-anchored at P1 (reanchor_source below):
        - a line range (`R/<file>:<a>-<b> @ <P0>`, `<P0>:R/<file>:<a>-<b>`,
          `NAMESPACE:<P0 blob>:<a>-<b>`) is located at P1 by searching for the
          same text: first the whole cited block verbatim, then the shortest
          unique leading and trailing runs of its lines. The citation is
          rewritten to the P1 line range (with the P1 blob sha), marked
          `identical` or `changed` in the log;
        - a range that cannot be located unambiguously, and a file-level
          citation (`R/<file> @ <P0>`), become a P1 file pin
          `R/<file> @ <P1> blob <sha>`;
        - a `<P0>:test/parity/...` citation names a GLLVModels.jl path under a
          gllvmTMB sha (the path does not exist in gllvmTMB at P0 or P1); it is
          rewritten to a GLLVModels.jl file pin with the file's git blob sha;
      the full per-citation log is written as comments under the header;
    * everything else (case-id registries, family rows, obligations) is
      carried verbatim from P0.

  covariance-batch-contract-p1.json
    * reference_commit and the four source_pins sha256 are recomputed at P1;
    * cases, expectations and negative controls are carried verbatim: no
      expectation is edited to make a P1 run pass. A P1 failure is recorded as
      a failure by the batch, never absorbed here.

  wave6-conversion-batch-contract-p1.json
    * reference_commit is set to P1; cases, r_call strings, tolerances,
      deferred rows, rejection cases and negative controls are carried
      verbatim. The whole 10-case batch runs, not only its two covariance
      cases, because the runner and verifier check the batch as one unit.

Usage:
  GLLVMTMB_DIR=/path/to/gllvmTMB python3 tools/core070_covariance_p1_contract.py [--check]

--check regenerates in memory and exits nonzero if any tracked P1 contract
differs, so a stale contract is caught without rewriting it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
P0_TOML = ROOT / "docs/dev-log/core070/frozen-r070-contract.toml"
P0_BATCH = ROOT / "docs/dev-log/core070/covariance-batch-contract.json"
OUT_DIR = ROOT / "docs/dev-log/core070/true-parity-latest"
P1_TOML = OUT_DIR / "frozen-r070-contract-p1.toml"
P1_BATCH = OUT_DIR / "covariance-batch-contract-p1.json"
P0_WAVE6 = ROOT / "docs/dev-log/core070/wave6-conversion-batch-contract.json"
P1_WAVE6 = OUT_DIR / "wave6-conversion-batch-contract-p1.json"
PINS_FILE = ROOT / "tools/core070_oracle_pins.toml"
PIN = "P1"
P1_SOURCE_INVENTORY = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/source.json"


def git_show(repo, sha, path):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(["git", "-C", str(repo), "show", f"{sha}:{path}"], check=True,
                          capture_output=True, env=env).stdout


def git_blob(repo, sha, path):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(["git", "-C", str(repo), "rev-parse", f"{sha}:{path}"], check=True,
                          capture_output=True, text=True, env=env).stdout.strip()


def pin_entry():
    table = tomllib.loads(PINS_FILE.read_text())
    return table["P0"], table[PIN]


def replace_line(text, key, value, log):
    pattern = re.compile(rf'^{re.escape(key)} = "([^"]*)"$', re.M)
    m = pattern.search(text)
    if m is None:
        raise SystemExit(f"P0 contract has no top-level `{key}` line")
    log.append({"field": key, "p0": m.group(1), "p1": value})
    return text[:m.start()] + f'{key} = "{value}"' + text[m.end():]


def git_text_lines(ctx, sha, path):
    key = (sha, path)
    if key not in ctx["cache"]:
        ctx["cache"][key] = git_show(ctx["repo"], sha, path).decode().split("\n")
    return ctx["cache"][key]


def find_unique(haystack, needle):
    """Index of the only occurrence of the contiguous run `needle` in `haystack`, or None."""
    hits = [i for i in range(len(haystack) - len(needle) + 1) if haystack[i:i + len(needle)] == needle]
    return hits[0] if len(hits) == 1 else None


def locate_block(old_lines, new_lines, a, b, max_anchor=12):
    """Locate P0 lines a..b (1-based, inclusive) in the P1 file.

    Returns (kind, a1, b1) with kind "range-identical" or "range-changed", or None
    when the block cannot be located unambiguously.
    """
    block = old_lines[a - 1:b]
    at = find_unique(new_lines, block)
    if at is not None:
        return "range-identical", at + 1, at + len(block)
    start = end = None
    for k in range(1, min(max_anchor, len(block)) + 1):
        start = find_unique(new_lines, block[:k])
        if start is not None:
            break
    for k in range(1, min(max_anchor, len(block)) + 1):
        hit = find_unique(new_lines, block[-k:])
        if hit is not None:
            end = hit + k - 1
            break
    if start is None or end is None or end < start:
        return None
    return "range-changed", start + 1, end + 1


def reanchor_part(part, ctx, anchors):
    """Rewrite one `;`-separated citation at P1 and append its log row to `anchors`."""
    p0, p1 = ctx["p0"], ctx["p1"]
    patterns = [
        (re.compile(rf"^(R/[^:@ ]+):(\d+)-(\d+) @ {p0}$"), "range"),
        (re.compile(rf"^{p0}:(R/[^:@ ]+):(\d+)-(\d+)$"), "range"),
        (re.compile(rf"^(NAMESPACE):{ctx['p0_ns_blob']}:(\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)$"), "ranges"),
        (re.compile(rf"^(R/[^:@ ]+) ?@ ?{p0}$"), "file"),
        (re.compile(rf"^{p0}:(test/parity/[^:@ ]+)$"), "glvmodels"),
    ]
    for pattern, kind in patterns:
        m = pattern.match(part)
        if m is None:
            continue
        path = m.group(1)
        if kind == "glvmodels":
            blob = subprocess.run(["git", "-C", str(ROOT), "hash-object", path], check=True,
                                  capture_output=True, text=True).stdout.strip()
            new = f"GLLVModels.jl:{path} @ blob {blob}"
            anchors.append({"kind": "glvmodels-file-pin", "p0": part, "p1": new})
            return new
        blob = git_blob(ctx["repo"], p1, path)
        if kind == "range":
            a, b = int(m.group(2)), int(m.group(3))
            hit = locate_block(git_text_lines(ctx, p0, path), git_text_lines(ctx, p1, path), a, b)
            if hit is not None:
                how, a1, b1 = hit
                new = f"{path}:{a1}-{b1} @ {p1} blob {blob}"
                anchors.append({"kind": how, "p0": part, "p1": new})
                return new
        if kind == "ranges":
            # NAMESPACE:<blob>:<list> -- a comma list of lines / line ranges; each segment
            # is located on its own, and all must be found or the file is pinned instead.
            segs = []
            for seg in m.group(2).split(","):
                a, _, b = seg.partition("-")
                hit = locate_block(git_text_lines(ctx, p0, path), git_text_lines(ctx, p1, path),
                                   int(a), int(b or a))
                if hit is None:
                    segs = None
                    break
                segs.append(hit)
            if segs:
                how = "range-identical" if all(h[0] == "range-identical" for h in segs) else "range-changed"
                spans = ",".join(str(h[1]) if h[1] == h[2] else f"{h[1]}-{h[2]}" for h in segs)
                new = f"{path}:{blob}:{spans} @ {p1}"
                anchors.append({"kind": how, "p0": part, "p1": new})
                return new
        new = f"{path} @ {p1} blob {blob}"
        anchors.append({"kind": "file-pin", "p0": part, "p1": new})
        return new
    if p0 in part or ctx["p0_ns_blob"] in part:
        raise SystemExit(f"unrecognised P0 citation form, cannot re-anchor: {part!r}")
    return part


def build_toml(repo):
    p0, p1 = pin_entry()
    sha = p1["reference_commit"]
    text = P0_TOML.read_text()
    head_end = text.index("\nfamily_smoke_case_ids")
    head, rest = text[:head_end], text[head_end:]
    log = []
    ns_bytes = git_show(repo, sha, "NAMESPACE")
    ns_sha = hashlib.sha256(ns_bytes).hexdigest()
    if ns_sha != p1["namespace_sha256"]:
        raise SystemExit(f"NAMESPACE at {sha} hashes to {ns_sha}, pins file says {p1['namespace_sha256']}")
    for key, value in (
        ("reference_commit", sha),
        ("reference_namespace_blob", git_blob(repo, sha, "NAMESPACE")),
        ("reference_namespace_sha256", ns_sha),
        ("reference_fit_multi_blob", git_blob(repo, sha, "R/fit-multi.R")),
        ("reference_families_blob", git_blob(repo, sha, "R/families.R")),
        ("reference_archive_sha256", p1["archive_sha256"]),
        ("reference_source_tree_sha256", p1["source_tree_sha256"]),
        ("reference_source_inventory", P1_SOURCE_INVENTORY),
        ("source", f"git show gllvmTMB:{sha}"),
    ):
        head = replace_line(head, key, value, log)
    p0_ns_blob = git_blob(repo, p0["reference_commit"], "NAMESPACE")
    anchors = []
    ctx = {"repo": repo, "p0": p0["reference_commit"], "p1": sha, "p0_ns_blob": p0_ns_blob, "cache": {}}

    def rewrite(m):
        value = m.group(2)
        if p0["reference_commit"] not in value and f"NAMESPACE:{p0_ns_blob}:" not in value:
            return m.group(0)
        parts = [reanchor_part(part.strip(), ctx, anchors) for part in value.split(";")]
        return f'{m.group(1)}"{"; ".join(parts)}"'

    rest = re.sub(r'^(source = )"([^"]*)"$', rewrite, rest, flags=re.M)
    leftover = len(re.findall(p0["reference_commit"], rest))
    if leftover:
        raise SystemExit(f"{leftover} P0 commit citation(s) were not re-anchored")
    kinds = {}
    for a in anchors:
        kinds[a["kind"]] = kinds.get(a["kind"], 0) + 1
    note = [
        "",
        "# P1 regeneration (tools/core070_covariance_p1_contract.py). The provenance header above",
        "# was recomputed at P1. Below it, every `source` citation that named the P0 commit",
        f"# ({p0['reference_commit']}) or the P0 NAMESPACE blob was re-anchored at P1",
        f"# ({len(anchors)} citations: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ").",
        "# `range-identical`: the cited block is byte-identical at P1, only its line numbers moved.",
        "# `range-changed`: the block's first and last lines were found uniquely at P1, but the code",
        "# between them changed; the citation spans the P1 block. `file-pin`: the file is cited by its",
        "# P1 blob sha, either because the P0 citation was file-level or because the range could not",
        "# be located unambiguously. `glvmodels-file-pin`: the P0 citation named a GLLVModels.jl path",
        "# under a gllvmTMB sha; it is now pinned by the GLLVModels.jl git blob sha of that file.",
        "# Everything else below is carried verbatim from docs/dev-log/core070/frozen-r070-contract.toml.",
        "# Header changes:",
    ]
    note += [f"#   {row['field']}: {row['p0']} -> {row['p1']}" for row in log]
    note.append("# Citation re-anchoring (distinct citations; kind: P0 -> P1):")
    seen = set()
    for a in anchors:
        if (a["p0"], a["p1"]) in seen:
            continue
        seen.add((a["p0"], a["p1"]))
        n = sum(1 for b in anchors if b["p0"] == a["p0"])
        note.append(f"#   {a['kind']} x{n}: {a['p0']} -> {a['p1']}")
    return head + "\n".join(note) + rest


def build_batch(repo):
    p0, p1 = pin_entry()
    sha = p1["reference_commit"]
    contract = json.loads(P0_BATCH.read_text())
    if contract["reference_commit"] != p0["reference_commit"]:
        raise SystemExit("P0 covariance batch contract is not pinned at P0")
    log = [{"field": "reference_commit", "p0": contract["reference_commit"], "p1": sha}]
    contract["reference_commit"] = sha
    pins = {}
    for rel, old in contract["source_pins"].items():
        new = hashlib.sha256(git_show(repo, sha, rel)).hexdigest()
        pins[rel] = new
        log.append({"field": f"source_pins[{rel}]", "p0": old, "p1": new, "changed": old != new})
    contract["source_pins"] = pins
    contract["status"] = "REGENERATED_AT_P1_BEFORE_RUN"
    contract["p0_contract"] = "docs/dev-log/core070/covariance-batch-contract.json"
    contract["p0_contract_sha256"] = hashlib.sha256(P0_BATCH.read_bytes()).hexdigest()
    contract["regeneration_log"] = {
        "generator": "tools/core070_covariance_p1_contract.py",
        "changes": log,
        "carried_verbatim": "cases, expected_covstructs, negative_controls, fixture, required_health_checks, "
                            "claim_boundary. No expectation was edited; a P1 mismatch is recorded by the batch as a failure.",
    }
    return json.dumps(contract, indent=2) + "\n"


WAVE6_NOBS_CASE = {
    "case_id": "CORE070-WAVE6-POSTFIT-NOBS-MULTI",
    "source_id": "postfit/POSTFIT-SURFACE-nobs.gllvmTMB_multi",
    "kind": "integer_equality",
    "fixture": "gaussian_small",
    "quantity": "nobs",
    "tolerance": 0.5,
    "r_call": "as.numeric(nobs(fit_g))",
    "julia_call": "GLLVModels.nobs(fit_g, Y_g)  # StatsAPI.nobs(fit::AnyGllvmFit, Y; mask), src/postfit.jl",
    "expected": "R and Julia return the same integer: both count observed response cells (p * n = 400 on "
                "gaussian_small, which has no missing cells)",
    "ruling": "maintainer ruling 2026-10-05 (D-319), item N2; comparison kind integer_equality under "
              "itchyshin/GLLVModels.jl#684 item 1 (tolerance 0.5 on integers means equal)",
    "notes": "Rewritten at P1 from the P0 own_receipt_defect case, which asserted each engine against its own "
             "formula (R == p*n, Julia == n). Julia now returns p*n, as R does, so that expectation is stale, not a "
             "parity gap. The case now asserts R nobs == Julia nobs exactly.",
}


def build_wave6():
    p0, p1 = pin_entry()
    contract = json.loads(P0_WAVE6.read_text())
    if contract["reference_commit"] != p0["reference_commit"]:
        raise SystemExit("P0 wave6 contract is not pinned at P0")
    contract["reference_commit"] = p1["reference_commit"]
    contract["p0_contract"] = "docs/dev-log/core070/wave6-conversion-batch-contract.json"
    contract["p0_contract_sha256"] = hashlib.sha256(P0_WAVE6.read_bytes()).hexdigest()
    changes = [{"field": "reference_commit", "p0": p0["reference_commit"], "p1": p1["reference_commit"]}]
    # Maintainer ruling 2026-10-05 (D-319), item N2: the nobs case's P0 expectation (Julia == n) is stale at
    # P1, so the case is rewritten as an R-vs-Julia integer equality (itchyshin/GLLVModels.jl#684 item 1).
    idx = next(i for i, c in enumerate(contract["cases"]) if c["case_id"] == WAVE6_NOBS_CASE["case_id"])
    old = contract["cases"][idx]
    if old["kind"] != "own_receipt_defect" or old["source_id"] != WAVE6_NOBS_CASE["source_id"]:
        raise SystemExit("P0 wave6 nobs case is not the own_receipt_defect case N2 rewrites")
    contract["cases"][idx] = WAVE6_NOBS_CASE
    changes.append({"field": f"cases[{WAVE6_NOBS_CASE['case_id']}]", "p0": "kind own_receipt_defect (R == p*n, Julia == n)",
                    "p1": "kind integer_equality (R nobs == Julia nobs, tolerance 0.5)",
                    "ruling": "maintainer ruling 2026-10-05 (D-319), item N2"})
    contract["regeneration_log"] = {
        "generator": "tools/core070_covariance_p1_contract.py",
        "changes": changes,
        "carried_verbatim": "status, the other nine cases (r_call, term_expr, quantity, tolerance), deferred, "
                            "rejection_cases, negative_controls, fixtures, runner. No tolerance was edited; the one "
                            "expectation changed is the nobs case, by the signed ruling named in changes.",
    }
    return json.dumps(contract, indent=2) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    repo = os.environ.get("GLLVMTMB_DIR")
    if not repo:
        raise SystemExit("set GLLVMTMB_DIR to a gllvmTMB clone that contains the P1 commit")
    outputs = {P1_TOML: build_toml(repo), P1_BATCH: build_batch(repo), P1_WAVE6: build_wave6()}
    tomllib.loads(outputs[P1_TOML])  # the regenerated manifest must still parse
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, t in outputs.items() if not p.exists() or p.read_text() != t]
        if stale:
            print("STALE " + " ".join(stale))
            sys.exit(1)
        print("CORE070_COVARIANCE_P1_CONTRACTS_CURRENT")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, t in outputs.items():
        p.write_text(t)
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

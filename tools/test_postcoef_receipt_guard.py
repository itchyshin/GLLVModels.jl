#!/usr/bin/env python3
"""Guard for the committed POST-COEF-EMPTY case receipt. Read-only; runs no fit.

usage: test_postcoef_receipt_guard.py --root TREE [--git-repo REPO]
  TREE      tree holding docs/ and tools/ (the worktree, or a patched copy)
  REPO      git repo used to resolve the receipt's "src/postfit.jl:N at commit SHA" citation (default: TREE)
Exit 0 only if every check passes AND every negative control is caught.
"""
import argparse, json, pathlib, re, subprocess, sys

LATEST = "docs/dev-log/core070/true-parity-latest"
CID = "CORE070-POSTFIT-COEF-EMPTY-NATIVE"
MARK = "DIAGNOSTIC, NON-BINDING"

def g_contract_names_method(contract):
    """The contract must state which R-side method it accepts; origin/main says a synthetic mock, a later clause may say a real fit."""
    d = contract["fixture"]["empty_design"]["description"].lower()
    return "real fitted r" in d or "synthetic mock" in d

def g_synthetic_is_labelled(rec):
    """A receipt whose R side is a synthetic object must say so up front, as diagnostic and non-binding."""
    if "synthetic" in rec["r_call"].lower():
        return rec["numeric_note"].lstrip().startswith(MARK)
    return True

def g_julia_citation_resolves(rec, line_at):
    """'src/postfit.jl:N at the measured commit SHA' must name the StatsAPI.coef(fit::GllvmFit) line at that commit."""
    m = re.search(r"src/postfit\.jl:(\d+) at the measured commit ([0-9a-f]{7,40})", rec["julia_surface"])
    if not m:
        return False
    return line_at(m.group(2), int(m.group(1))).startswith("StatsAPI.coef(fit::GllvmFit)")

def g_comparison_cannot_bind(rec):
    c = next(x for x in rec["comparison"]["cases"] if x["case_id"] == CID)
    zero_r = abs(c["r_value"]) < 1e-10
    return (not zero_r) or c.get("discriminating") is False   # a 0 = 0 comparison must stay non-discriminating

def g_tolerance_documented(rec, contract):
    c = rec["comparison"]["cases"][0]
    return (c["tolerance"] == 0.5 and contract["tolerances"]["integer_exact"] == 0
            and "integer_exact = 0" in c["tolerance_rule"])

def g_scoreboard_unbound(scoreboard_text):
    line = next(l for l in scoreboard_text.splitlines() if "postfit-policy/POST-COEF-EMPTY" in l and l.startswith("|"))
    return "NON-DISCRIMINATING" in line and "not bound" in line

def g_packet_not_overclaiming(packet_text, rec):
    """The decision packet may not say retained evidence tests real no-X behaviour while the only committed receipt is synthetic."""
    over = "retained evidence now tests real no-x behavior" in packet_text.lower()
    return not (over and "synthetic" in rec["r_call"].lower())

def run(root, repo):
    root, repo = pathlib.Path(root), pathlib.Path(repo)
    L = root / LATEST
    rec = json.loads((L / f"receipts/postfit/cases/{CID}.json").read_text())
    contract = json.loads((L / "postfit-policy-batch-contract-p1.json").read_text())
    p0 = json.loads((root / "docs/dev-log/core070/postfit-policy-batch-contract.json").read_text())
    sb = (L / "scoreboard.md").read_text()
    pk = root / "docs/dev-log/plans/2026-10-27-p1-current-main-decision-packet.md"
    packet = pk.read_text() if pk.exists() else ""   # packet lives on another branch; absent text cannot overclaim
    def line_at(sha, n):
        out = subprocess.run(["git", "-C", str(repo), "show", f"{sha}:src/postfit.jl"], capture_output=True, text=True)
        lines = out.stdout.splitlines()
        return lines[n - 1] if out.returncode == 0 and 0 < n <= len(lines) else ""
    checks = {
        "G1 contract states its R-side method (real fit or synthetic mock)": g_contract_names_method(contract),
        "G2 synthetic R side is labelled diagnostic/non-binding": g_synthetic_is_labelled(rec),
        "G3 julia_surface line citation resolves at its commit": g_julia_citation_resolves(rec, line_at),
        "G4 0 = 0 comparison stays non-discriminating": g_comparison_cannot_bind(rec),
        "G5 tolerance 0.5 vs contract integer_exact 0 is documented": g_tolerance_documented(rec, p0),
        "G6 scoreboard row NON-DISCRIMINATING and not bound": g_scoreboard_unbound(sb),
        "G7 packet (if present) does not overclaim retained real-fit evidence": g_packet_not_overclaiming(packet, rec),
    }
    # negative controls: each mutation must be CAUGHT by the named guard
    import copy
    def mut(f):
        r = copy.deepcopy(rec); f(r); return r
    neg = {
        "N1 unlabelled synthetic note is caught by G2":
            not g_synthetic_is_labelled(mut(lambda r: r.update(numeric_note="Integer equality ..."))),
        "N2 wrong line number (633 -> 640) caught by G3":
            not g_julia_citation_resolves(mut(lambda r: r.update(julia_surface=r["julia_surface"].replace(":652", ":640").replace(":633", ":640"))), line_at),
        "N3 0 = 0 marked discriminating caught by G4":
            not g_comparison_cannot_bind(mut(lambda r: r["comparison"]["cases"][0].update(discriminating=True))),
        "N4 tolerance 0.4 caught by G5":
            not g_tolerance_documented(mut(lambda r: r["comparison"]["cases"][0].update(tolerance=0.4)), p0),
        "N5 scoreboard row flipped to EVIDENCED caught by G6":
            not g_scoreboard_unbound(sb.replace("| NON-DISCRIMINATING | not bound", "| EVIDENCED | docs/x")),
        "N6 overclaiming packet caught by G7":
            not g_packet_not_overclaiming("Retained evidence now tests real no-X behavior", rec) or "synthetic" not in rec["r_call"].lower(),
    }
    return checks, neg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--git-repo")
    a = ap.parse_args()
    checks, neg = run(a.root, a.git_repo or a.root)
    for k, v in {**checks, **neg}.items():
        print(("PASS " if v else "FAIL ") + k)
    ok = all(checks.values()) and all(neg.values())
    print("POSTCOEF_RECEIPT_GUARD_" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()

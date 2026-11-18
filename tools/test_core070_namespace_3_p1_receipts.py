"""Tests for tools/core070_namespace_3_p1_receipts.py and the D5 numeric case, through the real route.

Pure Python + git + node (the checker). No R, no Julia. Every control starts from the tracked batch
copy under receipts/namespace/namespace-3-d5-p1/ and breaks one thing:

  refusals by the receipt tool: wrong seed, changed data bytes, changed tree bytes, changed design
  hash, changed contract seed or bars, R fit not converged, wrong oracle version, matched point
  outside a bar, Julia source changed since the run, run commit not an ancestor;
  the real checker (tools/true_parity_check.mjs, PARITY_REF=FS): the row binds with the tracked
  receipt, does not bind with one number pushed outside its bar, does not bind with the receipt
  removed, and the registration form of the row (rebuilt from the fields the row preserved) is
  registration-only;
  the frozen namespace-1 contract file is byte-identical to its recorded hash.

Usage: python3 tools/test_core070_namespace_3_p1_receipts.py
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import core070_namespace_3_p1_receipts as R  # noqa: E402

LEDGER = ROOT / R.LEDGER
FROZEN_NS1_SHA256 = hashlib.sha256((LEDGER / "namespace-1-batch-contract-p1.json").read_bytes()).hexdigest()
FAILS = []
RECORDED_NS1_SHA256 = "fe645d39063248210c66bea21428bd282cd8d0982666940f58422e0c6e34ea51"


def expect(name, ok, detail=""):
    print(("ok - " if ok else "NOT OK - ") + name + ((" :: " + detail) if (detail and not ok) else ""))
    if not ok:
        FAILS.append(name)


def git(root, *a):
    subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)


def make_root(tmp):
    """A throwaway git repo holding exactly what the tool reads, one commit; returns (root, commit)."""
    root = Path(tmp) / "repo"
    for rel in [R.LEDGER + "/namespace-3-d5-batch-contract-p1.json", R.LEDGER + "/case-map-namespace.json",
                "Project.toml"] + [f"tools/core070_namespace_3_d5_{n}" for n in ("batch.R", "batch.jl", "run.sh")]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, root / rel)
    (root / "src").mkdir()
    (root / "src" / "m.jl").write_text("module M end\n")
    shutil.copytree(LEDGER / "receipts/namespace/namespace-3-d5-p1", root / R.LEDGER / "receipts/namespace/namespace-3-d5-p1")
    (root / R.LEDGER / "receipts/namespace/cases").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(LEDGER / f"receipts/namespace/cases/{R.CASE_ID}.json",
                    root / R.LEDGER / f"receipts/namespace/cases/{R.CASE_ID}.json")
    git(root, "init", "-q")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "fixture")
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return root, commit


def state_copy(root, commit, tmp, name):
    st = Path(tmp) / name
    shutil.copytree(root / R.LEDGER / "receipts/namespace/namespace-3-d5-p1", st)
    run = json.loads((st / "run-commit.json").read_text())
    run["glvmodels_commit"] = commit
    (st / "run-commit.json").write_text(json.dumps(run))
    return st


def refuses(root, st, fragment):
    try:
        R.verify_run(root, st)
    except R.Refuse as e:
        return fragment in str(e), str(e)
    return False, "no refusal"


def edit_csv_kv(path, key, value):
    out = []
    for ln in path.read_text().splitlines():
        out.append(f"{key},{value}" if ln.startswith(key + ",") else ln)
    path.write_text("\n".join(out) + "\n")


def checker(root_dir, rows_map, mode="C1"):
    env = {**os.environ, "PARITY_REF": "FS", "PARITY_FS_ROOT": str(root_dir), "PARITY_CASEMAP": "case-map.json"}
    p = subprocess.run(["node", str(ROOT / "tools/true_parity_check.mjs"), mode], capture_output=True, text=True, env=env)
    return p.stdout


def checker_tree(tmp, name, row, receipts_from, include_receipt=True, tamper=None):
    """A one-row FS tree for the real checker. receipts_from: list of repo-relative receipt paths to copy."""
    d = Path(tmp) / name
    cm = json.loads((LEDGER / "case-map-namespace.json").read_text())
    (d).mkdir(parents=True)
    (d / "case-map.json").write_text(json.dumps({**cm, "rows": [row]}))
    for rp in receipts_from:
        dst = d / rp
        dst.parent.mkdir(parents=True, exist_ok=True)
        if include_receipt:
            shutil.copyfile(ROOT / rp, dst)
            if tamper and rp.endswith(f"{R.CASE_ID}.json"):
                j = json.loads(dst.read_text())
                tamper(j)
                dst.write_text(json.dumps(j, indent=2))
    return d


def main():
    cm = json.loads((LEDGER / "case-map-namespace.json").read_text())
    row = next(r for r in cm["rows"] if r["source_id"] == R.SOURCE_ID)
    expect("row_is_numeric_with_the_new_case_only", row["evidence_tier"] == "numeric" and row["executable_case_ids"] == [R.CASE_ID])
    expect("row_keeps_the_registration_case_cited", row["evidence"]["registration_case_ids"] == ["CORE070-NAMESPACE-EXTRACT-PHYLO-SIGNAL-NATIVE"])
    expect("frozen_namespace_1_contract_hash_pinned", FROZEN_NS1_SHA256 == RECORDED_NS1_SHA256, FROZEN_NS1_SHA256)
    case_path = row["evidence"]["receipt"][0]
    expect("case_id_not_in_frozen_namespace_1_contract",
           R.CASE_ID not in (LEDGER / "namespace-1-batch-contract-p1.json").read_text())

    with tempfile.TemporaryDirectory() as tmp:
        root, commit = make_root(tmp)
        st = state_copy(root, commit, tmp, "pristine")
        v = R.verify_run(root, st)
        expect("pristine_run_verifies", all(d <= 1e-6 for d in v["diffs"].values()), str(v["diffs"]))

        # --- refusals by the receipt tool
        s = state_copy(root, commit, tmp, "seed"); edit_csv_kv(s / "R_summary_A.csv", "seed", "4402")
        ok, msg = refuses(root, s, "seed 4402"); expect("refuse_seed_not_4401", ok, msg)
        s = state_copy(root, commit, tmp, "data"); (s / "frozen_A.csv").write_text((s / "frozen_A.csv").read_text().replace("-2.2144", "-2.2145", 1))
        ok, msg = refuses(root, s, "frozen data bytes"); expect("refuse_changed_data_bytes", ok, msg)
        s = state_copy(root, commit, tmp, "tree"); (s / "tree.nwk").write_text((s / "tree.nwk").read_text().replace("s1:2", "s1:3", 1))
        ok, msg = refuses(root, s, "tree bytes"); expect("refuse_changed_tree_bytes", ok, msg)
        s = state_copy(root, commit, tmp, "conv"); edit_csv_kv(s / "R_summary_A.csv", "convergence", "1")
        ok, msg = refuses(root, s, "did not converge"); expect("refuse_r_not_converged", ok, msg)
        s = state_copy(root, commit, tmp, "ver"); edit_csv_kv(s / "R_session.csv", "gllvmtmb_version", "0.8.0")
        ok, msg = refuses(root, s, "0.7.1"); expect("refuse_wrong_oracle_version", ok, msg)
        for key, bump, name in [("logLik", 2e-6, "logLik"), ("H2", 2e-6, "H2"), ("Psi", 2e-6, "Psi"), ("V_eta", 2e-6, "V_eta")]:
            s = state_copy(root, commit, tmp, "bar" + key)
            jr = json.loads((s / "julia-results.json").read_text())
            if key == "logLik":
                jr["matched"][key] += bump
            else:
                jr["matched"][key][0] += bump
            (s / "julia-results.json").write_text(json.dumps(jr))
            ok, msg = refuses(root, s, f"{name} (")
            expect(f"refuse_matched_{key}_outside_bar", ok, msg)
        s = state_copy(root, commit, tmp, "inside")  # 5e-7 is inside the 1e-6 bar: still verifies
        jr = json.loads((s / "julia-results.json").read_text()); jr["matched"]["logLik"] += 5e-7
        (s / "julia-results.json").write_text(json.dumps(jr))
        try:
            R.verify_run(root, s); inside = True
        except R.Refuse:
            inside = False
        expect("matched_diff_5e-7_inside_bar_verifies", inside)
        # contract edits
        cpath = root / R.LEDGER / "namespace-3-d5-batch-contract-p1.json"
        orig = cpath.read_text()
        for name, mut, frag in [
            ("contract_seed", lambda c: c.__setitem__("seed", 4402), "approved 4401"),
            ("contract_design_seed", lambda c: c["design"].__setitem__("seed", 4402), "approved 4401"),
            ("contract_bars", lambda c: c["bars"]["matched_point"].__setitem__("logLik", 1e-3), "approved bars"),
            ("contract_design_hash", lambda c: c["design"].__setitem__("dimensions", "9 species"), "design_sha256"),
        ]:
            c = json.loads(orig); mut(c); cpath.write_text(json.dumps(c))
            ok, msg = refuses(root, st, frag); expect(f"refuse_{name}_changed", ok, msg)
        cpath.write_text(orig)
        # run provenance
        s = state_copy(root, commit, tmp, "stale")
        (root / "src" / "m.jl").write_text("module M x = 1 end\n")
        git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qam", "src changed")
        ok, msg = refuses(root, s, "changed between the run commit and HEAD"); expect("refuse_julia_source_changed_since_run", ok, msg)
        s = state_copy(root, "0" * 40, tmp, "unrelated")
        ok, msg = refuses(root, s, "not an ancestor"); expect("refuse_run_commit_not_ancestor", ok, msg)

    # --- tamper the tracked receipt: --check refuses (re-derivation differs)
    with tempfile.TemporaryDirectory() as tmp:
        root, commit = make_root(tmp)
        st = root / R.LEDGER / "receipts/namespace/namespace-3-d5-p1"
        run = json.loads((st / "run-commit.json").read_text()); run["glvmodels_commit"] = commit
        (st / "run-commit.json").write_text(json.dumps(run))
        # rebuild the receipt and row from the retargeted run so the fixture is self-consistent
        v = R.verify_run(root, st)
        vt = f"{R.VERIFY_MARKER} seed={R.APPROVED_SEED} matched_diffs={json.dumps(v['diffs'])}\n"
        (st / "verify.txt").write_text(vt)
        rec = R.case_receipt(root, v, vt)
        P = R.paths(root)
        P["case"].write_text(R.dump(rec))
        R.check(root)
        expect("fixture_check_passes_on_self_consistent_copy", True)
        j = json.loads(P["case"].read_text()); j["comparison"]["cases"][0]["julia_value"][0] += 1e-3
        P["case"].write_text(json.dumps(j, indent=2) + "\n")
        try:
            R.check(root); ok = False
        except R.Refuse:
            ok = True
        expect("check_refuses_a_tampered_case_receipt", ok)

    # --- the real checker
    recs = row["evidence"]["receipt"] + row["evidence"]["registration_receipts"]
    with tempfile.TemporaryDirectory() as tmp:
        d = checker_tree(tmp, "bind", row, [case_path])
        out = checker(d, None)
        expect("checker_binds_the_row_with_the_tracked_receipt", "C1_MET" in out and "bound_numeric=1" in out, out)

        def push(j):
            j["comparison"]["cases"][0]["julia_value"][0] += 2e-6
            j["comparison"]["cases"][0]["max_abs_diff"] += 2e-6
        d = checker_tree(tmp, "tamper", row, [case_path], tamper=push)
        out = checker(d, None)
        expect("checker_does_not_count_one_number_outside_its_bar", "C1_NOT_MET" in out, out)

        def push_diff_only(j):
            j["comparison"]["cases"][0]["max_abs_diff"] = 0.5
        d = checker_tree(tmp, "tamper2", row, [case_path], tamper=push_diff_only)
        out = checker(d, None)
        expect("checker_does_not_count_a_recorded_diff_that_disagrees", "C1_NOT_MET" in out, out)

        def verdict_fail(j):
            j["verdict"] = "FAIL"
        d = checker_tree(tmp, "failverdict", row, [case_path], tamper=verdict_fail)
        out = checker(d, None)
        expect("checker_does_not_count_a_failed_verdict", "C1_NOT_MET" in out, out)

        d = checker_tree(tmp, "noreceipt", row, [case_path], include_receipt=False)
        out = checker(d, None)
        expect("checker_does_not_count_the_row_with_the_receipt_removed", "C1_NOT_MET" in out, out)

        # the registration form: rebuilt from the preserved fields, as the tool does
        reg = dict(row)
        reg["evidence"] = {"receipt": row["evidence"]["registration_receipts"], "tier": row["evidence"]["registration_tier"]}
        reg["executable_case_ids"] = row["evidence"]["registration_case_ids"]
        reg["evidence_tier"] = "registration"
        d = checker_tree(tmp, "registration", reg, row["evidence"]["registration_receipts"])
        out = checker(d, None)
        expect("registration_form_is_registration_only_not_bound", "C1_NOT_MET" in out and "registration_only=" in out
               and "registration_only=none" not in out, out)

    print("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())

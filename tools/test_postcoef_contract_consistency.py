"""Checks for the approved, P1-only POST-COEF accessor contract."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json"
P0 = ROOT / "docs/dev-log/core070/postfit-policy-batch-contract.json"
sys.path.insert(0, str(ROOT / "tools"))
import core070_verify_postfit_policy_batch as verifier

def healthy_fixture():
    return deepcopy(verifier.synthetic_report(json.loads(P1.read_text()))['cases']
                    ['CORE070-POSTFIT-COEF-EMPTY-NATIVE']['three_fit']['fit_health'])


def preliminary_health(test):
    """Bypass only the fit-health numerical-admission hold for this test.

    The real classifier never passes while held; non-health checks still run.
    """
    health = verifier.core070_postcoef_fit_health
    held = patch.object(health, 'classify', health._classify_recorded)
    held.start(); test.addCleanup(held.stop)


class PostcoefContract(unittest.TestCase):
    def test_depot_tree_hash_rejects_changes_but_ignores_generated_coverage(self):
        self.assertIsNotNone(importlib.util.find_spec("core070_postcoef_depot_check"))
        check = importlib.import_module("core070_postcoef_depot_check")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src/module.jl").write_text("x = 1\n")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "add", "src/module.jl"], check=True)
            expected = subprocess.check_output(["git", "-C", str(root), "write-tree"], text=True).strip()
            actual, excluded = check.source_tree_hash(root)
            self.assertEqual((actual, excluded), (expected, []))
            coverage = root / "src/module.jl.123.cov"
            coverage.write_text("        1 x = 1\n")
            actual, excluded = check.source_tree_hash(root)
            self.assertEqual(actual, expected)
            self.assertEqual(excluded, ["src/module.jl.123.cov"])
            coverage.write_text("        1 x = 999\n")
            with self.assertRaises(ValueError):
                check.source_tree_hash(root)
            coverage.write_text("        1 x = 1\n")
            (root / "src/legitimate.cov").write_text("not generated Julia coverage")
            self.assertNotEqual(check.source_tree_hash(root)[0], expected)
            (root / "src/legitimate.cov").unlink()
            (root / "src/link.jl.456.cov").symlink_to("module.jl")
            self.assertNotEqual(check.source_tree_hash(root)[0], expected)
            (root / "src/link.jl.456.cov").unlink()
            coverage.unlink()
            (root / "src/module.jl").write_text("x = 2\n")
            self.assertNotEqual(check.source_tree_hash(root)[0], expected)

    def test_depot_inventory_must_cover_every_manifest_tree(self):
        self.assertIsNotNone(importlib.util.find_spec("core070_postcoef_depot_check"))
        check = importlib.import_module("core070_postcoef_depot_check")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "module.jl").write_text("x = 1\n")
            digest, _ = check.source_tree_hash(root)
            uuid = "11111111-1111-1111-1111-111111111111"
            manifest = {"deps": {"Example": [{"uuid": uuid, "git-tree-sha1": digest}]}}
            records = [{"name": "Example", "uuid": uuid, "tree_hash": digest, "source": str(root)}]
            result = check.verify_manifest_sources(manifest, records)
            self.assertEqual(result["checked"], 1)
            with self.assertRaises(ValueError):
                check.verify_manifest_sources(manifest, [])
            with self.assertRaises(ValueError):
                check.verify_manifest_sources(manifest, records + records)
            (root / "module.jl").write_text("x = 2\n")
            with self.assertRaises(ValueError):
                check.verify_manifest_sources(manifest, records)

    def test_hashless_manifest_entries_must_be_installed_stdlibs(self):
        check = importlib.import_module("core070_postcoef_depot_check")
        with tempfile.TemporaryDirectory() as directory:
            stdlib = Path(directory) / "stdlib"
            package = stdlib / "TestStdlib"
            package.mkdir(parents=True)
            key = "22222222-2222-2222-2222-222222222222"
            (package / "Project.toml").write_text(f'name = "TestStdlib"\nuuid = "{key}"\nversion = "1.2.3"\n')
            manifest = {"deps": {"TestStdlib": [{"uuid": key, "version": "1.2.3"}]}}
            records = [{"name": "TestStdlib", "uuid": key, "tree_hash": None, "source": str(package)}]
            result = check.verify_manifest_sources(manifest, records, stdlib)
            self.assertEqual(result["stdlib_checked"], 1)
            bad = deepcopy(manifest)
            bad["deps"]["Unpinned"] = [{"uuid": "33333333-3333-3333-3333-333333333333"}]
            with self.assertRaises(ValueError):
                check.verify_manifest_sources(bad, records, stdlib)
            bad = deepcopy(records)
            bad[0]["source"] = str(Path(directory))
            with self.assertRaises(ValueError):
                check.verify_manifest_sources(manifest, bad, stdlib)

    def test_r_interpreter_matches_oracle_build(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "R"
            executable.write_bytes(b"runtime")
            receipt = {"r_executable_path": str(executable),
                       "r_executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
                       "r_version": "R version 4.6.1", "r_full_version": "R version 4.6.1 -- nickname",
                       "r_platform": "platform"}
            build = {"r_version": "R version 4.6.1 -- nickname"}
            verifier.check_postcoef_r_runtime(receipt, build)
            with self.assertRaises(ValueError):
                verifier.check_postcoef_r_runtime(receipt, {"r_version": "R version 4.6.0"})
            executable.write_bytes(b"different")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_r_runtime(receipt, build)

    def test_runtime_evidence_binds_launcher_environment_and_children(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo, state = base / "repo", base / "state"
            repo.mkdir()
            (repo / "tools").mkdir()
            (state / "runner-snapshots").mkdir(parents=True)
            project, manifest = repo / "Project.toml", repo / "Manifest.toml"
            executable, preload = base / "julia-dist/bin/julia", base / "libunwind.so"
            executable.parent.mkdir(parents=True)
            stdlib_root = base / "julia-dist/share/julia/stdlib/v1.10"
            stdlib_root.mkdir(parents=True)
            launcher = executable
            python_executable = base / "python"
            checker = repo / "tools/core070_postcoef_depot_check.py"
            package = base / "package"
            package.mkdir()
            (package / "module.jl").write_text("x = 1\n")
            depot = importlib.import_module("core070_postcoef_depot_check")
            tree, _ = depot.source_tree_hash(package)
            package_uuid = "11111111-1111-1111-1111-111111111111"
            manifest_text = f'[[deps.Example]]\nuuid = "{package_uuid}"\ngit-tree-sha1 = "{tree}"\n'
            for path, value in ((project, "project"), (manifest, manifest_text), (checker, "checker"),
                                (python_executable, "python"),
                                (executable, "julia"), (preload, "libunwind")):
                path.write_text(value)
            (state / "runner-snapshots/julia-launcher").write_text("julia")
            (state / "julia-Manifest.toml").write_text(manifest_text)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            inventory = [{"name": "Example", "uuid": package_uuid, "tree_hash": tree, "source": str(package)}]
            manifest_data = {"deps": {"Example": [{"uuid": package_uuid, "git-tree-sha1": tree}]}}
            source_summary = json.dumps(depot.source_summary(depot.verify_manifest_sources(manifest_data, inventory)),
                                        sort_keys=True)
            runtime = {"executable_path": str(executable), "executable_sha256": digest(executable),
                       "project_sha256": digest(project), "manifest_sha256": digest(manifest),
                       "preferences_sha256": "absent", "load_path": "@:@stdlib", "depot_path": "depot",
                       "stdlib_root": str(stdlib_root),
                       "depot_checker_sha256": digest(checker), "python_executable_path": str(python_executable),
                       "python_executable_sha256": digest(python_executable),
                       "depot_inventory": inventory, "depot_check": source_summary,
                       "preloads": [{"path": str(preload), "sha256": digest(preload)}], "julia_version": "1.10.12"}
            receipt = {"postcoef_runtime": runtime, "julia_launcher_path": str(launcher),
                       "julia_launcher_sha256": digest(launcher)}
            report = {"postcoef_runtime": deepcopy(runtime)}
            controls = {mode: deepcopy(report) for mode in ("always_empty", "spurious_coefficient")}
            verifier.check_postcoef_runtime(repo, state, receipt, report, controls)
            wrapper = base / "wrapper"
            wrapper.write_text("wrapper")
            bad = dict(receipt, julia_launcher_path=str(wrapper), julia_launcher_sha256=digest(wrapper))
            (state / "runner-snapshots/julia-launcher").write_text("wrapper")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_runtime(repo, state, bad, report, controls)
            (state / "runner-snapshots/julia-launcher").write_text("julia")
            (repo / "Manifest-v1.10.toml").write_text("unrecorded")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_runtime(repo, state, receipt, report, controls)
            (repo / "Manifest-v1.10.toml").unlink()
            controls["always_empty"]["postcoef_runtime"]["load_path"] = "other"
            with self.assertRaises(ValueError):
                verifier.check_postcoef_runtime(repo, state, receipt, report, controls)
            controls["always_empty"] = deepcopy(report)
            bad_runtime = deepcopy(runtime)
            bad_runtime["depot_check"] = "{}"
            bad_receipt = dict(receipt, postcoef_runtime=bad_runtime)
            bad_report = {"postcoef_runtime": deepcopy(bad_runtime)}
            bad_controls = {mode: deepcopy(bad_report) for mode in controls}
            with self.assertRaises(ValueError):
                verifier.check_postcoef_runtime(repo, state, bad_receipt, bad_report, bad_controls)
            (package / "module.jl").write_text("x = 2\n")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_runtime(repo, state, receipt, report, controls)
            (package / "module.jl").write_text("x = 1\n")
            preload.write_text("changed")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_runtime(repo, state, receipt, report, controls)

    def test_committed_capture_files_reject_unreviewed_runner_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo, state = base / "repo", base / "state"
            (repo / "tools").mkdir(parents=True)
            (repo / "docs").mkdir()
            (state / "runner-snapshots").mkdir(parents=True)
            for name in ("core070_postfit_policy_batch.R", "core070_postfit_policy_batch.jl"):
                (repo / "tools" / name).write_text(name)
                (state / "runner-snapshots" / name).write_text(name)
            (repo / "tools/core070_verify_postfit_policy_batch.py").write_text("verifier")
            for name in ("parity_oracle.py", "core070_source_pin_check.py", "core070_source_pin.R", "core070_oracle_pins.toml"):
                (repo / "tools" / name).write_text(name)
            (repo / "tools/core070_postcoef_depot_check.py").write_text("depot checker")
            health_helpers = ("core070_json_semantics.py", "core070_postcoef_fit_health.py", "core070_postcoef_fit_health.R",
                              "core070_postcoef_fit_health.jl")
            for name in health_helpers:
                (repo / "tools" / name).write_text(name)
            (repo / "Project.toml").write_text("project")
            (repo / "docs" / "contract.json").write_text("{}")
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "tools", "docs/contract.json", "Project.toml"], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                            "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
            commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            verifier.check_postcoef_committed_files(repo, state, commit, "docs/contract.json")
            for name in health_helpers:
                (repo / "tools" / name).write_text("tampered")
                with self.subTest(name=name), self.assertRaises(ValueError):
                    verifier.check_postcoef_committed_files(repo, state, commit, "docs/contract.json")
                (repo / "tools" / name).write_text(name)
            (repo / "tools/core070_verify_postfit_policy_batch.py").write_text("changed")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_committed_files(repo, state, commit, "docs/contract.json")
            (repo / "tools/core070_verify_postfit_policy_batch.py").write_text("verifier")
            (state / "runner-snapshots/core070_postfit_policy_batch.jl").write_text("changed")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_committed_files(repo, state, commit, "docs/contract.json")

    def test_oracle_build_attestation_checks_builder_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo, state, build = base / "repo", base / "state", base / "build"
            (repo / "tools").mkdir(parents=True)
            state.mkdir()
            (build / "library").mkdir(parents=True)
            builder = repo / "tools/core070_build_oracle.py"
            builder.write_text('print("CORE070_ORACLE_VERIFY_PASS")\n')
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "tools/core070_build_oracle.py"], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                            "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
            commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            (build / "build.json").write_text(json.dumps({"r_version": "R test -- nickname"}))
            r_executable = base / "R"
            r_executable.write_text("R")
            for name in ("oracle-verify-preload.log", "oracle-verify-before.log", "oracle-verify-after.log"):
                (state / name).write_text("CORE070_ORACLE_VERIFY_PASS\n")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            receipt = {"postcoef_capture_identity": {"source_commit": commit},
                       "frozen_library": str(build / "library"),
                       "oracle_build_sha256": digest(build / "build.json"),
                       "oracle_builder_sha256": digest(builder),
                       "r_executable_path": str(r_executable), "r_executable_sha256": digest(r_executable),
                       "r_version": "R test", "r_full_version": "R test -- nickname", "r_platform": "test",
                       "oracle_verify_preload_sha256": digest(state / "oracle-verify-preload.log"),
                       "oracle_verify_before_sha256": digest(state / "oracle-verify-before.log"),
                       "oracle_verify_after_sha256": digest(state / "oracle-verify-after.log")}
            verifier.check_postcoef_oracle_build(repo, state, receipt)
            preload = state / "oracle-verify-preload.log"
            preload.write_text("tampered\n")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_oracle_build(repo, state, receipt)
            preload.write_text("CORE070_ORACLE_VERIFY_PASS\n")
            (build / "build.json").write_text("changed")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_oracle_build(repo, state, receipt)

    def test_retained_outputs_bind_reports_logs_exits_and_library(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            lib = Path(directory) / "library"
            (state / "runner-snapshots").mkdir(parents=True)
            lib.mkdir()
            dll = lib / "gllvmTMB.so"
            dll.write_bytes(b"library")
            identity = {"source_commit": "a" * 40, "source_tree": "b" * 40,
                        "r_oracle_sha256": "c" * 64, "julia_runner_sha256": "d" * 64}
            def put(name, value):
                path = state / name
                path.write_text(value)
                return hashlib.sha256(path.read_bytes()).hexdigest()
            receipt = {"postcoef_capture_identity": identity, "frozen_library": str(lib),
                       "r_library_path": str(dll), "r_library_sha256": hashlib.sha256(dll.read_bytes()).hexdigest(),
                       "r_runner_sha256": put("runner-snapshots/core070_postfit_policy_batch.R", "R runner"),
                       "julia_results_sha256": put("julia-results.json", json.dumps({"postcoef_capture_identity": identity})),
                       "julia_stdout_sha256": put("julia-stdout.log", "ok"),
                       "julia_stderr_sha256": put("julia-stderr.log", ""), "postcoef_controls": {}}
            for mode in ("always_empty", "spurious_coefficient"):
                prefix = "julia-control-" + mode
                receipt["postcoef_controls"][mode] = {
                    "mode": mode, "exit_code": 1,
                    "report_sha256": put(prefix + ".json", json.dumps({"postcoef_capture_identity": identity})),
                    "stdout_sha256": put(prefix + ".stdout.log", "failed"),
                    "stderr_sha256": put(prefix + ".stderr.log", ""),
                    "exit_sha256": put(prefix + ".exit", "1\n")}
            verifier.check_postcoef_retained_outputs(state, receipt)
            bad = dict(receipt)
            bad.pop("postcoef_capture_identity")
            bad["julia_results_sha256"] = put("julia-results.json", "{}")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_retained_outputs(state, bad)
            put("julia-results.json", json.dumps({"postcoef_capture_identity": identity}))
            (state / "julia-control-always_empty.exit").write_text("0\n")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_retained_outputs(state, receipt)
            (state / "julia-control-always_empty.exit").write_text("1\n")
            dll.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_retained_outputs(state, receipt)

    def test_capture_identity_binds_files_and_git_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            repo, state = base / "repo", base / "state"
            (repo / "src").mkdir(parents=True)
            (state / "runner-snapshots").mkdir(parents=True)
            (repo / "src" / "source.jl").write_text("x = 1\n")
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "src/source.jl"], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                            "user.email=test@example.invalid", "commit", "-qm", "fixture"], check=True)
            commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            tree = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD:src"], text=True).strip()
            oracle = state / "r-oracle.json"
            runner = state / "runner-snapshots" / "core070_postfit_policy_batch.jl"
            oracle.write_text("{}")
            runner.write_text("runner")
            identity = {"source_commit": commit, "source_tree": tree,
                        "r_oracle_sha256": hashlib.sha256(oracle.read_bytes()).hexdigest(),
                        "julia_runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest()}
            verifier.check_postcoef_capture_artifacts(state, repo, identity)
            runner.write_text("altered")
            with self.assertRaises(ValueError):
                verifier.check_postcoef_capture_artifacts(state, repo, identity)
            runner.write_text("runner")
            wrong_tree = dict(identity, source_tree="0" * len(tree))
            with self.assertRaises(ValueError):
                verifier.check_postcoef_capture_artifacts(state, repo, wrong_tree)

    def test_control_run_must_use_the_mutant_and_fail(self):
        real = {"selected_accessor": "real", "r_lengths": [0, 5, 6],
                "julia_lengths": [0, 5, 6], "r_design_columns": [0, 5, 6],
                "r_values": [[], [0.0] * 5, [0.0] * 6], "r_names": [None] * 3,
                "julia_names": [None] * 3,
                "control_lengths": {"always_empty": [0, 0, 0],
                                    "spurious_coefficient": [1, 6, 7]}}
        real['fit_health'] = healthy_fixture()
        mutant = deepcopy(real)
        mutant["selected_accessor"] = "always_empty"
        mutant["julia_lengths"] = [0, 0, 0]
        identity = {"source_commit": "a" * 40, "source_tree": "b" * 40,
                    "r_oracle_sha256": "c" * 64, "julia_runner_sha256": "d" * 64}
        contract = json.loads(P1.read_text())
        report = verifier.synthetic_report(contract)
        report.update(status="FAIL", all_checks=False, all_positive_pass=False)
        report["cases"]["CORE070-POSTFIT-COEF-EMPTY-NATIVE"] = {"pass": False, "three_fit": mutant}
        with self.assertRaisesRegex(ValueError, "fit health"):
            verifier.check_postcoef_control_outcome(real, report, "always_empty", 1, identity, identity, contract)
        preliminary_health(self)
        verifier.check_postcoef_control_outcome(real, report, "always_empty", 1, identity, identity, contract)
        with self.assertRaises(ValueError):
            verifier.check_postcoef_control_outcome(real, report, "always_empty", 0, identity, identity, contract)
        broken = deepcopy(report)
        broken["cases"]["CORE070-POSTFIT-COEF-EMPTY-NATIVE"]["pass"] = True
        with self.assertRaises(ValueError):
            verifier.check_postcoef_control_outcome(real, broken, "always_empty", 1, identity, identity, contract)
        broken = deepcopy(report)
        broken["cases"].pop(next(key for key in broken["cases"] if key != "CORE070-POSTFIT-COEF-EMPTY-NATIVE"))
        with self.assertRaises(ValueError):
            verifier.check_postcoef_control_outcome(real, broken, "always_empty", 1, identity, identity, contract)
        broken = deepcopy(report)
        broken["scope"] = "wrong"
        with self.assertRaises(ValueError):
            verifier.check_postcoef_control_outcome(real, broken, "always_empty", 1, identity, identity, contract)
        other = dict(identity, source_commit="e" * 40)
        with self.assertRaises(ValueError):
            verifier.check_postcoef_control_outcome(real, report, "always_empty", 1, identity, other, contract)
        mutant["r_lengths"] = [0, 0, 0]
        with self.assertRaises(ValueError):
            verifier.check_postcoef_control_outcome(real, report, "always_empty", 1, identity, identity, contract)
        mutant = deepcopy(real)
        mutant["selected_accessor"] = "spurious_coefficient"
        mutant["julia_lengths"] = [1, 6, 7]
        report["cases"]["CORE070-POSTFIT-COEF-EMPTY-NATIVE"]["three_fit"] = mutant
        verifier.check_postcoef_control_outcome(real, report, "spurious_coefficient", 1, identity, identity, contract)

    def test_report_matches_the_retained_r_oracle(self):
        fits = [{"length": n, "fixed_columns": n, "values": [0.0] * n,
                 "names": None} for n in (0, 5, 6)]
        fit_health = healthy_fixture()
        for fit, health_row in zip(fits, fit_health['R']):
            fit['fit_health'] = deepcopy(health_row)
        oracle = {"postcoef_three_fit": {"fits": fits}}
        evidence = {"r_lengths": [0, 5, 6], "r_design_columns": [0, 5, 6],
                    "r_values": [[], [0.0] * 5, [0.0] * 6],
                    "r_names": [None, None, None], "fit_health": fit_health}
        verifier.check_postcoef_r_oracle_binding(evidence, oracle)
        bad = deepcopy(evidence)
        bad["r_values"][1][0] = 7.0
        with self.assertRaises(ValueError):
            verifier.check_postcoef_r_oracle_binding(bad, oracle)
        bad = deepcopy(oracle)
        bad["postcoef_three_fit"]["fits"].pop()
        with self.assertRaises(ValueError):
            verifier.check_postcoef_r_oracle_binding(evidence, bad)
        bad = deepcopy(oracle)
        bad["postcoef_three_fit"]["fits"][0]["length"] = False
        with self.assertRaises(ValueError):
            verifier.check_postcoef_r_oracle_binding(evidence, bad)
        bad = deepcopy(oracle)
        bad["postcoef_three_fit"]["fits"][1]["values"][0] = False
        with self.assertRaises(ValueError):
            verifier.check_postcoef_r_oracle_binding(evidence, bad)

    def test_three_fit_evidence_rejects_bad_lengths_and_controls(self):
        evidence = {
            "selected_accessor": "real",
            "r_lengths": [0, 5, 6], "julia_lengths": [0, 5, 6],
            "r_design_columns": [0, 5, 6],
            "r_values": [[], [0.0] * 5, [0.0] * 6],
            "r_names": [[], ["t1", "t2", "t3", "t4", "t5"],
                        ["t1", "t2", "t3", "t4", "t5", "x"]],
            "julia_names": [None, None, None],
            "control_lengths": {"always_empty": [0, 0, 0],
                                "spurious_coefficient": [1, 6, 7]},
            "fit_health": healthy_fixture(),
        }
        with self.assertRaisesRegex(ValueError, "fit health"):
            verifier.check_postcoef_three_fit_evidence(evidence)
        preliminary_health(self)
        verifier.check_postcoef_three_fit_evidence(evidence)
        absent_names = deepcopy(evidence)
        absent_names["r_names"] = [None, None, None]
        verifier.check_postcoef_three_fit_evidence(absent_names)
        mutations = [
            lambda x: x.update(selected_accessor="always_empty"),
            lambda x: x.update(r_lengths=[0, 0, 0]),
            lambda x: x.update(julia_lengths=[0, 5, 5]),
            lambda x: x.update(r_design_columns=[0, 4, 6]),
            lambda x: x.update(r_values=[[], [0.0], [0.0] * 6]),
            lambda x: x["r_values"][1].__setitem__(0, None),
            lambda x: x.update(r_names=[[], []]),
            lambda x: x.update(r_names=[["bad"], [], []]),
            lambda x: x.pop("julia_names"),
            lambda x: x["control_lengths"].update(always_empty=[0, 5, 6]),
            lambda x: x["control_lengths"].update(spurious_coefficient=[1, 5, 7]),
            lambda x: x["control_lengths"].pop("always_empty"),
        ]
        for mutate in mutations:
            bad = deepcopy(evidence)
            mutate(bad)
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                verifier.check_postcoef_three_fit_evidence(bad)

    def test_three_public_fits_and_controls_are_required(self):
        contract = json.loads(P1.read_text())
        case = next(c for c in contract["cases"] if c["case_id"] ==
                    "CORE070-POSTFIT-COEF-EMPTY-NATIVE")
        comparison = case.get("discriminating_comparison", {})
        self.assertEqual(comparison.get("implementation_status"), "pending_runner_and_receipt_verification")
        self.assertEqual(comparison.get("expected_lengths"), [0, 5, 6])
        self.assertEqual(comparison.get("length_tolerance"), 0)
        self.assertEqual(comparison.get("fit_designs"),
                         ["no_fixed_effects", "trait_intercepts", "trait_intercepts_and_shared_x"])
        self.assertEqual(comparison.get("accessor_controls"), ["always_empty", "spurious_coefficient"])
        self.assertEqual(comparison.get("name_comparison"), "record_separately")

    def test_frozen_p0_ancestry_is_unchanged(self):
        contract = json.loads(P1.read_text())
        self.assertEqual(contract["p0_contract_sha256"], hashlib.sha256(P0.read_bytes()).hexdigest())





if __name__ == "__main__":
    unittest.main()

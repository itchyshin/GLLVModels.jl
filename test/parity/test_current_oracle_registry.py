import hashlib
from copy import deepcopy
import json
import os
import subprocess
from pathlib import Path
import re
import runpy
import shutil
import tempfile
import tomllib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(str(ROOT / "tools/core070_family_p1_receipts.py"))
ORACLE = ROOT / "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle"


class CurrentOracleRegistry(unittest.TestCase):
    def test_committed_source_ignores_replacement_refs_and_ambient_git_dir(self):
        check = MODULE['matches_committed_oracle_source']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(['git','-C',str(root),*args], check=True, capture_output=True, text=True).stdout
            git('init','-q'); git('config','user.name','fixture'); git('config','user.email','fixture@example.invalid')
            (root / 'initial').write_text('initial'); (root / 'source.jl').write_text('original source')
            git('add','initial','source.jl'); git('commit','-qm','initial')
            original = git('rev-parse','HEAD').strip()
            rel = MODULE['ORACLE_SOURCE']; path = root / rel; path.parent.mkdir(parents=True); path.write_text('source receipt')
            (root / 'source.jl').write_text('changed source')
            git('add',rel,'source.jl'); git('commit','-qm','receipt'); descendant = git('rev-parse','HEAD').strip()
            git('replace',original,descendant)
            source = {'oracle_source_receipt_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch.dict(check.__globals__, {'ROOT':root}):
                self.assertFalse(check(source, commit=original))
                drift = MODULE['harness_drift']
                run = {'execution': {'entries': [{'path':'source.jl', 'sha256':hashlib.sha256(b'original source').hexdigest()}]}}
                self.assertEqual(drift(run, original), [])
                with patch.dict(os.environ, {'GIT_DIR':'/invalid/foreign/repo','GIT_NO_REPLACE_OBJECTS':'0'}):
                    self.assertTrue(check(source, commit=descendant))
                    self.assertFalse(check(source, commit=original))
                    self.assertEqual(drift(run, original), [])

    def test_family_verifier_child_receives_sanitized_git_environment(self):
        with patch.dict(os.environ, {'GIT_DIR':'/invalid/foreign/repo', 'GIT_NO_REPLACE_OBJECTS':'0'}):
            with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')) as call:
                MODULE['run_external']('family-links-p1', Path('/fixture'))
            env = call.call_args.kwargs['env']
            self.assertNotIn('GIT_DIR', env)
            self.assertEqual(env['GIT_NO_REPLACE_OBJECTS'], '1')
            self.assertEqual(env['GLLVM_PARITY_PIN'], 'P1')

    def test_original_receipt_preserved_and_new_receipt_registered(self):
        old = ORACLE / "build-totoro.json"
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),
                         "6aaf50b2b2f89c257494300cd3c25c3345f0c50d45092d5899f1e8c9bd65ad4f")
        new = ORACLE / "build-totoro-20261104.json"
        rel = str(new.relative_to(ROOT))
        self.assertIn(rel, MODULE["ORACLE_BUILDS"])
        build = json.loads(new.read_text())
        pin = MODULE["PINS"]["P1"]
        self.assertEqual(build["reference_commit"], pin["reference_commit"])
        self.assertEqual(build["gllvmTMB_version"], pin["version"])
        self.assertNotEqual(build["installed_tree_sha256"], json.loads(old.read_text())["installed_tree_sha256"])

    def test_build_identity_requires_matching_receipt_tree_and_marker(self):
        new = ORACLE / "build-totoro-20261104.json"
        build = json.loads(new.read_text())
        source = {
            "oracle_build_receipt_sha256": hashlib.sha256(new.read_bytes()).hexdigest(),
            "installed_tree_sha256": build["installed_tree_sha256"],
            "source_marker_sha256": build["marker_sha256"],
        }
        check = MODULE["matches_registered_oracle_build"]
        self.assertTrue(check(source))
        for key in source:
            with self.subTest(key=key):
                self.assertFalse(check(source | {key: "0" * 64}))
        # The fresh receipt exists only as an untracked candidate file, so the
        # parent commit cannot certify it even though its bytes are registered.
        self.assertFalse(check(source, commit="cac9657fb8b493f03e2d7dccdfc4499886725a9b"))

    def test_historical_receipt_is_bound_to_its_recorded_commit(self):
        run = ROOT / "docs/dev-log/core070/true-parity-latest/receipts/family/runparity-gaussian/run.toml"
        source = tomllib.loads(run.read_text())["source"]
        self.assertTrue(MODULE["matches_registered_oracle_build"](
            source, commit="4b78fa01245381f6c0ab8a8b9ea0bc581ade825a"))
        source_check = MODULE["matches_committed_oracle_source"]
        self.assertTrue(source_check(source, commit="4b78fa01245381f6c0ab8a8b9ea0bc581ade825a"))
        self.assertFalse(source_check(source, commit="374b58a1dbbb4295702b1f92fcf670fc16a7dcf1"))
        self.assertFalse(source_check(source | {"oracle_source_receipt_sha256": "0" * 64},
                                      commit="4b78fa01245381f6c0ab8a8b9ea0bc581ade825a"))

    def test_batch_check_rejects_tampered_installed_tree(self):
        original = ROOT / "docs/dev-log/core070/true-parity-latest/receipts/family/runparity-gaussian"
        check = MODULE["runparity_checks"]
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "runparity-gaussian"
            shutil.copytree(original, copied)
            with patch.dict(check.__wrapped__.__globals__, {"batch_rel": lambda _: str(copied)}):
                check.cache_clear()
                baseline = dict(check("runparity-gaussian"))
                label = "oracle build/source receipts are tracked P1 receipts (a registered build, the pin's source receipt)"
                self.assertTrue(baseline[label])
                harness = next(key for key in baseline if key.startswith("harness files in the execution inventory"))
                self.assertTrue(baseline[harness])
                commit_path = copied / "run-commit.json"
                original_commit = commit_path.read_text()
                altered = json.loads(original_commit)
                altered["dirty"] = ["src/fit.jl"]
                commit_path.write_text(json.dumps(altered))
                check.cache_clear()
                self.assertFalse(dict(check("runparity-gaussian"))[harness])
                commit_path.write_text(original_commit)
                run_path = copied / "run.toml"
                text = run_path.read_text()
                old = tomllib.loads(text)["source"]["installed_tree_sha256"]
                self.assertEqual(text.count(old), 1)
                run_path.write_text(text.replace(old, "0" * 64))
                check.cache_clear()
                self.assertFalse(dict(check("runparity-gaussian"))[label])
        check.cache_clear()

    def test_mixed_builds_rejected_before_existing_output_is_touched(self):
        main = MODULE["main"]
        glob = main.__globals__
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            runs = Path(tmp) / "runs"
            root.mkdir()
            runs.mkdir()
            for rel in MODULE["ORACLE_BUILDS"]:
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / rel, dest)
            hashes = [hashlib.sha256((root / rel).read_bytes()).hexdigest()
                      for rel in MODULE["ORACLE_BUILDS"]]
            for index, batch in enumerate(MODULE["RUNPARITY_RUNS"]):
                directory = runs / batch
                directory.mkdir()
                digest = hashes[index == 0]
                (directory / "run.toml").write_text(
                    f'[source]\noracle_build_receipt_sha256 = "{digest}"\n')
            existing = root / MODULE["REC_REL"] / "cases" / "keep.txt"
            existing.parent.mkdir(parents=True)
            existing.write_text("historical evidence")
            runtimes = Path(tmp) / "runtimes.json"
            runtimes.write_text("{}")
            changes = {"ROOT": root, "git_state": lambda: ("head", []),
                       "check_run_commit": lambda *_: None,
                       "copy_batch": lambda *_: self.fail("copy started before build validation")}
            with patch.dict(glob, changes), patch("sys.argv", ["tool", "--runs", str(runs), "--runtimes", str(runtimes)]):
                with self.assertRaisesRegex(SystemExit, "one registered oracle build"):
                    main()
            self.assertEqual(existing.read_text(), "historical evidence")

    def test_unreviewed_inventory_routine_fails_closed(self):
        batch = "runparity-gaussian"
        d = ROOT / MODULE["batch_rel"](batch)
        run = tomllib.loads((d / "run.toml").read_text())
        cells = {cid: tomllib.loads((d / f"cell-{cid}.toml").read_text())
                 for cid in MODULE["RUNPARITY_RUNS"][batch]}
        commit = json.loads((d / "run-commit.json").read_text())["glvmodels_commit"]
        check = MODULE["execution_inventory_intact"]
        self.assertTrue(check(run, commit, cells))
        original_git = check.__globals__["git"]

        def altered_git(*args, **kwargs):
            result = original_git(*args, **kwargs)
            if args[0] == "show" and args[1].endswith(":test/parity/parity_helpers.jl"):
                return SimpleNamespace(returncode=0, stdout=result.stdout.replace(
                    b"paths = String[", b"# changed inventory routine\n    paths = String[", 1))
            return result

        with patch.dict(check.__globals__, {"git": altered_git}):
            self.assertFalse(check(run, commit, cells))

    def test_source_tree_digest_must_match_inventory(self):
        batch = "runparity-gaussian"
        d = ROOT / MODULE["batch_rel"](batch)
        run = tomllib.loads((d / "run.toml").read_text())
        cells = {cid: tomllib.loads((d / f"cell-{cid}.toml").read_text())
                 for cid in MODULE["RUNPARITY_RUNS"][batch]}
        commit = json.loads((d / "run-commit.json").read_text())["glvmodels_commit"]
        check = MODULE["execution_inventory_intact"]
        self.assertTrue(check(run, commit, cells))
        run["source"]["julia_source_tree_sha256"] = "0" * 64
        self.assertFalse(check(run, commit, cells))

    def test_tracked_optional_manifest_must_be_in_the_inventory(self):
        batch = "runparity-gaussian"
        d = ROOT / MODULE["batch_rel"](batch)
        run = tomllib.loads((d / "run.toml").read_text())
        cells = {cid: tomllib.loads((d / f"cell-{cid}.toml").read_text())
                 for cid in MODULE["RUNPARITY_RUNS"][batch]}
        commit = json.loads((d / "run-commit.json").read_text())["glvmodels_commit"]
        check = MODULE["execution_inventory_intact"]
        original_git = check.__globals__["git"]

        def manifest_tracked(*args, **kwargs):
            if args[0] == "ls-tree" and "test/Manifest.toml" in args:
                result = original_git(*args, **kwargs)
                return SimpleNamespace(returncode=result.returncode,
                                       stdout=result.stdout + b"100644 blob " + b"a" * 40 + b"\ttest/Manifest.toml\0")
            return original_git(*args, **kwargs)

        with patch.dict(check.__globals__, {"git": manifest_tracked}):
            self.assertFalse(check(run, commit, cells))
            with_manifest = deepcopy(run)
            execution = with_manifest["execution"]
            execution["entries"].append({"path": "test/Manifest.toml", "sha256": "a" * 64})
            rows = sorted(f'{e["path"]}\0{e["sha256"]}' for e in execution["entries"])
            execution["manifest_sha256"] = hashlib.sha256("\n".join(rows).encode()).hexdigest()
            self.assertTrue(check(with_manifest, commit, cells))

    def test_batch_check_rejects_rehashed_omission_of_required_inputs(self):
        original = ROOT / "docs/dev-log/core070/true-parity-latest/receipts/family/runparity-gaussian"
        check = MODULE["runparity_checks"]
        helper = (ROOT / "test/parity/parity_helpers.jl").read_text()
        block = helper.split("function _core070_execution_paths", 1)[1].split("paths = String[", 1)[1].split("]", 1)[0]
        fixed = set(re.findall(r'"([^"\\]*)"', block)) - {"src"}
        cases = [(name, False) for name in sorted(fixed | {MODULE["MANIFEST"], "src/fit.jl"})]
        cases += [("test/parity/test_gaussian_original_required.jl", False),
                  ("test/parity/test_gaussian_original_required.jl", True)]
        for omitted, spoof in cases:
          with self.subTest(omitted=omitted, spoof=spoof), tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp) / "runparity-gaussian"
            shutil.copytree(original, copied)
            run_path = copied / "run.toml"
            text = run_path.read_text()
            pattern = (r'(?m)^    \[\[execution\.entries\]\]\n    sha256 = "[0-9a-f]{64}"\n'
                       + '    path = "' + re.escape(omitted) + '"\n')
            altered, count = re.subn(pattern, "", text)
            self.assertEqual(count, 1)
            if spoof:
                altered = altered.replace(omitted, "test/parity/parity_helpers.jl")
            execution = tomllib.loads(altered)["execution"]
            rows = sorted(f'{e["path"]}\0{e["sha256"]}' for e in execution["entries"])
            digest = hashlib.sha256("\n".join(rows).encode()).hexdigest()
            old = execution["manifest_sha256"]
            run_path.write_text(altered.replace(old, digest))
            for cell in copied.glob("cell-*.toml"):
                body = cell.read_text()
                self.assertEqual(body.count(old), 1)
                if spoof:
                    body = body.replace(omitted, "test/parity/parity_helpers.jl")
                cell.write_text(body.replace(old, digest))
            with patch.dict(check.__wrapped__.__globals__, {"batch_rel": lambda _: str(copied)}):
                check.cache_clear()
                results = dict(check("runparity-gaussian"))
                harness = next(k for k in results if k.startswith("harness files in the execution inventory"))
                self.assertFalse(results[harness])
        check.cache_clear()


if __name__ == "__main__":
    unittest.main()

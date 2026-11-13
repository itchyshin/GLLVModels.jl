"""Guard integration of the P1 three-fit coefficient-length receipt."""
import unittest
import json
import tempfile
import os
import shutil
import hashlib
import subprocess
from pathlib import Path
from unittest.mock import patch
import core070_postfit_p1_receipts as receipt


class CoefficientReceiptTests(unittest.TestCase):
    def test_integer_source_routes_only_bound_target_to_dedicated_batch(self):
        legacy = Path('/evidence/postfit-policy-p1')
        dedicated = Path('/evidence/postcoef-empty-p1')
        binding = {'receipt': f'{receipt.REC_REL}/postcoef-empty-p1/receipt.json', 'sha256': 'a' * 64}
        prefix = f'{receipt.REC_REL}/postcoef-empty-p1'
        case = {'postcoef_capture': binding, 'raw': [f'{prefix}/julia-results.json', f'{prefix}/r-oracle.json'],
                'batch_verifier': {'log': f'{prefix}/verify.txt', 'log_sha256': 'a' * 64}}
        self.assertEqual(receipt.integer_capture_state(receipt.COEF_EMPTY_CID, case, legacy, dedicated), dedicated)
        bad = dict(case, raw=[f'{receipt.REC_REL}/postfit-policy-p1/julia-results.json', case['raw'][1]])
        with self.assertRaisesRegex(SystemExit, 'capture pointers'):
            receipt.integer_capture_state(receipt.COEF_EMPTY_CID, bad, legacy, dedicated)
        self.assertEqual(receipt.integer_capture_state(receipt.COEF_EMPTY_CID, {}, legacy, dedicated), legacy)
        other = next(cid for cid in receipt.INTEGER_EQUALITY if cid != receipt.COEF_EMPTY_CID)
        self.assertEqual(receipt.integer_capture_state(other, {}, legacy, dedicated), legacy)
        for bad in ({}, {'receipt': f'{receipt.REC_REL}/postfit-policy-p1/receipt.json', 'sha256': 'a' * 64},
                    {'receipt': binding['receipt'], 'sha256': 'bad'}):
            with self.subTest(binding=bad), self.assertRaisesRegex(SystemExit, 'capture binding'):
                receipt.integer_capture_state(receipt.COEF_EMPTY_CID, {'postcoef_capture': bad}, legacy, dedicated)
    def test_three_fit_binding_uses_dedicated_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp)
            (state / 'receipt.json').write_bytes(b'capture')
            result = receipt.postcoef_capture_binding(receipt.COEF_EMPTY_CID, {'three_fit': {}}, state)
            self.assertEqual(result, {'receipt': f'{receipt.REC_REL}/postcoef-empty-p1/receipt.json',
                                      'sha256': hashlib.sha256(b'capture').hexdigest()})
            self.assertIsNone(receipt.postcoef_capture_binding(receipt.COEF_EMPTY_CID, {}, state))

    def test_policy_verifier_child_receives_sanitized_git_environment(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(receipt, "REC", Path(tmp)):
            with patch.dict(os.environ, {"GIT_DIR": "/invalid", "GIT_NO_REPLACE_OBJECTS": "0"}):
                with patch.object(receipt.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
                    receipt.run_verifier("postfit-policy-p1", state=tmp)
                env = run.call_args.kwargs["env"]
                self.assertNotIn("GIT_DIR", env)
                self.assertEqual(env["GIT_NO_REPLACE_OBJECTS"], "1")
                self.assertEqual(env["GLLVM_PARITY_PIN"], "P1")

    def test_full_ingestion_stops_before_writes_until_copy_is_transactional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); runs = root / 'runs'; raw = runs / 'postfit-policy-p1'; raw.mkdir(parents=True)
            (raw / 'julia-results.json').write_text(json.dumps({'cases': {receipt.COEF_EMPTY_CID: {'three_fit': {}}}}))
            runtimes = root / 'runtimes.json'; runtimes.write_text('{}')
            sentinel = root / 'sentinel'; sentinel.write_text('unchanged')
            with patch('sys.argv', ['tool', '--runs', str(runs), '--runtimes', str(runtimes)]):
                with patch.object(receipt, 'git_state', side_effect=AssertionError('preflight wrote or continued')):
                    with self.assertRaisesRegex(SystemExit, 'transactional'):
                        receipt.main()
            self.assertEqual(sentinel.read_text(), 'unchanged')

    def test_integer_overlay_stops_before_any_write_for_three_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Path(tmp); raw = rec / 'postfit-policy-p1'; raw.mkdir()
            (raw / 'julia-results.json').write_text(json.dumps({'cases': {receipt.COEF_EMPTY_CID: {'three_fit': {}}}}))
            (raw / 'r-oracle.json').write_text('{}')
            (raw / '.ignored-owner-evidence').write_bytes(b'private fixture bytes')
            (raw / '.ignored-owner-evidence').chmod(0o640)
            before = {p.relative_to(raw): (p.stat().st_mode, p.read_bytes())
                      for p in raw.rglob('*') if p.is_file()}
            sentinel = rec / 'sentinel'; sentinel.write_text('unchanged')
            with patch.object(receipt, 'REC', rec), patch.object(receipt, 'write_json', side_effect=AssertionError('write attempted')):
                with self.assertRaisesRegex(SystemExit, 'transactional'):
                    receipt.apply_integer_equality()
            self.assertEqual(sentinel.read_text(), 'unchanged')
            self.assertEqual({p.relative_to(raw): (p.stat().st_mode, p.read_bytes())
                              for p in raw.rglob('*') if p.is_file()}, before)

    def test_three_fit_copy_stops_before_creating_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir(); output = root / 'output'
            (source / 'julia-results.json').write_text(json.dumps({'cases': {receipt.COEF_EMPTY_CID: {'three_fit': {}}}}))
            with patch.object(receipt, 'REC', output):
                with self.assertRaisesRegex(SystemExit, 'transactional'):
                    receipt.copy_batch(source, 'postfit-policy-p1', ['julia-results.json'])
            self.assertFalse(output.exists())
            for alias in ('postfit-policy-p1/.', 'postfit-policy-p1/../postfit-policy-p1', '../outside'):
                with self.subTest(alias=alias), patch.object(receipt, 'REC', output):
                    with self.assertRaisesRegex(SystemExit, 'canonical'):
                        receipt.copy_batch(source, alias, ['julia-results.json'])
                self.assertFalse(output.exists())
            with patch.object(receipt, 'REC', output):
                with self.assertRaisesRegex(SystemExit, 'canonical relative'):
                    receipt.copy_batch(source, 'other', ['../postfit-policy-p1/julia-results.json'])
            self.assertFalse(output.exists())

    def test_batch_copy_rejects_symlinked_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir(); (source / 'result').write_text('new')
            output = root / 'output'; output.mkdir(); policy = output / 'postfit-policy-p1'; policy.mkdir()
            sentinel = policy / 'result'; sentinel.write_text('old')
            (output / 'other').symlink_to(policy, target_is_directory=True)
            with patch.object(receipt, 'REC', output):
                with self.assertRaisesRegex(SystemExit, 'symlink'):
                    receipt.copy_batch(source, 'other', ['result'])
            self.assertEqual(sentinel.read_text(), 'old')
            (output / 'other').unlink(); (output / 'other').mkdir()
            (output / 'other' / 'nested').symlink_to(policy, target_is_directory=True)
            (source / 'nested').mkdir(); (source / 'nested' / 'result').write_text('new')
            with patch.object(receipt, 'REC', output):
                with self.assertRaisesRegex(SystemExit, 'symlink'):
                    receipt.copy_batch(source, 'other', ['nested/result'])
            self.assertEqual(sentinel.read_text(), 'old')
            (output / 'other' / 'result').symlink_to(sentinel)
            with patch.object(receipt, 'REC', output):
                with self.assertRaisesRegex(SystemExit, 'symlink'):
                    receipt.copy_batch(source, 'other', ['result'])
            self.assertEqual(sentinel.read_text(), 'old')

    def test_batch_copy_preflights_all_files_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            (source / 'first').write_text('new first'); (source / 'bad').symlink_to('first')
            output = root / 'output'; target = output / 'other'; target.mkdir(parents=True)
            (target / 'first').write_text('old first')
            with patch.object(receipt, 'REC', output):
                with self.assertRaisesRegex(SystemExit, 'symlink'):
                    receipt.copy_batch(source, 'other', ['first', 'bad'])
            self.assertEqual((target / 'first').read_text(), 'old first')

    def test_batch_copy_rejects_source_enumeration_error_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; (source / 'nested').mkdir(parents=True)
            (source / 'first').write_text('new first'); output = root / 'output'; target = output / 'other'; target.mkdir(parents=True)
            (target / 'first').write_text('old first')
            def failed_walk(path, *, followlinks, onerror):
                onerror(PermissionError('denied'))
                yield from ()
            with patch.object(receipt, 'REC', output), patch.object(receipt.os, 'walk', side_effect=failed_walk):
                with self.assertRaisesRegex(SystemExit, 'cannot enumerate'):
                    receipt.copy_batch(source, 'other', ['first', 'nested'])
            self.assertEqual((target / 'first').read_text(), 'old first')

    def test_batch_copy_rejects_hardlinked_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir(); (source / 'first').write_text('new')
            output = root / 'output'; target = output / 'other'; target.mkdir(parents=True)
            protected = output / 'protected'; protected.write_text('old')
            os.link(protected, target / 'first')
            with patch.object(receipt, 'REC', output):
                with self.assertRaisesRegex(SystemExit, 'hardlinked'):
                    receipt.copy_batch(source, 'other', ['first'])
            self.assertEqual(protected.read_text(), 'old')

    def test_batch_copy_rejects_unreadable_file_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            (source / 'first').write_text('new first'); (source / 'bad').write_text('secret')
            output = root / 'output'; target = output / 'other'; target.mkdir(parents=True)
            (target / 'first').write_text('old first')
            original_open = Path.open
            def opened(path, *args, **kwargs):
                if path == source / 'bad':
                    raise PermissionError('denied')
                return original_open(path, *args, **kwargs)
            with patch.object(receipt, 'REC', output), patch.object(Path, 'open', opened):
                with self.assertRaisesRegex(SystemExit, 'cannot read'):
                    receipt.copy_batch(source, 'other', ['first', 'bad'])
            self.assertEqual((target / 'first').read_text(), 'old first')

    def test_batch_copy_rejects_inaccessible_named_path_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; (source / 'hidden').mkdir(parents=True)
            (source / 'first').write_text('new first'); (source / 'hidden' / 'last').write_text('secret')
            output = root / 'output'; target = output / 'other'; target.mkdir(parents=True)
            (target / 'first').write_text('old first')
            original_lstat = Path.lstat
            def lstat(path, *args, **kwargs):
                if path == source / 'hidden':
                    raise PermissionError('denied')
                return original_lstat(path, *args, **kwargs)
            with patch.object(receipt, 'REC', output), patch.object(Path, 'lstat', lstat):
                with self.assertRaisesRegex(SystemExit, 'cannot inspect'):
                    receipt.copy_batch(source, 'other', ['first', 'hidden/last'])
            self.assertEqual((target / 'first').read_text(), 'old first')

    def test_batch_copy_preflights_non_directory_ancestor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / 'source'; (source / 'nested').mkdir(parents=True)
            (source / 'first').write_text('new first'); (source / 'nested' / 'last').write_text('new last')
            output = root / 'output'; target = output / 'other'; target.mkdir(parents=True)
            (target / 'first').write_text('old first'); (target / 'nested').write_text('blocking file')
            with patch.object(receipt, 'REC', output):
                with self.assertRaisesRegex(SystemExit, 'non-directory ancestor'):
                    receipt.copy_batch(source, 'other', ['first', 'nested/last'])
            self.assertEqual((target / 'first').read_text(), 'old first')
            self.assertEqual((target / 'nested').read_text(), 'blocking file')

    def test_three_fit_cannot_bind_without_verified_capture(self):
        case = {'pass': True, 'r_length': 0, 'julia_length': 0,
                'three_fit': {'r_lengths': [0, 5, 6], 'julia_lengths': [0, 5, 6]}}
        with self.assertRaisesRegex(SystemExit, 'verif'):
            receipt.integer_entry(receipt.COEF_EMPTY_CID, case, {'empty_coef': []})

    def test_legacy_scalar_remains_nondiscriminating(self):
        case = {'pass': True, 'r_length': 0, 'julia_length': 0}
        result = receipt.integer_entry(receipt.COEF_EMPTY_CID, case, {'empty_coef': []})
        self.assertEqual(result['r_value'], 0)
        self.assertFalse(result['discriminating'])

    def test_overlay_rejects_unverified_capture_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rec = root / "receipts"; raw = rec / "postfit-policy-p1"
            cases = rec / "cases"; raw.mkdir(parents=True); cases.mkdir()
            oracle = {"df": 8, "loglik_nobs_attr": 400, "nobs": 400, "empty_coef": []}
            results = {}
            for cid, (key, _) in receipt.INTEGER_EQUALITY.items():
                if cid == receipt.COEF_EMPTY_CID:
                    jc = {"pass": True, "r_length": 0, "julia_length": 0,
                          "three_fit": {"r_lengths": [0,5,6], "julia_lengths": [0,5,6]}}
                else:
                    jc = {"pass": True, "r": oracle[key], "julia": oracle[key]}
                results[cid] = jc
                (cases / f"{cid}.json").write_text(json.dumps({"harness_fields": jc}))
            (raw / "r-oracle.json").write_text(json.dumps(oracle))
            (raw / "julia-results.json").write_text(json.dumps({"cases": results}))
            first = next(iter(receipt.INTEGER_EQUALITY))
            before = (cases / f"{first}.json").read_bytes()
            with patch.object(receipt, "REC", rec), patch.object(receipt, "ROOT", root):
                with self.assertRaisesRegex(SystemExit, "verif"):
                    receipt.apply_integer_equality()
            self.assertEqual(before, (cases / f"{first}.json").read_bytes())

    def test_capture_copy_retains_control_and_runner_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "source"; out = root / "out"
            (source / "runner-snapshots").mkdir(parents=True)
            (source / "runner-snapshots" / "runner.jl").write_text("pinned runner")
            (source / "julia-control-always_empty.json").write_text("{}")
            with patch.object(receipt, "REC", out):
                receipt.copy_batch(source, "policy", ["runner-snapshots", "julia-control-always_empty.json"])
            self.assertEqual((out / "policy" / "runner-snapshots" / "runner.jl").read_text(), "pinned runner")
            self.assertTrue((out / "policy" / "julia-control-always_empty.json").is_file())

    def test_nobs_comparison_unchanged(self):
        case = {'pass': True, 'r': 400, 'julia': 400}
        result = receipt.integer_entry('CORE070-POSTFIT-NOBS-COUNT-NATIVE', case, {'nobs': 400})
        self.assertEqual((result['r_value'], result['julia_value']), (400, 400))


@unittest.skipUnless(os.environ.get("P1_REAL_CAPTURE"), "requires retained source-bound P1 capture")
class RealCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / "state"
        shutil.copytree(os.environ["P1_REAL_CAPTURE"], self.state)
        self.report = json.loads((self.state / "julia-results.json").read_text())
        self.oracle = json.loads((self.state / "r-oracle.json").read_text())

    def entry(self):
        return receipt.integer_entry(receipt.COEF_EMPTY_CID,
                                     self.report["cases"][receipt.COEF_EMPTY_CID], self.oracle, self.state)

    def write_json(self, name, value):
        (self.state / name).write_text(json.dumps(value))

    def test_verified_capture_produces_three_comparisons(self):
        x = self.entry()
        self.assertEqual(x["r_value"], [0, 5, 6])
        self.assertEqual(x["julia_value"], [0, 5, 6])
        self.assertTrue(x["discriminating"])

    def test_policy_capture_uses_verified_identity_without_sidecar(self):
        self.assertTrue(hasattr(receipt, 'check_policy_run_commit'), 'policy identity check is missing')
        self.assertFalse((self.state / 'run-commit.json').exists())
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=receipt.ROOT, text=True).strip()
        receipt.check_policy_run_commit(self.state, head)

    def test_policy_capture_rejects_contradictory_sidecar(self):
        self.assertTrue(hasattr(receipt, 'check_policy_run_commit'), 'policy identity check is missing')
        (self.state / 'run-commit.json').write_text(json.dumps({'glvmodels_commit': '0' * 40, 'dirty': []}))
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=receipt.ROOT, text=True).strip()
        with self.assertRaises(SystemExit): receipt.check_policy_run_commit(self.state, head)

    def test_policy_capture_rejects_wrong_expected_head(self):
        self.assertTrue(hasattr(receipt, 'check_policy_run_commit'), 'policy identity check is missing')
        with self.assertRaises(SystemExit): receipt.check_policy_run_commit(self.state, '0' * 40)

    def test_copied_capture_remains_fully_verifiable(self):
        with tempfile.TemporaryDirectory(dir=receipt.ROOT) as temp:
            with patch.object(receipt, "REC", Path(temp)):
                receipt.copy_batch(self.state, "copy", receipt.POLICY_CAPTURE_ARTIFACTS)
            copied = Path(temp) / "copy"
            x = receipt.integer_entry(receipt.COEF_EMPTY_CID,
                                      self.report["cases"][receipt.COEF_EMPTY_CID], self.oracle, copied)
            self.assertEqual(x["julia_value"], [0, 5, 6])

    def test_policy_copy_stops_before_writes_until_transactional(self):
        with tempfile.TemporaryDirectory(dir=receipt.ROOT) as temp:
            with patch.object(receipt, "REC", Path(temp)):
                with self.assertRaisesRegex(SystemExit, "transactional"):
                    receipt.copy_batch(self.state, "postfit-policy-p1", receipt.POLICY_CAPTURE_ARTIFACTS)
            self.assertFalse((Path(temp) / "postfit-policy-p1").exists())

    @unittest.skip("three-fit overlay requires a reviewed transactional target-only path")
    def test_overlay_updates_only_temporary_row_without_stale_note(self):
        with tempfile.TemporaryDirectory(dir=receipt.ROOT) as temp:
            base = Path(temp); rec = base / "receipts"; out = base / "out"
            raw = rec / "postfit-policy-p1"; cases = rec / "cases"
            shutil.copytree(self.state, raw); cases.mkdir(parents=True); out.mkdir()
            rows = []
            for cid in receipt.INTEGER_EQUALITY:
                rec_data = {"harness_fields": self.report["cases"][cid], "verdict": "PASS",
                            "batch_verifier": {"status": "PASS"}}
                (cases / f"{cid}.json").write_text(json.dumps(rec_data))
                rows.append({"source_id": "postfit-policy/POST-COEF-EMPTY" if cid == receipt.COEF_EMPTY_CID else cid,
                             "executable_case_ids": [cid], "evidence_tier": "numeric_non_discriminating",
                             "note": "old non-discriminating zero only", "carry_scan_status": "DANGLING"})
            fallback = {"verdict": "PASS", "batch_verifier": {"status": "PASS"}}
            (cases / f"{receipt.FALLBACK_CID}.json").write_text(json.dumps(fallback))
            rows.append({"source_id": "fallback-test", "executable_case_ids": [receipt.FALLBACK_CID],
                         "evidence_tier": "partial_non_numeric_case", "carry_scan_status": "DANGLING"})
            (out / "case-map-postfit.json").write_text(json.dumps({"rows": rows, "counts": {}}))
            fresh = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build-totoro-20261104.json"
            # Synthetic injection exercises output wiring only. The actual
            # source lacks this tracked build and is tested separately below.
            with patch.object(receipt, "REC", rec), patch.object(receipt, "OUT", out), \
                 patch.object(receipt, "select_postcoef_build", return_value=fresh):
                receipt.apply_integer_equality()
            result = json.loads((out / "case-map-postfit.json").read_text())
            row = next(x for x in result["rows"] if x["source_id"] == "postfit-policy/POST-COEF-EMPTY")
            self.assertEqual(row["evidence_tier"], "numeric")
            self.assertNotIn("old non-discriminating", row.get("note", ""))
            payload = json.loads((cases / f"{receipt.COEF_EMPTY_CID}.json").read_text())
            self.assertEqual(payload["comparison"]["cases"][0]["r_value"], [0,5,6])
            self.assertIn("lengths only", payload["numeric_note"])
            self.assertEqual(payload["postcoef_capture"], {
                "receipt": f"{receipt.REC_REL}/postcoef-empty-p1/receipt.json",
                "sha256": hashlib.sha256((raw / "receipt.json").read_bytes()).hexdigest(),
            })
            self.assertEqual(payload["oracle_build_receipt"], fresh)

    def test_build_mismatch_and_overlay_guard_reject_before_receipt_write(self):
        record = json.loads((self.state / "receipt.json").read_text())
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=receipt.ROOT, text=True).strip()
        self.assertEqual(receipt.select_postcoef_build(record, receipt.ROOT, head),
                         "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build-totoro-20261104.json")
        record["oracle_build_sha256"] = "0" * 64
        with self.assertRaisesRegex(SystemExit, "matching oracle build receipt"):
            receipt.select_postcoef_build(record, receipt.ROOT, head)
        with tempfile.TemporaryDirectory(dir=receipt.ROOT) as temp:
            base = Path(temp); rec = base / "receipts"; out = base / "out"
            raw = rec / "postfit-policy-p1"; cases = rec / "cases"
            shutil.copytree(self.state, raw); cases.mkdir(parents=True); out.mkdir()
            sentinel = cases / "unchanged.json"; sentinel.write_text("unchanged\n")
            output_sentinel = out / "case-map-postfit.json"; output_sentinel.write_text("unchanged map\n")
            raw_before = {p.relative_to(raw): p.read_bytes() for p in raw.rglob('*') if p.is_file()}
            with patch.object(receipt, "REC", rec), patch.object(receipt, "OUT", out):
                with self.assertRaisesRegex(SystemExit, "transactional"):
                    receipt.apply_integer_equality()
            self.assertEqual(sentinel.read_text(), "unchanged\n")
            self.assertEqual(output_sentinel.read_text(), "unchanged map\n")
            self.assertEqual({p.relative_to(raw): p.read_bytes() for p in raw.rglob('*') if p.is_file()}, raw_before)
            self.assertEqual(sorted(p.name for p in cases.iterdir()), ["unchanged.json"])

    def test_missing_control_rejected(self):
        (self.state / "julia-control-always_empty.json").unlink()
        with self.assertRaises(SystemExit): self.entry()

    def test_changed_control_rejected(self):
        p = self.state / "julia-control-always_empty.json"
        p.write_text(p.read_text() + " ")
        with self.assertRaises(SystemExit): self.entry()

    def test_wrong_control_rejected_even_if_report_hash_updated(self):
        mode = "always_empty"; name = f"julia-control-{mode}.json"
        control = json.loads((self.state / name).read_text())
        control["cases"][receipt.COEF_EMPTY_CID]["three_fit"]["julia_lengths"] = [0, 5, 6]
        self.write_json(name, control)
        record = json.loads((self.state / "receipt.json").read_text())
        record["postcoef_controls"][mode]["report_sha256"] = hashlib.sha256((self.state / name).read_bytes()).hexdigest()
        self.write_json("receipt.json", record)
        with self.assertRaises(SystemExit): self.entry()

    def test_control_success_exit_rejected_with_matching_hash(self):
        mode = "always_empty"; exit_name = f"julia-control-{mode}.exit"
        (self.state / exit_name).write_text("0\n")
        record = json.loads((self.state / "receipt.json").read_text())
        record["postcoef_controls"][mode]["exit_code"] = 0
        record["postcoef_controls"][mode]["exit_sha256"] = hashlib.sha256((self.state / exit_name).read_bytes()).hexdigest()
        self.write_json("receipt.json", record)
        with self.assertRaises(SystemExit): self.entry()

    def test_oracle_design_mutation_rejected(self):
        self.oracle["postcoef_three_fit"]["fits"][1]["fixed_columns"] = 6
        with self.assertRaises(SystemExit): self.entry()

    def test_boolean_or_mismatched_vector_cannot_be_substituted(self):
        evidence = self.report["cases"][receipt.COEF_EMPTY_CID]["three_fit"]
        for values in ([False, 5, 6], [0, 0, 0], [1, 6, 7], [0, 5]):
            old = evidence["julia_lengths"]; evidence["julia_lengths"] = values
            with self.assertRaises(SystemExit): self.entry()
            evidence["julia_lengths"] = old

    def test_mismatched_supplied_oracle_rejected(self):
        self.oracle["empty_coef"] = [1]
        with self.assertRaises(SystemExit): self.entry()

    def test_changed_capture_identity_rejected(self):
        record = json.loads((self.state / "receipt.json").read_text())
        record["postcoef_capture_identity"]["source_commit"] = "0" * 40
        self.write_json("receipt.json", record)
        with self.assertRaises(SystemExit): self.entry()


class OracleBuildSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="postcoef-build-selection-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q")
        self.prefix = "docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/"
        old = self.root / (self.prefix + "build.json"); old.parent.mkdir(parents=True)
        old.write_text("{}\n"); self.commit(self.prefix + "build.json")
        pin = {"reference_commit": "1" * 40, "archive_sha256": "2" * 64,
               "source_tree_sha256": "3" * 64, "installed_tree_sha256": "4" * 64,
               "marker_sha256": "5" * 64, "version": "0.7.1"}
        build = {key: pin[key] for key in ("reference_commit", "archive_sha256", "source_tree_sha256",
                                           "installed_tree_sha256", "marker_sha256")}
        build["gllvmTMB_version"] = pin["version"]
        self.build_bytes = json.dumps(build).encode()
        self.record = {"source_pin": pin, "oracle_build_sha256": hashlib.sha256(self.build_bytes).hexdigest(),
                       "postcoef_capture_identity": {"source_commit": self.head()}}

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], check=True,
                              capture_output=True, text=True).stdout

    def commit(self, *paths):
        self.git("add", "--", *paths)
        self.git("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")

    def head(self):
        return self.git("rev-parse", "HEAD").strip()

    def test_missing_and_untracked_build_are_rejected(self):
        with self.assertRaises(SystemExit):
            receipt.select_postcoef_build(self.record, self.root, self.head())
        (self.root / (self.prefix + "build-totoro-20261104.json")).write_bytes(self.build_bytes)
        with self.assertRaises(SystemExit):
            receipt.select_postcoef_build(self.record, self.root, self.head())

    def test_tracked_build_and_evidence_only_descendant(self):
        rel = self.prefix + "build-totoro-20261104.json"
        (self.root / rel).write_bytes(self.build_bytes); self.commit(rel)
        self.record["postcoef_capture_identity"]["source_commit"] = self.head()
        self.assertEqual(receipt.select_postcoef_build(self.record, self.root, self.head()), rel)
        evidence = f"{receipt.REC_REL}/postcoef-empty-p1/results.tsv"
        path = self.root / evidence; path.parent.mkdir(parents=True); path.write_text("capture\n")
        self.commit(evidence)
        self.assertEqual(receipt.select_postcoef_build(self.record, self.root, self.head()), rel)
        (self.root / rel).write_bytes(self.build_bytes + b" ")
        with self.assertRaises(SystemExit):
            receipt.select_postcoef_build(self.record, self.root, self.head())

    def test_duplicate_matching_builds_are_rejected(self):
        paths = [self.prefix + "build-totoro.json", self.prefix + "build-totoro-20261104.json"]
        for rel in paths:
            (self.root / rel).write_bytes(self.build_bytes)
        self.commit(*paths)
        self.record["postcoef_capture_identity"]["source_commit"] = self.head()
        with self.assertRaises(SystemExit):
            receipt.select_postcoef_build(self.record, self.root, self.head())


class EvidenceDescendantTests(unittest.TestCase):
    RECEIPT = "docs/dev-log/core070/true-parity-latest/receipts/postfit/postcoef-empty-p1/receipt.json"
    LEGACY_RECEIPT = "docs/dev-log/core070/true-parity-latest/receipts/postfit/postfit-policy-p1/receipt.json"
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q"); self.git("config", "user.name", "Fixture"); self.git("config", "user.email", "fixture@example.invalid")
        self.write("src/fit.jl", "source"); self.write("tools/runner.py", "runner")
        self.git("add", "src/fit.jl", "tools/runner.py"); self.git("commit", "-qm", "source")
        self.source = self.git("rev-parse", "HEAD").strip()
    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True, text=True).stdout
    def write(self, rel, value):
        p = self.root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(value)
    def commit(self, rel, value):
        self.write(rel, value); self.git("add", rel); self.git("commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD").strip()
    def accepts(self, head):
        return receipt.postcoef_source_compatible(self.root, self.source, head)
    def test_same_and_empty_commit(self):
        self.assertTrue(self.accepts(self.source)); self.git("commit", "--allow-empty", "-qm", "empty")
        self.assertTrue(self.accepts(self.git("rev-parse", "HEAD").strip()))
    def test_exact_evidence_file(self):
        self.assertTrue(self.accepts(self.commit(self.RECEIPT, "evidence")))
    def test_preload_log_only_at_postcoef_path(self):
        rel = str(Path(self.RECEIPT).parent / "oracle-verify-preload.log")
        self.assertTrue(self.accepts(self.commit(rel, "CORE070_ORACLE_VERIFY_PASS\n")))
        self.assertNotIn("oracle-verify-preload.log", receipt.POLICY_CAPTURE_ARTIFACTS)
        other = str(Path(self.LEGACY_RECEIPT).parent / "oracle-verify-preload.log")
        self.assertFalse(self.accepts(self.commit(other, "CORE070_ORACLE_VERIFY_PASS\n")))
    def test_neighbor_of_preload_log_remains_unapproved(self):
        rel = str(Path(self.RECEIPT).parent / "oracle-verify-preload.log")
        self.assertTrue(self.accepts(self.commit(rel, "CORE070_ORACLE_VERIFY_PASS\n")))
        self.assertFalse(self.accepts(self.commit(rel + ".extra", "unexpected\n")))
    def test_legacy_batch_cannot_be_changed_as_evidence(self):
        self.assertFalse(self.accepts(self.commit(self.LEGACY_RECEIPT, '{"status":"PASS"}')))
    def test_unlisted_evidence(self):
        self.assertFalse(self.accepts(self.commit(str(Path(self.RECEIPT).parent / "unlisted.json"), "no")))
    def test_source_change_and_revert(self):
        self.assertFalse(self.accepts(self.commit("src/fit.jl", "changed")))
        self.assertFalse(self.accepts(self.commit("src/fit.jl", "source")))
    def test_evidence_deletion(self):
        source = self.commit(self.RECEIPT, "evidence"); self.git("rm", "-q", self.RECEIPT); self.git("commit", "-qm", "delete")
        self.assertFalse(receipt.postcoef_source_compatible(self.root, source, self.git("rev-parse", "HEAD").strip()))
    def test_sibling_and_invalid_hash(self):
        self.git("checkout", "-qb", "sibling", self.source); sibling = self.commit(self.RECEIPT, "a")
        self.git("checkout", "-qb", "other", self.source); head = self.commit(self.RECEIPT, "b")
        self.assertFalse(receipt.postcoef_source_compatible(self.root, sibling, head)); self.assertFalse(self.accepts("HEAD"))
    def test_unrelated_signed_failure_in_case_map_cannot_change(self):
        rel = "docs/dev-log/core070/true-parity-latest/case-map-postfit.json"
        target = {"source_id": "postfit-policy/POST-COEF-EMPTY", "disposition": None}
        other = {"source_id": "postfit/OTHER", "verdict": "FAIL", "signed_by": "maintainer"}
        baseline = self.commit(rel, json.dumps({"rows": [target, other]}))
        other["verdict"] = "PASS"
        head = self.commit(rel, json.dumps({"rows": [target, other]}))
        self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_target_case_map_only_change_is_rejected(self):
        rel = "docs/dev-log/core070/true-parity-latest/case-map-postfit.json"
        target = {"source_id": "postfit-policy/POST-COEF-EMPTY", "disposition": None, "evidence_tier": "partial"}
        other = {"source_id": "postfit/OTHER", "verdict": "FAIL", "signed_by": "maintainer"}
        baseline = self.commit(rel, json.dumps({"rows": [target, other]}))
        target["evidence_tier"] = "numeric"
        head = self.commit(rel, json.dumps({"rows": [target, other]}))
        self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_target_receipt_failure_cannot_be_rewritten(self):
        rel = "docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/CORE070-POSTFIT-COEF-EMPTY-NATIVE.json"
        d = {"case_id": "CORE070-POSTFIT-COEF-EMPTY-NATIVE", "verdict": "FAIL", "signed_by": "maintainer"}
        baseline = self.commit(rel, json.dumps(d)); d["verdict"] = "PASS"
        head = self.commit(rel, json.dumps(d)); self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_unrelated_batch_verifier_failure_cannot_change(self):
        rel = "docs/dev-log/core070/true-parity-latest/case-map-postfit.json"
        d = {"rows": [{"source_id": "postfit-policy/POST-COEF-EMPTY", "disposition": None}],
             "batch_verifiers": {"other-batch": {"status": "FAIL"}, "postfit-policy-p1": {"status": "PASS"}}}
        baseline = self.commit(rel, json.dumps(d)); d["batch_verifiers"]["other-batch"]["status"] = "PASS"
        head = self.commit(rel, json.dumps(d))
        self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_signed_target_failure_cannot_change(self):
        rel = "docs/dev-log/core070/true-parity-latest/case-map-postfit.json"
        d = {"rows": [{"source_id": "postfit-policy/POST-COEF-EMPTY", "disposition": "DISPOSITION-SIGNED", "verdict": "FAIL"}]}
        baseline = self.commit(rel, json.dumps(d)); d["rows"][0]["verdict"] = "PASS"
        head = self.commit(rel, json.dumps(d)); self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_boolean_failed_target_case_cannot_be_rewritten(self):
        rel = f"{receipt.REC_REL}/cases/{receipt.COEF_EMPTY_CID}.json"
        old = {'case_id': receipt.COEF_EMPTY_CID, 'harness_pass': False}
        baseline = self.commit(rel, json.dumps(old)); old['harness_pass'] = True
        head = self.commit(rel, json.dumps(old))
        self.assertFalse(receipt.postcoef_ref_compatible(self.root, baseline, head))

    def test_existing_nonjson_capture_evidence_is_immutable(self):
        prefix = f'{receipt.REC_REL}/postcoef-empty-p1/'
        for name in ('verify.txt', 'diagnostics.log'):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    def git(*args):
                        return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()
                    git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
                    (root / prefix).mkdir(parents=True)
                    (root / (prefix + 'receipt.json')).write_text('{"status":"PASS"}')
                    item = root / (prefix + name); item.write_text('SIGNED FAILURE: FAIL\n')
                    git('add', prefix + 'receipt.json', prefix + name); git('commit', '-qm', 'retained evidence')
                    baseline = git('rev-parse', 'HEAD')
                    item.write_text('PASS\n')
                    self.assertFalse(receipt.postcoef_source_compatible(root, baseline, baseline))
                    git('add', prefix + name); git('commit', '-qm', 'overwrite')
                    self.assertFalse(receipt.postcoef_ref_compatible(root, baseline, git('rev-parse', 'HEAD')))

    def test_existing_json_capture_evidence_is_immutable(self):
        prefix = f'{receipt.REC_REL}/postcoef-empty-p1/'
        for prior in ({'status': 'ERROR'}, {'pass': 0}, {'status': 'PASS'}):
            with self.subTest(prior=prior), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                def git(*args):
                    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()
                git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
                (root / prefix).mkdir(parents=True)
                (root / (prefix + 'receipt.json')).write_text('{"status":"PASS"}')
                artifact = root / (prefix + 'julia-results.json'); artifact.write_text(json.dumps(prior))
                git('add', prefix + 'receipt.json', prefix + 'julia-results.json'); git('commit', '-qm', 'retained evidence')
                baseline = git('rev-parse', 'HEAD')
                artifact.write_text('{"status":"PASS","changed":true}')
                self.assertFalse(receipt.postcoef_source_compatible(root, baseline, baseline))
                git('add', prefix + 'julia-results.json'); git('commit', '-qm', 'overwrite')
                self.assertFalse(receipt.postcoef_ref_compatible(root, baseline, git('rev-parse', 'HEAD')))

    def test_new_json_capture_evidence_is_allowed(self):
        baseline = self.commit(self.RECEIPT, json.dumps({'status': 'PASS'}))
        raw = str(Path(self.RECEIPT).parent / 'julia-results.json')
        head = self.commit(raw, json.dumps({'status': 'PASS'}))
        self.assertTrue(receipt.postcoef_ref_compatible(self.root, baseline, head))

    def test_new_nonjson_capture_evidence_is_allowed(self):
        baseline = self.commit(self.RECEIPT, json.dumps({'status': 'PASS'}))
        log = str(Path(self.RECEIPT).parent / 'verify.txt')
        head = self.commit(log, 'new verifier transcript')
        self.assertTrue(receipt.postcoef_ref_compatible(self.root, baseline, head))

    def test_target_case_error_and_malformed_pass_are_preserved(self):
        rel = f'{receipt.REC_REL}/cases/{receipt.COEF_EMPTY_CID}.json'
        for bad in ({'verdict':'ERROR'}, {'harness_pass':0}, {'verdict':'TIMED_OUT'}):
            with self.subTest(prior=bad), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                def git(*args):
                    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()
                git('init', '-q'); git('config', 'user.name', 'Fixture'); git('config', 'user.email', 'fixture@example.invalid')
                path = root / rel; path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'case_id':receipt.COEF_EMPTY_CID, **bad}))
                git('add', rel); git('commit', '-qm', 'prior case'); baseline = git('rev-parse', 'HEAD')
                path.write_text(json.dumps({'case_id':receipt.COEF_EMPTY_CID, 'verdict':'PASS', 'harness_pass':True}))
                self.assertFalse(receipt.postcoef_source_compatible(root, baseline, baseline))
                git('add', rel); git('commit', '-qm', 'replace case')
                self.assertFalse(receipt.postcoef_ref_compatible(root, baseline, git('rev-parse', 'HEAD')))

    def test_signed_failed_raw_receipt_and_cohort_cannot_be_rewritten(self):
        rel = self.RECEIPT
        baseline = self.commit(rel, json.dumps({'status':'FAIL', 'signed_by':'maintainer'}))
        head = self.commit(rel, json.dumps({'status':'PASS'}))
        self.assertFalse(receipt.postcoef_ref_compatible(self.root, baseline, head))

    def test_untracked_signed_raw_capture_blocks_dirty_source(self):
        self.write(self.RECEIPT, json.dumps({'status':'FAIL', 'signed_by':'maintainer'}))
        self.assertFalse(receipt.postcoef_source_compatible(self.root, self.source, self.source))

    def test_untracked_capture_directory_symlink_blocks_dirty_source(self):
        with tempfile.TemporaryDirectory() as outside:
            (Path(outside) / 'receipt.json').write_text(json.dumps({'status':'FAIL', 'signed_by':'maintainer'}))
            link = self.root / Path(self.RECEIPT).parent
            link.parent.mkdir(parents=True)
            link.symlink_to(outside, target_is_directory=True)
            self.assertFalse(receipt.postcoef_source_compatible(self.root, self.source, self.source))

    def test_ignored_untracked_raw_capture_blocks_dirty_source(self):
        baseline = self.commit('.gitignore', '**/receipt.json\n')
        self.write(self.RECEIPT, json.dumps({'status':'FAIL', 'signed_by':'maintainer'}))
        self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, baseline))

    def test_contradictory_control_receipt_is_protected(self):
        controls = {'always_empty': {'mode':'always_empty', 'passed':True, 'exit_code':0},
                    'spurious_coefficient': {'mode':'spurious_coefficient', 'passed':True, 'exit_code':1}}
        baseline = self.commit(self.RECEIPT, json.dumps({'status':'PASS', 'postcoef_controls':controls}))
        controls['always_empty']['exit_code'] = 1
        head = self.commit(self.RECEIPT, json.dumps({'status':'PASS', 'postcoef_controls':controls}))
        self.assertFalse(receipt.postcoef_ref_compatible(self.root, baseline, head))

    def test_signed_failed_raw_capture_blocks_changing_cohort(self):
        baseline = self.commit(self.RECEIPT, json.dumps({'status':'FAIL', 'signed_by':'maintainer'}))
        raw = 'docs/dev-log/core070/true-parity-latest/receipts/postfit/postcoef-empty-p1/diagnostics.log'
        head = self.commit(raw, 'new diagnostics')
        self.assertFalse(receipt.postcoef_ref_compatible(self.root, baseline, head))
        self.git('checkout', '--detach', baseline)
        self.write(self.RECEIPT, json.dumps({'status':'PASS'}))
        self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, baseline))

    def test_failed_raw_boolean_case_is_preserved(self):
        raw = 'docs/dev-log/core070/true-parity-latest/receipts/postfit/postcoef-empty-p1/'
        self.commit(raw+'receipt.json', json.dumps({'status':'PASS'}))
        baseline = self.commit(raw+'julia-results.json', json.dumps({'cases':{'OTHER':{'pass':False}}}))
        head = self.commit(raw+'julia-results.json', json.dumps({'cases':{'OTHER':{'pass':True}}}))
        self.assertFalse(receipt.postcoef_ref_compatible(self.root, baseline, head))

    def test_signed_target_pass_cannot_be_rewritten(self):
        rel = "docs/dev-log/core070/true-parity-latest/case-map-postfit.json"
        d = {"rows": [{"source_id": "postfit-policy/POST-COEF-EMPTY", "disposition": "DISPOSITION-SIGNED", "verdict": "PASS", "note": "signed"}]}
        baseline = self.commit(rel, json.dumps(d)); d["rows"][0]["note"] = "rewritten"
        head = self.commit(rel, json.dumps(d)); self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_signed_failed_policy_batch_cannot_be_rewritten(self):
        rel = "docs/dev-log/core070/true-parity-latest/case-map-postfit.json"
        d = {"rows": [{"source_id": "postfit-policy/POST-COEF-EMPTY"}],
             "batch_verifiers": {"postfit-policy-p1": {"status": "FAIL", "signed_by": "Maintainer"}}}
        baseline = self.commit(rel, json.dumps(d))
        d["batch_verifiers"]["postfit-policy-p1"] = {"status": "PASS"}
        head = self.commit(rel, json.dumps(d))
        self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_unquoted_extra_scoreboard_row_rejected(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        text = "| postfit-policy-POST-COEF-EMPTY `postfit-policy/POST-COEF-EMPTY` | required_core | PARTIAL | none | note |\n"
        baseline = self.commit(rel, text)
        head = self.commit(rel, text + "| other-row | required_core | EVIDENCED | receipt | note |\n")
        self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_unrelated_assembled_input_cannot_change(self):
        rel = "docs/dev-log/core070/true-parity-latest/case-map-assembled.json"
        d = {"rows": [{"source_id": "postfit-policy/POST-COEF-EMPTY", "disposition": None}],
             "inputs": [{"path": "case-map-other.json", "sha256": "old"},
                        {"path": "case-map-postfit.json", "sha256": "old"}]}
        baseline = self.commit(rel, json.dumps(d)); d["inputs"][0]["sha256"] = "new"
        head = self.commit(rel, json.dumps(d)); self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_unrelated_scoreboard_row_cannot_change(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        text = "| postfit-policy-POST-COEF-EMPTY `postfit-policy/POST-COEF-EMPTY` | required_core | PARTIAL | none | note |\n| other-id `other/signed` | required_core | FAIL | none | note |\n"
        baseline = self.commit(rel, text); head = self.commit(rel, text.replace("| FAIL |", "| PASS |"))
        self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_scoreboard_unverified_summary_change_is_rejected(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        text = "| `all` | 1 | 0 |\n| postfit-policy-POST-COEF-EMPTY `postfit-policy/POST-COEF-EMPTY` | required_core | PARTIAL | none | note |\n"
        baseline = self.commit(rel, text); head = self.commit(rel, text.replace("| `all` | 1 | 0 |", "| `all` | 0 | 1 |"))
        self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_signed_case_numeric_type_change_rejected(self):
        rel = f"{receipt.REC_REL}/cases/{receipt.COEF_EMPTY_CID}.json"
        old = {"case_id": receipt.COEF_EMPTY_CID, "review": {"signed_by": "maintainer"}, "value": 0}
        new = dict(old, value=False)
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, json.dumps(old).encode(), json.dumps(new).encode()))

    def test_unrelated_failed_row_numeric_type_change_rejected(self):
        rel = "docs/dev-log/core070/true-parity-latest/case-map-postfit.json"
        old = {"rows": [{"source_id": "postfit-policy/POST-COEF-EMPTY"}, {"source_id": "other", "verdict": "FAIL", "n": 0}]}
        new = json.loads(json.dumps(old)); new["rows"][1]["n"] = False
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, json.dumps(old).encode(), json.dumps(new).encode()))

    def test_unrelated_null_metadata_addition_rejected(self):
        rel = "docs/dev-log/core070/true-parity-latest/case-map-postfit.json"
        old = {"rows": [{"source_id": "postfit-policy/POST-COEF-EMPTY"}]}
        new = dict(old, foreign_metadata=None)
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, json.dumps(old).encode(), json.dumps(new).encode()))
        new = dict(old, batch_verifiers={"other": None})
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, json.dumps(old).encode(), json.dumps(new).encode()))

    def test_scoreboard_target_status_change_rejected(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        old = b"| postfit-policy-POST-COEF-EMPTY `postfit-policy/POST-COEF-EMPTY` | required_core | NON-DISCRIMINATING | none | note |\n"
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, old, old.replace(b"NON-DISCRIMINATING", b"EVIDENCED")))

    def test_scoreboard_target_promotion_rejected_without_cross_file_proof(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        old = (Path(receipt.ROOT) / rel).read_text()
        lines = old.splitlines()
        target = next(i for i, line in enumerate(lines) if line.startswith("| postfit-policy-POST-COEF-EMPTY "))
        postfit = next(i for i, line in enumerate(lines) if line.startswith("| `postfit` |"))
        total = next(i for i, line in enumerate(lines) if line.startswith("| `all` |"))
        lines[target] = lines[target].replace("| NON-DISCRIMINATING |", "| EVIDENCED |")
        lines[postfit] = lines[postfit].replace("| 45 | 1 | 4 | 0 | 1 | 1 |", "| 46 | 1 | 4 | 0 | 1 | 0 |")
        lines[total] = lines[total].replace("| 206 | 63 | 35 | 3 | 2 | 1 |", "| 207 | 63 | 35 | 3 | 2 | 0 |")
        summary_start = lines.index("## Totals by family") + 2
        for i in range(summary_start, total + 1):
            columns = lines[i].split("|")
            del columns[7]  # the now-empty NON-DISCRIMINATING status column
            lines[i] = "|".join(columns)
        new = "\n".join(lines) + "\n"
        self.assertNotEqual(old, new)
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, old.encode(), new.encode()))
        stale = new.replace(lines[total], lines[total].replace("| 207 |", "| 206 |"))
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, old.encode(), stale.encode()))
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, old.encode(), new.replace("Listed for the reader only.", "Invented claim.").encode()))
        spoofed = lines.copy()
        spoofed[target] = spoofed[target].replace("family postfit;", "family other;")
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, old.encode(), ("\n".join(spoofed) + "\n").encode()))

    def test_scoreboard_cannot_forge_case_map_digest(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        old = (Path(receipt.ROOT) / rel).read_text()
        lines = old.splitlines()
        item = next(i for i, line in enumerate(lines) if "case-map-postfit.json" in line and "sha256 `" in line)
        import re
        lines[item] = re.sub(r"(?<=sha256 `)[0-9a-f]{64}(?=`)", "0" * 64, lines[item])
        changed = "\n".join(lines) + "\n"
        self.assertNotEqual(old, changed)
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, old.encode(), changed.encode()))

    def test_scoreboard_cannot_forge_signature_in_unchanged_status(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        old = (Path(receipt.ROOT) / rel).read_text()
        lines = old.splitlines()
        target = next(i for i, line in enumerate(lines) if line.startswith("| postfit-policy-POST-COEF-EMPTY "))
        lines[target] = lines[target].replace("not bound; cited:", "Disposition: DISPOSITION-SIGNED; signed_by: Shinichi Nakagawa; cited:")
        changed = "\n".join(lines) + "\n"
        self.assertNotEqual(old, changed)
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, old.encode(), changed.encode()))

    def test_scoreboard_target_classification_and_cases_are_protected(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        old = b"| postfit-policy-POST-COEF-EMPTY `postfit-policy/POST-COEF-EMPTY` | required_core; cases: example | PARTIAL | none | family postfit; note |\n"
        for changed in (old.replace(b"required_core", b"optional"), old.replace(b"cases: example", b"cases: other")):
            self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, old, changed))

    def test_scoreboard_unknown_target_status_rejected(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        old = (Path(receipt.ROOT) / rel).read_bytes()
        changed = old.replace(b"NON-DISCRIMINATING", b"INVENTED-STATUS")
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, old, changed))

    def test_scoreboard_target_family_cannot_be_prefixed(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        old = b"| postfit-policy-POST-COEF-EMPTY `postfit-policy/POST-COEF-EMPTY` | required_core | PARTIAL | none | family postfit; evidence_tier x |\n"
        new = old.replace(b"family postfit;", b"family other; family postfit;")
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, old, new))

    def test_scoreboard_target_identity_cannot_include_extra_source(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        old = b"| postfit-policy-POST-COEF-EMPTY `postfit-policy/POST-COEF-EMPTY` | required_core | PARTIAL | none | note |\n"
        new = old.replace(b"`postfit-policy/POST-COEF-EMPTY`", b"`postfit-policy/POST-COEF-EMPTY` `other/falsified`")
        self.assertFalse(receipt.postcoef_shared_evidence_safe(rel, old, new))

    def test_scoreboard_target_identity_cannot_be_spoofed(self):
        rel = "docs/dev-log/core070/true-parity-latest/scoreboard.md"
        text = "| postfit-policy-POST-COEF-EMPTY `postfit-policy/POST-COEF-EMPTY` | required_core | PARTIAL | none | note |\n"
        baseline = self.commit(rel, text); head = self.commit(rel, text.replace("`postfit-policy/POST-COEF-EMPTY`", "`other/identity`"))
        self.assertFalse(receipt.postcoef_source_compatible(self.root, baseline, head))
    def test_dirty_tracked_source_rejected(self):
        self.write("src/fit.jl", "uncommitted change")
        self.assertFalse(self.accepts(self.source))
    def test_untracked_source_rejected(self):
        self.write("src/new.jl", "untracked code")
        self.assertFalse(self.accepts(self.source))
    def test_merge_rejected(self):
        self.git("checkout", "-qb", "side"); self.git("commit", "--allow-empty", "-qm", "side")
        self.git("checkout", "-q", "-"); self.git("commit", "--allow-empty", "-qm", "main")
        self.git("merge", "--no-ff", "-qm", "merge", "side")
        self.assertFalse(self.accepts(self.git("rev-parse", "HEAD").strip()))


if __name__ == '__main__':
    unittest.main()

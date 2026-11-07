"""Preserve the historical non-discriminating case when adding three-fit evidence."""
import json
import copy
import os
import hashlib
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import true_parity_assemble as assemble
import core070_postcoef_live_ref as live
from test_core070_postcoef_live_ref import fixture as live_fixture

import core070_postfit_p1_receipts as receipt

LEGACY = f'{receipt.REC_REL}/cases/CORE070-POSTFIT-COEF-EMPTY-NATIVE.json'
NEW = f'{receipt.REC_REL}/cases/CORE070-POSTFIT-COEF-THREE-FIT-NATIVE.json'

class HistoricalPreservationTests(unittest.TestCase):
    def test_historical_scalar_text_rederives_without_receipt_changes(self):
        historical = receipt.load(receipt.ROOT / LEGACY)
        raw = receipt.load(receipt.REC / 'postfit-policy-p1' / 'julia-results.json')
        expected = receipt.integer_text(receipt.COEF_EMPTY_CID, raw['cases'][receipt.COEF_EMPTY_CID])
        self.assertEqual({key: historical[key] for key in expected}, expected)

    def test_allowlist_has_only_new_case(self):
        paths = receipt.postcoef_evidence_paths()
        self.assertIn(NEW, paths)
        self.assertNotIn(LEGACY, paths)

    def test_evidence_descendant_preserves_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def git(*args):
                return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()
            git('init','-q'); git('config','user.name','Fixture'); git('config','user.email','fixture@example.invalid')
            legacy = root / LEGACY; legacy.parent.mkdir(parents=True); legacy.write_text(json.dumps({'case_id':'old','discriminating':False}))
            git('add', LEGACY); git('commit','-qm','historical fixture'); source = git('rev-parse','HEAD')
            new = root / NEW; new.write_text(json.dumps({'case_id':'new'})); git('add', NEW); git('commit','-qm','new case'); head = git('rev-parse','HEAD')
            self.assertTrue(receipt.postcoef_ref_compatible(root,source,head))
            new.write_text(json.dumps({'case_id':'changed'})); git('add', NEW); git('commit','-qm','rewrite new case'); rewritten = git('rev-parse','HEAD')
            self.assertFalse(receipt.postcoef_ref_compatible(root,source,rewritten))
            git('checkout','--quiet','--detach',head)
            legacy.write_text(json.dumps({'case_id':'changed'})); git('add', LEGACY); git('commit','-qm','change historical case'); changed = git('rev-parse','HEAD')
            self.assertFalse(receipt.postcoef_ref_compatible(root,source,changed))

    def test_dirty_or_staged_new_signed_failure_rejected(self):
        for staged in (False, True):
            with self.subTest(staged=staged), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                def git(*args):
                    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()
                git('init','-q'); git('config','user.name','Fixture'); git('config','user.email','fixture@example.invalid')
                legacy = root / LEGACY; legacy.parent.mkdir(parents=True); legacy.write_text('{"discriminating": false}')
                git('add', LEGACY); git('commit','-qm','historical'); source=git('rev-parse','HEAD')
                new=root / NEW; new.write_text('{"status":"FAIL","signed_by":"fixture"}'); git('add', NEW); git('commit','-qm','signed failure'); head=git('rev-parse','HEAD')
                new.write_text('{"status":"PASS"}')
                if staged: git('add', NEW)
                self.assertFalse(receipt.postcoef_source_compatible(root,source,head))

class SeparateCaseDraftTests(unittest.TestCase):
    def test_draft_has_distinct_receipt_and_exact_harness_identity(self):
        historical=(Path(__file__).resolve().parent.parent / LEGACY).read_bytes()
        complete, report, oracle = live_fixture()
        harness=copy.deepcopy(report['cases'][live.CID])
        comparison={'case_id':receipt.COEF_EMPTY_CID,'kind':'integer_equality','r_value':[0,5,6],
                    'julia_value':[0,5,6],'tolerance':0.5,'max_abs_diff':0,'discriminating':True}
        draft=receipt.draft_separate_postcoef_case(harness,comparison,historical)
        self.assertEqual(draft['case_id'],receipt.COEF_THREE_FIT_CID)
        self.assertEqual(draft['harness_case_id'],receipt.COEF_EMPTY_CID)
        self.assertEqual(draft['comparison']['cases'][0]['case_id'],receipt.COEF_EMPTY_CID)
        self.assertEqual(draft['verdict'],'UNQUALIFIED')
        complete.update(draft)
        complete['verdict']='PASS'; self.assertIsNone(live.check_comparison(complete,report,oracle))
        complete['verdict']='UNQUALIFIED'
        with self.assertRaisesRegex(ValueError,'identity or harness'): live.check_comparison(complete,report,oracle)
        harness['three_fit']['julia_lengths'][0]=9
        comparison['r_value'][0]=9
        self.assertEqual(draft['harness_fields']['three_fit']['julia_lengths'],[0,5,6])
        self.assertEqual(draft['comparison']['cases'][0]['r_value'],[0,5,6])
        harness['three_fit']['julia_lengths'][0]=0; comparison['r_value'][0]=0
        draft['harness_fields']['three_fit']['julia_lengths'][0]=8
        self.assertEqual(harness['three_fit']['julia_lengths'],[0,5,6])
        self.assertEqual(draft['historical_case']['sha256'],hashlib.sha256(historical).hexdigest())
        for invalid in [dict(comparison,case_id='spoof'),dict(comparison,r_value=[0,0,0]),dict(comparison,tolerance=1),dict(comparison,max_abs_diff=False),dict(comparison,r_value=[False,5,6])]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                receipt.draft_separate_postcoef_case(harness,invalid,historical)
        with self.assertRaises(ValueError): receipt.draft_separate_postcoef_case(harness,comparison,historical+b' ')
        bad_harness={'pass':True,'three_fit':{'julia_lengths':[0,0,0]}}
        with self.assertRaises(ValueError): receipt.draft_separate_postcoef_case(bad_harness,comparison,historical)

class PrivateCaptureVerifierTests(unittest.TestCase):
    def test_private_verifier_ignores_dirty_and_cached_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'repo'; root.mkdir(); tools=root/'tools'; tools.mkdir(); state=Path(tmp)/'state'; state.mkdir()
            def git(*args):
                return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()
            git('init','-q'); git('config','user.name','Fixture'); git('config','user.email','fixture@example.invalid')
            module=tools/'core070_postcoef_live_ref.py'
            markers=('CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10\nCORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4\n'
                     'CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5\nCORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2\n')
            module.write_text('def run_source_selftest(*args): return '+repr(markers)+'\n')
            git('add','tools/core070_postcoef_live_ref.py'); git('commit','-qm','source'); source=git('rev-parse','HEAD')
            module.write_text('raise RuntimeError("forged working source")\n')
            (state/'data.json').write_text('{"fixture":true}')
            result=receipt.run_private_capture_selftest(root,source,state,{'python_executable_path':sys.executable})
            self.assertEqual(result,markers)
            self.assertEqual(module.read_text(),'raise RuntimeError("forged working source")\n')
            self.assertFalse((tools/'__pycache__').exists())
            (state/'link').symlink_to(state/'data.json')
            with self.assertRaisesRegex(ValueError,'linked or non-regular'): receipt.run_private_capture_selftest(root,source,state,{'python_executable_path':sys.executable})

    def test_private_verifier_rejects_tracked_symlink_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'repo'; root.mkdir(); tools=root/'tools'; tools.mkdir(); state=Path(tmp)/'state'; state.mkdir()
            def git(*args): return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()
            git('init','-q'); git('config','user.name','Fixture'); git('config','user.email','fixture@example.invalid')
            markers=('CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10\nCORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4\n'
                     'CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5\nCORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2\n')
            external=Path(tmp)/'external.py'; external.write_text('def result(): return '+repr(markers)+'\n')
            (tools/'external_helper.py').symlink_to(external)
            (tools/'core070_postcoef_live_ref.py').write_text('import external_helper\ndef run_source_selftest(*args): return external_helper.result()\n')
            git('add','tools/core070_postcoef_live_ref.py','tools/external_helper.py'); git('commit','-qm','source'); source=git('rev-parse','HEAD')
            (state/'data.json').write_text('{"fixture":true}')
            with self.assertRaisesRegex(ValueError,'symlink or bytecode'):
                receipt.run_private_capture_selftest(root,source,state,{'python_executable_path':sys.executable})

    def test_private_verifier_rejects_tracked_bytecode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'repo'; root.mkdir(); tools=root/'tools'; tools.mkdir(); state=Path(tmp)/'state'; state.mkdir()
            def git(*args): return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()
            git('init','-q'); git('config','user.name','Fixture'); git('config','user.email','fixture@example.invalid')
            (tools/'core070_postcoef_live_ref.py').write_text('def run_source_selftest(*args): return ""\n')
            (tools/'stale.pyc').write_bytes(b'forged')
            git('add','tools/core070_postcoef_live_ref.py'); git('add','-f','tools/stale.pyc'); git('commit','-qm','source'); source=git('rev-parse','HEAD')
            (state/'data.json').write_text('{"fixture":true}')
            with self.assertRaisesRegex(ValueError,'symlink or bytecode'):
                receipt.run_private_capture_selftest(root,source,state,{'python_executable_path':sys.executable})

    def test_private_verifier_cannot_modify_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'repo'; root.mkdir(); tools=root/'tools'; tools.mkdir(); state=Path(tmp)/'state'; state.mkdir()
            def git(*args): return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()
            git('init','-q'); git('config','user.name','Fixture'); git('config','user.email','fixture@example.invalid')
            markers=('CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10\nCORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4\n'
                     'CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5\nCORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2\n')
            (tools/'core070_postcoef_live_ref.py').write_text('from pathlib import Path\ndef run_source_selftest(root,source,state,runtime):\n Path(state,"injected").write_text("x")\n return '+repr(markers)+'\n')
            git('add','tools/core070_postcoef_live_ref.py'); git('commit','-qm','source'); source=git('rev-parse','HEAD')
            (state/'data.json').write_text('{"fixture":true}')
            with self.assertRaisesRegex(ValueError,'private capture snapshot changed'):
                receipt.run_private_capture_selftest(root,source,state,{'python_executable_path':sys.executable})
            self.assertEqual(sorted(p.name for p in state.iterdir()),['data.json'])

class CaptureDraftIntegrationTests(unittest.TestCase):
    def test_draft_calls_private_verifier_and_is_unqualified(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp)
            harness={'pass':True,'three_fit':{'julia_lengths':[0,5,6]}}
            oracle={'postcoef_three_fit':{'fits':[{'length':n} for n in (0,5,6)]}}
            record={'status':'PASS','postcoef_capture_identity':{'source_commit':'a'*40,'source_tree':'b'*40},
                    'postcoef_runtime':{'python_executable_path':sys.executable}}
            for name,value in [('receipt.json',record),('julia-results.json',{'cases':{receipt.COEF_EMPTY_CID:harness}}),('r-oracle.json',oracle)]:
                (state/name).write_text(json.dumps(value))
            (state/'control.log').write_text('retained control')
            before={p.name:p.read_bytes() for p in state.iterdir()}
            with patch.object(receipt,'postcoef_source_compatible',return_value=True), patch.object(receipt,'run_private_capture_selftest',return_value='fixture transcript') as verify, patch.object(receipt,'select_postcoef_build',return_value='selected-build'):
                result=receipt.prepare_separate_postcoef_case(state)
            verify.assert_called_once_with(receipt.ROOT,'a'*40,state,record['postcoef_runtime'])
            self.assertEqual(result['verdict'],'UNQUALIFIED')
            self.assertEqual(result['batch_verifier']['status'],'UNVERIFIED')
            self.assertEqual(result['oracle_build_receipt'],'selected-build')
            self.assertEqual(before,{p.name:p.read_bytes() for p in state.iterdir()})
            with patch.object(receipt,'postcoef_source_compatible',return_value=False), patch.object(receipt,'run_private_capture_selftest',side_effect=AssertionError('must not run')):
                with self.assertRaisesRegex(ValueError,'not compatible'): receipt.prepare_separate_postcoef_case(state)
            with patch.object(receipt,'postcoef_source_compatible',side_effect=[True,False]), patch.object(receipt,'run_private_capture_selftest',return_value='fixture transcript'), patch.object(receipt,'select_postcoef_build',return_value='selected-build'):
                with self.assertRaisesRegex(ValueError,'selected source changed'): receipt.prepare_separate_postcoef_case(state)
            def changed_during_verification(*args):
                (state/'r-oracle.json').write_text('{"changed":true}')
                return 'fixture transcript'
            with patch.object(receipt,'postcoef_source_compatible',return_value=True), patch.object(receipt,'run_private_capture_selftest',side_effect=changed_during_verification), patch.object(receipt,'select_postcoef_build',return_value='selected-build'):
                with self.assertRaisesRegex(ValueError,'changed during'): receipt.prepare_separate_postcoef_case(state)
            (state/'r-oracle.json').write_bytes(before['r-oracle.json'])
            def changed_after_private_check(*args):
                (state/'control.log').write_text('replaced control')
                return 'selected-build'
            with patch.object(receipt,'postcoef_source_compatible',return_value=True), patch.object(receipt,'run_private_capture_selftest',return_value='fixture transcript'), patch.object(receipt,'select_postcoef_build',side_effect=changed_after_private_check):
                with self.assertRaisesRegex(ValueError,'changed during'): receipt.prepare_separate_postcoef_case(state)
            (state/'control.log').write_bytes(before['control.log'])
            (state/'receipt.json').write_text(json.dumps(dict(record,status='FAIL')))
            with patch.object(receipt,'run_private_capture_selftest',side_effect=AssertionError('must not run')):
                with self.assertRaisesRegex(ValueError,'not PASS'): receipt.prepare_separate_postcoef_case(state)

    def test_real_late_source_edit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); root=base/'repo'; root.mkdir(); state=base/'state'; state.mkdir()
            def git(*args): return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()
            git('init','-q'); git('config','user.name','Fixture'); git('config','user.email','fixture@example.invalid')
            historical=root/LEGACY; historical.parent.mkdir(parents=True)
            historical.write_bytes((Path(receipt.__file__).parent.parent / LEGACY).read_bytes())
            code=root/'tools/source.py'; code.parent.mkdir(); code.write_text('original\n')
            git('add',LEGACY,'tools/source.py'); git('commit','-qm','source'); source=git('rev-parse','HEAD')
            harness={'pass':True,'three_fit':{'julia_lengths':[0,5,6]}}
            record={'status':'PASS','postcoef_capture_identity':{'source_commit':source,'source_tree':'b'*40},'postcoef_runtime':{}}
            oracle={'postcoef_three_fit':{'fits':[{'length':n} for n in (0,5,6)]}}
            for name,value in [('receipt.json',record),('julia-results.json',{'cases':{receipt.COEF_EMPTY_CID:harness}}),('r-oracle.json',oracle)]:
                (state/name).write_text(json.dumps(value))
            def late_edit(*args):
                code.write_text('user edit\n')
                return 'synthetic transcript'
            with patch.object(receipt,'ROOT',root), patch.object(receipt,'run_private_capture_selftest',side_effect=late_edit), patch.object(receipt,'select_postcoef_build',return_value='fixture-build'):
                with self.assertRaisesRegex(ValueError,'selected source changed'): receipt.prepare_separate_postcoef_case(state)
            self.assertEqual(code.read_text(),'user edit\n')

class SeparateCaseAssemblerTests(unittest.TestCase):
    def test_only_separate_case_can_bind_existing_harness_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path=root / NEW; path.parent.mkdir(parents=True)
            harness = receipt.COEF_EMPTY_CID
            case={'case_id':receipt.COEF_THREE_FIT_CID,'harness_case_id':harness,'verdict':'PASS',
                  'comparison':{'pin':'P1','cases':[{'case_id':harness,'kind':'integer_equality',
                      'r_value':[0,5,6],'julia_value':[0,5,6],'tolerance':0.5,'max_abs_diff':0}]}}
            path.write_text(json.dumps(case))
            row={'source_id':'postfit-policy/POST-COEF-EMPTY','executable_case_ids':[harness],
                 'evidence':{'receipt':NEW}}
            with patch.object(assemble,'postcoef_capture_problem',return_value=None):
                self.assertIsNone(assemble.numeric_receipt_problem(row,root))
                for field, value in [('case_id',harness),('harness_case_id','spoofed')]:
                    forged=dict(case); forged[field]=value; path.write_text(json.dumps(forged))
                    self.assertEqual(assemble.numeric_receipt_problem(row,root),'POST-COEF case receipt missing')
                path.write_text(json.dumps(case))
                wrong=root / LEGACY; wrong.write_text(json.dumps(case))
                forged_row=dict(row,evidence={'receipt':LEGACY})
                self.assertEqual(assemble.numeric_receipt_problem(forged_row,root),'POST-COEF case receipt missing')
                case['comparison']['cases'][0]['case_id']=receipt.COEF_THREE_FIT_CID
                path.write_text(json.dumps(case))
                self.assertIn('case ids not compared',assemble.numeric_receipt_problem(row,root))

if __name__ == '__main__': unittest.main()

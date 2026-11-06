#!/usr/bin/env python3
"""Experimental offline capture check. Does not verify a live runtime or admit a row."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', required=True, type=Path)
p.add_argument('--state', required=True, type=Path)
p.add_argument('--receipt-sha256', required=True)
a = p.parse_args()
repo, state = a.repo.resolve(), a.state.resolve()
tool_root = Path(__file__).resolve().parents[1]
if not re.fullmatch(r'[0-9a-f]{64}', a.receipt_sha256):
    sys.exit('invalid independent receipt hash')
try:
    receipt_path = (state / 'receipt.json').resolve()
    if state not in receipt_path.parents or not receipt_path.is_file():
        raise ValueError('receipt is missing')
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != a.receipt_sha256:
        raise ValueError('receipt hash mismatch')
    commit = json.loads(receipt_bytes)['postcoef_capture_identity']['source_commit']
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', commit):
        raise ValueError('invalid captured source commit')
    for name in ('tools/core070_verify_postfit_policy_batch.py', 'tools/parity_oracle.py',
                 'tools/core070_source_pin_check.py', 'tools/core070_oracle_pins.toml'):
        local = (tool_root / name).resolve()
        if tool_root not in local.parents or not local.is_file():
            raise ValueError('source file missing: ' + name)
        original = subprocess.run(['git', '-C', str(repo), 'show', commit + ':' + name],
                                  check=True, capture_output=True,
                                  env={**{k: v for k, v in os.environ.items() if not k.startswith('GIT_')}, 'GIT_NO_REPLACE_OBJECTS': '1'}).stdout
        if local.read_bytes() != original:
            raise ValueError('source file differs from capture commit: ' + name)
except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
    sys.exit('ARTIFACT_INTEGRITY_FAILED: ' + str(exc))
for key in list(os.environ):
    if key.startswith('GIT_'):
        del os.environ[key]
os.environ['GIT_NO_REPLACE_OBJECTS'] = '1'
os.environ['GLLVM_PARITY_PIN'] = 'P1'
sys.path.insert(0, str(tool_root / 'tools'))
import core070_verify_postfit_policy_batch as v

def need(ok, why):
    if not ok: raise ValueError(why)
def checked(rel, expected):
    path = (state / rel).resolve()
    need(state in path.parents and path.is_file(), f'missing {rel}')
    need(isinstance(expected, str) and v.sha(path) == expected, f'hash mismatch: {rel}')
    return path

def main():
    rp = checked('receipt.json', a.receipt_sha256)
    rec = json.loads(rp.read_text())
    contract = v.load_contract(); v.verify_contract(contract)
    digest = v.sha(v.CONTRACT_PATH)
    need(rec.get('status') == 'PASS' and rec.get('source_unchanged') is True, 'capture status')
    need(rec.get('reference_commit') == contract['reference_commit'] and rec.get('contract_sha256') == digest and rec.get('source_pins') == contract['source_pins'], 'source pin or contract')
    need(rec.get('case_count') == contract['expected_case_count'] and rec.get('negative_control_count') == len(contract['negative_controls']), 'batch counts')
    need(sorted(rec.get('expected_case_ids', [])) == sorted(c['case_id'] for c in contract['cases']) and sorted(rec.get('negative_control_case_ids', [])) == sorted(c['control_id'] for c in contract['negative_controls']), 'batch case ids')
    need(rec.get('julia_exit_code') == 0, 'Julia exit')
    v.core070_source_pin_check.check_source_pin(rec, 'P1', need)
    for name, key in [('results.tsv','raw_sha256'), ('diagnostics.log','diagnostics_sha256'), ('julia-stdout.log','julia_stdout_sha256'), ('julia-stderr.log','julia_stderr_sha256'), ('runner-snapshots/core070_postfit_policy_batch.R','r_runner_sha256')]: checked(name, rec.get(key))
    need(not (state/'diagnostics.log').read_text().strip(), 'nonempty diagnostics')
    report = json.loads(checked('julia-results.json', rec.get('julia_results_sha256')).read_text())
    v.check_report(report, contract, digest)
    ident = rec.get('postcoef_capture_identity')
    v.check_postcoef_capture_artifacts(state, repo, ident)
    v.check_postcoef_committed_files(repo, state, ident['source_commit'], v.CONTRACT_PATH.relative_to(tool_root))
    runtime = rec.get('postcoef_runtime'); need(isinstance(runtime, dict) and report.get('postcoef_runtime') == runtime, 'runtime record mismatch')
    checked('julia-Manifest.toml', runtime.get('manifest_sha256'))
    checked('runner-snapshots/julia-launcher', runtime.get('executable_sha256'))
    for when in ('before','after'):
        log = checked(f'oracle-verify-{when}.log', rec.get(f'oracle_verify_{when}_sha256'))
        need('CORE070_ORACLE_VERIFY_PASS' in log.read_text().splitlines(), f'oracle {when} marker')
    controls = rec.get('postcoef_controls'); need(isinstance(controls, dict) and set(controls) == {'always_empty', 'spurious_coefficient'}, 'control set')
    evidence = report['cases']['CORE070-POSTFIT-COEF-EMPTY-NATIVE']['three_fit']
    v.check_postcoef_three_fit_evidence(evidence)
    oracle = json.loads((state / 'r-oracle.json').read_text()); v.check_postcoef_r_oracle_binding(evidence, oracle)
    need(report.get('postcoef_capture_identity') == ident, 'report identity')
    for mode, meta in controls.items():
        need(isinstance(meta, dict) and meta.get('mode') == mode and type(meta.get('exit_code')) is int and meta['exit_code'] == 1, 'control metadata')
        prefix = 'julia-control-' + mode
        report_control = json.loads(checked(prefix+'.json', meta.get('report_sha256')).read_text())
        checked(prefix+'.stdout.log', meta.get('stdout_sha256')); checked(prefix+'.stderr.log', meta.get('stderr_sha256'))
        need(checked(prefix+'.exit', meta.get('exit_sha256')).read_text().strip() == '1', 'control exit')
        need(report_control.get('postcoef_runtime') == runtime, 'control runtime record mismatch')
        v.check_postcoef_control_outcome(evidence, report_control, mode, meta['exit_code'], ident, report_control.get('postcoef_capture_identity'), contract)
    print('ARTIFACT_INTEGRITY_PASS_RUNTIME_NOT_VERIFIED')
    print('Offline artifact consistency only; oracle build, runtime and parity admission are not verified.')

try: main()
except (OSError, ValueError, KeyError, TypeError) as exc: sys.exit('ARTIFACT_INTEGRITY_FAILED: '+str(exc))

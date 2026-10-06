"""Verify the original Gaussian runner subset; never certify the draft programme."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tomllib

import core070_evidence as evidence
from core070_manifest_coverage import need

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / '.unlazy/core070-aghq/gaussian-required-02'
IDS = ['CORE070-FAMILY-00-IDENTITY-NATIVE-MODEL',
       'CORE070-FAMILY-00-IDENTITY-FORMULA-INTERFACE']
FIXTURE = 'test/parity/test_gaussian_original_required.jl'
DATA = 'test/parity/fixtures/core070_gaussian_original.toml'
DATA_SHA = '52a4664c99295ec675a7b666458350bf57ff57e338c77d5ada1cf9d2e5efba92'
MANIFEST = ROOT / '.unlazy/core070-aghq/aghq-public-poisson-env-01/attempt1/test/parity/Manifest.toml'
sha = evidence.digest
read = evidence.load_toml


def bounded(value, limit):
    return isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= limit


def check_semantics(native, formula, long):
    need(all(row['fixture_sha256'] == DATA_SHA for row in [native, formula, long]), 'stale data fixture')
    need(native['native_health'] is True and native['r_health'] is True, 'native/R health failed')
    need(native['r_random'] == ['z_B'] and native['r_fixed_columns'] == 0
         and native['r_sigma_mapped'] is True and native['native_dof'] == 8, 'different R model')
    need(native['r_parameter_names'].count('theta_rr_B') == 4
         and native['r_parameter_names'].count('theta_diag_B') == 4, 'different nuisance grouping')
    need(bounded(abs(native['native_loglik'] - native['r_loglik']), 1e-3), 'native likelihood mismatch')
    for key in ['native_gradient_max', 'r_gradient_max']:
        need(bounded(native[key], 1e-4), key)
    for key in ['point_value_delta', 'point_gradient_delta', 'r_endpoint_value_delta', 'native_endpoint_cross_delta']:
        need(bounded(native[key], 1e-6), key)
    need(bounded(native['delta_loglik'], 1e-3), 'reported likelihood delta')
    need(math.isclose(native['fixed_residual_sd'], 0.0008837756554325282, rel_tol=0, abs_tol=1e-14), 'fixed residual scale')
    c = native['fixed_residual_sd']
    need(len(native['native_unique_variances']) == len(native['native_total_variances']) == 4, 'variance dimensions')
    need(all(u > 0 and math.isclose(t, u+c*c, rel_tol=0, abs_tol=1e-12)
             for u,t in zip(native['native_unique_variances'],native['native_total_variances'])), 'unique/total scale')
    need(formula['formula'] == 'y ~ 0' and formula['pervar'] is True
         and formula['fixed_effect_count'] == 0 and formula['dof'] == 8, 'formula model differs')
    need(formula['converged'] is True and long['converged'] is True, 'formula convergence')
    need(bounded(formula['formula_gradient_max'],1e-4) and bounded(long['gradient_max'],1e-4), 'formula health')
    need(formula['fixed_residual_sd'] == c, 'formula residual scale')
    for value in [formula['formula_loglik'], long['loglik']]:
        need(bounded(abs(value-native['native_loglik']),1e-7), 'formula likelihood differs')
    theta = native['native_parameters_in_r_scale']
    for other in [formula['formula_parameters'], long['parameters']]:
        need(len(theta) == len(other) == 8 and all(bounded(abs(a-b),1e-7) for a,b in zip(theta,other)), 'formula coordinates differ')


def check_run(run, cells, contract_hash, execution_hash):
    need(run['status'] == 'success' and run['exit_code'] == 0
         and run['success_marker'] == 'CORE070_PARITY_SUCCESS', 'failed required run')
    need(run['requested_case_ids'] == IDS and run['completed_case_ids'] == sorted(IDS), 'missing/changed requested cases')
    need(run['scope'] == 'subset' and run['selected_family_count'] == 0, 'scope overstated')
    need(run['contract_sha256'] == contract_hash, 'stale contract')
    need(set(cells) == set(IDS) and run['cells'] == cells, 'missing or changed cell receipt')
    for cid, cell in cells.items():
        need(cell['id'] == cid and cell['run_id'] == run['run_id'] and cell['status'] == 'success', 'transplanted cell')
        need(set(cell['execution_case_ids']) == set(IDS) and len(cell['execution_case_ids']) == 2, 'incomplete execution group')
        need(cell['fixture'] == FIXTURE and cell['fixture_sha256'] == sha(ROOT/FIXTURE), 'stale test fixture')
        need(cell['contract_sha256'] == contract_hash and cell['execution_manifest_sha256'] == execution_hash, 'stale cell pins')
        need(cell['assertions'] == dict(passed=31,failed=0,errored=0,broken=0), 'assertion failure/count drift')
    need(evidence.execution_assertion_counts(cells) == dict(passed=31,failed=0,errored=0,broken=0)
         and run['actual_assertions'] == 31, 'double-counted shared execution')


def verify(self_test=False):
    folder = STATE/'attempt1/receipts'
    run,cells = evidence._load_receipts(folder)
    contract = evidence.load_manifest(evidence.DEFAULT_MANIFEST)
    need(contract['status'] == 'DRAFT_INCOMPLETE_NOT_FROZEN', 'subset cannot freeze programme')
    cases = {r['id']:r for r in contract['executable_case']}
    need(set(cases) == set(IDS), 'executable Gaussian contracts differ')
    need(sha(ROOT/DATA) == DATA_SHA, 'original fixture changed')
    for cid in IDS:
        need(cases[cid]['fixture_sha256'] == sha(ROOT/FIXTURE), 'declared test fixture stale')
        need(cases[cid]['data_fixture_sha256'] == DATA_SHA, 'declared data stale')
    inventory = evidence.execution_inventory(cases, IDS, evidence.DEFAULT_MANIFEST)
    entries = inventory['entries'] + [dict(path='test/parity/Manifest.toml',sha256=sha(MANIFEST))]
    entries.sort(key=lambda row:row['path'])
    inventory = dict(entries=entries,manifest_sha256=evidence._hash_inventory(entries))
    need(run['execution'] == inventory, 'stale execution source/helper/manifest')
    check_run(run,cells,sha(evidence.DEFAULT_MANIFEST),inventory['manifest_sha256'])
    evidence._validate_source(run['source'],contract,inventory,folder)
    need(run['source']['julia_threads'] == run['source']['blas_threads'] == 1, 'thread budget')
    process_path = STATE/'attempt1/process/process-receipt.json'
    evidence.verify_process(folder,process_path,inventory)
    process=json.loads(process_path.read_text())
    plan=json.loads((STATE/'plan.json').read_text())
    need(process['plan_sha256'] == sha(STATE/'plan.json') and process['environment_overrides'] == plan['env'], 'stale local plan/environment')
    need(plan['env']['CORE070_PARITY_CASE_IDS'] == ','.join(IDS)
         and plan['env']['CORE070_PARITY_REQUIRED'] == plan['env']['GLLVM_PARITY_TESTS'] == '1', 'optional/wrong requested run')
    need([r['id'] for r in process['results']] == ['oracle-before','registry','pair','oracle-after'], 'missing qualification step')
    pair=next(row for row in process['results'] if row['id']=='pair')
    need(pair['argv'][-1] == 'test/parity/runparity.jl', 'not the required runner')
    log=(process_path.parent/pair['log']).read_text()
    records=[]
    for name in ['gaussian-native.toml','gaussian-formula.toml','gaussian-long.toml']:
        path=folder/name
        need(log.splitlines().count('GAUSSIAN_ARTIFACT_SHA256 '+name+' '+sha(path)) == 1, 'unbound semantic artifact '+name)
        records.append(read(path))
    check_semantics(*records)
    if self_test:
        changes=[('native_health',False),('r_gradient_max',0.01),('native_loglik',0.0),
                 ('fixed_residual_sd',0.0),('fixture_sha256','0'*64)]
        count=0
        for key,value in changes:
            altered=deepcopy(records);altered[0][key]=value
            try:check_semantics(*altered)
            except ValueError:count+=1
            else:raise AssertionError('accepted '+key)
        for change in [lambda r:r['completed_case_ids'].pop(),lambda r:r.update(exit_code=7),
                       lambda r:r.update(contract_sha256='0'*64),lambda r:r.update(actual_assertions=62),
                       lambda r:r['cells'][IDS[0]]['assertions'].update(failed=1)]:
            altered=deepcopy(run);change(altered)
            try:check_run(altered,altered['cells'],sha(evidence.DEFAULT_MANIFEST),inventory['manifest_sha256'])
            except ValueError:count+=1
            else:raise AssertionError('accepted altered run')
        print('GAUSSIAN_REQUIRED_NEGATIVE_CONTROLS_PASS',count)
    print('GAUSSIAN_REQUIRED_RUNNER_VERIFIED 2 cases; 31 assertions; 1 shared execution; bridge unpaid')
    return dict(status='VERIFIED_GAUSSIAN_NATIVE_FORMULA_SUBSET_NOT_FULL_FAMILY',
                case_ids=IDS,actual_assertions=31,source_facts_bound=1,
                execution_manifest_sha256=inventory['manifest_sha256'],
                contract_sha256=sha(evidence.DEFAULT_MANIFEST),
                process_receipt_sha256=sha(process_path),run_sha256=sha(folder/'run.toml'),
                delta_loglik=records[0]['delta_loglik'],
                native_gradient_max=records[0]['native_gradient_max'],r_gradient_max=records[0]['r_gradient_max'],
                elapsed_seconds=pair['elapsed_seconds'],scope_boundary='No public R bridge, wider family or full manifest certification.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=verify(args.self_test)
    if args.output:
        with args.output.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')

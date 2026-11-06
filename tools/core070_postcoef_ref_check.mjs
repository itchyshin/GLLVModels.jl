// Inspect selected-ref POST-COEF artifact bindings. This is not runtime verification.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { isDeepStrictEqual } from 'node:util';
import { readRefBytes, resolveGitRef, safeGitEnv } from './core070_ref_bytes.mjs';

const PREFIX = 'docs/dev-log/core070/true-parity-latest/receipts/postfit/postcoef-empty-p1';
const CID = 'CORE070-POSTFIT-COEF-EMPTY-NATIVE';
const RECEIPT_CID = 'CORE070-POSTFIT-COEF-THREE-FIT-NATIVE';
const HISTORICAL_CASE = `docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/${CID}.json`;
const HISTORICAL_SHA256 = '72fdb2739730d2ca763a8b5d291442bd5847403b3e30ac49c82b0ae1a31bab80';
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
const hex = (x) => typeof x === 'string' && /^[0-9a-f]{64}$/.test(x);
const sameKeys = (obj, ids) => obj && typeof obj === 'object' && !Array.isArray(obj) &&
  isDeepStrictEqual(Object.keys(obj).sort(), [...ids].sort());

// This compares identity only. The caller must have run the live verifier.
export function liveIdentityMatches(result, expected) {
  const keys=['selected_commit','selected_tree','source_commit','case_commit','case_sha256',
    'capture_sha256','oracle_build_sha256','runtime_record','runtime_root','python_executable'];
  return result?.status==='POSTCOEF_LIVE_REF_VERIFIED' && expected &&
    keys.every((key)=>Object.hasOwn(expected,key) && Object.hasOwn(result,key) &&
      isDeepStrictEqual(result[key],expected[key])) && result.capture_commit===expected.source_commit;
}

function validThreeFit(evidence, mode) {
  const expected=[0,5,6];
  const selected=mode==='real' ? expected : mode==='always_empty' ? [0,0,0] : [1,6,7];
  if (!evidence || evidence.selected_accessor!==mode ||
      !isDeepStrictEqual(evidence.r_lengths,expected) ||
      !isDeepStrictEqual(evidence.r_design_columns,expected) ||
      !isDeepStrictEqual(evidence.julia_lengths,selected) ||
      !isDeepStrictEqual(evidence.control_lengths,{always_empty:[0,0,0],spurious_coefficient:[1,6,7]})) return false;
  if (!Array.isArray(evidence.r_values) || evidence.r_values.length!==3 ||
      !evidence.r_values.every((v,i)=>Array.isArray(v) && v.length===expected[i] &&
        v.every((x)=>typeof x==='number' && Number.isFinite(x)))) return false;
  return ['r_names','julia_names'].every((key)=>Array.isArray(evidence[key]) && evidence[key].length===3 &&
    evidence[key].every((v,i)=>v===null || (Array.isArray(v) && v.length===expected[i] &&
      v.every((x)=>typeof x==='string'))));
}

function reportMatchesContract(report, contract, control) {
  const caseIds = contract.cases.map((c) => c.case_id);
  const negativeIds = contract.negative_controls.map((c) => c.control_id);
  return new Set(caseIds).size === caseIds.length && new Set(negativeIds).size === negativeIds.length &&
    report?.area === 'postfit-policy' && report.scope === 'CORE070_POSTFIT_POLICY_BATCH' &&
    report.case_count === caseIds.length && report.negative_control_count === negativeIds.length &&
    report.status === (control ? 'FAIL' : 'PASS') && report.all_checks === !control &&
    report.all_positive_pass === !control && report.negative_controls_behaved_as_expected === true &&
    sameKeys(report.cases, caseIds) && sameKeys(report.negative_controls, negativeIds) &&
    caseIds.every((id) => report.cases[id]?.pass === !(control && id === CID)) &&
    negativeIds.every((id) => report.negative_controls[id]?.behaved === true);
}

// An unsigned local record can be checked for consistency but does not attest
// that a remote host ran the command. It never authorizes a row by itself.
export function inspectLiveVerificationRecord({root, record, captureHash}) {
  const failure = (reason) => ({ok:false, attested:false, reason});
  try {
    if (!record || record.status!=='LIVE_VERIFICATION_PASS_REVIEW_ONLY' || record.pin!=='P1' ||
        record.exit_code!==0 || record.git_status_before!=='' || record.git_status_after!=='' ||
        !hex(captureHash) || record.capture_receipt_sha256!==captureHash || !hex(record.verifier_sha256))
      return failure('record identity or status');
    if (typeof record.source_commit!=='string' || !/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(record.source_commit))
      return failure('record source commit');
    const verifier=readRefBytes({root,ref:record.source_commit,path:'tools/core070_verify_postfit_policy_batch.py'});
    if (!verifier || digest(verifier)!==record.verifier_sha256) return failure('record verifier source');
    const utc = (s) => typeof s==='string' && /(?:Z|\+00:00)$/.test(s) ? Date.parse(s) : NaN;
    const start=utc(record.started_utc), end=utc(record.ended_utc);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end<start || end-start>180000)
      return failure('record interval');
    const argv=record.argv;
    if (!Array.isArray(argv) || argv.length!==5 || argv[0]!=='python3' ||
        typeof argv[1]!=='string' || !argv[1].startsWith('/') ||
        !argv[1].endsWith('/tools/core070_verify_postfit_policy_batch.py') || argv[2]!=='--state' ||
        typeof argv[3]!=='string' || !argv[3].startsWith('/') || argv[4]!=='--self-test' ||
        typeof record.python_executable!=='string' || !record.python_executable.startsWith('/'))
      return failure('record command');
    const required=['CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10',
      'CORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4',
      'CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5',
      'CORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2'];
    if (typeof record.stdout!=='string' || record.stderr!=='' ||
        !required.every((line)=>record.stdout.split('\n').includes(line))) return failure('record output');
    return {ok:true, attested:false, reason:'unsigned record consistency only'};
  } catch (error) { return failure(String(error.message || error)); }
}

export function inspectPostcoefArtifacts({root, ref, caseReceipt}) {
  const failure = (reason) => ({ok:false, runtimeVerified:false, reason});
  try {
    const pinned = ref === 'FS' ? 'FS' : resolveGitRef(root,ref);
    if (!pinned) return failure('selected ref unavailable');
    if (caseReceipt?.case_id !== RECEIPT_CID || caseReceipt?.harness_case_id !== CID ||
        caseReceipt?.historical_case?.path !== HISTORICAL_CASE ||
        caseReceipt?.historical_case?.sha256 !== HISTORICAL_SHA256) return failure('case identity or historical binding');
    const binding = caseReceipt?.postcoef_capture;
    if (binding?.receipt !== `${PREFIX}/receipt.json` || !hex(binding?.sha256)) return failure('capture binding');
    if (!isDeepStrictEqual(caseReceipt?.raw, [`${PREFIX}/julia-results.json`, `${PREFIX}/r-oracle.json`]) ||
        caseReceipt?.batch_verifier?.log !== `${PREFIX}/verify.txt`) return failure('capture pointers');
    const read = (name, expected) => {
      if (!hex(expected)) throw new Error(`invalid hash: ${name}`);
      const bytes = readRefBytes({root,ref:pinned,path:`${PREFIX}/${name}`});
      if (!bytes || digest(bytes) !== expected) throw new Error(`missing or changed: ${name}`);
      return bytes;
    };
    const verifierLog = read('verify.txt', caseReceipt.batch_verifier?.log_sha256).toString('utf8').split('\n');
    for (const marker of ['CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10',
      'CORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4',
      'CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5',
      'CORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2']) {
      if (!verifierLog.includes(marker)) return failure('verifier transcript');
    }
    const captureBytes = read('receipt.json',binding.sha256);
    const receipt = JSON.parse(captureBytes.toString('utf8'));
    if (receipt.status !== 'PASS' || receipt.julia_exit_code !== 0 || receipt.source_unchanged !== true)
      return failure('capture status');
    const id=receipt.postcoef_capture_identity, runtime=receipt.postcoef_runtime;
    if (!id || !runtime) return failure('capture identity');
    if (typeof id.source_commit!=='string' || !/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(id.source_commit) ||
        typeof id.source_tree!=='string' || !/^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(id.source_tree))
      return failure('capture source identity');
    const original=readRefBytes({root,ref:id.source_commit,path:HISTORICAL_CASE});
    const selectedHistorical=readRefBytes({root,ref:pinned,path:HISTORICAL_CASE});
    if (!original || !selectedHistorical || digest(original)!==HISTORICAL_SHA256 ||
        !original.equals(selectedHistorical)) return failure('historical case differs');
    const gitOpts={cwd:root,stdio:['ignore','pipe','pipe'],maxBuffer:64 << 20,env:safeGitEnv()};
    const sourceTree=execFileSync('git',['rev-parse','--verify',`${id.source_commit}:src`],gitOpts).toString('utf8').trim();
    if (sourceTree!==id.source_tree ||
        execFileSync('git',['cat-file','-t',sourceTree],gitOpts).toString('utf8').trim()!=='tree')
      return failure('capture source tree');
    const contractPath='docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json';
    const contractBytes=readRefBytes({root,ref:pinned,path:contractPath});
    if (!contractBytes || !hex(receipt.contract_sha256) || digest(contractBytes)!==receipt.contract_sha256)
      return failure('contract hash');
    const contract=JSON.parse(contractBytes.toString('utf8'));
    for (const [path,hash] of [[contractPath,receipt.contract_sha256],
      ['tools/core070_postfit_policy_batch.R',receipt.r_runner_sha256],
      ['tools/core070_postfit_policy_batch.jl',id.julia_runner_sha256]]) {
      const committed=readRefBytes({root,ref:id.source_commit,path});
      if (!committed || !hex(hash) || digest(committed)!==hash) return failure(`capture commit file: ${path}`);
    }
    if (receipt.reference_commit!=='9539352f66f2db2cc26b1c393e67212a359b60c9' ||
        receipt.reference_commit!==contract.reference_commit || !isDeepStrictEqual(receipt.source_pins,contract.source_pins) ||
        !Array.isArray(contract.cases) || !Array.isArray(contract.negative_controls) ||
        receipt.case_count!==contract.cases.length || receipt.negative_control_count!==contract.negative_controls.length)
      return failure('contract identity');
    const files = [
      ['julia-results.json',receipt.julia_results_sha256], ['results.tsv',receipt.raw_sha256],
      ['diagnostics.log',receipt.diagnostics_sha256], ['julia-stdout.log',receipt.julia_stdout_sha256],
      ['julia-stderr.log',receipt.julia_stderr_sha256], ['r-oracle.json',id.r_oracle_sha256],
      ['runner-snapshots/core070_postfit_policy_batch.R',receipt.r_runner_sha256],
      ['runner-snapshots/core070_postfit_policy_batch.jl',id.julia_runner_sha256],
      ['runner-snapshots/julia-launcher',runtime.executable_sha256],
      ['julia-Manifest.toml',runtime.manifest_sha256],
      ['oracle-verify-before.log',receipt.oracle_verify_before_sha256],
      ['oracle-verify-after.log',receipt.oracle_verify_after_sha256],
    ];
    const values = new Map(files.map(([name,hash]) => [name,read(name,hash)]));
    const report=JSON.parse(values.get('julia-results.json').toString('utf8'));
    const oracle=JSON.parse(values.get('r-oracle.json').toString('utf8'));
    if (!isDeepStrictEqual(report.postcoef_capture_identity,id) || !isDeepStrictEqual(report.postcoef_runtime,runtime))
      return failure('real report identity');
    const actual=report.cases?.[CID], cmp=caseReceipt?.comparison?.cases;
    const evidence=actual?.three_fit;
    if (!reportMatchesContract(report,contract,false) || actual?.pass!==true ||
        !validThreeFit(evidence,'real'))
      return failure('real three-fit evidence');
    const fits=oracle.postcoef_three_fit?.fits;
    if (!Array.isArray(fits) || fits.length!==3 ||
        !['length','fixed_columns','values','names'].every((field,i)=>
          isDeepStrictEqual(fits.map((f)=>f[field]),evidence[['r_lengths','r_design_columns','r_values','r_names'][i]])))
      return failure('R oracle binding');
    if (!actual || !isDeepStrictEqual(caseReceipt.harness_fields,actual) || !Array.isArray(cmp) || cmp.length!==1 ||
        cmp[0].case_id!==CID || !isDeepStrictEqual(cmp[0].julia_value,actual.three_fit?.julia_lengths) ||
        !isDeepStrictEqual(cmp[0].r_value,oracle.postcoef_three_fit?.fits?.map((f)=>f.length)))
      return failure('case differs from capture');
    const controls=receipt.postcoef_controls;
    if (!controls || Object.keys(controls).sort().join(',') !== 'always_empty,spurious_coefficient')
      return failure('control set');
    for (const mode of ['always_empty','spurious_coefficient']) {
      const meta=controls[mode], prefix=`julia-control-${mode}`;
      if (meta?.mode!==mode || meta.exit_code!==1 || meta.passed!==true) return failure(`control metadata: ${mode}`);
      const control=JSON.parse(read(`${prefix}.json`,meta.report_sha256).toString('utf8'));
      read(`${prefix}.stdout.log`,meta.stdout_sha256);
      read(`${prefix}.stderr.log`,meta.stderr_sha256);
      if (read(`${prefix}.exit`,meta.exit_sha256).toString('utf8').trim()!=='1') return failure(`control exit: ${mode}`);
      const observed=control.cases?.[CID];
      if (!reportMatchesContract(control,contract,true) || observed?.pass!==false ||
          !validThreeFit(observed.three_fit,mode) ||
          !['r_lengths','r_design_columns','r_values','r_names'].every((key)=>
            isDeepStrictEqual(observed.three_fit?.[key],evidence[key])) ||
          !isDeepStrictEqual(control.postcoef_capture_identity,id) ||
          !isDeepStrictEqual(control.postcoef_runtime,runtime)) return failure(`control report: ${mode}`);
    }
    return {ok:true, runtimeVerified:false, sourceCommit:id.source_commit,
            reason:'artifact hashes and selected fields only'};
  } catch (error) { return failure(String(error.message || error)); }
}

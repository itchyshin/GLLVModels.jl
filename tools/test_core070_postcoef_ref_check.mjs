#!/usr/bin/env node
// Synthetic selected-ref controls. No R, Julia, retained capture or network is needed.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { inspectPostcoefArtifacts, liveIdentityMatches } from './core070_postcoef_ref_check.mjs';

const root=mkdtempSync(join(tmpdir(),'postcoef-ref-fixture-'));
const prefix='docs/dev-log/core070/true-parity-latest/receipts/postfit/postcoef-empty-p1';
const legacy='docs/dev-log/core070/true-parity-latest/receipts/postfit/postfit-policy-p1';
const contractPath='docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json';
const cid='CORE070-POSTFIT-COEF-EMPTY-NATIVE';
const receiptCid='CORE070-POSTFIT-COEF-THREE-FIT-NATIVE';
const historical=`docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/${cid}.json`;
const historicalBytes=readFileSync(new URL(`../${historical}`,import.meta.url));
const pin='9539352f66f2db2cc26b1c393e67212a359b60c9';
const sha=(bytes)=>createHash('sha256').update(bytes).digest('hex');
const bytes=(value)=>Buffer.from(JSON.stringify(value));
const write=(path,value)=>{ const target=join(root,path); mkdirSync(dirname(target),{recursive:true}); writeFileSync(target,value); };
const git=(...args)=>execFileSync('git',args,{cwd:root,stdio:'pipe'}).toString('utf8').trim();
try {
  git('init','-q');
  const rRunner=Buffer.from('R runner\n'), jRunner=Buffer.from('Julia runner\n');
  const contract={reference_commit:pin,source_pins:{},cases:[{case_id:cid},{case_id:'OTHER'}],negative_controls:[{control_id:'NEG'}]};
  write(historical,historicalBytes);
  write('src/example.jl','source\n'); write('tools/core070_postfit_policy_batch.R',rRunner);
  write('tools/core070_postfit_policy_batch.jl',jRunner); write(contractPath,bytes(contract));
  git('add',historical,'src/example.jl','tools/core070_postfit_policy_batch.R','tools/core070_postfit_policy_batch.jl',contractPath);
  git('-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm','source');
  const source=git('rev-parse','HEAD'), tree=git('rev-parse','HEAD:src');
  const fits=[0,5,6].map((n)=>({length:n,fixed_columns:n,values:Array(n).fill(0.25),names:Array(n).fill('x')}));
  const evidence={selected_accessor:'real',r_lengths:[0,5,6],julia_lengths:[0,5,6],r_design_columns:[0,5,6],
    r_values:fits.map((f)=>f.values),r_names:fits.map((f)=>f.names),julia_names:[null,null,null],
    control_lengths:{always_empty:[0,0,0],spurious_coefficient:[1,6,7]}};
  const oracle=bytes({postcoef_three_fit:{fits}}), launcher=Buffer.from([0,255,128,10]), manifest=Buffer.from('manifest\n');
  const runtime={executable_sha256:sha(launcher),manifest_sha256:sha(manifest)};
  const identity={source_commit:source,source_tree:tree,r_oracle_sha256:sha(oracle),julia_runner_sha256:sha(jRunner)};
  const report=(mode)=>({status:mode?'FAIL':'PASS',area:'postfit-policy',scope:'CORE070_POSTFIT_POLICY_BATCH',
    case_count:2,negative_control_count:1,all_checks:!mode,all_positive_pass:!mode,negative_controls_behaved_as_expected:true,
    postcoef_capture_identity:identity,postcoef_runtime:runtime,
    cases:{[cid]:{pass:!mode,three_fit:mode?{...evidence,selected_accessor:mode,julia_lengths:mode==='always_empty'?[0,0,0]:[1,6,7]}:evidence},OTHER:{pass:true}},
    negative_controls:{NEG:{behaved:true}}});
  const real=report(null), record={status:'PASS',julia_exit_code:0,source_unchanged:true,reference_commit:pin,
    contract_sha256:sha(bytes(contract)),source_pins:{},case_count:2,negative_control_count:1,
    postcoef_capture_identity:identity,postcoef_runtime:runtime,postcoef_controls:{}};
  const verifyLog=Buffer.from('CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10\nCORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4\nCORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5\nCORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2\n');
  const files={
    'verify.txt':verifyLog,
    'julia-results.json':bytes(real),'results.tsv':Buffer.from('results\n'),'diagnostics.log':Buffer.alloc(0),
    'julia-stdout.log':Buffer.from('ok\n'),'julia-stderr.log':Buffer.alloc(0),'r-oracle.json':oracle,
    'runner-snapshots/core070_postfit_policy_batch.R':rRunner,'runner-snapshots/core070_postfit_policy_batch.jl':jRunner,
    'runner-snapshots/julia-launcher':launcher,'julia-Manifest.toml':manifest,
    'oracle-verify-before.log':Buffer.from('verified\n'),'oracle-verify-after.log':Buffer.from('verified\n')};
  for (const [field,name] of Object.entries({julia_results_sha256:'julia-results.json',raw_sha256:'results.tsv',
    diagnostics_sha256:'diagnostics.log',julia_stdout_sha256:'julia-stdout.log',julia_stderr_sha256:'julia-stderr.log',
    r_runner_sha256:'runner-snapshots/core070_postfit_policy_batch.R',oracle_verify_before_sha256:'oracle-verify-before.log',
    oracle_verify_after_sha256:'oracle-verify-after.log'})) record[field]=sha(files[name]);
  for (const mode of ['always_empty','spurious_coefficient']) {
    const p=`julia-control-${mode}`;
    files[`${p}.json`]=bytes(report(mode)); files[`${p}.stdout.log`]=Buffer.from('failed as expected\n');
    files[`${p}.stderr.log`]=Buffer.alloc(0); files[`${p}.exit`]=Buffer.from('1\n');
    record.postcoef_controls[mode]={mode,exit_code:1,passed:true,report_sha256:sha(files[`${p}.json`]),
      stdout_sha256:sha(files[`${p}.stdout.log`]),stderr_sha256:sha(files[`${p}.stderr.log`]),exit_sha256:sha(files[`${p}.exit`])};
  }
  for (const [name,value] of Object.entries(files)) write(`${prefix}/${name}`,value);
  const recordBytes=bytes(record); write(`${prefix}/receipt.json`,recordBytes);
  const caseReceipt={case_id:receiptCid,harness_case_id:cid,historical_case:{path:historical,sha256:sha(historicalBytes)},postcoef_capture:{receipt:`${prefix}/receipt.json`,sha256:sha(recordBytes)},
    raw:[`${prefix}/julia-results.json`,`${prefix}/r-oracle.json`],
    batch_verifier:{status:'PASS',log:`${prefix}/verify.txt`,log_sha256:sha(verifyLog)},
    harness_fields:real.cases[cid],comparison:{cases:[{case_id:cid,r_value:[0,5,6],julia_value:[0,5,6]}]}};
  git('add',prefix); git('-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm','capture');
  const inspect=(ref='HEAD',candidate=caseReceipt)=>inspectPostcoefArtifacts({root,ref,caseReceipt:candidate});
  assert.equal(inspect().ok,true); assert.equal(inspect().runtimeVerified,false);
  for (const bad of [{...caseReceipt,case_id:cid},{...caseReceipt,harness_case_id:'spoofed'}, {...caseReceipt,historical_case:{path:historical,sha256:'a'.repeat(64)}}]) assert.equal(inspect('HEAD',bad).ok,false,'spoofed identity');
  const historicalPath=join(root,historical);
  assert.equal(inspect('FS').ok,true,'unmodified FS baseline');
  try { writeFileSync(historicalPath,Buffer.from('forged')); const bad=inspect('FS'); assert.equal(bad.ok,false); assert.equal(bad.reason,'historical case differs'); }
  finally { writeFileSync(historicalPath,historicalBytes); }
  for (const mixed of [
    {...caseReceipt,postcoef_capture:{...caseReceipt.postcoef_capture,receipt:`${legacy}/receipt.json`}},
    {...caseReceipt,raw:[`${legacy}/julia-results.json`,`${prefix}/r-oracle.json`]},
    {...caseReceipt,raw:[`${prefix}/julia-results.json`,`${legacy}/r-oracle.json`]},
    {...caseReceipt,batch_verifier:{status:'PASS',log:`${legacy}/verify.txt`}},
  ]) assert.equal(inspect('HEAD',mixed).ok,false,'mixed capture directories must be rejected');
  const logPath=join(root,prefix,'verify.txt');
  try {
    rmSync(logPath);
    assert.equal(inspect('FS').ok,false,'missing verifier transcript');
    const forged=Buffer.from('forged transcript\n'); writeFileSync(logPath,forged);
    const forgedCase={...caseReceipt,batch_verifier:{...caseReceipt.batch_verifier,log_sha256:sha(forged)}};
    assert.equal(inspect('FS',forgedCase).ok,false,'rehashed forged transcript');
  } finally { writeFileSync(logPath,verifyLog); }
  for (const mode of ['always_empty','spurious_coefficient']) for (const suffix of ['.json','.stdout.log','.stderr.log','.exit']) {
    const path=join(root,prefix,`julia-control-${mode}${suffix}`), original=readFileSync(path); rmSync(path);
    try { assert.equal(inspect('FS').ok,false); assert.equal(inspect('HEAD').ok,true); }
    finally { writeFileSync(path,original); }
  }
  const controlPath=join(root,prefix,'julia-control-always_empty.json');
  const originalControl=readFileSync(controlPath), receiptPath=join(root,prefix,'receipt.json');
  try {
    const bad=JSON.parse(originalControl); bad.cases[cid].three_fit.julia_lengths=[0,5,6];
    const changed=bytes(bad); writeFileSync(controlPath,changed);
    const rec=structuredClone(record); rec.postcoef_controls.always_empty.report_sha256=sha(changed);
    const recBytes=bytes(rec); writeFileSync(receiptPath,recBytes);
    const changedCase={...caseReceipt,postcoef_capture:{...caseReceipt.postcoef_capture,sha256:sha(recBytes)}};
    assert.equal(inspect('FS',changedCase).ok,false,'rehashed non-discriminating control');
  } finally { writeFileSync(controlPath,originalControl); writeFileSync(receiptPath,recordBytes); }
  const expectedIdentity={selected_commit:'a',selected_tree:'b',source_commit:'c',case_commit:'d',case_sha256:'e',
    capture_sha256:'f',oracle_build_sha256:'g',runtime_record:{path:'x'},runtime_root:'/r',python_executable:'/p'};
  const verifiedIdentity={...expectedIdentity,status:'POSTCOEF_LIVE_REF_VERIFIED',capture_commit:'c'};
  assert.equal(liveIdentityMatches(verifiedIdentity,expectedIdentity),true);
  for (const key of Object.keys(expectedIdentity)) {
    assert.equal(liveIdentityMatches({...verifiedIdentity,[key]:'wrong'},expectedIdentity),false,key);
    const missing={...verifiedIdentity}; delete missing[key];
    assert.equal(liveIdentityMatches(missing,expectedIdentity),false,key);
  }
  assert.equal(liveIdentityMatches({...verifiedIdentity,capture_commit:'other'},expectedIdentity),false);
  console.log('POSTCOEF_SYNTHETIC_REF_CHECK_PASS_RUNTIME_UNVERIFIED');
} finally { rmSync(root,{recursive:true,force:true}); }

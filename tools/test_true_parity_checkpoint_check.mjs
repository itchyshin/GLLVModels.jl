#!/usr/bin/env node
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { BASELINE, P1, REQUIRED, APPROVED_C6_EXTRAS, APPROVED_C6_SCOPED, HELD_C6_NAMES, MAIN_C6_EXPORT_ADDITIONS, parseBoard, compareCheckpoint, compareC6Decisions, provenanceProblem, rowMaps } from './true_parity_checkpoint_check.mjs';
const board = execFileSync('git',['show',`${BASELINE}:docs/dev-log/core070/true-parity-latest/scoreboard.md`],{encoding:'utf8'});
const base = parseBoard(board);
const maps = rowMaps(BASELINE);
assert.equal(maps.rows.length,317,'generated assembled map must not be counted twice');
assert.equal(new Set(maps.rows.map(r=>r.source_id)).size,317);
const reverseGapPath='docs/dev-log/core070/true-parity-latest/reverse-gap.json';
const c6Base=JSON.parse(execFileSync('git',['show',`${BASELINE}:${reverseGapPath}`],{encoding:'utf8'}));
const c6Candidate=JSON.parse(readFileSync(new URL(`../${reverseGapPath}`,import.meta.url),'utf8'));
assert.deepEqual(compareC6Decisions(c6Base,c6Candidate),{extras:17,scoped:11,held:9,export_additions:3});
for (const name of MAIN_C6_EXPORT_ADDITIONS) {
  const missing=c6Candidate.filter(x=>x.name!==name);
  assert.throws(()=>compareC6Decisions(c6Base,missing),/admission set changed/);
  for (const key of ['basis','ruling','derived_from','kind','decision','source_id']) {
    const bad=structuredClone(c6Candidate);
    bad.find(x=>x.name===name)[key]='fabricated';
    assert.throws(()=>compareC6Decisions(c6Base,bad),/invalid main-export census addition/);
  }
}
const invented=structuredClone(c6Candidate);
invented.push({...invented.find(x=>x.name==='optimizer_info'),name:'unsigned',source_id:'julia-export/unsigned'});
assert.throws(()=>compareC6Decisions(c6Base,invented),/admission set changed/);
const staleCensus=structuredClone(c6Candidate);
staleCensus.find(x=>x.name==='AnBSparseSolver').derived_from='fabricated';
assert.throws(()=>compareC6Decisions(c6Base,staleCensus),/census provenance changed/);
const missingExtra=structuredClone(c6Candidate);
missingExtra.find(x=>x.name===APPROVED_C6_EXTRAS[0]).decision=null;
assert.throws(()=>compareC6Decisions(c6Base,missingExtra),/approved C6 extra not applied exactly/);
const overSigned=structuredClone(c6Candidate);
overSigned.find(x=>x.name===HELD_C6_NAMES[0]).decision='SCOPED_R_TWIN';
assert.throws(()=>compareC6Decisions(c6Base,overSigned),/approved C6 hold not applied exactly/);
const signedHolds=structuredClone(c6Candidate);
for (const name of HELD_C6_NAMES) Object.assign(signedHolds.find(x=>x.name===name), {
  decision:'HOLD_PENDING_EVIDENCE', status:'held',
  basis:'held pending route-specific fit or behavioural evidence; no twin or parity claim; no scoreboard row admitted',
  ruling:{ref:'docs/dev-log/decisions/2026-10-17-p1-c6-and-c3-c4-contract-approval.md',
    signed_by:'Shinichi Nakagawa',signed_on:'2026-10-17'},
});
assert.deepEqual(compareC6Decisions(c6Base,signedHolds),{extras:17,scoped:11,held:9,export_additions:3});
const holdOnUnapproved=structuredClone(signedHolds);
holdOnUnapproved.find(x=>x.name==='StudentT').decision='HOLD_PENDING_EVIDENCE';
assert.throws(()=>compareC6Decisions(c6Base,holdOnUnapproved),/approved C6 scope not applied exactly/);
const wrongScope=structuredClone(c6Candidate);
wrongScope.find(x=>x.name===APPROVED_C6_SCOPED[0]).decision='KEPT_AS_JULIA_EXTRA';
assert.throws(()=>compareC6Decisions(c6Base,wrongScope),/approved C6 scope not applied exactly/);
const widenedScope=structuredClone(c6Candidate);
widenedScope.find(x=>x.name==='fit_gllvm').scope.exclusions=['none'];
widenedScope.find(x=>x.name==='fit_gllvm').scope.data='all families, all links';
assert.throws(()=>compareC6Decisions(c6Base,widenedScope),/approved C6 scope contents changed: fit_gllvm/);
const wrongReceipt=structuredClone(c6Candidate);
wrongReceipt.find(x=>x.name==='fit_gllvm').receipt_ids=['UNRELATED-RECEIPT'];
assert.throws(()=>compareC6Decisions(c6Base,wrongReceipt),/approved C6 scope contents changed: fit_gllvm/);
const wrongScopedRuling=structuredClone(c6Candidate);
wrongScopedRuling.find(x=>x.name==='fit_gllvm').ruling={ref:'wrong',signed_by:'someone else',signed_on:'1900-01-01'};
assert.throws(()=>compareC6Decisions(c6Base,wrongScopedRuling),/approved C6 scope contents changed: fit_gllvm/);
const candidate = () => {
  const rows = new Map([...base].map(([k,v])=>[k,{...v}]));
  for (const id of REQUIRED) rows.get(id).status = id.startsWith('family/') ? 'EVIDENCED' : 'EVIDENCED-BEHAVIOURAL';
  return rows;
};
assert.doesNotThrow(()=>compareCheckpoint(base,candidate()));
assert.throws(()=>compareCheckpoint(base,base),/required exact row/);
const wrong = candidate();
wrong.get(REQUIRED[0]).status=base.get(REQUIRED[0]).status;
const substitute=[...base].find(([k,v])=>!REQUIRED.includes(k)&&v.status==='NOT-MEASURED');
assert.ok(substitute,'baseline supplies an alternative gap');
wrong.get(substitute[0]).status='EVIDENCED';
assert.throws(()=>compareCheckpoint(base,wrong),/required exact row|unrelated status/);
const dropped=candidate();dropped.delete(substitute[0]);
assert.throws(()=>compareCheckpoint(base,dropped),/admission set/);
const changed=candidate();changed.get(REQUIRED[1]).requires='excluded';
assert.throws(()=>compareCheckpoint(base,changed),/admission or cases/);
const signed=candidate();signed.get(REQUIRED[0]).status='DISPOSITION-SIGNED';
assert.throws(()=>compareCheckpoint(base,signed),/required exact row/);
const raw='observed public refusal\n';
const read=p=>p==='raw' ? raw : JSON.stringify({reference_commit:P1});
const valid={pin:'P1',reference_commit:P1,read_from:{raw:createHash('sha256').update(raw).digest('hex')},oracle_build_receipt:'build',oracle_source_receipt:'source',evidence_kind:'r_public_bridge_boundary',verdict:'R_BOUNDARY_UNCHANGED'};
assert.equal(provenanceProblem(valid,read),null);
assert.match(provenanceProblem({...valid,reference_commit:'P0'},read),/P1/);
assert.match(provenanceProblem({...valid,read_from:{}},read),/raw hashes/);
assert.match(provenanceProblem({...valid,read_from:{raw:'0'.repeat(64)}},read),/hash mismatch/);
assert.match(provenanceProblem({...valid,verdict:'NOT_EXECUTED'},read),/bridge boundary/);
assert.match(provenanceProblem({...valid,comparison:{}},read),/bridge boundary/);
assert.match(provenanceProblem({...valid,oracle_source_receipt:undefined},read),/oracle_source/);
assert.match(provenanceProblem(valid,p=>p==='build' ? JSON.stringify({reference_commit:'P0'}) : read(p)),/oracle_build_receipt is not P1/);
console.log('P1_CHECKPOINT_CONTROLS_PASS: exact rows, unchanged contracts, P1 provenance, raw hashes and bridge context');

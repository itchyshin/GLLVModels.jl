#!/usr/bin/env node
// Negative-control tests for tools/true_parity_check.mjs (docs/dev-log/core070/true-parity-latest/GATES.md).
// Runs the checker in PARITY_REF=FS mode against small fixture trees under
// test/fixtures/true_parity/, so it needs no gllvmTMB clone and no network access.
//
// Five controls (ultra-plan.md "Acceptance ledger" / A0a brief):
//   (a) a dangling receipt fails C1
//   (b) a C8 capability with no scoreboard row fails C2
//   (c) a name-only match fails C8
//   (d) a stale carried receipt fails its row (C1)
//   (e) a positive control where everything holds prints MET, for every mode
//
// Each control asserts BOTH the verdict (MET/NOT_MET) and that the printed reason names the
// specific row/path responsible — a control that merely fails is not enough; it must fail for
// the right reason.
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import assert from 'node:assert/strict';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '..');
const CHECKER = join(REPO_ROOT, 'tools', 'true_parity_check.mjs');
const FIXTURES = join(REPO_ROOT, 'test', 'fixtures', 'true_parity');
const ALL_MODES = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'X2'];

function run(fixture, mode) {
  try {
    const out = execFileSync('node', [CHECKER, mode], {
      encoding: 'utf8',
      env: { ...process.env, PARITY_REF: 'FS', PARITY_FS_ROOT: join(FIXTURES, fixture) },
    });
    return { stdout: out, code: 0 };
  } catch (e) {
    return { stdout: e.stdout || '', code: e.status };
  }
}

let failures = 0;
function test(name, fn) {
  try {
    fn();
    console.log(`ok - ${name}`);
  } catch (e) {
    failures++;
    console.log(`NOT OK - ${name}`);
    console.log(String(e.stack || e).split('\n').map((l) => `    ${l}`).join('\n'));
  }
}

// (e) positive control: every mode MET, on the base fixture, exit 0.
test('(e) positive control: base fixture is MET on every mode', () => {
  for (const mode of ALL_MODES) {
    const { stdout, code } = run('base', mode);
    assert.equal(code, 0, `${mode} exited ${code}, expected 0\n${stdout}`);
    assert.match(stdout, new RegExp(`${mode}_MET$`, 'm'), `${mode} did not print ${mode}_MET:\n${stdout}`);
  }
});

// (a) a dangling receipt fails C1, naming the row and the missing path.
test('(a) dangling receipt fails C1', () => {
  const { stdout, code } = run('dangling_receipt', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /dangling_receipts=isdm\/CAP-ISDM-1FO-PREDICT-EXPORT:.*ghost-receipt-does-not-exist\.json/);
});

// (b) a C8-relevant capability with no scoreboard row fails C2, naming the capability.
test('(b) capability with no scoreboard row fails C2', () => {
  const { stdout, code } = run('missing_scoreboard_row', 'C2');
  assert.equal(code, 0);
  assert.match(stdout, /C2_NOT_MET$/m);
  assert.match(stdout, /capabilities_missing_scoreboard_row=CAP-GHOST/);
});

// (c) a name-only match (semantic_divergence with executable_case_ids but no signed
// disposition) fails C8, naming the row and the reason.
test('(c) name-only match fails C8', () => {
  const { stdout, code } = run('name_only_match', 'C8');
  assert.equal(code, 0);
  assert.match(stdout, /C8_NOT_MET$/m);
  assert.match(stdout, /zi\/zi_star_r_semantics_export:NAME_ONLY_MATCH_NOT_SIGNED/);
});

// (d) a stale carried receipt (sha256_at_p1 != sha256_at_p0) fails its row under C1.
test('(d) stale carried receipt fails its row', () => {
  const { stdout, code } = run('stale_carry', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /stale_carries=temporal\/CAP-TEMPORAL-1FO-EXPORT:PARTIAL_STALE_AT_P1/);
});

// An empty selection is never a pass (independent sanity check, not one of the five named
// controls, but the same GATES.md rule and cheap to verify here).
test('an empty scoreboard selection is never a pass', () => {
  const { stdout, code } = run('empty_scoreboard', 'X2');
  assert.equal(code, 0);
  assert.match(stdout, /EMPTY_SELECTION \(vacuous; not a pass\)/);
  assert.match(stdout, /X2_NOT_MET$/m);
});

// Unknown mode and missing ref file both exit 2 (MEASUREMENT_FAILED), never a silent pass.
test('an unknown mode is a measurement failure (exit 2)', () => {
  const { stdout, code } = run('base', 'NOPE');
  assert.equal(code, 2);
  assert.match(stdout, /^MEASUREMENT_FAILED/m);
});

test('a missing case-map at the ref is a measurement failure (exit 2)', () => {
  const { stdout, code } = run('missing_scoreboard_row_nonexistent_fixture_name', 'C1');
  assert.equal(code, 2);
  assert.match(stdout, /^MEASUREMENT_FAILED/m);
});

if (failures > 0) {
  console.log(`\n${failures} control(s) FAILED`);
  process.exit(1);
}
console.log('\nAll true-parity negative controls passed.');

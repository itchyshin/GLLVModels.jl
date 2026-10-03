#!/usr/bin/env node
// Negative-control tests for tools/true_parity_check.mjs (docs/dev-log/core070/true-parity-latest/GATES.md).
// Runs the checker in PARITY_REF=FS mode against small fixture trees under
// test/fixtures/true_parity/ (no gllvmTMB clone, no network access needed), plus a git-mode
// block that builds a real temp `git init` repo so `show`, `existsAsBlob` and `listDir` are
// exercised through actual `git show` / `git cat-file` / `git ls-tree`, the same code path CI
// runs against origin/main, not just the FS fallback path.
//
// These controls came out of an independent BLOCKING review of the first cut of this tool
// (PR #523): it accepted labels in place of evidence -- a receipt that resolved to a directory
// counted as present (git `cat-file -e` succeeds on trees too), an EVIDENCED row with no
// extractable receipt path counted as done, a bare `DISPOSITION-SIGNED` label or bare
// `executable_case_ids` counted as bound/twinned with no receipt, signer or date,
// `outside_boundary` rows vanished from C8 entirely, the carry rule compared author-typed
// strings with no hash-format check, and `git ls-tree` without a trailing slash never actually
// listed a workflow directory's contents in git mode. Each control below reproduces one of
// those bugs on a fixture and asserts both the verdict and the specific printed reason -- a
// control that merely fails is not enough; it must fail for the stated reason.
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { mkdtempSync, rmSync, cpSync, readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
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

// Runs the checker against a real git ref inside `repoDir` (git mode: PARITY_REF defaults to
// origin/main, so pass an explicit local ref/branch here -- 'HEAD' after a commit).
function runGit(repoDir, mode, ref = 'HEAD') {
  try {
    const out = execFileSync('node', [CHECKER, mode], {
      encoding: 'utf8',
      cwd: repoDir,
      env: { ...process.env, PARITY_REF: ref },
    });
    return { stdout: out, code: 0 };
  } catch (e) {
    return { stdout: e.stdout || '', code: e.status };
  }
}

function makeGitRepo(fixtureName) {
  const dir = mkdtempSync(join(tmpdir(), 'true-parity-git-'));
  cpSync(join(FIXTURES, fixtureName), dir, { recursive: true });
  execFileSync('git', ['init', '-q'], { cwd: dir });
  execFileSync('git', ['-c', 'user.email=test@test.invalid', '-c', 'user.name=test', 'add', '-A'], { cwd: dir });
  execFileSync('git', ['-c', 'user.email=test@test.invalid', '-c', 'user.name=test', 'commit', '-q', '-m', 'fixture'], { cwd: dir });
  return dir;
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

// --- positive control: every mode MET, on the base fixture, exit 0 (FS mode) ---
test('positive control: base fixture is MET on every mode (FS mode)', () => {
  for (const mode of ALL_MODES) {
    const { stdout, code } = run('base', mode);
    assert.equal(code, 0, `${mode} exited ${code}, expected 0\n${stdout}`);
    assert.match(stdout, new RegExp(`${mode}_MET$`, 'm'), `${mode} did not print ${mode}_MET:\n${stdout}`);
  }
});

// --- item 1: a receipt that resolves to a directory is never a valid receipt ---
test('item 1: a directory receipt fails C1 (not silently present)', () => {
  const { stdout, code } = run('receipt_is_directory', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /dangling_receipts=isdm\/CAP-ISDM-1FO-PREDICT-EXPORT:.*receipts(?!\/)/);
});
test('item 1: a directory receipt fails C8', () => {
  const { stdout, code } = run('receipt_is_directory', 'C8');
  assert.equal(code, 0);
  assert.match(stdout, /C8_NOT_MET$/m);
  assert.match(stdout, /isdm\/CAP-ISDM-1FO-PREDICT-EXPORT:DANGLING_RECEIPT/);
});

// --- item 2: EVIDENCED needs an extracted, resolving path; an uncaptured path-like receipt
// reference is a measurement failure, not a silent pass or a silent not-done ---
test('item 2: an uncaptured path-like receipt reference is MEASUREMENT_FAILED, not a pass', () => {
  const { stdout, code } = run('scoreboard_uncaptured_pathlike_receipt', 'C2');
  assert.equal(code, 2);
  assert.match(stdout, /^MEASUREMENT_FAILED/m);
  assert.match(stdout, /looks path-like but no path was extracted/);
});
test('item 2: an EVIDENCED row with an empty receipt cell is not done', () => {
  const { stdout, code } = run('scoreboard_empty_receipt_cell', 'X2');
  assert.equal(code, 0);
  assert.match(stdout, /X2_NOT_MET$/m);
  assert.match(stdout, /CAP-ISDM-1FO-PREDICT:NO_RECEIPT_PATH/);
});

// --- item 3(a): DISPOSITION-SIGNED with no signed_by/signed_on never counts as signed ---
test('item 3(a): an unsigned DISPOSITION-SIGNED label fails C1', () => {
  const { stdout, code } = run('disposition_signed_unverified', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /"DISPOSITION-SIGNED-UNVERIFIED":1/);
});
test('item 3(a): an unsigned DISPOSITION-SIGNED label fails C8', () => {
  const { stdout, code } = run('disposition_signed_unverified', 'C8');
  assert.equal(code, 0);
  assert.match(stdout, /C8_NOT_MET$/m);
  assert.match(stdout, /x\/self_signed_no_evidence:NOT_TWINNED_NOT_SIGNED/);
});

// --- item 3(b): executable_case_ids alone (no resolving receipt) never counts as bound/twinned ---
test('item 3(b): case ids without a receipt fail C1', () => {
  const { stdout, code } = run('caseids_without_receipt', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /free=1/);
});
test('item 3(b): case ids without a receipt fail C8 (not twinned)', () => {
  const { stdout, code } = run('caseids_without_receipt', 'C8');
  assert.equal(code, 0);
  assert.match(stdout, /C8_NOT_MET$/m);
  assert.match(stdout, /isdm\/CAP-ISDM-1FO-PREDICT-EXPORT:NOT_TWINNED_NOT_SIGNED/);
});

// --- item 3(c): outside_boundary without a signed disposition fails C8 (it does not vanish) ---
test('item 3(c): an unsigned outside_boundary row fails C8', () => {
  const { stdout, code } = run('outside_boundary_unsigned', 'C8');
  assert.equal(code, 0);
  assert.match(stdout, /C8_NOT_MET$/m);
  assert.match(stdout, /spatial\/spatial_dep_export:OUTSIDE_BOUNDARY_NOT_SIGNED/);
});

// --- a name-only match (semantic_divergence, unsigned) still fails C8 ---
test('a name-only match without a signed disposition fails C8', () => {
  const { stdout, code } = run('name_only_match_unsigned', 'C8');
  assert.equal(code, 0);
  assert.match(stdout, /C8_NOT_MET$/m);
  assert.match(stdout, /zi\/zi_star_export:NAME_ONLY_MATCH_NOT_SIGNED/);
});

// --- a capability with no scoreboard row fails C2 even though every scoreboard row is done ---
test('a capability with no scoreboard row fails C2', () => {
  const { stdout, code } = run('missing_scoreboard_row', 'C2');
  assert.equal(code, 0);
  assert.match(stdout, /C2_NOT_MET$/m);
  assert.match(stdout, /capabilities_missing_scoreboard_row=CAP-GHOST/);
});

// --- item 4: C0 finds the real P1 job by id + tag convention, not any P1-mentioning workflow ---
test('item 4: an advisory (continue-on-error) job with the wrong id fails C0', () => {
  const { stdout, code } = run('c0_wrong_job_advisory', 'C0');
  assert.equal(code, 0);
  assert.match(stdout, /C0_NOT_MET$/m);
  assert.match(stdout, /p1_twin_job_file=none/);
});
test('item 4: the smoke workflow content under a different filename still fails C0', () => {
  const { stdout, code } = run('c0_smoke_workflow_renamed', 'C0');
  assert.equal(code, 0);
  assert.match(stdout, /C0_NOT_MET$/m);
  assert.match(stdout, /p1_twin_job_file=none/);
});
test('item 4: a missing CAPABILITY_LEDGER_REF fails C0', () => {
  const { stdout, code } = run('c0_missing_capability_ledger_ref', 'C0');
  assert.equal(code, 0);
  assert.match(stdout, /C0_NOT_MET$/m);
  assert.match(stdout, /capability_ledger_ref_present=false/);
});

// --- false-MET fix: a switch existing is not the same as the default actually being P1.
// PR #524's real shape names the live default as a single token, _DEFAULT_PIN; C0 must read
// it directly rather than accept "a switch exists" as sufficient (that accepted #524's own
// current state -- default still P0 by design -- as C0_MET, the exact failure this tool
// exists to catch). ---
test('the false-MET fix: _DEFAULT_PIN = "P0" (the switch exists, but the default has not been flipped) fails C0', () => {
  const { stdout, code } = run('c0_default_pin_p0', 'C0');
  assert.equal(code, 0);
  assert.match(stdout, /C0_NOT_MET$/m);
  assert.match(stdout, /default_pin=P0\b/);
});
test('the false-MET fix: _DEFAULT_PIN = "P1" (the same shape, one token flipped) is what C0_MET actually requires', () => {
  const { stdout, code } = run('c0_default_pin_p1', 'C0');
  assert.equal(code, 0);
  assert.match(stdout, /C0_MET$/m);
  assert.match(stdout, /default_pin=P1\b/);
});
// Regression found while verifying the fix above against the real sibling branch: an
// unanchored regex matched the token where it appears in the module's own docstring prose
// ("flip `_DEFAULT_PIN = "P1"` below") before it ever reached the real assignment further
// down the file (`_DEFAULT_PIN = "P0"`), misreading the file as already flipped -- the exact
// false-MET this clause exists to prevent.
test('a docstring mentioning the token in prose is not mistaken for the real assignment', () => {
  const { stdout, code } = run('c0_default_pin_mentioned_in_prose_only', 'C0');
  assert.equal(code, 0);
  assert.match(stdout, /C0_NOT_MET$/m);
  assert.match(stdout, /default_pin=P0\b/);
});

// --- item 5(i)(ii): measured_against + validated 64-hex carry hashes, not opt-in strings ---
test('item 5(i): a row missing measured_against is stale, not fresh by default', () => {
  const { stdout, code } = run('carry_missing_measured_against', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /PARTIAL_STALE_AT_P1\(missing measured_against\)/);
});
test('item 5(ii): a carried row with no carry block is stale', () => {
  const { stdout, code } = run('carry_no_carry_block', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /PARTIAL_STALE_AT_P1\(no carry\.source_pins\)/);
});
test('item 5(ii): empty carry.source_pins is stale', () => {
  const { stdout, code } = run('carry_empty_source_pins', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /PARTIAL_STALE_AT_P1\(no carry\.source_pins\)/);
});
test('item 5(ii): a non-64-hex carry hash is stale (author-typed strings no longer pass)', () => {
  const { stdout, code } = run('carry_hash_not_64hex', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /PARTIAL_STALE_AT_P1\(hash mismatch or not 64-hex sha256\)/);
});
test('item 5(ii): a mismatched valid-format carry hash is stale', () => {
  const { stdout, code } = run('carry_hash_mismatch', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /PARTIAL_STALE_AT_P1\(hash mismatch or not 64-hex sha256\)/);
});

// --- item 7: enumerated C6 decision vocabulary; exact C7 heading ---
test('item 7: a placeholder decision ("TBD", blank) fails C6', () => {
  const { stdout, code } = run('c6_placeholder_decision', 'C6');
  assert.equal(code, 0);
  assert.match(stdout, /C6_NOT_MET$/m);
  assert.match(stdout, /A:"TBD"/);
  assert.match(stdout, /B:""/);
});
test('item 7: an unrelated "is not" heading no longer satisfies C7', () => {
  const { stdout, code } = run('c7_unrelated_is_not_heading', 'C7');
  assert.equal(code, 0);
  assert.match(stdout, /C7_NOT_MET$/m);
});

// --- an empty selection is never a pass ---
test('an empty scoreboard selection is never a pass', () => {
  const { stdout, code } = run('empty_scoreboard', 'X2');
  assert.equal(code, 0);
  assert.match(stdout, /EMPTY_SELECTION \(vacuous; not a pass\)/);
  assert.match(stdout, /X2_NOT_MET$/m);
});

// Unknown mode and a missing ref file both exit 2 (MEASUREMENT_FAILED), never a silent pass.
test('an unknown mode is a measurement failure (exit 2)', () => {
  const { stdout, code } = run('base', 'NOPE');
  assert.equal(code, 2);
  assert.match(stdout, /^MEASUREMENT_FAILED/m);
});
test('a missing case-map at the ref is a measurement failure (exit 2)', () => {
  const { stdout, code } = run('this-fixture-does-not-exist', 'C1');
  assert.equal(code, 2);
  assert.match(stdout, /^MEASUREMENT_FAILED/m);
});

// --- evidence tier (D-295 row 5, review of #559): a registration-only (name/export) match is
// never a numeric twin. C1 reports bound_numeric / bound_registration_only and is MET only when
// no bound row is registration-only (unless that row carries a signed disposition); a missing
// evidence_tier is fail-closed (counted as registration-only). C8 does not count it as twinned. ---
test('evidence tier: the base fixture\'s numeric rows count as bound_numeric', () => {
  const { stdout, code } = run('base', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_MET$/m);
  assert.match(stdout, /bound_numeric=2 bound_registration_only=0\b/);
});
test('evidence tier: a registration-only row does not make C1 MET', () => {
  const { stdout, code } = run('c1_registration_only', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /bound_numeric=1 bound_registration_only=1\b/);
  assert.match(stdout, /registration_only=isdm\/CAP-ISDM-1FO-PREDICT-EXPORT/);
});
test('evidence tier: a registration-only row is not twinned for C8', () => {
  const { stdout, code } = run('c1_registration_only', 'C8');
  assert.equal(code, 0);
  assert.match(stdout, /C8_NOT_MET$/m);
  assert.match(stdout, /isdm\/CAP-ISDM-1FO-PREDICT-EXPORT:REGISTRATION_ONLY_NOT_TWINNED/);
});
test('evidence tier: a missing evidence_tier is fail-closed (counted as registration-only)', () => {
  const { stdout, code } = run('c1_evidence_tier_missing', 'C1');
  assert.equal(code, 0);
  assert.match(stdout, /C1_NOT_MET$/m);
  assert.match(stdout, /bound_registration_only=1\b/);
});
test('evidence tier: a registration-only row with a real signed disposition does not block C1 or C8', () => {
  for (const mode of ['C1', 'C8']) {
    const { stdout, code } = run('c1_registration_only_signed', mode);
    assert.equal(code, 0);
    assert.match(stdout, new RegExp(`${mode}_MET$`, 'm'), `${mode}:\n${stdout}`);
  }
});

// --- numeric tier verified against the receipt (review of #561, BLOCKING): the "numeric" label
// was trusted on its own, so flipping one word on a registration row made C1_MET and C8_MET. A
// "numeric" row now needs a receipt with a machine-readable comparison block (pin P1, per-case
// abs_diff or r_value/julia_value, finite tolerance > 0, abs_diff <= tolerance, every
// executable_case_id covered); otherwise NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT. ---
for (const [fixture, why] of [
  ['c1_numeric_label_registration_receipt', /no comparison block in any receipt/],
  ['c1_numeric_label_malformed_comparison', /tolerance not a finite number > 0/],
  ['c1_numeric_label_over_tolerance', /abs_diff 0\.5 > tolerance/],
]) {
  test(`numeric tier: ${fixture} fails C1 (label without a numeric receipt does not bind)`, () => {
    const { stdout, code } = run(fixture, 'C1');
    assert.equal(code, 0);
    assert.match(stdout, /C1_NOT_MET$/m);
    assert.match(stdout, /bound_numeric=1\b/);
    assert.match(stdout, /numeric_label_without_numeric_receipt=isdm\/CAP-ISDM-1FO-PREDICT-EXPORT\(/);
    assert.match(stdout, why);
  });
  test(`numeric tier: ${fixture} fails C8 (NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT)`, () => {
    const { stdout, code } = run(fixture, 'C8');
    assert.equal(code, 0);
    assert.match(stdout, /C8_NOT_MET$/m);
    assert.match(stdout, /isdm\/CAP-ISDM-1FO-PREDICT-EXPORT:NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT/);
  });
}
// The reviewer's mutation, on the real row and its real receipts: namespace/S3method/coef,
// gllvmTMB_multi relabelled "numeric" in an otherwise faithful copy. Before this fix it printed
// C1_MET and C8_MET; it must now be NOT_MET on both.
test('numeric tier: the real coef,gllvmTMB_multi row relabelled "numeric" is NOT_MET on C1 and C8', () => {
  const cmPath = 'docs/dev-log/core070/true-parity-latest/case-map-namespace.json';
  const cm = JSON.parse(readFileSync(join(REPO_ROOT, cmPath), 'utf8'));
  const row = cm.rows.find((r) => r.source_id === 'namespace/S3method/coef,gllvmTMB_multi');
  assert.ok(row, 'real row not found in case-map-namespace.json');
  const dir = mkdtempSync(join(tmpdir(), 'true-parity-mut-'));
  try {
    for (const rp of row.evidence.receipt) {
      mkdirSync(dirname(join(dir, rp)), { recursive: true });
      cpSync(join(REPO_ROOT, rp), join(dir, rp));
    }
    writeFileSync(join(dir, 'case-map.json'), JSON.stringify({ ...cm, rows: [{ ...row, evidence_tier: 'numeric' }] }));
    for (const mode of ['C1', 'C8']) {
      let stdout = '';
      try {
        stdout = execFileSync('node', [CHECKER, mode], {
          encoding: 'utf8',
          env: { ...process.env, PARITY_REF: 'FS', PARITY_FS_ROOT: dir, PARITY_CASEMAP: 'case-map.json' },
        });
      } catch (e) { stdout = e.stdout || ''; }
      assert.match(stdout, new RegExp(`${mode}_NOT_MET$`, 'm'), `${mode}:\n${stdout}`);
      assert.match(stdout, /NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT|numeric_label_without_numeric_receipt=namespace/);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// --- receipt status (review of #567, tamper test "verdict = FAIL, comparison intact"): a
// comparison block within tolerance is not enough when the receipt itself says the run did not
// pass. A status/verdict/batch_status/harness_pass field that is not a pass value, at the top
// level or inside the comparison block, fails the row as NUMERIC_RECEIPT_NOT_PASSED; only a
// maintainer-signed receipt_status_exception on the row waives it, and then the row counts in
// bound_signed=, never in bound_numeric=. ---
for (const [fixture, why] of [
  ['c1_numeric_receipt_verdict_fail', /verdict="FAIL" in docs\/dev-log\/core070\/true-parity-latest\/receipts\/r1\.json/],
  ['c1_numeric_receipt_comparison_status_fail', /comparison\.batch_status="FAIL" in /],
]) {
  test(`receipt status: ${fixture} fails C1 (NUMERIC_RECEIPT_NOT_PASSED)`, () => {
    const { stdout, code } = run(fixture, 'C1');
    assert.equal(code, 0);
    assert.match(stdout, /C1_NOT_MET$/m);
    assert.match(stdout, /bound_numeric=1\b/);
    assert.match(stdout, /numeric_label_without_numeric_receipt=none\b/);
    assert.match(stdout, /numeric_receipt_not_passed=isdm\/CAP-ISDM-1FO-PREDICT-EXPORT\(/);
    assert.match(stdout, why);
  });
  test(`receipt status: ${fixture} fails C8 (NUMERIC_RECEIPT_NOT_PASSED)`, () => {
    const { stdout, code } = run(fixture, 'C8');
    assert.equal(code, 0);
    assert.match(stdout, /C8_NOT_MET$/m);
    assert.match(stdout, /isdm\/CAP-ISDM-1FO-PREDICT-EXPORT:NUMERIC_RECEIPT_NOT_PASSED\(/);
  });
}
test('receipt status: a maintainer-signed receipt_status_exception binds the row as bound_signed, not bound_numeric', () => {
  const c1 = run('c1_numeric_receipt_fail_signed_exception', 'C1');
  assert.equal(c1.code, 0);
  assert.match(c1.stdout, /C1_MET$/m);
  assert.match(c1.stdout, /bound=1 bound_numeric=1 bound_registration_only=0 bound_signed=2\b/);
  assert.match(c1.stdout, /numeric_receipt_not_passed=none\b/);
  const c8 = run('c1_numeric_receipt_fail_signed_exception', 'C8');
  assert.match(c8.stdout, /C8_MET$/m);
});
test('receipt status: a receipt_status_exception signed by an agent does not waive the FAIL', () => {
  const c1 = run('c1_numeric_receipt_fail_exception_by_agent', 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /numeric_receipt_not_passed=isdm\/CAP-ISDM-1FO-PREDICT-EXPORT\(verdict="FAIL" .*; DISPOSITION-SIGNER-NOT-ALLOWED\)/);
  assert.match(c1.stdout, /bound_signed=1\b/);
  const c8 = run('c1_numeric_receipt_fail_exception_by_agent', 'C8');
  assert.match(c8.stdout, /C8_NOT_MET$/m);
  assert.match(c8.stdout, /NUMERIC_RECEIPT_NOT_PASSED\(verdict="FAIL"/);
});

// --- recorded diff cross-checked (review of #567, tamper test "stale max_abs_diff, vectors
// disagree by 1"): when a case records both r_value and julia_value, the tool computes the
// difference itself and fails the row if a recorded abs_diff/max_abs_diff disagrees with it
// beyond 1e-12 relative (NUMERIC_RECORDED_DIFF_MISMATCH), whether or not either is within
// tolerance. ---
for (const [fixture, why] of [
  // vectors disagree by 1, recorded max_abs_diff 6e-11 (stale): used to bind on the recorded value
  ['c1_numeric_recorded_diff_stale', /case CASE-1: recorded max_abs_diff 6e-11 != recomputed 1\.0000000000/],
  // recorded 1e-7, recomputed 4e-7: both within tolerance 1e-6, still a mismatch
  ['c1_numeric_recorded_diff_mismatch_within_tol', /case CASE-1: recorded abs_diff 1e-7 != recomputed 3\.99999999/],
]) {
  test(`recorded diff: ${fixture} fails C1 (NUMERIC_RECORDED_DIFF_MISMATCH)`, () => {
    const { stdout, code } = run(fixture, 'C1');
    assert.equal(code, 0);
    assert.match(stdout, /C1_NOT_MET$/m);
    assert.match(stdout, /bound_numeric=1\b/);
    assert.match(stdout, /numeric_recorded_diff_mismatch=isdm\/CAP-ISDM-1FO-PREDICT-EXPORT\(/);
    assert.match(stdout, why);
  });
  test(`recorded diff: ${fixture} fails C8 (NUMERIC_RECORDED_DIFF_MISMATCH)`, () => {
    const { stdout, code } = run(fixture, 'C8');
    assert.equal(code, 0);
    assert.match(stdout, /C8_NOT_MET$/m);
    assert.match(stdout, /isdm\/CAP-ISDM-1FO-PREDICT-EXPORT:NUMERIC_RECORDED_DIFF_MISMATCH\(/);
  });
}

// --- signed-disposition hatch (review of #561): the signer must be on the maintainer allow-list
// (an agent name is refused), the date must be a real calendar date not in the future, and a
// signed row is counted in bound_signed=, never in bound= ---
test('signed hatch: a signed row is counted in bound_signed=, not in bound=', () => {
  const { stdout } = run('base', 'C1');
  assert.match(stdout, /C1 required=3 bound=2 bound_numeric=2 bound_registration_only=0 bound_signed=1\b/);
});
for (const [fixture, reason] of [
  ['c1_signed_by_agent', 'DISPOSITION-SIGNER-NOT-ALLOWED'],
  ['c1_signed_bad_date', 'DISPOSITION-SIGNED-BAD-DATE'],
  ['c1_signed_future_date', 'DISPOSITION-SIGNED-BAD-DATE'],
]) {
  test(`signed hatch: ${fixture} does not resolve the row (C1 ${reason}, C8 not signed)`, () => {
    const c1 = run(fixture, 'C1');
    assert.equal(c1.code, 0);
    assert.match(c1.stdout, /C1_NOT_MET$/m);
    assert.match(c1.stdout, new RegExp(`"${reason}":1`));
    assert.match(c1.stdout, /bound_signed=1\b/); // only the legacy row's real signature counts
    const c8 = run(fixture, 'C8');
    assert.match(c8.stdout, /C8_NOT_MET$/m);
    assert.match(c8.stdout, /isdm\/CAP-ISDM-1FO-PREDICT-EXPORT:REGISTRATION_ONLY_NOT_TWINNED/);
  });
}

// --- C8 checks receipts before signatures (review of #589, finding 2): C1 rejects a dangling
// receipt or a stale carry before it looks at a signed disposition, so C8 must too. Otherwise a
// validly signed row whose receipt is missing, or whose P0 receipt has no valid carry, passes C8
// while failing C1. Fixtures are derived from `base` in a temp dir by editing the signed
// required_core row (legacy/legacy_export_disposition, maintainer-signed, no receipt in base). ---
function runDerived(mutateRow, mode) {
  const dir = mkdtempSync(join(tmpdir(), 'true-parity-derived-'));
  try {
    cpSync(join(FIXTURES, 'base'), dir, { recursive: true });
    const cm = join(dir, 'docs', 'dev-log', 'core070', 'true-parity-latest', 'case-map.json');
    const d = JSON.parse(readFileSync(cm, 'utf8'));
    const row = d.rows.find((r) => r.source_id === 'legacy/legacy_export_disposition');
    mutateRow(row);
    writeFileSync(cm, JSON.stringify(d, null, 1));
    try {
      const out = execFileSync('node', [CHECKER, mode], {
        encoding: 'utf8',
        env: { ...process.env, PARITY_REF: 'FS', PARITY_FS_ROOT: dir },
      });
      return { stdout: out, code: 0 };
    } catch (e) {
      return { stdout: e.stdout || '', code: e.status };
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
test('C8 receipts first: a validly signed row with a dangling receipt fails C8 (and C1)', () => {
  const mutate = (r) => {
    assert.equal(r.signed_by, 'Shinichi Nakagawa');
    r.evidence = { receipt: 'docs/dev-log/core070/true-parity-latest/receipts/missing.json' };
    r.measured_against = '9539352f66f2db2cc26b1c393e67212a359b60c9';
  };
  const c8 = runDerived(mutate, 'C8');
  assert.equal(c8.code, 0);
  assert.match(c8.stdout, /C8_NOT_MET$/m);
  assert.match(c8.stdout, /legacy\/legacy_export_disposition:DANGLING_RECEIPT/);
  const c1 = runDerived(mutate, 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /dangling_receipts=legacy\/legacy_export_disposition:/);
});
test('C8 receipts first: a validly signed row citing a P0 receipt with no carry fails C8 (and C1)', () => {
  const mutate = (r) => {
    assert.equal(r.signed_by, 'Shinichi Nakagawa');
    r.evidence = { receipt: 'docs/dev-log/core070/true-parity-latest/receipts/r1.json' };
    r.measured_against = 'b4d5fee64def88bc768dda1f1f77c29b295edd86';
  };
  const c8 = runDerived(mutate, 'C8');
  assert.equal(c8.code, 0);
  assert.match(c8.stdout, /C8_NOT_MET$/m);
  assert.match(c8.stdout, /legacy\/legacy_export_disposition:STALE_CARRY\(PARTIAL_STALE_AT_P1\(no carry\.source_pins\)\)/);
  const c1 = runDerived(mutate, 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /stale_carries=legacy\/legacy_export_disposition:PARTIAL_STALE_AT_P1\(no carry\.source_pins\)/);
});

// --- family-prefixed ids: the assembled scoreboard writes `data-RD-01` / `grouping-GRP-..`, so C4
// and C5 must select by the tier marker AFTER an optional `<family>-` prefix. Fixtures are the base
// scoreboard with row ids and statuses rewritten in a temp copy. ---
function runScoreboard(edit, mode) {
  const dir = mkdtempSync(join(tmpdir(), 'true-parity-board-'));
  try {
    cpSync(join(FIXTURES, 'base'), dir, { recursive: true });
    const sb = join(dir, 'docs', 'dev-log', 'core070', 'true-parity-latest', 'scoreboard.md');
    writeFileSync(sb, edit(readFileSync(sb, 'utf8')));
    try {
      const out = execFileSync('node', [CHECKER, mode], {
        encoding: 'utf8',
        env: { ...process.env, PARITY_REF: 'FS', PARITY_FS_ROOT: dir },
      });
      return { stdout: out, code: 0 };
    } catch (e) {
      return { stdout: e.stdout || '', code: e.status };
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
const rowDone = (id, tier) => (t) => t.replace(`| ${id} |`, `| ${tier} |`);
const rowNotDone = (id, tier) => (t) => t.replace(new RegExp(`\\| ${id} \\|([^|]*)\\| EVIDENCED \\|`), `| ${tier} |$1| NOT_DONE |`);
for (const [mode, baseId, prefixed, label] of [
  ['C4', 'RD-1', 'data-RD-01', 'real-data'],
  ['C5', 'GRP-1', 'grouping-GRP-LEVEL-01', 'grouping'],
]) {
  test(`prefixed ${label} row (${prefixed}) that is NOT_DONE makes ${mode} NOT_MET (selected, not vacuous)`, () => {
    const { stdout, code } = runScoreboard(rowNotDone(baseId, prefixed), mode);
    assert.equal(code, 0);
    assert.match(stdout, new RegExp(`${mode}_NOT_MET$`, 'm'));
    assert.match(stdout, new RegExp(`rows=1 done=0 not_done=${prefixed}:NOT_DONE`));
    assert.doesNotMatch(stdout, /EMPTY_SELECTION/);
  });
  test(`prefixed ${label} row (${prefixed}) that is DONE makes ${mode} MET when it is the only row`, () => {
    const { stdout, code } = runScoreboard(rowDone(baseId, prefixed), mode);
    assert.equal(code, 0);
    assert.match(stdout, new RegExp(`${mode}_MET$`, 'm'));
    assert.match(stdout, /rows=1 done=1 not_done=none/);
  });
  test(`prefixed ${label} row is excluded from C2's board (selected by its own tier, not C2)`, () => {
    const { stdout } = runScoreboard(rowNotDone(baseId, prefixed), 'C2');
    assert.doesNotMatch(stdout, new RegExp(`${prefixed}:NOT_DONE`));
  });
}
test('a multi-hyphen family prefix (postfit-policy-RD-01) NOT_DONE selects C4 with rows=1', () => {
  const { stdout, code } = runScoreboard(rowNotDone('RD-1', 'postfit-policy-RD-01'), 'C4');
  assert.equal(code, 0);
  assert.match(stdout, /C4_NOT_MET$/m);
  assert.match(stdout, /rows=1 done=0 not_done=postfit-policy-RD-01:NOT_DONE/);
});
test('a row whose id only contains RD mid-word (family-NB2RD-X) is NOT selected by C4', () => {
  // Rename the only RD row to the mid-word id: C4 must see no rows (vacuous), not select it.
  const { stdout, code } = runScoreboard(rowNotDone('RD-1', 'family-NB2RD-X'), 'C4');
  assert.equal(code, 0);
  assert.match(stdout, /C4_NOT_MET$/m);
  assert.match(stdout, /C4 real-data workflows rows=0 EMPTY_SELECTION/);
  // ...and it stays in C2's board, where an unselected non-tier row belongs.
  const c2 = runScoreboard(rowNotDone('RD-1', 'family-NB2RD-X'), 'C2');
  assert.match(c2.stdout, /family-NB2RD-X:NOT_DONE/);
});
test('a row whose id only contains GRP mid-word (family-NB2GRP-X) is NOT selected by C5', () => {
  const { stdout } = runScoreboard(rowNotDone('GRP-1', 'family-NB2GRP-X'), 'C5');
  assert.match(stdout, /C5 grouping levels rows=0 EMPTY_SELECTION/);
});
test('a family-prefixed RSZ suffix row still selects C3 (suffix rule unchanged)', () => {
  const { stdout } = runScoreboard(rowNotDone('CAP-X-RSZ', 'size-CAP-X-RSZ'), 'C3');
  assert.match(stdout, /C3_NOT_MET$/m);
  assert.match(stdout, /rows=1 done=0 not_done=size-CAP-X-RSZ:NOT_DONE/);
});

// --- item 4 / git-mode control: show, existsAsBlob and listDir exercised through real git,
// the same code path CI runs against origin/main, not the FS fallback ---
{
  let goodRepo, dirReceiptRepo, defaultPinP0Repo;
  test('git mode: positive control base fixture is MET on every mode via a real git ref', () => {
    goodRepo = makeGitRepo('base');
    for (const mode of ALL_MODES) {
      const { stdout, code } = runGit(goodRepo, mode);
      assert.equal(code, 0, `${mode} exited ${code}, expected 0\n${stdout}`);
      assert.match(stdout, new RegExp(`${mode}_MET$`, 'm'), `${mode} did not print ${mode}_MET (git mode):\n${stdout}`);
    }
  });
  test('git mode: a directory receipt fails C1 via real `git cat-file -t` (was the original bug: `cat-file -e` succeeds on trees)', () => {
    dirReceiptRepo = makeGitRepo('receipt_is_directory');
    const { stdout, code } = runGit(dirReceiptRepo, 'C1');
    assert.equal(code, 0);
    assert.match(stdout, /C1_NOT_MET$/m);
    assert.match(stdout, /dangling_receipts=isdm\/CAP-ISDM-1FO-PREDICT-EXPORT/);
  });
  test('git mode: C0 finds the P1 job via `git ls-tree` with the trailing-slash fix', () => {
    const { stdout, code } = runGit(goodRepo, 'C0');
    assert.equal(code, 0);
    assert.match(stdout, /C0_MET$/m);
    assert.match(stdout, /p1_twin_job_file=parity-p1-twin\.yml/);
  });
  test('git mode: the false-MET fix -- _DEFAULT_PIN = "P0" fails C0 even with everything else in place (matches PR #524\'s real current state)', () => {
    defaultPinP0Repo = makeGitRepo('c0_default_pin_p0');
    const { stdout, code } = runGit(defaultPinP0Repo, 'C0');
    assert.equal(code, 0);
    assert.match(stdout, /C0_NOT_MET$/m);
    assert.match(stdout, /default_pin=P0\b/);
  });
  let regOnlyRepo;
  test('git mode: a registration-only row does not make C1 MET via a real git ref; the numeric base does', () => {
    regOnlyRepo = makeGitRepo('c1_registration_only');
    const bad = runGit(regOnlyRepo, 'C1');
    assert.equal(bad.code, 0);
    assert.match(bad.stdout, /C1_NOT_MET$/m);
    assert.match(bad.stdout, /bound_registration_only=1\b/);
    const good = runGit(goodRepo, 'C1');
    assert.match(good.stdout, /C1_MET$/m);
    assert.match(good.stdout, /bound_numeric=2 bound_registration_only=0\b/);
  });
  let numLabelRepo;
  test('git mode: a "numeric" label over a registration receipt does not make C1 MET via a real git ref', () => {
    numLabelRepo = makeGitRepo('c1_numeric_label_registration_receipt');
    const { stdout, code } = runGit(numLabelRepo, 'C1');
    assert.equal(code, 0);
    assert.match(stdout, /C1_NOT_MET$/m);
    assert.match(stdout, /numeric_label_without_numeric_receipt=isdm\/CAP-ISDM-1FO-PREDICT-EXPORT\(no comparison block/);
  });
  for (const dir of [goodRepo, dirReceiptRepo, defaultPinP0Repo, regOnlyRepo, numLabelRepo]) {
    if (dir) rmSync(dir, { recursive: true, force: true });
  }
}

// --- Rulings of 2026-10-02 (itchyshin/GLLVModels.jl#684): integer equality, the behavioural tier,
// and C6 decisions. Each control derives a fixture from `base` in a temp dir (runTree), so a
// control is a few lines of mutation and the failing reason is asserted, not just the verdict. ---
const L = 'docs/dev-log/core070/true-parity-latest';
const P1_FULL = '9539352f66f2db2cc26b1c393e67212a359b60c9';
function runTree(mutate, mode) {
  const dir = mkdtempSync(join(tmpdir(), 'true-parity-tree-'));
  try {
    cpSync(join(FIXTURES, 'base'), dir, { recursive: true });
    const readJ = (rel) => JSON.parse(readFileSync(join(dir, L, rel), 'utf8'));
    const writeJ = (rel, v) => { mkdirSync(dirname(join(dir, L, rel)), { recursive: true }); writeFileSync(join(dir, L, rel), JSON.stringify(v, null, 1)); };
    mutate({ dir, readJ, writeJ });
    try {
      return { stdout: execFileSync('node', [CHECKER, mode], { encoding: 'utf8', env: { ...process.env, PARITY_REF: 'FS', PARITY_FS_ROOT: dir } }), code: 0 };
    } catch (e) {
      return { stdout: e.stdout || '', code: e.status };
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

// Ruling 1: integer equality. The base row isdm/CAP-ISDM-1FO-PREDICT-EXPORT cites receipts/r1.json,
// CASE-1; each control rewrites that one case.
function intCase(over) {
  return ({ readJ, writeJ }) => {
    const r1 = readJ('receipts/r1.json');
    r1.comparison.cases = [{ case_id: 'CASE-1', quantity: 'df', ...over }];
    writeJ('receipts/r1.json', r1);
  };
}
test('integer equality: 15 vs 15 with tolerance 0.5 binds as numeric (C1 and C8 MET)', () => {
  const m = intCase({ kind: 'integer_equality', r_value: 15, julia_value: 15, tolerance: 0.5 });
  const c1 = runTree(m, 'C1');
  assert.match(c1.stdout, /C1_MET$/m, c1.stdout);
  assert.match(c1.stdout, /bound=2 bound_numeric=2\b/);
  assert.match(runTree(m, 'C8').stdout, /C8_MET$/m);
});
test('integer equality: equal integer vectors with tolerance 0.5 bind', () => {
  const m = intCase({ kind: 'integer_equality', r_value: [15, 3, 40], julia_value: [15, 3, 40], tolerance: 0.5 });
  assert.match(runTree(m, 'C1').stdout, /C1_MET$/m);
});
for (const [name, over, why] of [
  ['off by one (15 vs 16)', { kind: 'integer_equality', r_value: 15, julia_value: 16, tolerance: 0.5 }, /abs_diff 1 > tolerance 0\.5 \(integer_equality: the integers differ\)/],
  ['non-integer values (15.2 vs 15.2)', { kind: 'integer_equality', r_value: 15.2, julia_value: 15.2, tolerance: 0.5 }, /integer_equality needs integer r_value and julia_value/],
  ['tolerance 1 instead of 0.5', { kind: 'integer_equality', r_value: 15, julia_value: 15, tolerance: 1 }, /integer_equality needs tolerance exactly 0\.5/],
  ['vector length mismatch', { kind: 'integer_equality', r_value: [15, 3], julia_value: [15], tolerance: 0.5 }, /same shape and length/],
  ['abs_diff only, no values', { kind: 'integer_equality', abs_diff: 0, tolerance: 0.5 }, /integer_equality needs integer r_value and julia_value/],
  ['unknown kind', { kind: 'close_enough', r_value: 15, julia_value: 15, tolerance: 0.5 }, /unknown comparison kind "close_enough"/],
]) {
  test(`integer equality: ${name} fails C1 and C8 for the stated reason`, () => {
    const m = intCase(over);
    const c1 = runTree(m, 'C1');
    assert.match(c1.stdout, /C1_NOT_MET$/m);
    assert.match(c1.stdout, /numeric_label_without_numeric_receipt=isdm\/CAP-ISDM-1FO-PREDICT-EXPORT\(/);
    assert.match(c1.stdout, why);
    const c8 = runTree(m, 'C8');
    assert.match(c8.stdout, /C8_NOT_MET$/m);
    assert.match(c8.stdout, /NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT/);
  });
}
test('integer equality: unsafe integers (2^53 and beyond) are refused, so the checker and the assembler agree', () => {
  for (const [r, j] of [[9007199254740993, 9007199254740992], [9007199254740992, 9007199254740992], [-9007199254740992, -9007199254740992]]) {
    const m = intCase({ kind: 'integer_equality', r_value: r, julia_value: j, tolerance: 0.5 });
    const c1 = runTree(m, 'C1');
    assert.match(c1.stdout, /C1_NOT_MET$/m, `${r} ${j}`);
    assert.match(c1.stdout, /integer_equality needs integer r_value and julia_value/);
  }
  const edge = intCase({ kind: 'integer_equality', r_value: 9007199254740991, julia_value: 9007199254740991, tolerance: 0.5 });
  assert.match(runTree(edge, 'C1').stdout, /C1_MET$/m);
});
test('integer equality: a case with no kind keeps today rule (tolerance 0.5 on non-integers still judged by difference)', () => {
  const m = intCase({ r_value: 15.2, julia_value: 15.2, tolerance: 0.5 });
  assert.match(runTree(m, 'C1').stdout, /C1_MET$/m);
});

// Ruling 2: the behavioural tier. bTree() adds one required_core row, inference/CI-ROUTE-001, of
// tier "behavioural", citing receipts/b1.json, to the base map.
const B_ROW = 'inference/CI-ROUTE-001';
function bTree({ row = {}, receipt, equivalence, extraReceipts = {}, extraRows = [] } = {}) {
  return ({ readJ, writeJ }) => {
    const cm = readJ('case-map.json');
    const base = {
      source_id: B_ROW, classification: 'required_core', capability: 'CAP-ISDM-1FO-PREDICT',
      executable_case_ids: ['CASE-B'], evidence: { receipt: `${L}/receipts/b1.json` },
      measured_against: P1_FULL, disposition: null, evidence_tier: 'behavioural',
    };
    cm.rows.push({ ...base, ...row });
    for (const extra of extraRows) cm.rows.push({ ...base, ...extra }); // more rows citing the same case id
    writeJ('case-map.json', cm);
    writeJ('receipts/b1.json', receipt ?? bReceipt());
    for (const [rel, v] of Object.entries(extraReceipts)) writeJ(rel, v);
    if (equivalence !== undefined) writeJ('behaviour-equivalence.json', equivalence);
  };
}
function bCase(over = {}) {
  return { case_id: 'CASE-B', kind: 'route', r_observed: 'wald', julia_observed: 'wald', ...over };
}
function bReceipt(top = {}, blockOver = {}, cases = [bCase()]) {
  return { case_id: 'B1', result: 'PASS', ...top, behaviour: { pin: 'P1', cases, ...blockOver } };
}
const eq = (...classes) => ({ schema: 1, pin: 'P1', classes });
const waldClass = { kind: 'route', canonical: 'wald', r: ['.confint_lambda:wald'], julia: ['wald_packed'], basis: 'R .confint_lambda and Julia confint_lambda route the Wald interval.' };

test('behavioural: matching labels bind; counted in bound_behavioural, never bound= or bound_numeric=', () => {
  const c1 = runTree(bTree(), 'C1');
  assert.match(c1.stdout, /C1_MET$/m, c1.stdout);
  assert.match(c1.stdout, /C1 required=4 bound=2 bound_numeric=2 bound_registration_only=0 bound_signed=1 free=0\b/);
  assert.match(c1.stdout, /bound_behavioural=1 behavioural_label_without_behavioural_receipt=none$/m);
  assert.match(runTree(bTree(), 'C8').stdout, /C8_MET$/m);
});
test('behavioural: a route mismatch with no equivalence class fails C1 and C8', () => {
  const m = bTree({ receipt: bReceipt({}, {}, [bCase({ r_observed: '.confint_lambda:wald', julia_observed: 'wald_packed' })]) });
  const c1 = runTree(m, 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /bound_behavioural=0 /);
  assert.match(c1.stdout, /behavioural_label_without_behavioural_receipt=inference\/CI-ROUTE-001\(case CASE-B \(route\): R ".confint_lambda:wald" vs Julia "wald_packed" differ after canonicalisation/);
  const c8 = runTree(m, 'C8');
  assert.match(c8.stdout, /C8_NOT_MET$/m);
  assert.match(c8.stdout, /inference\/CI-ROUTE-001:BEHAVIOURAL_LABEL_WITHOUT_BEHAVIOURAL_RECEIPT\(/);
});
test('behavioural: the same mismatch is rescued by an equivalence class', () => {
  const m = bTree({ receipt: bReceipt({}, {}, [bCase({ r_observed: '.confint_lambda:wald', julia_observed: 'wald_packed' })]), equivalence: eq(waldClass) });
  assert.match(runTree(m, 'C1').stdout, /C1_MET$/m);
  assert.match(runTree(m, 'C8').stdout, /C8_MET$/m);
});
test('behavioural: a class of another kind does not rescue (kinds are separate)', () => {
  const m = bTree({ receipt: bReceipt({}, {}, [bCase({ kind: 'refusal', r_observed: '.confint_lambda:wald', julia_observed: 'wald_packed' })]), equivalence: eq(waldClass) });
  const c1 = runTree(m, 'C1').stdout;
  assert.match(c1, /C1_NOT_MET$/m);
  assert.match(c1, /bound_behavioural=0 /);
  // the reason is the missing class (a refusal lookup finds no route class), not any other failure of the fixture
  assert.match(c1, /case CASE-B \(refusal\): R ".confint_lambda:wald" vs Julia "wald_packed" differ after canonicalisation \(R: no class; Julia: no class\)/);
});
test('behavioural: an ambiguous equivalence table is MEASUREMENT_FAILED (exit 2) on C1 and C8', () => {
  const dup = { ...waldClass, canonical: 'wald_other' };
  const m = bTree({ equivalence: eq(waldClass, dup) });
  for (const mode of ['C1', 'C8']) {
    const r = runTree(m, mode);
    assert.equal(r.code, 2, r.stdout);
    assert.match(r.stdout, /MEASUREMENT_FAILED .*ambiguous table, route r label ".confint_lambda:wald" is in classes wald and wald_other/);
  }
});
test('behavioural: a class with an empty basis is MEASUREMENT_FAILED', () => {
  const r = runTree(bTree({ equivalence: eq({ ...waldClass, basis: ' ' }) }), 'C1');
  assert.equal(r.code, 2);
  assert.match(r.stdout, /empty basis/);
});
test('behavioural: an executable case id with no behaviour entry fails', () => {
  const m = bTree({ row: { executable_case_ids: ['CASE-B', 'CASE-B2'] } });
  const c1 = runTree(m, 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /case ids without an applicable behaviour entry: CASE-B2/);
  assert.match(runTree(m, 'C8').stdout, /C8_NOT_MET$/m);
});
test('behavioural: an entry scoped to another source_id does not cover this row; one scoped to it does', () => {
  const other = bTree({ receipt: bReceipt({}, {}, [bCase({ source_id: 'inference/CI-OTHER' })]) });
  const c1 = runTree(other, 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /case ids without an applicable behaviour entry: CASE-B/);
  assert.match(runTree(other, 'C8').stdout, /C8_NOT_MET$/m);
  const own = bTree({ receipt: bReceipt({}, {}, [bCase({ source_id: B_ROW })]) });
  assert.match(runTree(own, 'C1').stdout, /C1_MET$/m);
});
test('behavioural: a scoped mismatching entry for this row fails even if an unscoped entry matches', () => {
  const m = bTree({ receipt: bReceipt({}, {}, [bCase(), bCase({ source_id: B_ROW, julia_observed: 'profile' })]) });
  const c1 = runTree(m, 'C1').stdout;
  assert.match(c1, /C1_NOT_MET$/m);
  assert.match(c1, /bound_behavioural=0 /);
  assert.match(c1, /case CASE-B \(route\): R "wald" vs Julia "profile" differ after canonicalisation/);
});
test('behavioural: a receipt verdict of FAIL does not bind (top level and inside the block)', () => {
  for (const m of [bTree({ receipt: bReceipt({ verdict: 'FAIL' }) }), bTree({ receipt: bReceipt({}, { batch_status: 'FAIL' }) })]) {
    const c1 = runTree(m, 'C1');
    assert.match(c1.stdout, /C1_NOT_MET$/m);
    assert.match(c1.stdout, /receipt did not pass: (verdict="FAIL"|behaviour\.batch_status="FAIL") in /);
    const c8 = runTree(m, 'C8');
    assert.match(c8.stdout, /C8_NOT_MET$/m);
    assert.match(c8.stdout, /BEHAVIOURAL_LABEL_WITHOUT_BEHAVIOURAL_RECEIPT\(receipt did not pass/);
  }
});
test('behavioural: tier behavioural but a receipt with only a numeric comparison block fails', () => {
  const m = bTree({ receipt: { case_id: 'B1', result: 'PASS', comparison: { pin: 'P1', cases: [{ case_id: 'CASE-B', r_value: 1, julia_value: 1, tolerance: 1e-6 }] } } });
  const c1 = runTree(m, 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /no behaviour block in any receipt/);
  assert.match(c1.stdout, /bound_numeric=2\b/);
  assert.match(runTree(m, 'C8').stdout, /BEHAVIOURAL_LABEL_WITHOUT_BEHAVIOURAL_RECEIPT\(no behaviour block/);
});
test('behavioural: a behaviour block not pinned to P1 fails', () => {
  const m = bTree({ receipt: bReceipt({}, { pin: 'P0' }) });
  const c1 = runTree(m, 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /behaviour not pinned to P1/);
  assert.match(runTree(bTree({ receipt: bReceipt({}, { pin: P1_FULL }) }), 'C1').stdout, /C1_MET$/m);
});
test('behavioural: an empty or blank observed label fails', () => {
  for (const over of [{ r_observed: '' }, { julia_observed: '   ' }, { r_observed: ['wald', ''], julia_observed: ['wald', 'x'] }, { r_observed: [], julia_observed: [] }, { r_observed: 3, julia_observed: 3 }]) {
    const c1 = runTree(bTree({ receipt: bReceipt({}, {}, [bCase(over)]) }), 'C1');
    assert.match(c1.stdout, /C1_NOT_MET$/m, JSON.stringify(over));
    assert.match(c1.stdout, /must be a non-empty string or an array of non-empty strings/);
  }
});
test('behavioural: array observed labels of different lengths fail; equal arrays compared elementwise', () => {
  const bad = runTree(bTree({ receipt: bReceipt({}, {}, [bCase({ kind: 'printed_fields', r_observed: ['call', 'df', 'aic'], julia_observed: ['call', 'df'] })]) }), 'C1');
  assert.match(bad.stdout, /C1_NOT_MET$/m);
  assert.match(bad.stdout, /r_observed has 3 labels, julia_observed 2/);
  const shape = runTree(bTree({ receipt: bReceipt({}, {}, [bCase({ r_observed: ['wald'], julia_observed: 'wald' })]) }), 'C1');
  assert.match(shape.stdout, /same shape/);
  const ok = bTree({ receipt: bReceipt({}, {}, [bCase({ kind: 'printed_fields', r_observed: ['call', 'df', 'aic'], julia_observed: ['call', 'df', 'aic'] })]) });
  assert.match(runTree(ok, 'C1').stdout, /C1_MET$/m);
  const off = bTree({ receipt: bReceipt({}, {}, [bCase({ kind: 'printed_fields', r_observed: ['call', 'df', 'aic'], julia_observed: ['call', 'df', 'bic'] })]) });
  const offOut = runTree(off, 'C1').stdout;
  assert.match(offOut, /C1_NOT_MET$/m);
  assert.match(offOut, /R "aic" vs Julia "bic"/);
});
test('behavioural: an invalid kind fails', () => {
  const c1 = runTree(bTree({ receipt: bReceipt({}, {}, [bCase({ kind: 'vibes' })]) }), 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /kind "vibes" is not one of route\|refusal\|error_class\|printed_fields/);
});
test('behavioural: a stale carry fails before the receipt is read', () => {
  const m = bTree({ row: { measured_against: 'b4d5fee64def88bc768dda1f1f77c29b295edd86' } });
  const c1 = runTree(m, 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /stale_carries=inference\/CI-ROUTE-001:PARTIAL_STALE_AT_P1/);
  const c8 = runTree(m, 'C8');
  assert.match(c8.stdout, /inference\/CI-ROUTE-001:STALE_CARRY/);
});
test('behavioural: a dangling receipt fails (C1 dangling_receipts)', () => {
  const m = bTree({ row: { evidence: { receipt: `${L}/receipts/not-there.json` } } });
  const c1 = runTree(m, 'C1');
  assert.match(c1.stdout, /dangling_receipts=inference\/CI-ROUTE-001:/);
});
test('behavioural: a row with no executable_case_ids is free, not bound', () => {
  const c1 = runTree(bTree({ row: { executable_case_ids: [] } }), 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /free=1\b/);
  assert.match(c1.stdout, /bound_behavioural=0 /);
});
test('behavioural: an unknown tier is still registration-only (the new tier did not loosen it)', () => {
  const c1 = runTree(bTree({ row: { evidence_tier: 'behavioral' } }), 'C1');
  assert.match(c1.stdout, /bound_registration_only=1\b/);
  assert.match(c1.stdout, /bound_behavioural=0 /);
});
test('behavioural: a behavioural label cannot ride on another row numeric receipt (numeric row, behaviour-only receipt)', () => {
  const m = ({ readJ, writeJ }) => {
    const r1 = readJ('receipts/r1.json');
    delete r1.comparison;
    r1.behaviour = { pin: 'P1', cases: [{ case_id: 'CASE-1', kind: 'route', r_observed: 'a', julia_observed: 'a' }] };
    writeJ('receipts/r1.json', r1);
  };
  const c1 = runTree(m, 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /numeric_label_without_numeric_receipt=isdm\/CAP-ISDM-1FO-PREDICT-EXPORT\(no comparison block/);
});

// Scope of the behavioural tier (review of the first cut: any row could relabel itself behavioural).
// Only inference/* rows and the four named C1 rows are covered by itchyshin/GLLVModels.jl#684 item 2.
test('behavioural scope: a row outside the frozen list (59 inference rows, four named rows) does not bind, even with a valid block', () => {
  for (const sid of ['isdm/CAP-ISDM-1FO-PREDICT-EXPORT-2', 'postfit/POSTFIT-SURFACE-extract_proportions', 'data/RD-01', 'inference2/CI-ROUTE-001', 'x/inference/CI-ROUTE-001',
    // the four inference rows #684 item 2 does not name (two numeric, two partial): a prefix rule admitted them
    'inference/CI-ROUTE-008', 'inference/CI-ROUTE-009', 'inference/CI-ROUTE-010', 'inference/CI-ROUTE-011',
    // a bare prefix, a path trick, a new unlisted inference id, a gap in the numbering, whitespace and case variants of a listed id
    'inference/', 'inference/../isdm/X', 'inference/CI-ROUTE-999', 'inference/CI-ROUTE-005', 'inference/CI-ROUTE-001 ', ' inference/CI-ROUTE-001', 'Inference/CI-ROUTE-001']) {
    const m = ({ readJ, writeJ }) => { bTree({ row: { source_id: sid } })({ readJ, writeJ }); };
    const c1 = runTree(m, 'C1');
    assert.match(c1.stdout, /C1_NOT_MET$/m, sid);
    assert.match(c1.stdout, /bound_behavioural=0 /, sid);
    assert.match(c1.stdout, new RegExp(`behavioural_label_without_behavioural_receipt=${sid.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&')}\\(source_id not covered by itchyshin/GLLVModels\\.jl#684 item 2`), sid);
    const c8 = runTree(m, 'C8');
    assert.match(c8.stdout, /C8_NOT_MET$/m, sid);
    assert.match(c8.stdout, /BEHAVIOURAL_LABEL_WITHOUT_BEHAVIOURAL_RECEIPT\(source_id not covered/, sid);
  }
});
test('behavioural scope: each of the four named C1 rows binds like an inference row', () => {
  for (const sid of ['latent-scores/extract_latent_scores.default', 'select-lv/print.gllvmTMB_select_lv', 'model-comparison/print.anova.gllvmTMB_multi', 'model-comparison/update.gllvmTMB_multi', 'inference/CI-ROUTE-001', 'inference/CI-ROUTE-084']) {
    const m = bTree({ row: { source_id: sid } });
    const c1 = runTree(m, 'C1');
    assert.match(c1.stdout, /C1_MET$/m, `${sid}\n${c1.stdout}`);
    assert.match(c1.stdout, /bound_behavioural=1 /, sid);
    assert.match(runTree(m, 'C8').stdout, /C8_MET$/m, sid);
  }
});
test('behavioural scope: a cited receipt whose own comparison is out of tolerance does not bind a relabelled row', () => {
  const bad = { pin: 'P1', cases: [{ case_id: 'CASE-B', quantity: 'q', r_value: 1, julia_value: 99, tolerance: 1e-6 }] };
  const m = bTree({ receipt: { ...bReceipt(), comparison: bad } });
  const c1 = runTree(m, 'C1');
  assert.match(c1.stdout, /C1_NOT_MET$/m);
  assert.match(c1.stdout, /bound_behavioural=0 /);
  assert.match(c1.stdout, /cited receipt's own comparison fails: case CASE-B: abs_diff 98 > tolerance 0\.000001/);
  assert.match(runTree(m, 'C8').stdout, /C8_NOT_MET$/m);
  const good = { pin: 'P1', cases: [{ case_id: 'CASE-B', quantity: 'q', r_value: 1, julia_value: 1, tolerance: 1e-6 }] };
  assert.match(runTree(bTree({ receipt: { ...bReceipt(), comparison: good } }), 'C1').stdout, /C1_MET$/m);
});
test('behavioural: a case-level verdict of FAIL does not bind', () => {
  for (const over of [{ verdict: 'FAIL' }, { harness_pass: false }, { status: 'error' }]) {
    const c1 = runTree(bTree({ receipt: bReceipt({}, {}, [bCase(over)]) }), 'C1');
    assert.match(c1.stdout, /C1_NOT_MET$/m, JSON.stringify(over));
    assert.match(c1.stdout, /receipt did not pass: behaviour\.cases\[CASE-B\]\./);
  }
});
test('behavioural: canonicalisation is side-specific (R label in the Julia slot does not match)', () => {
  const eqv = eq(waldClass);
  const swapped = bTree({ receipt: bReceipt({}, {}, [bCase({ r_observed: 'wald_packed', julia_observed: '.confint_lambda:wald' })]), equivalence: eqv });
  assert.match(runTree(swapped, 'C1').stdout, /C1_NOT_MET$/m);
  const ok = bTree({ receipt: bReceipt({}, {}, [bCase({ r_observed: '.confint_lambda:wald', julia_observed: 'wald_packed' })]), equivalence: eqv });
  assert.match(runTree(ok, 'C1').stdout, /C1_MET$/m);
});
test('behavioural: the carry rule is applied inside the behavioural check for a row outside C1 scope (C8)', () => {
  const m = bTree({ row: { classification: 'extra_surface', measured_against: 'b4d5fee64def88bc768dda1f1f77c29b295edd86' } });
  const c8 = runTree(m, 'C8');
  assert.match(c8.stdout, /C8_NOT_MET$/m);
  assert.match(c8.stdout, /inference\/CI-ROUTE-001:BEHAVIOURAL_LABEL_WITHOUT_BEHAVIOURAL_RECEIPT\(PARTIAL_STALE_AT_P1/);
  const fresh = runTree(bTree({ row: { classification: 'extra_surface' } }), 'C8');
  assert.match(fresh.stdout, /C8_MET$/m, fresh.stdout);
});
test('behaviour-equivalence.json is validated whether or not a behavioural row exists (C1 and C8)', () => {
  const noRows = (equivalence) => ({ writeJ }) => writeJ('behaviour-equivalence.json', equivalence);
  const cases = [
    ['schema 2', { schema: 2, pin: 'P1', classes: [] }, /schema must be 1/],
    ['pin P0', { schema: 1, pin: 'P0', classes: [] }, /pin must be P1/],
    ['classes not an array', { schema: 1, pin: 'P1', classes: {} }, /classes must be an array/],
    ['ambiguous label', eq(waldClass, { ...waldClass, canonical: 'other' }), /ambiguous table/],
    ['duplicate canonical in one kind', eq(waldClass, { ...waldClass, r: ['profile'], julia: ['profile_x'] }), /duplicate canonical "wald" for kind route/],
    ['empty basis', eq({ ...waldClass, basis: '' }), /empty basis/],
  ];
  for (const [name, table, why] of cases) {
    for (const mode of ['C1', 'C8']) {
      const r = runTree(noRows(table), mode);
      assert.equal(r.code, 2, `${name} ${mode}: ${r.stdout}`);
      assert.match(r.stdout, why, name);
    }
  }
  const sameCanonicalOtherKind = runTree(noRows(eq(waldClass, { ...waldClass, kind: 'refusal' })), 'C1');
  assert.equal(sameCanonicalOtherKind.code, 0, 'one canonical label may serve two kinds');
  assert.equal(runTree(noRows(eq()), 'C1').code, 0);
});

// Scoreboard: EVIDENCED-BEHAVIOURAL is done only on a row ruling 2 covers (an inference- id) and is counted
// in done_behavioural; BEHAVIOURAL-UNVERIFIED is not done; it never closes a C3, C4 or C5 row.
const RECEIPT_CELL = `${L}/receipts/r1.json`;
function runBoard(status, mode, id = 'inference-CI-ROUTE-001') {
  return runTree(({ dir }) => {
    const sb = join(dir, L, 'scoreboard.md');
    writeFileSync(sb, `${readFileSync(sb, 'utf8')}| ${id} | routing | ${status} | ${RECEIPT_CELL} | fixture |\n`);
  }, mode);
}
function runBoardEdit(status, mode, rowId) {
  return runTree(({ dir }) => {
    const sb = join(dir, L, 'scoreboard.md');
    const re = new RegExp(`^(\\| ${rowId} \\|[^|]*\\| )EVIDENCED( \\|)`, 'm');
    const t = readFileSync(sb, 'utf8');
    assert.match(t, re);
    writeFileSync(sb, t.replace(re, `$1${status}$2`));
  }, mode);
}
test('scoreboard: EVIDENCED-BEHAVIOURAL on an inference- row is done and reported as done_behavioural', () => {
  const x2 = runBoard('EVIDENCED-BEHAVIOURAL', 'X2');
  assert.match(x2.stdout, /X2_MET$/m, x2.stdout);
  assert.match(x2.stdout, /rows=6 done=6 not_done=none done_behavioural=1$/m);
  const c2 = runBoard('EVIDENCED-BEHAVIOURAL', 'C2');
  assert.match(c2.stdout, /C2 P1-boundary capabilities rows=3 done=3 not_done=none done_behavioural=1$/m);
  assert.match(c2.stdout, /C2_MET$/m);
});
test('scoreboard: EVIDENCED-BEHAVIOURAL never closes C3, C4 or C5 (campaign clauses are numeric)', () => {
  for (const [mode, rowId, label] of [['C3', 'CAP-X-RSZ', 'realistic-size'], ['C4', 'RD-1', 'real-data workflows'], ['C5', 'GRP-1', 'grouping levels']]) {
    const r = runBoardEdit('EVIDENCED-BEHAVIOURAL', mode, rowId);
    assert.match(r.stdout, new RegExp(`^${mode} ${label} rows=1 done=0 not_done=${rowId}:BEHAVIOURAL_NOT_ALLOWED_FOR_THIS_ROW done_behavioural=0$`, 'm'), r.stdout);
    assert.match(r.stdout, new RegExp(`${mode}_NOT_MET$`, 'm'));
    const x2 = runBoardEdit('EVIDENCED-BEHAVIOURAL', 'X2', rowId);
    assert.match(x2.stdout, /X2_NOT_MET$/m);
  }
  const rsz = runBoard('EVIDENCED-BEHAVIOURAL', 'C3', 'inference-X-RSZ');
  assert.match(rsz.stdout, /inference-X-RSZ:BEHAVIOURAL_NOT_ALLOWED_FOR_THIS_ROW/);
  assert.match(rsz.stdout, /C3_NOT_MET$/m);
});
test('scoreboard: EVIDENCED-BEHAVIOURAL on a row ruling 2 does not cover is not done (C2 and X2)', () => {
  const r = runBoardEdit('EVIDENCED-BEHAVIOURAL', 'C2', 'CAP-ISDM-1FO-PREDICT');
  assert.match(r.stdout, /C2_NOT_MET$/m);
  assert.match(r.stdout, /CAP-ISDM-1FO-PREDICT:BEHAVIOURAL_NOT_ALLOWED_FOR_THIS_ROW/);
  assert.match(runBoardEdit('EVIDENCED-BEHAVIOURAL', 'X2', 'CAP-TEMPORAL-1FO').stdout, /X2_NOT_MET$/m);
  assert.match(runBoard('EVIDENCED-BEHAVIOURAL', 'X2', 'inference2-CI').stdout, /inference2-CI:BEHAVIOURAL_NOT_ALLOWED_FOR_THIS_ROW/);
  assert.match(runBoard('EVIDENCED-BEHAVIOURAL', 'X2', 'select-lv-print-gllvmTMB_select_lv').stdout, /X2_MET$/m);
});
test('scoreboard: BEHAVIOURAL-UNVERIFIED is not done; plain EVIDENCED reports done_behavioural=0', () => {
  const bad = runBoard('BEHAVIOURAL-UNVERIFIED', 'X2');
  assert.match(bad.stdout, /X2_NOT_MET$/m);
  assert.match(bad.stdout, /inference-CI-ROUTE-001:NOT_DONE/);
  assert.match(bad.stdout, /done_behavioural=0$/m);
  assert.match(runBoard('EVIDENCED', 'X2').stdout, /done=6 not_done=none done_behavioural=0$/m);
});
// Follow-up to the review of #687: the assembler writes "not bound; cited: ..." in the receipt cell of every row it
// did not bind, and a disposition once became a Status word on such a row ("EVIDENCED" with a receipt that exists).
// The checker reads the Status word alone, so it also refuses a done word on a row the assembler wrote as not bound.
function runBoardCell(status, receiptCell, mode, id = 'inference-CI-ROUTE-001') {
  return runTree(({ dir }) => {
    const sb = join(dir, L, 'scoreboard.md');
    writeFileSync(sb, `${readFileSync(sb, 'utf8')}| ${id} | routing | ${status} | ${receiptCell} | fixture |\n`);
  }, mode);
}
for (const status of ['EVIDENCED', 'EVIDENCED-BEHAVIOURAL', 'DISPOSITION-SIGNED']) {
  test(`scoreboard: ${status} on a row whose receipt cell says "not bound" is not done (X2 and C2), though the cited path exists`, () => {
    const cell = `not bound; cited: ${RECEIPT_CELL}`;
    const x2 = runBoardCell(status, cell, 'X2');
    assert.match(x2.stdout, /X2_NOT_MET$/m, x2.stdout);
    assert.match(x2.stdout, /inference-CI-ROUTE-001:STATUS_NOT_BOUND/);
    assert.match(x2.stdout, /done=5 not_done=inference-CI-ROUTE-001:STATUS_NOT_BOUND done_behavioural=0$/m);
    const c2 = runBoardCell(status, cell, 'C2');
    assert.match(c2.stdout, /C2_NOT_MET$/m, c2.stdout);
    assert.match(c2.stdout, /inference-CI-ROUTE-001:STATUS_NOT_BOUND/);
  });
}
test('scoreboard: the same done word with a bound receipt cell is still done, so only "not bound" rows are refused', () => {
  for (const status of ['EVIDENCED', 'EVIDENCED-BEHAVIOURAL']) {
    const r = runBoardCell(status, RECEIPT_CELL, 'X2');
    assert.match(r.stdout, /X2_MET$/m, r.stdout);
  }
  const signed = runBoardCell('DISPOSITION-SIGNED', 'Disposition: outside_boundary; signed_by: Shinichi Nakagawa; signed_on: 2026-09-27', 'X2', 'inference-CI-ROUTE-001');
  assert.match(signed.stdout, /X2_MET$/m, signed.stdout);
});
test('scoreboard: a status that is not a done word stays not done whatever its receipt cell says', () => {
  const r = runBoardCell('NOT-MEASURED', `not bound; cited: ${RECEIPT_CELL}`, 'X2');
  assert.match(r.stdout, /X2_NOT_MET$/m);
  assert.match(r.stdout, /inference-CI-ROUTE-001:NOT_DONE/);
});
test('scoreboard: an EVIDENCED-BEHAVIOURAL row with no receipt path is not done', () => {
  const r = runTree(({ dir }) => {
    const sb = join(dir, L, 'scoreboard.md');
    writeFileSync(sb, `${readFileSync(sb, 'utf8')}| inference-CI-ROUTE-001 | routing | EVIDENCED-BEHAVIOURAL | | fixture |\n`);
  }, 'X2');
  assert.match(r.stdout, /X2_NOT_MET$/m);
  assert.match(r.stdout, /inference-CI-ROUTE-001:NO_RECEIPT_PATH/);
});

// Ruling 3: C6 decisions. The base fixture's two items now carry a basis and a ruling.
const RULING = { ref: 'itchyshin/GLLVModels.jl#684 item 3', signed_by: 'Shinichi Nakagawa', signed_on: '2026-10-02' };
function c6Items(items) {
  return ({ dir }) => writeFileSync(join(dir, L, 'reverse-gap.json'), JSON.stringify(items));
}
const helper = (over = {}) => ({ source_id: 'julia-export/internal_thing', decision: 'EXCLUDED_INTERNAL_HELPER', basis: 'no docstring in docs/src/low-level-reference.md', ruling: RULING, ...over });
// A documented extra cites a docs/src file that exists (the base fixture has docs/src/gllvmtmb-parity.md).
const kept = (over = {}) => helper({ source_id: 'julia-export/doc', decision: 'KEPT_AS_JULIA_EXTRA', basis: 'documented in docs/src/gllvmtmb-parity.md', ...over });
test('C6: EXCLUDED_INTERNAL_HELPER with a basis and the maintainer ruling is a valid decision, with counts', () => {
  const r = runTree(c6Items([helper(), helper({ source_id: 'julia-export/x2' }), kept()]), 'C6');
  assert.match(r.stdout, /C6_MET$/m, r.stdout);
  assert.match(r.stdout, /invalid_decision=none /);
  assert.match(r.stdout, /unsigned_decision=none decision_counts=KEPT_AS_JULIA_EXTRA:1,PORT_TO_MATCH_R:0,DEPRECATE_AND_REMOVE:0,RENAME_TO_AVOID_COLLISION:0,EXCLUDED_INTERNAL_HELPER:2$/m);
});
for (const [name, item, why] of [
  ['a decided item with no ruling', helper({ ruling: undefined }), /unsigned_decision=julia-export\/internal_thing\(no ruling\)/],
  ['a ruling without a ref', helper({ ruling: { ...RULING, ref: '' } }), /\(ruling without a ref\)/],
  ['a ruling signed by an agent name', helper({ ruling: { ...RULING, signed_by: 'Claude Opus (agent)' } }), /\(DISPOSITION-SIGNER-NOT-ALLOWED\)/],
  ['a ruling signed by a name outside the allow-list', helper({ ruling: { ...RULING, signed_by: 'Someone Else' } }), /\(DISPOSITION-SIGNER-NOT-ALLOWED\)/],
  ['a ruling with a future date', helper({ ruling: { ...RULING, signed_on: '2999-01-01' } }), /\(DISPOSITION-SIGNED-BAD-DATE\)/],
  ['a ruling with no signer', helper({ ruling: { ref: RULING.ref } }), /\(DISPOSITION-SIGNED-UNVERIFIED\)/],
  ['a decided item without a basis', helper({ basis: '' }), /\(no basis\)/],
  ['a decided item with a missing basis field', helper({ basis: undefined }), /\(no basis\)/],
]) {
  test(`C6: ${name} is reported under unsigned_decision and fails`, () => {
    const r = runTree(c6Items([item]), 'C6');
    assert.equal(r.code, 0);
    assert.match(r.stdout, /C6_NOT_MET$/m);
    assert.match(r.stdout, /invalid_decision=none /);
    assert.match(r.stdout, why);
  });
}
// Scope of the C6 signature (review of the first cut): the ruling ref is an allow-list, a ruling covers only
// the words it names, and signed_on is pinned to the ruling's date.
for (const [name, item, why] of [
  ['an unrecognised ruling ref', helper({ ruling: { ...RULING, ref: 'anything' } }), /\(ruling ref "anything" is not a recognised signed ruling\)/],
  ['a bare #684 reference with no item', helper({ ruling: { ...RULING, ref: '#684' } }), /not a recognised signed ruling/],
  ['PORT_TO_MATCH_R under the item 3 ruling', helper({ decision: 'PORT_TO_MATCH_R' }), /\(decision PORT_TO_MATCH_R is not covered by itchyshin\/GLLVModels\.jl#684 item 3\)/],
  ['DEPRECATE_AND_REMOVE under the item 3 ruling', helper({ decision: 'DEPRECATE_AND_REMOVE' }), /decision DEPRECATE_AND_REMOVE is not covered/],
  ['RENAME_TO_AVOID_COLLISION under the item 3 ruling', helper({ decision: 'RENAME_TO_AVOID_COLLISION' }), /decision RENAME_TO_AVOID_COLLISION is not covered/],
  ['a signed_on date other than the ruling date', helper({ ruling: { ...RULING, signed_on: '2026-01-01' } }), /\(ruling signed_on "2026-01-01" is not the date of itchyshin\/GLLVModels\.jl#684 item 3\)/],
  ['a signed_on date after the ruling date', helper({ ruling: { ...RULING, signed_on: '2026-10-03' } }), /is not the date of|BAD-DATE/],
]) {
  test(`C6 scope: ${name} is unsigned_decision and fails`, () => {
    const r = runTree(c6Items([item]), 'C6');
    assert.equal(r.code, 0);
    assert.match(r.stdout, /C6_NOT_MET$/m);
    assert.match(r.stdout, /invalid_decision=none /);
    assert.match(r.stdout, why);
  });
}
test('C6 scope: both covered words stay valid with the pinned ref and date', () => {
  const r = runTree(c6Items([helper(), kept({ source_id: 'julia-export/x' })]), 'C6');
  assert.match(r.stdout, /C6_MET$/m, r.stdout);
});
// A ruling ref is looked up by own property: C6_RULINGS is an object literal, so `ref in C6_RULINGS` would accept the
// names every object inherits and then crash on `.words.has` (exit 1, no verdict). Each of these must be reported as an
// unrecognised ruling with the verdict C6_NOT_MET and exit 0.
test('C6 scope: a ruling ref that is an inherited Object.prototype name is not a recognised ruling (own-property lookup)', () => {
  for (const ref of ['constructor', '__proto__', 'toString', 'hasOwnProperty', 'valueOf', 'isPrototypeOf']) {
    const r = runTree(c6Items([helper({ ruling: { ...RULING, ref } })]), 'C6');
    assert.equal(r.code, 0, `${ref}: exit ${r.code}, stdout ${r.stdout}`);
    assert.match(r.stdout, /C6_NOT_MET$/m, ref);
    assert.match(r.stdout, /invalid_decision=none /, ref);
    assert.match(r.stdout, new RegExp(`\\(ruling ref "${ref}" is not a recognised signed ruling\\)`), ref);
  }
});
test('C6: an unknown decision word still fails (EXCLUDED_HELPER, and a case variant)', () => {
  for (const word of ['EXCLUDED_HELPER', 'excluded_internal_helper', 'EXCLUDED_INTERNAL_HELPER ']) {
    const r = runTree(c6Items([helper({ decision: word })]), 'C6');
    assert.match(r.stdout, /C6_NOT_MET$/m);
    assert.match(r.stdout, new RegExp(`invalid_decision=julia-export/internal_thing:${JSON.stringify(word).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`));
  }
});
test('C6: an undecided item (decision null) still fails, as before', () => {
  const r = runTree(c6Items([helper(), { source_id: 'julia-export/undecided', decision: null, status: 'unsigned' }]), 'C6');
  assert.match(r.stdout, /C6_NOT_MET$/m);
  assert.match(r.stdout, /invalid_decision=julia-export\/undecided:null /);
});

// --- Fix round on PR #687 (three adversarial reviews). Each control below fails on the head before the
// round (6f546fb00) for the reason it names, and passes after. ---

// Scope: the behavioural tier is a frozen list, not a prefix. Scoreboard side: the same list, by scoreboard id.
test('behavioural scope (scoreboard): EVIDENCED-BEHAVIOURAL on CI-ROUTE-008..011, a new inference id or a bare prefix is not done', () => {
  for (const id of ['inference-CI-ROUTE-008', 'inference-CI-ROUTE-009', 'inference-CI-ROUTE-010', 'inference-CI-ROUTE-011', 'inference-CI-ROUTE-999', 'inference']) {
    const r = runBoard('EVIDENCED-BEHAVIOURAL', 'X2', id);
    assert.match(r.stdout, /X2_NOT_MET$/m, id);
    assert.match(r.stdout, new RegExp(`${id}:BEHAVIOURAL_NOT_ALLOWED_FOR_THIS_ROW`), id);
    assert.match(r.stdout, /done_behavioural=0$/m, id);
  }
  assert.match(runBoard('EVIDENCED-BEHAVIOURAL', 'X2', 'inference-CI-ROUTE-084').stdout, /X2_MET$/m);
});

// Label collision: compare class identity, not canonical strings (external review F8a, F8b, F8e, F8f).
const waldCollide = { kind: 'route', canonical: 'wald', r: ['r_wald'], julia: ['jl_wald'], basis: 'R .confint_lambda and Julia confint_lambda route the Wald interval.' };
const stopClass = { kind: 'refusal', canonical: 'stop', r: ['stop_dup'], julia: ['throw_X'], basis: 'R stop() and Julia throw both refuse the call.' };
function labelRun(klass, kind, r, j, mode = 'C1') {
  return runTree(bTree({ receipt: bReceipt({}, {}, [bCase({ kind, r_observed: r, julia_observed: j })]), equivalence: eq(klass) }), mode);
}
test('label identity: Julia raw "wald" (unlisted) does not match R "r_wald" listed in a class named wald (F8a)', () => {
  const c1 = labelRun(waldCollide, 'route', 'r_wald', 'wald').stdout;
  assert.match(c1, /C1_NOT_MET$/m);
  assert.match(c1, /bound_behavioural=0 /);
  assert.match(c1, /R "r_wald" vs Julia "wald" differ after canonicalisation \(R: class "wald"; Julia: no class\)/);
  assert.match(labelRun(waldCollide, 'route', 'r_wald', 'wald', 'C8').stdout, /C8_NOT_MET$/m);
});
test('label identity: R raw "wald" (unlisted) does not match Julia "jl_wald" listed in a class named wald (F8b)', () => {
  const c1 = labelRun(waldCollide, 'route', 'wald', 'jl_wald').stdout;
  assert.match(c1, /C1_NOT_MET$/m);
  assert.match(c1, /R "wald" vs Julia "jl_wald" differ after canonicalisation \(R: no class; Julia: class "wald"\)/);
});
test('label identity: refusal class "stop": R raw "stop" does not match Julia "throw_X" (F8f)', () => {
  const c1 = labelRun(stopClass, 'refusal', 'stop', 'throw_X').stdout;
  assert.match(c1, /C1_NOT_MET$/m);
  assert.match(c1, /R "stop" vs Julia "throw_X" differ after canonicalisation \(R: no class; Julia: class "stop"\)/);
});
test('label identity: both engines emitting the same literal string always match, even when it is listed on one side only (F8e)', () => {
  for (const [r, j] of [['r_wald', 'r_wald'], ['jl_wald', 'jl_wald'], ['wald', 'wald']]) {
    const c1 = labelRun(waldCollide, 'route', r, j).stdout;
    assert.match(c1, /C1_MET$/m, `${r} ${j}\n${c1}`);
    assert.match(c1, /bound_behavioural=1 /);
    assert.match(labelRun(waldCollide, 'route', r, j, 'C8').stdout, /C8_MET$/m);
  }
});
test('label identity: the two listed members of one class match; listed members of two classes do not', () => {
  assert.match(labelRun(waldCollide, 'route', 'r_wald', 'jl_wald').stdout, /C1_MET$/m);
  const two = { ...waldCollide, canonical: 'profile', r: ['r_prof'], julia: ['jl_prof'] };
  const c1 = runTree(bTree({ receipt: bReceipt({}, {}, [bCase({ r_observed: 'r_wald', julia_observed: 'jl_prof' })]), equivalence: eq(waldCollide, two) }), 'C1').stdout;
  assert.match(c1, /C1_NOT_MET$/m);
  assert.match(c1, /R "r_wald" vs Julia "jl_prof" differ after canonicalisation \(R: class "wald"; Julia: class "profile"\)/);
});
test('label identity: arrays are compared element by element with the same rule', () => {
  const bad = runTree(bTree({ receipt: bReceipt({}, {}, [bCase({ kind: 'route', r_observed: ['r_wald', 'wald'], julia_observed: ['jl_wald', 'jl_wald'] })]), equivalence: eq(waldCollide) }), 'C1').stdout;
  assert.match(bad, /C1_NOT_MET$/m);
  assert.match(bad, /R "wald" vs Julia "jl_wald" differ after canonicalisation \(R: no class; Julia: class "wald"\)/);
  const ok = runTree(bTree({ receipt: bReceipt({}, {}, [bCase({ kind: 'route', r_observed: ['r_wald', 'x'], julia_observed: ['jl_wald', 'x'] })]), equivalence: eq(waldCollide) }), 'C1').stdout;
  assert.match(ok, /C1_MET$/m);
});

// Visible text: one definition for both ports (at least one character in Unicode L, N, P or S).
const INVISIBLE = [['U+FEFF', '﻿'], ['U+200B', '​'], ['U+0085', '\u0085'], ['U+00A0', ' '], ['U+2060', '⁠'], ['spaces, tab, newline', ' \t\n']];
test('visible text: an observed label with no visible character fails, whatever the invisible character (U+FEFF, U+200B, U+0085, ...)', () => {
  for (const [name, ch] of INVISIBLE) {
    for (const over of [{ r_observed: ch, julia_observed: ch }, { r_observed: [ch], julia_observed: [ch] }, { r_observed: ['wald', ch], julia_observed: ['wald', ch] }]) {
      const c1 = runTree(bTree({ receipt: bReceipt({}, {}, [bCase(over)]) }), 'C1').stdout;
      assert.match(c1, /C1_NOT_MET$/m, `${name} ${JSON.stringify(over)}`);
      assert.match(c1, /bound_behavioural=0 /, name);
      assert.match(c1, /must be a non-empty string or an array of non-empty strings/, name);
    }
  }
  const padded = runTree(bTree({ receipt: bReceipt({}, {}, [bCase({ r_observed: '​wald﻿', julia_observed: '​wald﻿' })]) }), 'C1').stdout;
  assert.match(padded, /C1_MET$/m, 'a label with a visible character is visible');
});
test('visible text: an equivalence class with an invisible canonical, basis or label is MEASUREMENT_FAILED on C1 and C8', () => {
  for (const [name, ch] of INVISIBLE) {
    for (const [what, klass, why] of [
      ['canonical', { ...waldClass, canonical: ch }, /has no canonical label/],
      ['basis', { ...waldClass, basis: ch }, /empty basis/],
      ['r label', { ...waldClass, r: [ch] }, /r must be an array of non-empty labels/],
      ['julia label', { ...waldClass, julia: ['wald_packed', ch] }, /julia must be an array of non-empty labels/],
    ]) {
      for (const mode of ['C1', 'C8']) {
        const r = runTree(({ writeJ }) => writeJ('behaviour-equivalence.json', eq(klass)), mode);
        assert.equal(r.code, 2, `${name} ${what} ${mode}\n${r.stdout}`);
        assert.match(r.stdout, why, `${name} ${what}`);
      }
    }
  }
});
test('visible text: a C6 basis or ruling ref with no visible character is not a decision', () => {
  for (const [name, ch] of INVISIBLE) {
    const b = runTree(c6Items([helper({ basis: ch })]), 'C6');
    assert.match(b.stdout, /C6_NOT_MET$/m, name);
    assert.match(b.stdout, /unsigned_decision=julia-export\/internal_thing\(no basis\)/, name);
    const kb = runTree(c6Items([kept({ basis: ch })]), 'C6');
    assert.match(kb.stdout, /unsigned_decision=julia-export\/doc\(no basis\)/, name);
    const rf = runTree(c6Items([helper({ ruling: { ...RULING, ref: ch } })]), 'C6');
    assert.match(rf.stdout, /C6_NOT_MET$/m, name);
    assert.match(rf.stdout, /\(ruling without a ref\)/, name);
  }
  // A one-character visible basis is visible (the rule is visibility, not substance; review decides substance).
  assert.match(runTree(c6Items([helper({ basis: '.' })]), 'C6').stdout, /C6_MET$/m);
});

// Strict JSON types: the table's schema is the number 1.
test('behaviour-equivalence.json: schema must be the number 1 (true, "1", [1], null, 0, 2 are refused on C1 and C8)', () => {
  for (const schema of [true, '1', [1], null, 0, 2]) {
    for (const mode of ['C1', 'C8']) {
      const r = runTree(({ writeJ }) => writeJ('behaviour-equivalence.json', { schema, pin: 'P1', classes: [] }), mode);
      assert.equal(r.code, 2, `${JSON.stringify(schema)} ${mode}\n${r.stdout}`);
      assert.match(r.stdout, /schema must be 1/, JSON.stringify(schema));
    }
  }
  assert.equal(runTree(({ writeJ }) => writeJ('behaviour-equivalence.json', { schema: 1, pin: 'P1', classes: [] }), 'C1').code, 0);
});

// Behavioural receipt failure detection (the numeric tier's list is unchanged by this PR).
test('behavioural receipt: a top-level result that is not a pass value does not bind', () => {
  for (const result of ['FAIL', 'error', null, false, 0, { family: 'gaussian' }, ['PASS']]) {
    const m = bTree({ receipt: bReceipt({ result }) });
    const c1 = runTree(m, 'C1').stdout;
    assert.match(c1, /C1_NOT_MET$/m, JSON.stringify(result));
    assert.match(c1, /bound_behavioural=0 /, JSON.stringify(result));
    assert.match(c1, /receipt did not pass: result=.* in docs\/dev-log\/core070\/true-parity-latest\/receipts\/b1\.json/, JSON.stringify(result));
    const c8 = runTree(m, 'C8').stdout;
    assert.match(c8, /C8_NOT_MET$/m, JSON.stringify(result));
    assert.match(c8, /BEHAVIOURAL_LABEL_WITHOUT_BEHAVIOURAL_RECEIPT\(receipt did not pass: result=/, JSON.stringify(result));
  }
  for (const result of ['PASS', 'pass', true]) assert.match(runTree(bTree({ receipt: bReceipt({ result }) }), 'C1').stdout, /C1_MET$/m, JSON.stringify(result));
});
test('behavioural receipt: a failing result inside the behaviour block does not bind', () => {
  const c1 = runTree(bTree({ receipt: bReceipt({}, { result: 'FAIL' }) }), 'C1').stdout;
  assert.match(c1, /C1_NOT_MET$/m);
  assert.match(c1, /receipt did not pass: behaviour\.result="FAIL" in /);
});
test('behavioural receipt: a nested batch_verifier.status that is not PASS does not bind', () => {
  for (const bv of [{ status: 'FAIL', exit_code: 1 }, { status: null }, { status: 'PASSED' }, { status: {} }]) {
    const c1 = runTree(bTree({ receipt: bReceipt({ verdict: 'PASS', batch_verifier: bv }) }), 'C1').stdout;
    assert.match(c1, /C1_NOT_MET$/m, JSON.stringify(bv));
    assert.match(c1, /receipt did not pass: batch_verifier\.status=.* in /, JSON.stringify(bv));
  }
  const notObj = runTree(bTree({ receipt: bReceipt({ batch_verifier: 'FAIL' }) }), 'C1').stdout;
  assert.match(notObj, /C1_NOT_MET$/m);
  assert.match(notObj, /batch_verifier="FAIL" is not an object/);
  for (const bv of [{ status: 'PASS', exit_code: 0 }, { exit_code: 0 }]) assert.match(runTree(bTree({ receipt: bReceipt({ batch_verifier: bv }) }), 'C1').stdout, /C1_MET$/m, JSON.stringify(bv));
});
test('behavioural receipt: a comparison block whose status or verdict is not PASS does not bind', () => {
  const cmp = (over) => ({ pin: 'P1', cases: [{ case_id: 'CASE-B', quantity: 'q', r_value: 1, julia_value: 1, tolerance: 1e-6 }], ...over });
  for (const over of [{ status: 'FAIL' }, { verdict: 'FAIL' }, { batch_status: 'FAIL' }, { harness_pass: false }, { result: 'FAIL' }]) {
    const c1 = runTree(bTree({ receipt: { ...bReceipt(), comparison: cmp(over) } }), 'C1').stdout;
    assert.match(c1, /C1_NOT_MET$/m, JSON.stringify(over));
    assert.match(c1, /receipt did not pass: comparison\.(status|verdict|batch_status|harness_pass|result)=/, JSON.stringify(over));
  }
  assert.match(runTree(bTree({ receipt: { ...bReceipt(), comparison: cmp({ status: 'PASS', verdict: 'PASS' }) } }), 'C1').stdout, /C1_MET$/m);
});
test('behavioural receipt: a behaviour case with status not PASS, or match not true, does not bind', () => {
  for (const over of [{ status: 'FAIL' }, { status: 'error' }, { result: 'FAIL' }, { match: false }, { match: null }, { match: 'yes' }, { match: 0 }]) {
    const c1 = runTree(bTree({ receipt: bReceipt({}, {}, [bCase(over)]) }), 'C1').stdout;
    assert.match(c1, /C1_NOT_MET$/m, JSON.stringify(over));
    assert.match(c1, /receipt did not pass: behaviour\.cases\[CASE-B\]\.(status|result|match)=/, JSON.stringify(over));
  }
  for (const over of [{ status: 'PASS' }, { match: true }, { status: 'PASS', match: true }]) assert.match(runTree(bTree({ receipt: bReceipt({}, {}, [bCase(over)]) }), 'C1').stdout, /C1_MET$/m, JSON.stringify(over));
});
test('numeric receipt: this PR leaves the numeric status list as it was (a top-level result of FAIL is not read)', () => {
  const m = ({ readJ, writeJ }) => { const r1 = readJ('receipts/r1.json'); r1.result = 'FAIL'; writeJ('receipts/r1.json', r1); };
  const c1 = runTree(m, 'C1');
  assert.match(c1.stdout, /C1_MET$/m, c1.stdout);
  assert.match(c1.stdout, /bound_numeric=2 /);
});

// Unscoped entries: an entry without source_id covers none of several rows citing the same case id.
const SECOND_ROW = { source_id: 'inference/CI-ROUTE-002' };
test('unscoped entry: one entry without source_id covers none of two rows that cite the same case id', () => {
  const m = bTree({ extraRows: [SECOND_ROW] });
  const c1 = runTree(m, 'C1').stdout;
  assert.match(c1, /C1_NOT_MET$/m);
  assert.match(c1, /bound_behavioural=0 /);
  assert.match(c1, /inference\/CI-ROUTE-001\(case ids without an applicable behaviour entry: CASE-B \(cited by 2 rows, so an entry without source_id covers none of them; scope each entry with source_id\)\)/);
  assert.match(c1, /inference\/CI-ROUTE-002\(case ids without an applicable behaviour entry: CASE-B \(cited by 2 rows/);
  const c8 = runTree(m, 'C8').stdout;
  assert.match(c8, /C8_NOT_MET$/m);
  assert.match(c8, /inference\/CI-ROUTE-001:BEHAVIOURAL_LABEL_WITHOUT_BEHAVIOURAL_RECEIPT\(case ids without an applicable behaviour entry: CASE-B \(cited by 2 rows/);
});
test('unscoped entry: each row needs its own source_id-scoped entry; one scoped row binds, the other does not', () => {
  const both = bTree({ extraRows: [SECOND_ROW], receipt: bReceipt({}, {}, [bCase({ source_id: B_ROW }), bCase({ source_id: 'inference/CI-ROUTE-002' })]) });
  const ok = runTree(both, 'C1').stdout;
  assert.match(ok, /C1_MET$/m, ok);
  assert.match(ok, /bound_behavioural=2 /);
  assert.match(runTree(both, 'C8').stdout, /C8_MET$/m);
  const one = bTree({ extraRows: [SECOND_ROW], receipt: bReceipt({}, {}, [bCase({ source_id: B_ROW })]) });
  const c1 = runTree(one, 'C1').stdout;
  assert.match(c1, /C1_NOT_MET$/m);
  assert.match(c1, /bound_behavioural=1 /);
  assert.match(c1, /inference\/CI-ROUTE-002\(case ids without an applicable behaviour entry: CASE-B\)/);
});
test('unscoped entry: an unscoped entry still covers a case id that only one row cites; a row of another tier citing it counts', () => {
  assert.match(runTree(bTree(), 'C1').stdout, /C1_MET$/m);
  // the numeric base row CASE-1 shares its id with a second row: an unscoped entry for CASE-1 would be ambiguous too
  const m = bTree({ row: { executable_case_ids: ['CASE-1'] }, receipt: bReceipt({}, {}, [bCase({ case_id: 'CASE-1' })]) });
  const c1 = runTree(m, 'C1').stdout;
  assert.match(c1, /C1_NOT_MET$/m);
  assert.match(c1, /CASE-1 \(cited by 2 rows, so an entry without source_id covers none of them/);
});

// Integer equality: tolerance is exactly 0.5 (mutation: a `> 0.5` test would accept 0.4).
for (const tolerance of [0.4, 0.49, 0.5000001, 0.51]) {
  test(`integer equality: tolerance ${tolerance} is refused (only exactly 0.5 means equality)`, () => {
    const m = intCase({ kind: 'integer_equality', r_value: 15, julia_value: 15, tolerance });
    const c1 = runTree(m, 'C1').stdout;
    assert.match(c1, /C1_NOT_MET$/m);
    assert.match(c1, /case CASE-1: integer_equality needs tolerance exactly 0\.5/);
    const c8 = runTree(m, 'C8').stdout;
    assert.match(c8, /C8_NOT_MET$/m);
    assert.match(c8, /NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT/);
  });
}

// C6: a documented extra cites a docs/src file that resolves; a helper's basis is visible text.
for (const [name, basis, why] of [
  ['no path at all', 'documented in the README', /\(KEPT_AS_JULIA_EXTRA basis must cite a docs\/src\/\.\.\. file\)/],
  ['a bare dot', '.', /\(KEPT_AS_JULIA_EXTRA basis must cite a docs\/src\/\.\.\. file\)/],
  ['a path outside docs/src', 'see tools/true_parity_check.mjs', /must cite a docs\/src/],
  ['a dot-dot path', 'see docs/src/../README.md', /must cite a docs\/src/],
  ['a dot-leading segment', 'see docs/src/.hidden.md', /must cite a docs\/src/],
  ['a docs/src path that does not exist', 'see docs/src/no-such-page.md', /\(KEPT_AS_JULIA_EXTRA basis cites docs\/src\/no-such-page\.md, which does not resolve at the ref\)/],
  ['one existing and one missing path', 'docs/src/gllvmtmb-parity.md and docs/src/gone.md', /cites docs\/src\/gone\.md, which does not resolve/],
  ['a directory, not a file', 'see docs/src/sub.md', /which does not resolve/],
]) {
  test(`C6: a KEPT_AS_JULIA_EXTRA basis with ${name} is unsigned_decision and fails`, () => {
    const r = runTree(({ dir, ...rest }) => { mkdirSync(join(dir, 'docs/src/sub.md'), { recursive: true }); c6Items([kept({ basis })])({ dir, ...rest }); }, 'C6');
    assert.equal(r.code, 0);
    assert.match(r.stdout, /C6_NOT_MET$/m);
    assert.match(r.stdout, /invalid_decision=none /);
    assert.match(r.stdout, why);
  });
}
test('C6: a KEPT_AS_JULIA_EXTRA basis citing existing docs/src files binds; so does an EXCLUDED_INTERNAL_HELPER basis that cites no file', () => {
  for (const basis of ['docs/src/gllvmtmb-parity.md', 'see docs/src/gllvmtmb-parity.md, section extras.', 'x docs/src/gllvmtmb-parity.md#extras']) {
    assert.match(runTree(c6Items([kept({ basis })]), 'C6').stdout, /C6_MET$/m, basis);
  }
  assert.match(runTree(c6Items([helper({ basis: 'internal; no docstring' })]), 'C6').stdout, /C6_MET$/m);
});
test('C6: the docs/src check applies to KEPT_AS_JULIA_EXTRA only, in git mode too (resolved with git cat-file at the ref)', () => {
  const dir = mkdtempSync(join(tmpdir(), 'true-parity-c6git-'));
  try {
    cpSync(join(FIXTURES, 'base'), dir, { recursive: true });
    writeFileSync(join(dir, L, 'reverse-gap.json'), JSON.stringify([kept(), kept({ source_id: 'julia-export/gone', basis: 'docs/src/gone.md' })]));
    execFileSync('git', ['init', '-q'], { cwd: dir });
    execFileSync('git', ['-c', 'user.email=t@t.invalid', '-c', 'user.name=t', 'add', '-A'], { cwd: dir });
    execFileSync('git', ['-c', 'user.email=t@t.invalid', '-c', 'user.name=t', 'commit', '-q', '-m', 'fixture'], { cwd: dir });
    const r = runGit(dir, 'C6');
    assert.match(r.stdout, /C6_NOT_MET$/m);
    assert.match(r.stdout, /unsigned_decision=julia-export\/gone\(KEPT_AS_JULIA_EXTRA basis cites docs\/src\/gone\.md, which does not resolve at the ref\)/);
    assert.doesNotMatch(r.stdout, /julia-export\/doc\(/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// --- the generated scoreboard is tied back to the case maps (review of #589, finding 5) ---
// C2/X2 read the scoreboard's status word by design; tools/true_parity_assemble.py --check is
// what ties that word back to the per-family case maps. Running it here means a hand-edited
// scoreboard row (e.g. NON-NUMERIC -> EVIDENCED with no case-map change) fails this run.
{
  const ASSEMBLER = join(REPO_ROOT, 'tools', 'true_parity_assemble.py');
  const LEDGER_REL = join('docs', 'dev-log', 'core070', 'true-parity-latest');
  function runAssemble(root) {
    try {
      return { stdout: execFileSync('python3', [ASSEMBLER, '--root', root, '--check'], { encoding: 'utf8' }), code: 0 };
    } catch (e) {
      return { stdout: e.stdout || '', code: e.status };
    }
  }
  test('assembler --check: the tracked scoreboard, assembled case map and reverse gap are current', () => {
    const { stdout, code } = runAssemble(REPO_ROOT);
    assert.equal(code, 0, stdout);
    assert.match(stdout, /^ASSEMBLE_OK \d+ rows current$/m);
  });
  test('assembler --check: a hand-edited EVIDENCED scoreboard row fails (copy of the tree)', () => {
    const tmp = mkdtempSync(join(tmpdir(), 'true-parity-assemble-'));
    try {
      // Receipts cited by the maps live under docs/dev-log/core070/; fixtures are listed in the board.
      cpSync(join(REPO_ROOT, 'docs', 'dev-log', 'core070'), join(tmp, 'docs', 'dev-log', 'core070'), { recursive: true });
      cpSync(join(REPO_ROOT, 'test', 'fixtures'), join(tmp, 'test', 'fixtures'), { recursive: true });
      // The C6 decisions file names the script that wrote it and, for a documented Julia extra, the docs/src
      // pages that render it; the assembler requires both in the tree.
      cpSync(join(REPO_ROOT, 'docs', 'src'), join(tmp, 'docs', 'src'), { recursive: true });
      const decisionsPath = join(REPO_ROOT, LEDGER_REL, 'reverse-gap-decisions.json');
      if (existsSync(decisionsPath)) {
        const gen = JSON.parse(readFileSync(decisionsPath, 'utf8')).generator;
        mkdirSync(dirname(join(tmp, gen)), { recursive: true });
        cpSync(join(REPO_ROOT, gen), join(tmp, gen));
      }
      const clean = runAssemble(tmp);
      assert.equal(clean.code, 0, clean.stdout); // positive control on the copy
      const sb = join(tmp, LEDGER_REL, 'scoreboard.md');
      const txt = readFileSync(sb, 'utf8');
      const edited = txt.replace(/^(\| \S+ `[^`]*` \| [^|]* \| )(?!EVIDENCED )[A-Z-]+( \|)/m, '$1EVIDENCED$2');
      assert.notEqual(edited, txt, 'no non-EVIDENCED row to hand-edit');
      writeFileSync(sb, edited);
      const bad = runAssemble(tmp);
      assert.equal(bad.code, 1, bad.stdout);
      assert.match(bad.stdout, /^ASSEMBLE_STALE scoreboard\.md$/m);
    } finally {
      rmSync(tmp, { recursive: true, force: true });
    }
  });
}

if (failures > 0) {
  console.log(`\n${failures} control(s) FAILED`);
  process.exit(1);
}
console.log('\nAll true-parity negative controls passed.');

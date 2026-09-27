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
import { mkdtempSync, rmSync, cpSync } from 'node:fs';
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

// --- item 4 / git-mode control: show, existsAsBlob and listDir exercised through real git,
// the same code path CI runs against origin/main, not the FS fallback ---
{
  let goodRepo, dirReceiptRepo;
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
  for (const dir of [goodRepo, dirReceiptRepo]) {
    if (dir) rmSync(dir, { recursive: true, force: true });
  }
}

if (failures > 0) {
  console.log(`\n${failures} control(s) FAILED`);
  process.exit(1);
}
console.log('\nAll true-parity negative controls passed.');

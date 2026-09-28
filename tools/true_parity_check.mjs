#!/usr/bin/env node
// True-parity acceptance oracle for GLLVModels.jl vs gllvmTMB, pinned per-pin (P0 frozen at
// 0.7.0 b4d5fee64def88bc768dda1f1f77c29b295edd86; P1 = gllvmTMB main
// 9539352f66f2db2cc26b1c393e67212a359b60c9, 0.7.1 candidate, D-294/D-295, 2026-09-27).
//
// One mode per destination clause in ultra-plan.md "Destination (the stopping condition)":
// C0 C1 C2 C3 C4 C5 C6 C7 C8 X2. Prints measured numbers, then <mode>_MET only when the
// clause holds. Exit 0 on a clean measurement regardless of MET/NOT_MET (the gate is decided
// by the printed verdict, not the exit code); exit 2 if the measurement itself could not be
// made (missing file, malformed JSON, a receipt reference the tool cannot resolve either way,
// etc -- MEASUREMENT_FAILED). An empty row selection is never a pass.
//
// Reads a git ref (default origin/main), never the working tree, so a gate cannot pass on an
// unmerged branch. PARITY_REF=FS switches to a filesystem fixture tree rooted at
// PARITY_FS_ROOT, for the negative-control tests in test/fixtures/true_parity/.
//
// Ledger tracked in git (was gitignored under .unlazy/true-parity/ and lost receipts; see
// docs/dev-log/core070/true-parity-latest/GATES.md). Copied from and extends
// ~/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/true-parity/check.mjs
// (untracked, kept as-is; this file is the tracked continuation, not an edit of that one).
//
// Post-review hardening (2026-09-27, independent review of PR #523, BLOCKING): the first cut
// accepted labels in place of evidence -- a receipt path that resolved to a directory counted
// as present; a scoreboard cell with no extractable path counted as done anyway; a bare
// `DISPOSITION-SIGNED` label or a bare `executable_case_ids` entry counted as bound with no
// receipt, signer, or date; `outside_boundary` rows vanished from C8 entirely; the carry rule
// compared author-typed strings with no hash-format check and was opt-in; `git ls-tree` without
// a trailing slash never actually listed a directory's contents, so C0's CI-job scan silently
// scanned nothing in git mode. Every one of those is fixed below. What is intentionally still
// deferred (name-twin detection from receipt content; a CARRY_VERIFY mode that re-hashes the
// gllvmTMB tree itself) is stated as a gap, not silently patched over -- see
// docs/dev-log/core070/true-parity-latest/GATES.md.
//
// Evidence tier (review of #559, D-295 row 5 "a name match never counts"): a case-map row carries
// `evidence_tier` ("numeric" or "registration"). C1 prints bound_numeric / bound_registration_only
// and is MET only when no bound row is registration-only (a signed disposition still resolves a
// row); C8 does not count a registration-only row as twinned. A missing tier is fail-closed.
//
// Numeric tier verified against the receipt (review of #561, BLOCKING): the label alone was
// trusted, so a one-word edit ("registration" -> "numeric") on a namespace row made C1_MET and
// C8_MET. A row labelled "numeric" now counts as numeric only if its cited receipts carry a
// machine-readable `comparison` block (schema in numericReceiptStatus below and in GATES.md)
// pinned to P1 and covering every executable_case_id, each case within its tolerance.
// Otherwise the row reads NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT and does not bind.
//
// Receipt status (review of #567's tamper tests): a comparison block within tolerance in a
// receipt whose own verdict/status/batch_status/harness_pass is not a pass value (e.g. "FAIL")
// no longer binds: NUMERIC_RECEIPT_NOT_PASSED, unless the row carries a maintainer-signed
// `receipt_status_exception`, in which case it counts in bound_signed=, not bound_numeric=.
// And a recorded abs_diff/max_abs_diff that disagrees with the difference the tool recomputes
// from r_value/julia_value fails the row as NUMERIC_RECORDED_DIFF_MISMATCH.
import { execFileSync } from 'node:child_process';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const REF = process.env.PARITY_REF || 'origin/main';
const PIN = process.env.PARITY_PIN || 'P1';
const FS_ROOT = process.env.PARITY_FS_ROOT || '.';

const P0_SHA = 'b4d5fee64def88bc768dda1f1f77c29b295edd86';
const P1_SHA = '9539352f66f2db2cc26b1c393e67212a359b60c9';
const SHA256_RE = /^[0-9a-f]{64}$/;

// Per-pin ledger paths. "Configurable per pin": add a pin here, or override with
// PARITY_SCOREBOARD / PARITY_CASEMAP / PARITY_REVERSE_GAP for one-off runs.
const PIN_PATHS = {
  P0: {
    scoreboard: 'docs/dev-log/core070/true-parity-gate-tier-scoreboard-2026-09-25.md',
    casemap: 'docs/dev-log/core070/required-source-case-map.json',
    reverseGap: null, // P0's reverse-gap list is tool-produced by tools/parity_ledger.py, not tracked as a file
  },
  P1: {
    scoreboard: 'docs/dev-log/core070/true-parity-latest/scoreboard.md',
    casemap: 'docs/dev-log/core070/true-parity-latest/case-map.json',
    reverseGap: 'docs/dev-log/core070/true-parity-latest/reverse-gap.json',
  },
};
const PATHS = {
  scoreboard: process.env.PARITY_SCOREBOARD || (PIN_PATHS[PIN] || PIN_PATHS.P1).scoreboard,
  casemap: process.env.PARITY_CASEMAP || (PIN_PATHS[PIN] || PIN_PATHS.P1).casemap,
  reverseGap: process.env.PARITY_REVERSE_GAP || (PIN_PATHS[PIN] || PIN_PATHS.P1).reverseGap,
};
const PARITY_PAGE = 'docs/src/gllvmtmb-parity.md';
const ORACLE_PY = 'tools/parity_oracle.py';
const WORKFLOWS_DIR = '.github/workflows';
// This tool's own CI smoke test never counts as "the P1 twin job" for C0 -- it runs fixtures,
// not real P1 twin tests. Excluded by name as a backstop; the content check below (job id
// `p1-twin-tests` plus the `gllvm-parity-tag: P1` discovery convention from PR #524) would not
// match it anyway, but a filename exclusion is cheap insurance against that check drifting.
const SMOKE_WORKFLOW_FILE = 'true-parity-check.yml';

const DONE = new Set(['EVIDENCED', 'DISPOSITION-SIGNED']);
// C6's decision vocabulary is enumerated, not free text: a reverse-gap item's "decision" must
// be exactly one of these, or it does not count as decided (a placeholder like "TBD" must not
// pass just because the field is non-empty).
const C6_DECISION_VOCAB = new Set([
  'KEPT_AS_JULIA_EXTRA',
  'PORT_TO_MATCH_R',
  'DEPRECATE_AND_REMOVE',
  'RENAME_TO_AVOID_COLLISION',
]);

const git = (...a) => execFileSync('git', a, { encoding: 'utf8', maxBuffer: 64 << 20, stdio: ['ignore', 'pipe', 'pipe'] });
const die = (m) => { console.log(`MEASUREMENT_FAILED ${m}`); process.exit(2); };

function show(p) {
  try {
    return REF === 'FS' ? readFileSync(join(FS_ROOT, p), 'utf8') : git('show', `${REF}:${p}`);
  } catch {
    return null;
  }
}

// A receipt (or any cited path) must resolve to a FILE, never a directory: git `cat-file -e`
// succeeds on a tree too, which is how a directory silently passed as "the receipt" before
// this fix. `cat-file -t` reports the object type; only "blob" counts.
function existsAsBlob(p) {
  if (REF === 'FS') {
    try { return statSync(join(FS_ROOT, p)).isFile(); } catch { return false; }
  }
  try { return git('cat-file', '-t', `${REF}:${p}`).trim() === 'blob'; } catch { return false; }
}

function listDir(dir) {
  if (REF === 'FS') {
    try { return readdirSync(join(FS_ROOT, dir)); } catch { return []; }
  }
  try {
    // `git ls-tree --name-only REF -- dir` (no trailing slash) returns the directory entry
    // itself, not its contents, so nothing was ever scanned in git mode. A trailing slash (or
    // -r) makes it list the directory's contents instead.
    const d = dir.endsWith('/') ? dir : `${dir}/`;
    return git('ls-tree', '--name-only', REF, '--', d).split('\n').filter(Boolean).map((p) => p.split('/').pop());
  } catch {
    return [];
  }
}

// --- shared parsers -------------------------------------------------------

// Extracts path-like tokens (docs/..., tools/..., test/..., src/...) from free text, e.g. a
// scoreboard's "Receipt / disposition" column, so receipt resolution can check they exist.
// Deliberately narrow (a known top-level dir plus a known extension) to avoid pulling prose
// into a false receipt; `looksPathLike` below catches text this misses so it fails loudly
// instead of silently passing.
function extractPaths(text) {
  if (!text) return [];
  const m = text.match(/\b(?:docs|tools|test|src)\/[A-Za-z0-9._\-/]+\.(?:md|json|toml|jl|py|mjs)\b/g);
  return m ? [...new Set(m)] : [];
}

// True for any text that contains a slash-separated token (".unlazy/x/y.json", "docs/x.txt",
// a bare directory) -- i.e. text a human plainly intended as a path reference, whether or not
// `extractPaths`'s narrower pattern captured it.
function looksPathLike(text) {
  if (!text) return false;
  return /[A-Za-z0-9_.-]+\/[A-Za-z0-9_./-]+/.test(text);
}

// A DISPOSITION-SIGNED scoreboard row (markdown has no separate signed_by/signed_on columns)
// records its signer and date as plain tokens in the same free-text cell, e.g.
// "Disposition: outside_boundary; signed_by: Shinichi Nakagawa; signed_on: 2026-09-27".
// The same signer allow-list and date check as the case-map rows apply here.
function hasSignedTokens(text) {
  if (!text) return false;
  const by = text.match(/signed_by:\s*([^;|]+)/);
  const on = text.match(/signed_on:\s*([^;|\s]+)/);
  return !!by && !!on && signatureProblem(by[1], on[1]) === null;
}

function loadCasemap() {
  const j = show(PATHS.casemap);
  if (j === null) die(`${PATHS.casemap} not on ${REF}`);
  let d;
  try { d = JSON.parse(j); } catch (e) { die(`${PATHS.casemap} is not valid JSON: ${e.message}`); }
  return Array.isArray(d.rows) ? d.rows : die(`${PATHS.casemap} has no "rows" array`);
}

function rowReceiptPaths(row) {
  const r = row.evidence && row.evidence.receipt;
  if (!r) return [];
  return Array.isArray(r) ? r : [r];
}

// A case-map row's evidence tier: "numeric" (the receipt records an actual R-vs-Julia output
// comparison) or "registration" (export/existence registration only, e.g. the namespace Tier 0
// batch). Anything but "numeric" -- including a missing field -- is treated as registration-only.
function isNumericTier(row) { return row.evidence_tier === 'numeric'; }

// The label can only lower a row, never raise it: "numeric" must be backed by receipt content.
// A numeric receipt is a JSON file with a top-level `comparison` block:
//
//   "comparison": {
//     "pin": "P1",                      // or the full P1 sha
//     "cases": [
//       { "case_id": "CASE-1", "quantity": "coef",
//         "abs_diff": 2.4e-06,          // or "max_abs_diff" (the core070 receipt spelling), or
//                                       // "r_value" + "julia_value" (finite numbers, or equal-
//                                       // length arrays of finite numbers; the tool computes the
//                                       // max absolute difference itself)
//         "tolerance": 1e-04 }          // finite number > 0
//     ]
//   }
//
// Every case must satisfy abs_diff <= tolerance, and the union of case_ids across the row's
// receipts must cover every executable_case_id. A malformed block in any cited receipt fails the
// row (it is never skipped). Returns { ok } or { ok: false, reason }.
//
// Recorded diff cross-check (review of #567, tamper test "stale max_abs_diff, vectors disagree by
// 1"): when a case carries a comparable r_value + julia_value pair, the tool's own difference is
// the one judged against tolerance, and every recorded abs_diff / max_abs_diff on that case must
// agree with it to 1e-12 relative, or the row fails as NUMERIC_RECORDED_DIFF_MISMATCH. A recorded
// value is used on its own only when the case carries no comparable pair.
const RECORDED_DIFF_REL_TOL = 1e-12;

// Returns { d } (the difference to judge), or { d: null } when there is none, or
// { mismatch: reason } when a recorded difference disagrees with the recomputed one.
function comparisonCaseDiff(c) {
  const fin = (x) => typeof x === 'number' && Number.isFinite(x);
  const finArr = (x) => Array.isArray(x) && x.length > 0 && x.every(fin);
  let computed = null;
  if (fin(c.r_value) && fin(c.julia_value)) computed = Math.abs(c.r_value - c.julia_value);
  else if (finArr(c.r_value) && finArr(c.julia_value) && c.r_value.length === c.julia_value.length) {
    computed = Math.max(...c.r_value.map((r, i) => Math.abs(r - c.julia_value[i])));
  }
  const recorded = [];
  for (const k of ['abs_diff', 'max_abs_diff']) {
    if (c[k] === undefined) continue;
    if (!fin(c[k]) || c[k] < 0) return { d: null };
    recorded.push([k, c[k]]);
  }
  if (computed === null) return { d: recorded.length ? recorded[0][1] : null };
  for (const [k, v] of recorded) {
    if (Math.abs(v - computed) > RECORDED_DIFF_REL_TOL * Math.max(Math.abs(v), Math.abs(computed))) {
      return { mismatch: `case ${c.case_id}: recorded ${k} ${v} != recomputed ${computed} from r_value/julia_value` };
    }
  }
  return { d: computed };
}

// Receipt status (review of #567, tamper test "verdict = FAIL, comparison intact"): a comparison
// block within tolerance is not enough if the receipt itself says the run did not pass. These are
// the status fields real core070 receipts write (case receipts: `verdict`, `batch_status`,
// `harness_pass`; batch receipts: `status`). Each one present at the top level of a cited JSON
// receipt, or inside its `comparison` block, must hold a pass value; anything else (including
// "FAIL", null, or an object) fails the row as NUMERIC_RECEIPT_NOT_PASSED unless the row carries a
// maintainer-signed `receipt_status_exception` (see receiptStatusExceptionProblem).
const RECEIPT_STATUS_FIELDS = ['status', 'verdict', 'batch_status', 'harness_pass'];
const isPassValue = (v) => v === 'PASS' || v === 'pass' || v === true;

function receiptNotPassed(j, p) {
  const where = [[j, ''], [j.comparison && typeof j.comparison === 'object' ? j.comparison : null, 'comparison.']];
  for (const [obj, prefix] of where) {
    if (!obj) continue;
    for (const f of RECEIPT_STATUS_FIELDS) {
      if (Object.prototype.hasOwnProperty.call(obj, f) && !isPassValue(obj[f])) {
        return `${prefix}${f}=${JSON.stringify(obj[f])} in ${p}`;
      }
    }
  }
  return null;
}

function numericReceiptStatus(row) {
  const paths = rowReceiptPaths(row);
  if (paths.length === 0) return { ok: false, reason: 'no receipt' };
  const covered = new Set();
  let blocks = 0;
  let notPassed = null;
  for (const p of paths) {
    const txt = show(p);
    if (txt === null) return { ok: false, reason: `unreadable ${p}` };
    let j;
    try { j = JSON.parse(txt); } catch { continue; } // a non-JSON receipt carries no comparison
    if (!j || typeof j !== 'object') continue;
    if (notPassed === null) notPassed = receiptNotPassed(j, p);
    if (j.comparison === undefined) continue;
    const cmp = j.comparison;
    if (!cmp || typeof cmp !== 'object' || Array.isArray(cmp)) return { ok: false, reason: `malformed comparison in ${p}` };
    if (cmp.pin !== 'P1' && cmp.pin !== P1_SHA) return { ok: false, reason: `comparison not pinned to P1 in ${p}` };
    if (!Array.isArray(cmp.cases) || cmp.cases.length === 0) return { ok: false, reason: `comparison has no cases in ${p}` };
    for (const c of cmp.cases) {
      if (!c || typeof c.case_id !== 'string' || c.case_id.length === 0) return { ok: false, reason: `comparison case without case_id in ${p}` };
      const tol = c.tolerance;
      if (typeof tol !== 'number' || !Number.isFinite(tol) || tol <= 0) return { ok: false, reason: `case ${c.case_id}: tolerance not a finite number > 0` };
      const cd = comparisonCaseDiff(c);
      if (cd.mismatch) return { ok: false, kind: 'diff_mismatch', reason: `${cd.mismatch} in ${p}` };
      const d = cd.d;
      if (d === null) return { ok: false, reason: `case ${c.case_id}: no finite abs_diff or r_value/julia_value` };
      if (d > tol) return { ok: false, reason: `case ${c.case_id}: abs_diff ${d} > tolerance ${tol}` };
      covered.add(c.case_id);
    }
    blocks++;
  }
  if (blocks === 0) return { ok: false, reason: 'no comparison block in any receipt' };
  const ids = Array.isArray(row.executable_case_ids) ? row.executable_case_ids : [row.executable_case_ids];
  const missing = ids.filter((id) => !covered.has(id));
  if (missing.length) return { ok: false, reason: `case ids not compared: ${missing.join(',')}` };
  // The comparison itself holds; the row still does not bind if a cited receipt says it failed.
  if (notPassed !== null) return { ok: false, kind: 'not_passed', reason: notPassed };
  return { ok: true };
}

// A maintainer may accept a receipt whose status is not a pass (e.g. a batch receipt that reads
// FAIL because of one unrelated case) by signing the row: `receipt_status_exception:
// { "reason": "...", "signed_by": "...", "signed_on": "YYYY-MM-DD" }`. Same signer allow-list and
// date rule as a signed disposition, plus a non-empty reason. It only waives the status check:
// the comparison block must still be valid and within tolerance. A row bound this way counts in
// bound_signed=, never in bound= / bound_numeric=. Returns null when acceptable, else why not.
function receiptStatusExceptionProblem(row) {
  const e = row.receipt_status_exception;
  if (e === undefined || e === null) return 'no receipt_status_exception';
  if (typeof e !== 'object' || typeof e.reason !== 'string' || e.reason.trim().length === 0) return 'receipt_status_exception without a reason';
  return signatureProblem(e.signed_by, e.signed_on);
}

function isValidSha256(s) { return typeof s === 'string' && SHA256_RE.test(s); }

// A DISPOSITION-SIGNED row (case-map JSON schema) needs an actual signer and date on the row,
// not just the label. The tool cannot verify that the named person actually signed -- that
// happens in PR review, by a human reading the diff (GATES.md says so) -- but it refuses an
// unsigned label, a signer outside the maintainer allow-list, and a date that is not a real
// calendar date or lies in the future (review of #561: "Claude Fable (agent)" / "9999-99-99"
// both passed before).
const SIGNER_ALLOW = new Set(['Shinichi Nakagawa', 'itchyshin']);
const SIGNER_DENY_RE = /agent|claude|codex|cursor|fable/i;

// A real YYYY-MM-DD calendar date, no later than the current date anywhere on earth (UTC+14),
// so a signer's local date is never refused for time-zone reasons alone.
function isRealPastIsoDate(s) {
  if (typeof s !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(s)) return false;
  const d = new Date(`${s}T00:00:00Z`);
  if (Number.isNaN(d.getTime()) || d.toISOString().slice(0, 10) !== s) return false;
  const latestToday = new Date(Date.now() + 14 * 3600 * 1000).toISOString().slice(0, 10);
  return s <= latestToday;
}

// Returns null when the signature is acceptable, else the reason it is not.
function signatureProblem(signedBy, signedOn) {
  if (typeof signedBy !== 'string' || signedBy.trim().length === 0 || typeof signedOn !== 'string' || signedOn.trim().length === 0) {
    return 'DISPOSITION-SIGNED-UNVERIFIED';
  }
  const who = signedBy.trim();
  if (SIGNER_DENY_RE.test(who) || !SIGNER_ALLOW.has(who)) return 'DISPOSITION-SIGNER-NOT-ALLOWED';
  if (!isRealPastIsoDate(signedOn.trim())) return 'DISPOSITION-SIGNED-BAD-DATE';
  return null;
}

function dispositionSignedProperly(row) {
  return row.disposition === 'DISPOSITION-SIGNED' && signatureProblem(row.signed_by, row.signed_on) === null;
}

// Receipt carry rule (Packet 1 row 1): a row not measured directly against P1 needs a `carry`
// block whose source_pins are non-empty, real 64-lowercase-hex sha256 pairs, and equal (byte-
// identical at the two pins) -- author-typed strings like "abc123"/"abc123" no longer pass, an
// empty source_pins array no longer passes, and a row with no `carry` block at all (or no
// `measured_against` at all) is no longer treated as fresh by default.
function carryStatus(row) {
  const ma = row.measured_against;
  if (ma === 'P1' || ma === P1_SHA) return { stale: false };
  if (ma === undefined || ma === null) return { stale: true, reason: 'PARTIAL_STALE_AT_P1(missing measured_against)' };
  const c = row.carry;
  if (!c || !Array.isArray(c.source_pins) || c.source_pins.length === 0) {
    return { stale: true, reason: 'PARTIAL_STALE_AT_P1(no carry.source_pins)' };
  }
  const bad = c.source_pins.some((sp) => !isValidSha256(sp.sha256_at_p0) || !isValidSha256(sp.sha256_at_p1) || sp.sha256_at_p0 !== sp.sha256_at_p1);
  if (bad) return { stale: true, reason: 'PARTIAL_STALE_AT_P1(hash mismatch or not 64-hex sha256)' };
  return { stale: false };
}

function scoreboardRows() {
  const t = show(PATHS.scoreboard);
  if (t === null) die(`${PATHS.scoreboard} not on ${REF}`);
  const rows = t.split('\n')
    .filter((l) => /^\|\s*[A-Za-z0-9][A-Za-z0-9_-]*\b/.test(l) && !/^\|\s*-+\s*\|/.test(l) && !/^\|\s*(Row id|Capability id)\b/i.test(l))
    .map((l) => l.split('|').map((s) => s.trim()))
    // Guard against unrelated smaller tables in the same file (e.g. a "Status | Count"
    // summary table): a real capability/gate-tier row has at least id, requires, status,
    // and a receipt/disposition column.
    .filter((c) => c.length >= 6)
    .map((c) => ({ id: (c[1].match(/^[A-Za-z0-9][A-Za-z0-9_-]*/) || [c[1]])[0], requires: c[2], status: c[3], receiptText: c[4], notes: c[5] }));
  return rows;
}

// Evaluates one scoreboard row. Returns { ok } for a settled result, or { fatal, reason } when
// the row's own text cannot be trusted enough to call it either done or not-done (a path-like
// receipt reference that extractPaths could not capture): that is a MEASUREMENT_FAILED, not a
// silent "not done".
function evaluateScoreboardRow(r) {
  if (!DONE.has(r.status)) return { ok: false, reason: 'NOT_DONE' };
  const extracted = extractPaths(r.receiptText);
  if (extracted.length === 0) {
    if (r.status === 'DISPOSITION-SIGNED' && hasSignedTokens(r.receiptText)) return { ok: true };
    if (looksPathLike(r.receiptText)) {
      return { fatal: true, reason: `receipt text looks path-like but no path was extracted: "${r.receiptText}"` };
    }
    return { ok: false, reason: 'NO_RECEIPT_PATH' };
  }
  const dangling = extracted.filter((p) => !existsAsBlob(p));
  if (dangling.length) return { ok: false, reason: `DANGLING:${dangling.join(',')}` };
  return { ok: true };
}

function report(tag, rows, pick) {
  const sel = pick ? rows.filter(pick) : rows;
  if (sel.length === 0) { console.log(`${tag} rows=0 EMPTY_SELECTION (vacuous; not a pass)`); return false; }
  const notDone = [];
  for (const r of sel) {
    const ev = evaluateScoreboardRow(r);
    if (ev.fatal) die(`row ${r.id}: ${ev.reason}`);
    if (!ev.ok) notDone.push(`${r.id}:${ev.reason}`);
  }
  console.log(`${tag} rows=${sel.length} done=${sel.length - notDone.length} not_done=${notDone.join(',') || 'none'}`);
  return notDone.length === 0;
}

// --- C0: additive P1 oracle + required CI job -----------------------------

function checkC0() {
  const py = show(ORACLE_PY);
  if (py === null) die(`${ORACLE_PY} not on ${REF}`);
  const hasP0 = new RegExp(`FROZEN_GLLVMTMB_ORACLE\\s*=\\s*["']${P0_SHA}["']`).test(py);
  const hasP1 = new RegExp(`P1_GLLVMTMB_ORACLE\\s*=\\s*["']${P1_SHA}["']`).test(py);
  // The switch (a named-pins map plus an env-var-driven lookup, GLLVM_PARITY_PIN) is necessary
  // but NOT sufficient for C0: the plan's own C0 text requires DEFAULT_R_REF and
  // CAPABILITY_LEDGER_REF to point AT P, not merely be switchable to it on request. A branch
  // that only adds the switch while leaving the live default at P0 is exactly the false-MET
  // this clause exists to catch -- reported as `default_pin_switch_present` below, but it does
  // not by itself satisfy the clause.
  const hasP1Switch = /R_REF_PINS/.test(py) && /os\.environ\.get\(/.test(py) && /DEFAULT_R_REF\s*=\s*R_REF_PINS(\.get\(|\[)/.test(py);
  const hasCapabilityLedgerRef = /CAPABILITY_LEDGER_REF\s*=\s*["'][^"']+["']/.test(py);
  // tools/parity_oracle.py (PR #524) names the live default as a single token, `_DEFAULT_PIN`,
  // precisely so the ACTUAL default can be read directly instead of inferred from the presence
  // of a switch mechanism. C0 is met only once this reads "P1". Anchored to the start of a
  // line (module's own docstring/comments mention the token in prose, e.g. "flip
  // `_DEFAULT_PIN = "P1"` below" -- an unanchored match would find that prose occurrence
  // first and misread the file, the exact false-MET shape this clause exists to prevent).
  const defaultPinMatch = py.match(/^[ \t]*_DEFAULT_PIN\s*=\s*["'](P0|P1)["']/m);
  const defaultPin = defaultPinMatch ? defaultPinMatch[1] : 'MISSING';
  const defaultIsP1 = defaultPin === 'P1';

  const wfFiles = listDir(WORKFLOWS_DIR).filter((f) => /\.ya?ml$/.test(f) && f !== SMOKE_WORKFLOW_FILE);
  let p1JobFile = null;
  for (const f of wfFiles) {
    const t = show(`${WORKFLOWS_DIR}/${f}`);
    if (!t) continue;
    // Recognises exactly PR #524's job (by id and by the tagged-test discovery convention it
    // runs), not any workflow that merely mentions "P1" or "true-parity" in passing.
    const hasJobId = /(^|\n)\s{2}p1-twin-tests:/.test(t);
    const runsTaggedP1 = /gllvm-parity-tag:\s*P1/.test(t);
    const notAdvisory = !/continue-on-error:\s*true/i.test(t);
    if (hasJobId && runsTaggedP1 && notAdvisory) { p1JobFile = f; break; }
  }
  console.log(`C0 p0_pin_present=${hasP0} p1_pin_present=${hasP1} default_pin=${defaultPin} default_pin_switch_present=${hasP1Switch} capability_ledger_ref_present=${hasCapabilityLedgerRef} p1_twin_job_file=${p1JobFile || 'none'} (workflows scanned: ${wfFiles.join(',') || 'none'}; NOTE: GitHub's "required check" status is branch-protection configuration on main, not something readable from repo content -- this only checks the job exists, runs the P1-tagged convention, and is not continue-on-error)`);
  return hasP0 && hasP1 && defaultIsP1 && hasP1Switch && hasCapabilityLedgerRef && !!p1JobFile;
}

// --- C1: required rows bound, receipts resolve, carry rule applied --------

function checkC1() {
  const rows = loadCasemap();
  const req = rows.filter((r) => ['required_core', 'compatibility_adapter'].includes(r.classification));
  let bound = 0, free = 0, boundNumeric = 0, boundSigned = 0;
  const unsigned = {};
  const registrationOnly = [];
  const numericLabelOnly = [];
  const notPassed = [];
  const diffMismatch = [];
  const dangling = [];
  const stale = [];
  for (const r of req) {
    const paths = rowReceiptPaths(r);
    if (paths.length > 0) {
      const dang = paths.filter((p) => !existsAsBlob(p));
      if (dang.length) { dangling.push(`${r.source_id}:${dang.join(',')}`); continue; }
      // The carry/staleness gate only applies to rows that actually cite a receipt: a pure
      // signed disposition (no numeric evidence) has nothing that can go stale.
      const cs = carryStatus(r);
      if (cs.stale) { stale.push(`${r.source_id}:${cs.reason}`); continue; }
    }
    const d = r.disposition ?? null;
    if (d === 'DISPOSITION-SIGNED') {
      // A signed disposition resolves the row but is counted in bound_signed=, never in bound=
      // (bound= is numeric evidence only), so a signed registration row cannot pose as a twin.
      const why = signatureProblem(r.signed_by, r.signed_on);
      if (why === null) boundSigned++; else unsigned[why] = (unsigned[why] || 0) + 1;
      continue;
    }
    if (d !== null && d !== undefined) { unsigned[d] = (unsigned[d] || 0) + 1; continue; }
    // d === null: bound requires BOTH case ids and a resolving receipt -- case ids alone, or a
    // receipt alone, are not enough.
    const caseIdsPresent = Array.isArray(r.executable_case_ids) ? r.executable_case_ids.length > 0 : !!r.executable_case_ids;
    // Evidence tier (D-295 row 5, review of #559): only a row whose receipt records a numeric
    // R-vs-Julia comparison (`evidence_tier: "numeric"`) counts as bound. A registration/existence
    // match ("registration") is a name match, and a missing tier is fail-closed as the same.
    // The label must also be backed by a numeric comparison block in the receipt (review of #561).
    if (caseIdsPresent && paths.length > 0) {
      if (!isNumericTier(r)) { registrationOnly.push(r.source_id); continue; }
      const ns = numericReceiptStatus(r);
      if (ns.ok) { bound++; boundNumeric++; } else if (ns.kind === 'not_passed') {
        const why = receiptStatusExceptionProblem(r);
        if (why === null) boundSigned++; else notPassed.push(`${r.source_id}(${ns.reason}; ${why})`);
      } else if (ns.kind === 'diff_mismatch') diffMismatch.push(`${r.source_id}(${ns.reason})`);
      else numericLabelOnly.push(`${r.source_id}(${ns.reason})`);
    } else {
      free++;
    }
  }
  const nUnsigned = Object.values(unsigned).reduce((a, b) => a + b, 0);
  console.log(`C1 required=${req.length} bound=${bound} bound_numeric=${boundNumeric} bound_registration_only=${registrationOnly.length} bound_signed=${boundSigned} free=${free} unsigned_or_blocked=${nUnsigned} ${JSON.stringify(unsigned)} dangling_receipts=${dangling.join(';') || 'none'} stale_carries=${stale.join(';') || 'none'} registration_only=${registrationOnly.join(';') || 'none'} numeric_label_without_numeric_receipt=${numericLabelOnly.join(';') || 'none'} numeric_receipt_not_passed=${notPassed.join(';') || 'none'} numeric_recorded_diff_mismatch=${diffMismatch.join(';') || 'none'}`);
  return req.length > 0 && nUnsigned === 0 && free === 0 && dangling.length === 0 && stale.length === 0 && registrationOnly.length === 0 && numericLabelOnly.length === 0 && notPassed.length === 0 && diffMismatch.length === 0;
}

// --- C2..C5: scoreboard tiers, plus C2's boundary-capability cross-check --

function checkC2() {
  const rows = scoreboardRows();
  const boardOk = report('C2 P1-boundary capabilities', rows, (r) => !/-RSZ$/i.test(r.id) && !/^RD-/i.test(r.id) && !/^GRP-/i.test(r.id));
  // Cross-check against the case-map: every row carrying a "capability" tag must resolve to
  // a scoreboard row id, or C2 fails even if every scoreboard row itself is done (control (b)).
  const cm = loadCasemap();
  const boardIds = new Set(rows.map((r) => r.id));
  const missing = [...new Set(cm.filter((r) => r.capability && !boardIds.has(r.capability)).map((r) => r.capability))];
  console.log(`C2 capabilities_missing_scoreboard_row=${missing.join(',') || 'none'}`);
  return boardOk && missing.length === 0;
}

function checkC3() { return report('C3 realistic-size', scoreboardRows(), (r) => /-RSZ$/i.test(r.id)); }
function checkC4() { return report('C4 real-data workflows', scoreboardRows(), (r) => /^RD-/i.test(r.id)); }
function checkC5() { return report('C5 grouping levels', scoreboardRows(), (r) => /^GRP-/i.test(r.id)); }

// --- C6: reverse-gap list, one written decision per item, from a fixed vocabulary ----

function checkC6() {
  if (!PATHS.reverseGap) die('no reverse-gap list path configured for this pin');
  const j = show(PATHS.reverseGap);
  if (j === null) die(`${PATHS.reverseGap} not on ${REF}`);
  let items;
  try { items = JSON.parse(j); } catch (e) { die(`${PATHS.reverseGap} is not valid JSON: ${e.message}`); }
  if (!Array.isArray(items) || items.length === 0) { console.log('C6 items=0 EMPTY_SELECTION (vacuous; not a pass)'); return false; }
  const invalid = items.filter((it) => !C6_DECISION_VOCAB.has(it.decision)).map((it) => `${it.source_id || it.name || '?'}:${JSON.stringify(it.decision)}`);
  console.log(`C6 items=${items.length} invalid_decision=${invalid.join(',') || 'none'} (vocabulary: ${[...C6_DECISION_VOCAB].join('|')})`);
  return invalid.length === 0;
}

// --- C7: parity page states the exact "what parity does not mean" heading, pin-independent --

function checkC7() {
  const p = show(PARITY_PAGE);
  if (p === null) die(`${PARITY_PAGE} missing`);
  const hit = /^#+\s*What parity does not mean\s*$/im.test(p);
  console.log(`C7 parity_page_has_exact_not_mean_heading=${hit}`);
  return hit;
}

// --- C8: every export twinned or signed; name matches and unsigned dispositions never count --

function checkC8() {
  const rows = loadCasemap();
  if (rows.length === 0) { console.log('C8 rows=0 EMPTY_SELECTION (vacuous; not a pass)'); return false; }
  const failing = [];
  for (const r of rows) {
    // Receipts before signatures, in C1's order (review of #589, finding 2): a cited receipt must
    // resolve, and a C1-scope row citing one must pass the carry rule, before a signature counts.
    // Otherwise a signed row with a missing receipt or a stale P0 carry passes C8 while failing C1.
    const paths = rowReceiptPaths(r);
    if (paths.length > 0) {
      const dangling = paths.filter((p) => !existsAsBlob(p));
      if (dangling.length) { failing.push(`${r.source_id}:DANGLING_RECEIPT`); continue; }
      if (['required_core', 'compatibility_adapter'].includes(r.classification)) {
        const cs = carryStatus(r);
        if (cs.stale) { failing.push(`${r.source_id}:STALE_CARRY(${cs.reason})`); continue; }
      }
    }
    if (dispositionSignedProperly(r)) continue; // a real, signed-and-dated disposition resolves the row once its receipts pass
    if (r.classification === 'semantic_divergence') {
      // Name matches alone never count, however many executable_case_ids exist, and however
      // it is classified elsewhere: this branch fires on the label itself.
      failing.push(`${r.source_id}:NAME_ONLY_MATCH_NOT_SIGNED`);
      continue;
    }
    if (r.classification === 'outside_boundary') {
      // A capability placed outside the P1 boundary still needs a signed disposition; it does
      // not get to leave the ledger silently.
      failing.push(`${r.source_id}:OUTSIDE_BOUNDARY_NOT_SIGNED`);
      continue;
    }
    const caseIdsPresent = Array.isArray(r.executable_case_ids) ? r.executable_case_ids.length > 0 : !!r.executable_case_ids;
    const twinned = caseIdsPresent && paths.length > 0;
    if (!twinned) { failing.push(`${r.source_id}:NOT_TWINNED_NOT_SIGNED`); continue; }
    // A registration-only receipt is a name match, which never counts as a twin (D-295 row 5).
    if (!isNumericTier(r)) { failing.push(`${r.source_id}:REGISTRATION_ONLY_NOT_TWINNED`); continue; }
    // A "numeric" label without a numeric comparison block in the receipt is still a name match.
    const ns = numericReceiptStatus(r);
    if (ns.ok) continue;
    if (ns.kind === 'not_passed') {
      // A receipt that says it did not pass is not a twin, unless the maintainer signed for it.
      if (receiptStatusExceptionProblem(r) !== null) failing.push(`${r.source_id}:NUMERIC_RECEIPT_NOT_PASSED(${ns.reason})`);
      continue;
    }
    if (ns.kind === 'diff_mismatch') { failing.push(`${r.source_id}:NUMERIC_RECORDED_DIFF_MISMATCH(${ns.reason})`); continue; }
    failing.push(`${r.source_id}:NUMERIC_LABEL_WITHOUT_NUMERIC_RECEIPT`);
  }
  console.log(`C8 rows=${rows.length} failing=${failing.join(';') || 'none'}`);
  return failing.length === 0;
}

// --- X2: every scoreboard row done ----------------------------------------

function checkX2() { return report('X2 all scoreboard rows', scoreboardRows(), null); }

const MODES = { C0: checkC0, C1: checkC1, C2: checkC2, C3: checkC3, C4: checkC4, C5: checkC5, C6: checkC6, C7: checkC7, C8: checkC8, X2: checkX2 };

const mode = process.argv[2];
if (!MODES[mode]) die(`unknown mode ${mode} (expected one of ${Object.keys(MODES).join(', ')})`);
const met = MODES[mode]();
console.log(met ? `${mode}_MET` : `${mode}_NOT_MET`);

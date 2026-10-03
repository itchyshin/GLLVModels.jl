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
//
// Maintainer rulings of 2026-10-02 (itchyshin/GLLVModels.jl#684, signed by Shinichi Nakagawa), the
// only signature recorded here. Prose and schemas: GATES.md, "Rulings of 2026-10-02".
//   Ruling 1 (integer equality): a numeric comparison case may carry `"kind": "integer_equality"`.
//     Then r_value and julia_value must both be integers (or equal-length integer arrays) and
//     tolerance must be exactly 0.5, i.e. exact equality. The row keeps evidence_tier "numeric".
//   Ruling 2 (behavioural tier): a row with `evidence_tier: "behavioural"` (a refusal, a printed
//     summary, a routing decision or an error class) binds when its receipts carry a `behaviour`
//     block (schema in behaviouralReceiptStatus below) in which both engines gave the same
//     canonicalised label for every executable case id, via behaviour-equivalence.json. It counts
//     in C1 `bound_behavioural=`, never in bound= or bound_numeric=. A behavioural label without a
//     valid matching block is BEHAVIOURAL_LABEL_WITHOUT_BEHAVIOURAL_RECEIPT. The scoreboard status
//     EVIDENCED-BEHAVIOURAL counts as done (printed as done_behavioural=), but only on a row the
//     ruling covers and never on a C3, C4 or C5 row.
//     Scope (review of PR #687): the tier binds only for source_id inference/* and the four named C1
//     rows (behaviouralEligibleSourceId); a comparison block in a cited receipt must itself hold; the
//     equivalence table is validated by C1 and C8 even with no behavioural row.
//   Ruling 3 (C6): EXCLUDED_INTERNAL_HELPER joins the decision vocabulary. A decided reverse-gap
//     item also needs a non-empty basis and a `ruling` {ref, signed_by, signed_on}: the ref must be
//     a ruling in C6_RULINGS (only itchyshin/GLLVModels.jl#684 item 3, dated 2026-10-02, which
//     covers KEPT_AS_JULIA_EXTRA and EXCLUDED_INTERNAL_HELPER), the date must be that ruling's, and
//     the signer must pass the signature rule, else it is listed under unsigned_decision=.
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

const DONE = new Set(['EVIDENCED', 'EVIDENCED-BEHAVIOURAL', 'DISPOSITION-SIGNED']);
// C6's decision vocabulary is enumerated, not free text: a reverse-gap item's "decision" must
// be exactly one of these, or it does not count as decided (a placeholder like "TBD" must not
// pass just because the field is non-empty).
const C6_DECISION_VOCAB = new Set([
  'KEPT_AS_JULIA_EXTRA',
  'PORT_TO_MATCH_R',
  'DEPRECATE_AND_REMOVE',
  'RENAME_TO_AVOID_COLLISION',
  'EXCLUDED_INTERNAL_HELPER',
]);
// C6 decisions are signed by a ruling, and a ruling covers only the words it names. The table is the
// ONLY signature the tool accepts on a reverse-gap decision: itchyshin/GLLVModels.jl#684 item 3, signed
// 2026-10-02, covers exactly these two words. A new ruling is a new entry here, added in review.
const C6_RULINGS = {
  'itchyshin/GLLVModels.jl#684 item 3': { signed_on: '2026-10-02', words: new Set(['KEPT_AS_JULIA_EXTRA', 'EXCLUDED_INTERNAL_HELPER']) },
};
const BEHAVIOUR_EQUIVALENCE = 'docs/dev-log/core070/true-parity-latest/behaviour-equivalence.json';
const BEHAVIOUR_KINDS = new Set(['route', 'refusal', 'error_class', 'printed_fields']);
// Scope of the behavioural tier (itchyshin/GLLVModels.jl#684 item 2): the inference routing and
// error-class rows (source_id inference/...) and the four named C1 rows. Any other row that says
// `evidence_tier: "behavioural"` is not covered by the ruling and does not bind.
const BEHAVIOURAL_NAMED_SOURCE_IDS = new Set([
  'latent-scores/extract_latent_scores.default',
  'select-lv/print.gllvmTMB_select_lv',
  'model-comparison/print.anova.gllvmTMB_multi',
  'model-comparison/update.gllvmTMB_multi',
]);
const behaviouralEligibleSourceId = (sid) => typeof sid === 'string' && (sid.startsWith('inference/') || BEHAVIOURAL_NAMED_SOURCE_IDS.has(sid));
// The assembler writes a scoreboard id as the source_id with every run of other characters turned into '-'.
const scoreboardSlug = (sid) => sid.replace(/[^A-Za-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');
const BEHAVIOURAL_ELIGIBLE_BOARD_IDS = new Set([...BEHAVIOURAL_NAMED_SOURCE_IDS].map(scoreboardSlug));
const behaviouralEligibleBoardId = (id) => id.startsWith('inference-') || BEHAVIOURAL_ELIGIBLE_BOARD_IDS.has(id);

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
function isBehaviouralTier(row) { return row.evidence_tier === 'behavioural'; }

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

function receiptNotPassed(j, p, blockKey = 'comparison') {
  const blk = j[blockKey];
  const where = [[j, ''], [blk && typeof blk === 'object' ? blk : null, `${blockKey}.`]];
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

// Ruling 1 (itchyshin/GLLVModels.jl#684 item 1): exact-integer rows (a degrees of freedom, an
// observation count) may record tolerance 0.5 instead of a small float. The case must say so with
// `"kind": "integer_equality"`; r_value and julia_value must then both be integers (finite numbers
// with Number.isSafeInteger so 2^53 and beyond are refused, or equal-length non-empty arrays of them) and tolerance exactly 0.5, so
// "within tolerance" can only mean "equal". A case with no `kind` is judged as before; any other
// kind fails the row (it is never read as numeric by default). Returns null, or why not.
function integerEqualityProblem(c) {
  if (c.kind === undefined) return null;
  if (c.kind !== 'integer_equality') return `unknown comparison kind ${JSON.stringify(c.kind)}`;
  const isInt = (x) => typeof x === 'number' && Number.isSafeInteger(x);
  const ints = (x) => isInt(x) || (Array.isArray(x) && x.length > 0 && x.every(isInt));
  if (!ints(c.r_value) || !ints(c.julia_value)) return 'integer_equality needs integer r_value and julia_value';
  if (Array.isArray(c.r_value) !== Array.isArray(c.julia_value) || (Array.isArray(c.r_value) && c.r_value.length !== c.julia_value.length)) {
    return 'integer_equality needs r_value and julia_value of the same shape and length';
  }
  if (c.tolerance !== 0.5) return 'integer_equality needs tolerance exactly 0.5';
  return null;
}

// One `comparison` block, judged case by case (pin, tolerance, integer equality, recorded difference,
// within tolerance). Adds each compared case id to `covered`. Returns null when the block holds, else
// { ok: false, reason, kind? }. numericReceiptStatus uses it for numeric rows; behaviouralReceiptStatus
// uses it so that a behavioural row cannot cite a receipt whose own comparison is out of tolerance.
function checkComparisonBlock(cmp, p, covered) {
  if (!cmp || typeof cmp !== 'object' || Array.isArray(cmp)) return { ok: false, reason: `malformed comparison in ${p}` };
  if (cmp.pin !== 'P1' && cmp.pin !== P1_SHA) return { ok: false, reason: `comparison not pinned to P1 in ${p}` };
  if (!Array.isArray(cmp.cases) || cmp.cases.length === 0) return { ok: false, reason: `comparison has no cases in ${p}` };
  for (const c of cmp.cases) {
    if (!c || typeof c.case_id !== 'string' || c.case_id.length === 0) return { ok: false, reason: `comparison case without case_id in ${p}` };
    const tol = c.tolerance;
    if (typeof tol !== 'number' || !Number.isFinite(tol) || tol <= 0) return { ok: false, reason: `case ${c.case_id}: tolerance not a finite number > 0` };
    const intProblem = integerEqualityProblem(c);
    if (intProblem) return { ok: false, reason: `case ${c.case_id}: ${intProblem}` };
    const cd = comparisonCaseDiff(c);
    if (cd.mismatch) return { ok: false, kind: 'diff_mismatch', reason: `${cd.mismatch} in ${p}` };
    const d = cd.d;
    if (d === null) return { ok: false, reason: `case ${c.case_id}: no finite abs_diff or r_value/julia_value` };
    if (d > tol) return { ok: false, reason: `case ${c.case_id}: abs_diff ${d} > tolerance ${tol}${c.kind === 'integer_equality' ? ' (integer_equality: the integers differ)' : ''}` };
    covered.add(c.case_id);
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
    const bad = checkComparisonBlock(j.comparison, p, covered);
    if (bad) return bad;
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

// --- Ruling 2: the behavioural tier (itchyshin/GLLVModels.jl#684 item 2) ------------------
//
// A row whose R behaviour is a refusal, a printed summary, a routing decision or an error class has
// no number to compare. It binds when a cited receipt shows both engines giving the same refusal,
// route, error class or printed fields. The receipt carries a top-level `behaviour` block:
//
//   "behaviour": {
//     "pin": "P1",                              // or the full P1 sha
//     "cases": [
//       { "case_id": "CORE070-...",             // non-empty string
//         "source_id": "inference/CI-ROUTE-001",// OPTIONAL: when present, applies only to that row
//         "kind": "route" | "refusal" | "error_class" | "printed_fields",
//         "r_observed": "<label>" | ["<label>", ...],       // non-empty strings
//         "julia_observed": same shape and length }
//     ]
//   }
//
// Each label is what that engine actually produced, read from a raw artefact of the run. The two
// sides are compared after canonicalisation through behaviour-equivalence.json: a class of one kind
// lists the labels each engine uses for the same behaviour, with a one-sentence basis naming the R
// and Julia functions. A label in no class stands for itself. A label listed in two classes of the
// same kind and side makes the table ambiguous: MEASUREMENT_FAILED. The row binds when every
// executable case id has at least one applicable entry (same case_id, and either no source_id or
// the row's own), every applicable entry matches, the carry is fresh at P1, and no cited receipt
// has a failed status field (status, verdict, batch_status, harness_pass at the top level or in the
// behaviour block). A malformed block in any cited receipt fails the row.
let equivalenceCache = null;
function loadEquivalence() {
  if (equivalenceCache) return equivalenceCache;
  const txt = show(BEHAVIOUR_EQUIVALENCE);
  const index = {}; // index[kind][side][label] = { canonical, cls }
  if (txt !== null) {
    let t;
    try { t = JSON.parse(txt); } catch (e) { die(`${BEHAVIOUR_EQUIVALENCE} is not valid JSON: ${e.message}`); }
    if (!t || typeof t !== 'object' || t.schema !== 1) die(`${BEHAVIOUR_EQUIVALENCE}: schema must be 1`);
    if (t.pin !== 'P1' && t.pin !== P1_SHA) die(`${BEHAVIOUR_EQUIVALENCE}: pin must be P1`);
    if (!Array.isArray(t.classes)) die(`${BEHAVIOUR_EQUIVALENCE}: classes must be an array`);
    const nonEmpty = (x) => typeof x === 'string' && x.trim().length > 0;
    const canonicalSeen = new Set();
    t.classes.forEach((cls, i) => {
      if (!cls || typeof cls !== 'object' || !BEHAVIOUR_KINDS.has(cls.kind)) die(`${BEHAVIOUR_EQUIVALENCE}: class ${i} has no valid kind`);
      if (!nonEmpty(cls.canonical)) die(`${BEHAVIOUR_EQUIVALENCE}: class ${i} has no canonical label`);
      // Two classes of one kind with the same canonical label would merge unrelated labels silently.
      if (canonicalSeen.has(`${cls.kind}\u0000${cls.canonical}`)) die(`${BEHAVIOUR_EQUIVALENCE}: duplicate canonical ${JSON.stringify(cls.canonical)} for kind ${cls.kind} (add the labels to the existing class)`);
      canonicalSeen.add(`${cls.kind}\u0000${cls.canonical}`);
      if (!nonEmpty(cls.basis)) die(`${BEHAVIOUR_EQUIVALENCE}: class ${i} (${cls.canonical}) has an empty basis`);
      for (const side of ['r', 'julia']) {
        if (!Array.isArray(cls[side]) || !cls[side].every(nonEmpty)) die(`${BEHAVIOUR_EQUIVALENCE}: class ${i} (${cls.canonical}) ${side} must be an array of non-empty labels`);
        const bucket = ((index[cls.kind] ||= {})[side] ||= new Map());
        for (const label of cls[side]) {
          const prior = bucket.get(label);
          if (prior && prior.cls !== i) die(`${BEHAVIOUR_EQUIVALENCE}: ambiguous table, ${cls.kind} ${side} label ${JSON.stringify(label)} is in classes ${prior.canonical} and ${cls.canonical}`);
          bucket.set(label, { canonical: cls.canonical, cls: i });
        }
      }
    });
  }
  equivalenceCache = index;
  return index;
}

function canonicalLabel(kind, side, label) {
  const hit = ((loadEquivalence()[kind] || {})[side] || new Map()).get(label);
  return hit ? hit.canonical : label;
}

// Returns null when the observed labels are well formed, else why not.
function observedShapeProblem(c) {
  const nonEmpty = (x) => typeof x === 'string' && x.trim().length > 0;
  for (const k of ['r_observed', 'julia_observed']) {
    const v = c[k];
    if (!(nonEmpty(v) || (Array.isArray(v) && v.length > 0 && v.every(nonEmpty)))) return `${k} must be a non-empty string or an array of non-empty strings`;
  }
  if (Array.isArray(c.r_observed) !== Array.isArray(c.julia_observed)) return 'r_observed and julia_observed must have the same shape';
  if (Array.isArray(c.r_observed) && c.r_observed.length !== c.julia_observed.length) return `r_observed has ${c.r_observed.length} labels, julia_observed ${c.julia_observed.length}`;
  return null;
}

function behaviouralReceiptStatus(row) {
  const paths = rowReceiptPaths(row);
  if (paths.length === 0) return { ok: false, reason: 'no receipt' };
  // Scope: the ruling covers the inference routing and error-class rows and four named C1 rows only.
  // Any other row labelled behavioural (a numeric row, a campaign row) does not bind on typed labels.
  if (!behaviouralEligibleSourceId(row.source_id)) return { ok: false, reason: 'source_id not covered by itchyshin/GLLVModels.jl#684 item 2 (inference/* and four named C1 rows only)' };
  loadEquivalence(); // a malformed or ambiguous table is a measurement failure, whatever the row says
  const cs = carryStatus(row);
  if (cs.stale) return { ok: false, reason: cs.reason };
  const ids = Array.isArray(row.executable_case_ids) ? row.executable_case_ids : (row.executable_case_ids ? [row.executable_case_ids] : []);
  if (ids.length === 0) return { ok: false, reason: 'no executable_case_ids' };
  const entries = [];
  let blocks = 0;
  let notPassed = null;
  for (const p of paths) {
    const txt = show(p);
    if (txt === null) return { ok: false, reason: `unreadable ${p}` };
    let j;
    try { j = JSON.parse(txt); } catch { continue; }
    if (!j || typeof j !== 'object') continue;
    if (notPassed === null) notPassed = receiptNotPassed(j, p, 'behaviour');
    // A comparison block in a cited receipt must itself hold: a numeric failure is not hidden by relabelling the row.
    if (j.comparison !== undefined) {
      const bad = checkComparisonBlock(j.comparison, p, new Set());
      if (bad) return { ok: false, reason: `cited receipt's own comparison fails: ${bad.reason}` };
    }
    if (j.behaviour === undefined) continue;
    const b = j.behaviour;
    if (!b || typeof b !== 'object' || Array.isArray(b)) return { ok: false, reason: `malformed behaviour block in ${p}` };
    if (b.pin !== 'P1' && b.pin !== P1_SHA) return { ok: false, reason: `behaviour not pinned to P1 in ${p}` };
    if (!Array.isArray(b.cases) || b.cases.length === 0) return { ok: false, reason: `behaviour has no cases in ${p}` };
    for (const c of b.cases) {
      if (!c || typeof c.case_id !== 'string' || c.case_id.length === 0) return { ok: false, reason: `behaviour case without case_id in ${p}` };
      if (c.source_id !== undefined && (typeof c.source_id !== 'string' || c.source_id.length === 0)) return { ok: false, reason: `case ${c.case_id}: source_id must be a non-empty string` };
      if (!BEHAVIOUR_KINDS.has(c.kind)) return { ok: false, reason: `case ${c.case_id}: kind ${JSON.stringify(c.kind)} is not one of ${[...BEHAVIOUR_KINDS].join('|')}` };
      const shape = observedShapeProblem(c);
      if (shape) return { ok: false, reason: `case ${c.case_id}: ${shape}` };
      if (notPassed === null) {
        const f = RECEIPT_STATUS_FIELDS.find((k) => Object.prototype.hasOwnProperty.call(c, k) && !isPassValue(c[k]));
        if (f) notPassed = `behaviour.cases[${c.case_id}].${f}=${JSON.stringify(c[f])} in ${p}`;
      }
      entries.push(c);
    }
    blocks++;
  }
  if (blocks === 0) return { ok: false, reason: 'no behaviour block in any receipt' };
  const uncovered = [];
  for (const id of ids) {
    const app = entries.filter((e) => e.case_id === id && (e.source_id === undefined || e.source_id === row.source_id));
    if (app.length === 0) { uncovered.push(id); continue; }
    for (const e of app) {
      const rs = Array.isArray(e.r_observed) ? e.r_observed : [e.r_observed];
      const js = Array.isArray(e.julia_observed) ? e.julia_observed : [e.julia_observed];
      for (let i = 0; i < rs.length; i++) {
        const rc = canonicalLabel(e.kind, 'r', rs[i]);
        const jc = canonicalLabel(e.kind, 'julia', js[i]);
        if (rc !== jc) return { ok: false, reason: `case ${id} (${e.kind}): R ${JSON.stringify(rs[i])} vs Julia ${JSON.stringify(js[i])} differ after canonicalisation (${JSON.stringify(rc)} vs ${JSON.stringify(jc)})` };
      }
    }
  }
  if (uncovered.length) return { ok: false, reason: `case ids without an applicable behaviour entry: ${uncovered.join(',')}` };
  if (notPassed !== null) return { ok: false, reason: `receipt did not pass: ${notPassed}` };
  return { ok: true };
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
  // EVIDENCED-BEHAVIOURAL is only for the rows ruling 2 covers (inference-*, or a named C1 row); it never
  // closes a C3 (realistic size), C4 (real data) or C5 (grouping) row, which are numeric campaign clauses.
  if (r.status === 'EVIDENCED-BEHAVIOURAL' && (isRSZ(r) || isRD(r) || isGRP(r) || !behaviouralEligibleBoardId(r.id))) {
    return { ok: false, reason: 'BEHAVIOURAL_NOT_ALLOWED_FOR_THIS_ROW' };
  }
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
  let doneBehavioural = 0;
  for (const r of sel) {
    const ev = evaluateScoreboardRow(r);
    if (ev.fatal) die(`row ${r.id}: ${ev.reason}`);
    if (!ev.ok) notDone.push(`${r.id}:${ev.reason}`);
    else if (r.status === 'EVIDENCED-BEHAVIOURAL') doneBehavioural++;
  }
  console.log(`${tag} rows=${sel.length} done=${sel.length - notDone.length} not_done=${notDone.join(',') || 'none'} done_behavioural=${doneBehavioural}`);
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
  loadEquivalence(); // a malformed or ambiguous equivalence table is a measurement failure even with no behavioural row yet
  const rows = loadCasemap();
  const req = rows.filter((r) => ['required_core', 'compatibility_adapter'].includes(r.classification));
  let bound = 0, free = 0, boundNumeric = 0, boundSigned = 0, boundBehavioural = 0;
  const unsigned = {};
  const registrationOnly = [];
  const numericLabelOnly = [];
  const notPassed = [];
  const diffMismatch = [];
  const behaviouralLabelOnly = [];
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
      // Ruling 2 (itchyshin/GLLVModels.jl#684 item 2): a behavioural row binds on a matching `behaviour` block, counted apart from numeric.
      if (isBehaviouralTier(r)) {
        const bs = behaviouralReceiptStatus(r);
        if (bs.ok) boundBehavioural++; else behaviouralLabelOnly.push(`${r.source_id}(${bs.reason})`);
        continue;
      }
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
  console.log(`C1 required=${req.length} bound=${bound} bound_numeric=${boundNumeric} bound_registration_only=${registrationOnly.length} bound_signed=${boundSigned} free=${free} unsigned_or_blocked=${nUnsigned} ${JSON.stringify(unsigned)} dangling_receipts=${dangling.join(';') || 'none'} stale_carries=${stale.join(';') || 'none'} registration_only=${registrationOnly.join(';') || 'none'} numeric_label_without_numeric_receipt=${numericLabelOnly.join(';') || 'none'} numeric_receipt_not_passed=${notPassed.join(';') || 'none'} numeric_recorded_diff_mismatch=${diffMismatch.join(';') || 'none'} bound_behavioural=${boundBehavioural} behavioural_label_without_behavioural_receipt=${behaviouralLabelOnly.join(';') || 'none'}`);
  return req.length > 0 && nUnsigned === 0 && free === 0 && dangling.length === 0 && stale.length === 0 && registrationOnly.length === 0 && numericLabelOnly.length === 0 && notPassed.length === 0 && diffMismatch.length === 0 && behaviouralLabelOnly.length === 0;
}

// --- C2..C5: scoreboard tiers, plus C2's boundary-capability cross-check --

// The assembled scoreboard prefixes every id with its family (`data-RD-01`, `grouping-GRP-..`), so a
// tier marker is recognised after any `<family>-` prefix (one or more hyphenated segments). A marker buried mid-word
// (`family-NB2RD-X`) is deliberately not matched.
const isRD = (r) => /^(?:[a-z0-9_]+-)*RD-/i.test(r.id);
const isGRP = (r) => /^(?:[a-z0-9_]+-)*GRP-/i.test(r.id);
const isRSZ = (r) => /-RSZ$/i.test(r.id);

function checkC2() {
  const rows = scoreboardRows();
  const boardOk = report('C2 P1-boundary capabilities', rows, (r) => !isRSZ(r) && !isRD(r) && !isGRP(r));
  // Cross-check against the case-map: every row carrying a "capability" tag must resolve to
  // a scoreboard row id, or C2 fails even if every scoreboard row itself is done (control (b)).
  const cm = loadCasemap();
  const boardIds = new Set(rows.map((r) => r.id));
  const missing = [...new Set(cm.filter((r) => r.capability && !boardIds.has(r.capability)).map((r) => r.capability))];
  console.log(`C2 capabilities_missing_scoreboard_row=${missing.join(',') || 'none'}`);
  return boardOk && missing.length === 0;
}

function checkC3() { return report('C3 realistic-size', scoreboardRows(), isRSZ); }
function checkC4() { return report('C4 real-data workflows', scoreboardRows(), isRD); }
function checkC5() { return report('C5 grouping levels', scoreboardRows(), isGRP); }

// --- C6: reverse-gap list, one written decision per item, from a fixed vocabulary ----

function checkC6() {
  if (!PATHS.reverseGap) die('no reverse-gap list path configured for this pin');
  const j = show(PATHS.reverseGap);
  if (j === null) die(`${PATHS.reverseGap} not on ${REF}`);
  let items;
  try { items = JSON.parse(j); } catch (e) { die(`${PATHS.reverseGap} is not valid JSON: ${e.message}`); }
  if (!Array.isArray(items) || items.length === 0) { console.log('C6 items=0 EMPTY_SELECTION (vacuous; not a pass)'); return false; }
  const invalid = items.filter((it) => !C6_DECISION_VOCAB.has(it.decision)).map((it) => `${it.source_id || it.name || '?'}:${JSON.stringify(it.decision)}`);
  // Ruling 3 (itchyshin/GLLVModels.jl#684 item 3): a decided item needs a written basis and a
  // ruling {ref, signed_by, signed_on} whose signer and date pass the signature rule. The tool
  // cannot check that the named person signed (PR review does), but it refuses an item with no
  // basis, no ruling reference, or a signer outside the allow-list.
  const unsignedDecision = [];
  const counts = Object.fromEntries([...C6_DECISION_VOCAB].map((d) => [d, 0]));
  for (const it of items) {
    if (!C6_DECISION_VOCAB.has(it.decision)) continue;
    counts[it.decision]++;
    const id = it.source_id || it.name || '?';
    const rl = it.ruling;
    const nonEmpty = (x) => typeof x === 'string' && x.trim().length > 0;
    let why = null;
    if (!nonEmpty(it.basis)) why = 'no basis';
    else if (!rl || typeof rl !== 'object' || Array.isArray(rl)) why = 'no ruling';
    else if (!nonEmpty(rl.ref)) why = 'ruling without a ref';
    else if (!Object.prototype.hasOwnProperty.call(C6_RULINGS, rl.ref)) why = `ruling ref ${JSON.stringify(rl.ref)} is not a recognised signed ruling`;
    else if (!C6_RULINGS[rl.ref].words.has(it.decision)) why = `decision ${it.decision} is not covered by ${rl.ref}`;
    else why = signatureProblem(rl.signed_by, rl.signed_on);
    if (why === null && rl.signed_on.trim() !== C6_RULINGS[rl.ref].signed_on) why = `ruling signed_on ${JSON.stringify(rl.signed_on)} is not the date of ${rl.ref}`;
    if (why !== null) unsignedDecision.push(`${id}(${why})`);
  }
  console.log(`C6 items=${items.length} invalid_decision=${invalid.join(',') || 'none'} (vocabulary: ${[...C6_DECISION_VOCAB].join('|')}) unsigned_decision=${unsignedDecision.join(',') || 'none'} decision_counts=${Object.entries(counts).map(([d, n]) => `${d}:${n}`).join(',')}`);
  return invalid.length === 0 && unsignedDecision.length === 0;
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
  loadEquivalence();
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
    // Ruling 2: a behavioural row is twinned (behaviourally) when its receipts show the same
    // refusal, route, error class or printed fields from both engines.
    if (isBehaviouralTier(r)) {
      const bs = behaviouralReceiptStatus(r);
      if (!bs.ok) failing.push(`${r.source_id}:BEHAVIOURAL_LABEL_WITHOUT_BEHAVIOURAL_RECEIPT(${bs.reason})`);
      continue;
    }
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

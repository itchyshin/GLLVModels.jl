#!/usr/bin/env node
// True-parity acceptance oracle for GLLVModels.jl vs gllvmTMB, pinned per-pin (P0 frozen at
// 0.7.0 b4d5fee64def88bc768dda1f1f77c29b295edd86; P1 = gllvmTMB main
// 9539352f66f2db2cc26b1c393e67212a359b60c9, 0.7.1 candidate, D-294/D-295, 2026-09-27).
//
// One mode per destination clause in ultra-plan.md "Destination (the stopping condition)":
// C0 C1 C2 C3 C4 C5 C6 C7 C8 X2. Prints measured numbers, then <mode>_MET only when the
// clause holds. Exit 0 on a clean measurement regardless of MET/NOT_MET (the gate is decided
// by the printed verdict, not the exit code); exit 2 if the measurement itself could not be
// made (missing file, malformed JSON, etc — MEASUREMENT_FAILED). An empty row selection is
// never a pass.
//
// Reads a git ref (default origin/main), never the working tree, so a gate cannot pass on an
// unmerged branch. PARITY_REF=FS switches to a filesystem fixture tree rooted at
// PARITY_FS_ROOT, for the negative-control tests in test/fixtures/true_parity/.
//
// Ledger tracked in git (was gitignored under .unlazy/true-parity/ and lost receipts; see
// docs/dev-log/core070/true-parity-latest/GATES.md). Copied from and extends
// ~/local-scratch/lanes/GLLVM.jl-gllvm-backlog-20260926/.unlazy/true-parity/check.mjs
// (untracked, kept as-is; this file is the tracked continuation, not an edit of that one).
import { execFileSync } from 'node:child_process';
import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

const REF = process.env.PARITY_REF || 'origin/main';
const PIN = process.env.PARITY_PIN || 'P1';
const FS_ROOT = process.env.PARITY_FS_ROOT || '.';

const P0_SHA = 'b4d5fee64def88bc768dda1f1f77c29b295edd86';
const P1_SHA = '9539352f66f2db2cc26b1c393e67212a359b60c9';

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

const DONE = new Set(['EVIDENCED', 'DISPOSITION-SIGNED']);

const git = (...a) => execFileSync('git', a, { encoding: 'utf8', maxBuffer: 64 << 20, stdio: ['ignore', 'pipe', 'pipe'] });
const die = (m) => { console.log(`MEASUREMENT_FAILED ${m}`); process.exit(2); };

function show(p) {
  try {
    return REF === 'FS' ? readFileSync(join(FS_ROOT, p), 'utf8') : git('show', `${REF}:${p}`);
  } catch {
    return null;
  }
}

function exists(p) {
  if (REF === 'FS') return existsSync(join(FS_ROOT, p));
  try { git('cat-file', '-e', `${REF}:${p}`); return true; } catch { return false; }
}

function listDir(dir) {
  if (REF === 'FS') {
    try { return readdirSync(join(FS_ROOT, dir)); } catch { return []; }
  }
  try { return git('ls-tree', '--name-only', REF, '--', dir).split('\n').filter(Boolean).map((p) => p.split('/').pop()); } catch { return []; }
}

// --- shared parsers -------------------------------------------------------

// Extracts path-like tokens (docs/..., tools/..., test/..., src/...) from free text, e.g. a
// scoreboard's "Receipt / disposition" column, so receipt resolution can check they exist.
function extractPaths(text) {
  if (!text) return [];
  const m = text.match(/\b(?:docs|tools|test|src)\/[A-Za-z0-9._\-/]+\.(?:md|json|toml|jl|py|mjs)\b/g);
  return m ? [...new Set(m)] : [];
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

// A row carried from P0 counts at P1 only if every source-pin file is byte-identical between
// the gllvmTMB commit it was measured against and P1 (the two hashes are recorded on the row
// itself when it is re-measured; this tool never reaches into an external gllvmTMB clone).
function carryIsStale(row) {
  const c = row.carry;
  if (!c || !Array.isArray(c.source_pins) || c.source_pins.length === 0) return false;
  return c.source_pins.some((sp) => !sp.sha256_at_p1 || sp.sha256_at_p0 !== sp.sha256_at_p1);
}

function danglingReceipts(row) {
  return rowReceiptPaths(row).filter((p) => !exists(p));
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

function report(tag, rows, pick) {
  const sel = pick ? rows.filter(pick) : rows;
  if (sel.length === 0) { console.log(`${tag} rows=0 EMPTY_SELECTION (vacuous; not a pass)`); return false; }
  const dangling = [];
  const open = sel.filter((r) => {
    if (!DONE.has(r.status)) return true;
    const bad = extractPaths(r.receiptText).filter((p) => !exists(p));
    if (bad.length) { dangling.push(`${r.id}:${bad.join(',')}`); return true; }
    return false;
  });
  console.log(`${tag} rows=${sel.length} done=${sel.length - open.length} not_done=${open.map((r) => r.id).join(',') || 'none'}` + (dangling.length ? ` dangling_receipts=${dangling.join(';')}` : ''));
  return open.length === 0;
}

// --- C0: additive P1 oracle + required CI job -----------------------------

function checkC0() {
  const py = show(ORACLE_PY);
  if (py === null) die(`${ORACLE_PY} not on ${REF}`);
  const hasP0 = new RegExp(`FROZEN_GLLVMTMB_ORACLE\\s*=\\s*["']${P0_SHA}["']`).test(py);
  const hasP1 = new RegExp(`P1_GLLVMTMB_ORACLE\\s*=\\s*["']${P1_SHA}["']`).test(py);
  const defaultsToP1 = /DEFAULT_R_REF\s*=\s*P1_GLLVMTMB_ORACLE/.test(py);
  const wfFiles = listDir(WORKFLOWS_DIR).filter((f) => /\.ya?ml$/.test(f));
  let requiredP1Job = false;
  for (const f of wfFiles) {
    const t = show(`${WORKFLOWS_DIR}/${f}`);
    if (!t) continue;
    // Best-effort, not a full YAML parse: a job block mentioning the P1 pin or "true-parity"
    // that is not marked continue-on-error counts as a required job.
    const blocks = t.split(/\n(?=\s{2}\S)/);
    for (const b of blocks) {
      if (/true[-_]parity|P1\b/i.test(b) && !/continue-on-error:\s*true/i.test(b)) { requiredP1Job = true; break; }
    }
    if (requiredP1Job) break;
  }
  console.log(`C0 p0_pin_present=${hasP0} p1_pin_present=${hasP1} default_r_ref_is_p1=${defaultsToP1} required_p1_ci_job=${requiredP1Job} (workflows scanned: ${wfFiles.join(',') || 'none'})`);
  return hasP0 && hasP1 && defaultsToP1 && requiredP1Job;
}

// --- C1: required rows bound, receipts resolve, carry rule applied --------

function checkC1() {
  const rows = loadCasemap();
  const req = rows.filter((r) => ['required_core', 'compatibility_adapter'].includes(r.classification));
  let bound = 0, free = 0;
  const unsigned = {};
  const dangling = [];
  const stale = [];
  for (const r of req) {
    const dang = danglingReceipts(r);
    if (dang.length) { dangling.push(`${r.source_id}:${dang.join(',')}`); continue; }
    if (carryIsStale(r)) { stale.push(r.source_id); continue; }
    const d = r.disposition ?? null;
    const hasReceipt = (Array.isArray(r.executable_case_ids) ? r.executable_case_ids.length > 0 : !!r.executable_case_ids) || rowReceiptPaths(r).length > 0;
    if (d === null || d === undefined) { hasReceipt ? bound++ : free++; continue; }
    if (/^(BLOCKED|PARTIAL)/.test(d)) unsigned[d] = (unsigned[d] || 0) + 1;
    else if (d === 'DISPOSITION-SIGNED') bound++;
    else unsigned[d] = (unsigned[d] || 0) + 1;
  }
  const nUnsigned = Object.values(unsigned).reduce((a, b) => a + b, 0);
  console.log(`C1 required=${req.length} bound=${bound} free=${free} unsigned_or_blocked=${nUnsigned} ${JSON.stringify(unsigned)} dangling_receipts=${dangling.join(';') || 'none'} stale_carries=${stale.map((s) => `${s}:PARTIAL_STALE_AT_P1`).join(';') || 'none'}`);
  return req.length > 0 && nUnsigned === 0 && free === 0 && dangling.length === 0 && stale.length === 0;
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

// --- C6: reverse-gap list, one written decision per item ------------------

function checkC6() {
  if (!PATHS.reverseGap) die('no reverse-gap list path configured for this pin');
  const j = show(PATHS.reverseGap);
  if (j === null) die(`${PATHS.reverseGap} not on ${REF}`);
  let items;
  try { items = JSON.parse(j); } catch (e) { die(`${PATHS.reverseGap} is not valid JSON: ${e.message}`); }
  if (!Array.isArray(items) || items.length === 0) { console.log('C6 items=0 EMPTY_SELECTION (vacuous; not a pass)'); return false; }
  const undecided = items.filter((it) => !it.decision).map((it) => it.source_id || it.name || '?');
  console.log(`C6 items=${items.length} undecided=${undecided.join(',') || 'none'}`);
  return undecided.length === 0;
}

// --- C7: parity page states what parity does not mean, pin-independent ---

function checkC7() {
  const p = show(PARITY_PAGE);
  if (p === null) die(`${PARITY_PAGE} missing`);
  const hit = /^#+ .*(does not mean|is not|not (a )?parity claim)/im.test(p);
  console.log(`C7 parity_page_has_not_mean_section=${hit}`);
  return hit;
}

// --- C8: every export twinned or signed; name-only matches never count ---

function checkC8() {
  const rows = loadCasemap();
  const relevant = rows.filter((r) => r.classification !== 'outside_boundary');
  if (relevant.length === 0) { console.log('C8 rows=0 EMPTY_SELECTION (vacuous; not a pass)'); return false; }
  const failing = [];
  for (const r of relevant) {
    const signed = r.disposition === 'DISPOSITION-SIGNED';
    if (signed) continue;
    if (r.classification === 'semantic_divergence') {
      // Name matches alone never count, however many executable_case_ids exist.
      failing.push(`${r.source_id}:NAME_ONLY_MATCH_NOT_SIGNED`);
      continue;
    }
    const dang = danglingReceipts(r);
    if (dang.length) { failing.push(`${r.source_id}:DANGLING_RECEIPT`); continue; }
    const twinned = (Array.isArray(r.executable_case_ids) ? r.executable_case_ids.length > 0 : !!r.executable_case_ids);
    if (!twinned) failing.push(`${r.source_id}:NOT_TWINNED_NOT_SIGNED`);
  }
  console.log(`C8 rows=${relevant.length} failing=${failing.join(';') || 'none'}`);
  return failing.length === 0;
}

// --- X2: every scoreboard row done ----------------------------------------

function checkX2() { return report('X2 all scoreboard rows', scoreboardRows(), null); }

const MODES = { C0: checkC0, C1: checkC1, C2: checkC2, C3: checkC3, C4: checkC4, C5: checkC5, C6: checkC6, C7: checkC7, C8: checkC8, X2: checkX2 };

const mode = process.argv[2];
if (!MODES[mode]) die(`unknown mode ${mode} (expected one of ${Object.keys(MODES).join(', ')})`);
const met = MODES[mode]();
console.log(met ? `${mode}_MET` : `${mode}_NOT_MET`);

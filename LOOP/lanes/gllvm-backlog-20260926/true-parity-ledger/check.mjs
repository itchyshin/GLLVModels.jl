#!/usr/bin/env node
// True-parity acceptance oracle for GLLVModels.jl vs frozen gllvmTMB 0.7.0 (b4d5fee64).
// One mode per destination clause in docs/dev-log/core070/true-parity-decision-map.md.
// Prints measured numbers, then C<n>_MET only when the clause holds. Exit 0 always on a
// clean measurement (the gate is decided by EXPECT), exit 2 if the measurement itself failed.
// Reads origin/main (never the working tree), so a gate cannot pass on an unmerged branch.
import { execFileSync } from 'node:child_process';

const REF = process.env.PARITY_REF || 'origin/main';
const SCOREBOARD = 'docs/dev-log/core070/true-parity-gate-tier-scoreboard-2026-09-25.md';
const CASEMAP = 'docs/dev-log/core070/required-source-case-map.json';
const PARITY_PAGE = 'docs/src/gllvmtmb-parity.md';
const DONE = new Set(['EVIDENCED', 'DISPOSITION-SIGNED']);

const git = (...a) => execFileSync('git', a, { encoding: 'utf8', maxBuffer: 64 << 20 });
import { readFileSync } from 'node:fs';
const show = (p) => { try { return REF === 'FS' ? readFileSync(`${process.env.PARITY_FS_ROOT}/${p}`, 'utf8') : git('show', `${REF}:${p}`); } catch { return null; } };
const die = (m) => { console.log(`MEASUREMENT_FAILED ${m}`); process.exit(2); };

function scoreboardRows(prefix) {
  const t = show(SCOREBOARD);
  if (t === null) die(`${SCOREBOARD} not on ${REF} (PR #487 unmerged?)`);
  const rows = t.split('\n').filter((l) => /^\| [ABCD]\d+ /.test(l)).map((l) => {
    const c = l.split('|').map((s) => s.trim());
    const parts = c[1].split(' '); return { id: parts[0], name: c[1], status: c[3] };
  });
  if (rows.length !== 32) die(`scoreboard has ${rows.length} rows, expected 32`);
  return prefix ? rows.filter((r) => prefix.some((p) => r.id.startsWith(p) && /^\d+$/.test(r.id.slice(p.length)))) : rows;
}
function report(tag, rows, pick) {
  const sel = pick ? rows.filter(pick) : rows;
  const open = sel.filter((r) => !DONE.has(r.status));
  console.log(`${tag} rows=${sel.length} done=${sel.length - open.length} not_done=${open.map((r) => `${r.id}:${r.status}`).join(',') || 'none'}`);
  if (sel.length === 0) { console.log(`${tag} EMPTY_SELECTION (vacuous; not a pass)`); return false; }
  return open.length === 0;
}

const mode = process.argv[2];
let met = false;
switch (mode) {
  case 'C1': { // every required ledger row bound or maintainer-signed; BLOCKED_*/PARTIAL_* are NOT signed
    const j = show(CASEMAP); if (j === null) die(`${CASEMAP} missing`);
    const rows = JSON.parse(j).rows || [];
    const req = rows.filter((r) => ['required_core', 'compatibility_adapter'].includes(r.classification));
    const unsigned = {};
    let bound = 0, free = 0;
    for (const r of req) {
      const d = r.disposition ?? null;
      const hasReceipt = Array.isArray(r.executable_case_ids) ? r.executable_case_ids.length > 0 : !!r.executable_case_ids;
      if (d === null || d === undefined) { hasReceipt ? bound++ : free++; continue; }
      if (/^(BLOCKED|PARTIAL)/.test(d)) unsigned[d] = (unsigned[d] || 0) + 1;
    }
    const nUnsigned = Object.values(unsigned).reduce((a, b) => a + b, 0);
    console.log(`C1 required=${req.length} bound=${bound} free=${free} unsigned_or_blocked=${nUnsigned} ${JSON.stringify(unsigned)}`);
    met = req.length > 0 && nUnsigned === 0 && free === 0;
    break;
  }
  case 'C2': met = report('C2 paired first/second-order (A-rows, not RSZ)', scoreboardRows(['A']), (r) => !/RSZ/.test(r.name)); break;
  case 'C3': met = report('C3 realistic-size', scoreboardRows(['A']), (r) => /RSZ/.test(r.name)); break;
  case 'C4': met = report('C4 real-data workflows', scoreboardRows(['C'])); break;
  case 'C5': met = report('C5 grouping levels', scoreboardRows(['B'])); break;
  case 'C6': met = report('C6 reverse-gap dispositions', scoreboardRows(['D']), (r) => r.id === 'D8'); break;
  case 'C7': {
    const p = show(PARITY_PAGE); if (p === null) die(`${PARITY_PAGE} missing`);
    const hit = /^#+ .*(does not mean|is not|not (a )?parity claim)/im.test(p);
    console.log(`C7 parity_page_has_not_mean_section=${hit}`);
    met = hit; break;
  }
  case 'EXTRACT': met = report('D1-D7 bridge/extractor/fit-input', scoreboardRows(['D']), (r) => r.id !== 'D8'); break;
  case 'ALL32': met = report('ALL gate-tier rows', scoreboardRows(null)); break;
  default: die(`unknown mode ${mode}`);
}
console.log(met ? `${mode}_MET` : `${mode}_NOT_MET`);

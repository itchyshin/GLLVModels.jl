#!/usr/bin/env node
// Internal first-seven checkpoint. Measurements use the approved baseline checker,
// so a candidate cannot improve its result by weakening checker policy.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
export const BASELINE = 'f220379d0937d0afffc6a030023c63c0f715e168';
export const P1 = '9539352f66f2db2cc26b1c393e67212a359b60c9';
export const REQUIRED = Object.freeze([
  'family/FAMILY-11-LOG', 'postfit/POSTFIT-SURFACE-check_auto_residual',
  'isdm/ISDM-COUNT', 'isdm/ISDM-EXTRA-SOURCE', 'isdm/ISDM-MISSING-IN-TRAIT',
  'isdm/ISDM-MISSING-SOURCE', 'isdm/ISDM-WRAPPER-LAW',
]);
const LEDGER = 'docs/dev-log/core070/true-parity-latest/';
const MODES = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','X2'];
const fail = message => { throw new Error(message); };
const git = (...args) => execFileSync('git', args, { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 });
const blob = (ref, path) => {
  if (git('cat-file', '-t', `${ref}:${path}`).trim() !== 'blob') fail(`not a blob: ${path}`);
  return git('show', `${ref}:${path}`);
};
const digest = text => createHash('sha256').update(text).digest('hex');
export function parseBoard(text) {
  const rows = new Map();
  for (const line of text.split('\n')) {
    const c = line.split('|').map(x => x.trim());
    const sid = c[1]?.match(/`([^`]+)`/)?.[1];
    if (!sid?.includes('/') || c.length < 6) continue;
    if (rows.has(sid)) fail(`duplicate scoreboard source: ${sid}`);
    rows.set(sid, { id: c[1].split(' ')[0], requires: c[2], status: c[3] });
  }
  return rows;
}
export function compareCheckpoint(base, candidate) {
  if (base.size !== 317 || candidate.size !== 317) fail('expected identical 317-row admission set');
  const expected = new Set(REQUIRED);
  for (const [sid, before] of base) {
    const after = candidate.get(sid);
    if (!after) fail(`missing baseline row: ${sid}`);
    if (before.id !== after.id || before.requires !== after.requires) fail(`admission or cases changed: ${sid}`);
    if (expected.has(sid)) {
      if (['EVIDENCED','EVIDENCED-BEHAVIOURAL','DISPOSITION-SIGNED'].includes(before.status)) fail(`baseline already closed: ${sid}`);
      const status = sid.startsWith('family/') ? 'EVIDENCED' : 'EVIDENCED-BEHAVIOURAL';
      if (after.status !== status) fail(`required exact row not closed as ${status}: ${sid}`);
    } else if (after.status !== before.status) fail(`unrelated status changed: ${sid}`);
  }
  for (const sid of expected) if (!base.has(sid)) fail(`checkpoint source absent: ${sid}`);
}
export function provenanceProblem(rec, read) {
  if (rec.pin !== 'P1' || rec.reference_commit !== P1) return 'missing P1 receipt provenance';
  // Legacy case receipts preserve the original run fields. New paired public-door
  // evidence owns its independent byte hashes in behaviour.read_from. Validate
  // every populated hash block without relabelling the historical batch.
  const blocks = [rec.read_from, rec.behaviour?.read_from].filter(x => x && typeof x === 'object');
  if (!blocks.some(x => Object.keys(x).length)) return 'missing raw hashes';
  for (const [path, sha] of blocks.flatMap(x => Object.entries(x))) {
    if (!/^[0-9a-f]{64}$/.test(sha) || digest(read(path)) !== sha) return `raw hash mismatch: ${path}`;
  }
  for (const key of ['oracle_build_receipt', 'oracle_source_receipt']) {
    if (typeof rec[key] !== 'string') return `missing ${key}`;
    const oracle = JSON.parse(read(rec[key]));
    if (oracle.reference_commit !== P1) return `${key} is not P1`;
  }
  if (rec.evidence_kind === 'r_public_bridge_boundary' &&
      (rec.verdict !== 'R_BOUNDARY_UNCHANGED' || rec.comparison !== undefined)) return 'invalid bridge boundary context';
  if (rec.verdict === 'NOT_EXECUTED') return 'NOT_EXECUTED is not evidence';
  return null;
}
function cleanEnv(ref) {
  const env = { ...process.env };
  for (const key of Object.keys(env)) if (key.startsWith('PARITY_')) delete env[key];
  return { ...env, PARITY_REF: ref, PARITY_PIN: 'P1' };
}
function canonical(source, ref, mode, suffix = '') {
  return execFileSync(process.execPath, ['--input-type=module', '-', mode], {
    input: source + suffix, encoding: 'utf8', env: cleanEnv(ref), maxBuffer: 8 * 1024 * 1024,
  });
}
export function rowMaps(ref) {
  const paths = git('ls-tree','-r','--name-only',ref,'--',LEDGER).trim().split('\n')
    .filter(p => /^case-map-(?!assembled\.json$)[^/]+\.json$/.test(p.slice(LEDGER.length)));
  const rows = [];
  for (const path of paths) rows.push(...JSON.parse(blob(ref,path)).rows);
  return { paths, rows };
}
export function runCheckpoint(ref = 'HEAD', baseline = BASELINE) {
  if (baseline !== BASELINE) fail('baseline is the approved immutable checkpoint, not configurable');
  if (ref.startsWith('-') || !/^[A-Za-z0-9_/.~^-]+$/.test(ref)) fail('invalid git revision');
  const target = git('rev-parse','--verify',`${ref}^{commit}`).trim();
  const source = blob(BASELINE,'tools/true_parity_check.mjs');
  const candidateBoard = parseBoard(blob(target,LEDGER+'scoreboard.md'));
  compareCheckpoint(parseBoard(blob(BASELINE,LEDGER+'scoreboard.md')), candidateBoard);
  // Admission/classification signatures and case IDs are checked independently of
  // markdown, since a substituted or newly excluded row must not count as progress.
  const base = rowMaps(BASELINE), next = rowMaps(target);
  const old = new Map(base.rows.map(r => [r.source_id,r]));
  if (old.size !== next.rows.length) fail('case-map admission set changed or duplicated');
  for (const r of next.rows) {
    const b = old.get(r.source_id);
    if (!b) fail(`new case-map source: ${r.source_id}`);
    for (const key of ['classification','signed_by','signed_on','disposition','executable_case_ids']) {
      if (JSON.stringify(b[key]) !== JSON.stringify(r[key])) fail(`unsigned admission or contract change: ${r.source_id}.${key}`);
    }
    if (!REQUIRED.includes(r.source_id)) continue;
    if (!['P1',P1].includes(r.measured_against)) fail(`row lacks P1 provenance: ${r.source_id}`);
    const evidence = [...(Array.isArray(r.evidence?.receipt) ? r.evidence.receipt : [r.evidence?.receipt]), ...(r.evidence?.boundary_context_receipts || [])].filter(Boolean);
    if (!evidence.length) fail(`missing receipts: ${r.source_id}`);
    for (const path of evidence) {
      const why = provenanceProblem(JSON.parse(blob(target,path)), p => blob(target,p));
      if (why) fail(`${r.source_id}: ${why}`);
    }
  }
  for (const mode of MODES) {
    const out = canonical(source,target,mode);
    process.stdout.write(out);
    if (['C0','C1','C5','C7','C8'].includes(mode) && !out.includes(`${mode}_MET\n`)) fail(`baseline gate regressed: ${mode}`);
    const counts = { C2: [297,290], C3: [8,6], C4: [8,4], X2: [317,304] }[mode];
    if (counts && !out.includes(`rows=${counts[0]} done=${counts[1]} `)) fail(`unexpected canonical ${mode} counts`);
    if (mode === 'C6' && out !== canonical(source,BASELINE,mode)) fail('C6 changed before signature');
  }
  // Reuse the canonical receipt predicates on the exact family-map rows too;
  // ordinary C1 intentionally measures only the 24 primary-map required rows.
  const receiptBoundIds = [...candidateBoard].filter(([,r])=>['EVIDENCED','EVIDENCED-BEHAVIOURAL'].includes(r.status)).map(([sid])=>sid);
  const suffix = `\nconst cpRows=${JSON.stringify(next.rows)}; const cpCites=caseCitations(cpRows);
  for (const row of cpRows.filter(r=>${JSON.stringify(receiptBoundIds)}.includes(r.source_id))) {
    const result = row.evidence_tier==='behavioural' ? behaviouralReceiptStatus(row,cpCites) : numericReceiptStatus(row);
    if (!result.ok) die('checkpoint receipt does not bind: '+row.source_id+' '+JSON.stringify(result));
  }
  console.log('CHECKPOINT_EXACT_RECEIPTS_VALID');\n`;
  const leaf = canonical(source,target,'C1',suffix);
  if (!leaf.includes('CHECKPOINT_EXACT_RECEIPTS_VALID')) fail('exact receipt validation did not run');
  console.log(`P1_CHECKPOINT_MET ref=${target} seven=${REQUIRED.length} X2=304/317 C2=290/297`);
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    let ref = 'HEAD', baseline = BASELINE;
    const args = process.argv.slice(2);
    while (args.length) {
      const flag = args.shift(), value = args.shift();
      if (!value || !['--ref','--baseline'].includes(flag)) fail('usage: --ref REF --baseline APPROVED_SHA');
      if (flag === '--ref') ref = value; else baseline = value;
    }
    runCheckpoint(ref,baseline);
  } catch (e) { console.error(`P1_CHECKPOINT_NOT_MET: ${e.message}`); process.exitCode = 1; }
}

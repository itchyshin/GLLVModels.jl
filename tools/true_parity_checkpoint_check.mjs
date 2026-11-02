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
export const APPROVED_C6_EXTRAS = Object.freeze([
  '@formula', 'bridge_capabilities', 'excess_kurtosis', 'gaussian_marginal_loglik_sparse_phy',
  'lognormal_marginal_loglik', 'observed_mask', 'ppca_init', 'qq_max_dev', 'rank_sum_z',
  'spearman', 'truncated_nbinom2_marginal_loglik_laplace',
  'truncated_nbinom2_pertrait_marginal_loglik_laplace', 'welch_t', 'GllvmModel', 'StatsAPI',
  'random_balanced_tree', 'shrinkage_factor',
]);
export const APPROVED_C6_SCOPED = Object.freeze([
  'bridge_fit', 'fit_animal_dep_gllvm', 'fit_dep_gllvm', 'fit_gllvm',
  'fit_kernel_dep_gllvm', 'fit_kernel_indep_gllvm', 'StudentT', 'StudentTFamily', 'TruncatedNegBin2',
  'BetaBinom', 'NB1',
]);
// SHA-256 of each canonical counterpart/receipt/scope/ruling projection in reverse-gap.json.
// Keep these pinned beside the approved names so the checkpoint rejects a widened
// scope, changed counterpart, or substituted receipt even when decision/status agree.
export const APPROVED_C6_SCOPED_SHA256 = Object.freeze({
  "BetaBinom": "9cd3b96b383762cbb5b55d7d4614beedb0e4fb8b735eab9aa59d26b9710264c5",
  "NB1": "ba276a261732c8125bcd33765ff1db054139fcab46b4ac98d2e3b5e36d7cb7fc",
  "StudentT": "b6321a8f647f1ee2a646c6951bb85011b1b01f04b391e55bf19b13136d544279",
  "StudentTFamily": "e571e8fa54d7a51b2d9504cdd2cf8d96cbdb9ab65895285d82820908852b5e46",
  "TruncatedNegBin2": "174a5994c47565f760dc413a5a76a660f6de077db824011cb7b918679bea31c3",
  "bridge_fit": "5b5c1f8e067571073f71038d0263fbb63201e19d0fe854a2ff39a77903fb3222",
  "fit_animal_dep_gllvm": "fb8bd42aeaa88c3d3ea79ee37c8f542fe52837967513cb4b6a6efa3e2186788e",
  "fit_dep_gllvm": "c0baa3610de08e061b21bb117e93a2faccc6c51b03fccd2debcb57b7129e0125",
  "fit_gllvm": "a666827b39c386b01a5ef76d48ea5d3d4f354fb22f71d0d5baaaed8716193488",
  "fit_kernel_dep_gllvm": "33475a40838b1a5fc85d26dada32b69f80bc65c0468cd1f3a07ae227e349c756",
  "fit_kernel_indep_gllvm": "13dcb057d0794cb3b806f1825025db2817708ec402d8b13a68c5b2338c5427b8"
});
// Three exports introduced on main by #939. These use the existing documented-extra
// rule (#684 item 3), separately from the seventeen individually signed extras.
export const C6_CENSUS_COMMIT = 'c62b877f06b22e30f2c16df38becca34d746f5c7';
export const MAIN_C6_EXPORT_ADDITIONS = Object.freeze(['OptimizerRecord', 'OptimizerStage', 'optimizer_info']);
export const HELD_C6_NAMES = Object.freeze([
  'NBFit', 'Ordinal', 'TweedieFit', 'ZIB', 'ZINegBin',
  'fit_animal_latent_gllvm', 'fit_kernel_latent_gllvm', 'gllvm', 'link_residual',
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
const canonicalizeDecision = value => Array.isArray(value) ? value.map(canonicalizeDecision)
  : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(k => [k, canonicalizeDecision(value[k])]))
  : value;
const canonicalDigest = value => digest(JSON.stringify(canonicalizeDecision(value)));
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
export function compareC6Decisions(baseItems, candidateItems) {
  if (!Array.isArray(baseItems) || !Array.isArray(candidateItems)) fail('C6 reverse-gap list must be arrays');
  const index = items => {
    const out = new Map();
    for (const item of items) {
      if (!item || typeof item.source_id !== 'string' || out.has(item.source_id)) fail('C6 reverse-gap admission set changed or duplicated');
      out.set(item.source_id, item);
    }
    return out;
  };
  const before = index(baseItems), after = index(candidateItems);
  if (after.size !== before.size + MAIN_C6_EXPORT_ADDITIONS.length) fail('C6 reverse-gap admission set changed');
  const censusPrefix = `names(GLLVModels) at ${C6_CENSUS_COMMIT.slice(0,12)};`;
  for (const name of MAIN_C6_EXPORT_ADDITIONS) {
    const a = after.get(`julia-export/${name}`);
    const expected = {
      source_id: `julia-export/${name}`, name,
      kind: name === 'optimizer_info' ? 'Function' : 'Type',
      status: 'decided', decision: 'KEPT_AS_JULIA_EXTRA',
      derived_from: `${censusPrefix} no gllvmTMB export or S3 generic of the same name at P1 (compared with underscores and dots removed and lower-cased, as tools/parity_ledger.py norm()) and not named as the Julia side of any tracked namespace receipt`,
      basis: 'not classed internal by tools/parity_ledger.py --ref 9539352f6; has a docstring, rendered in an @docs block on docs/src/api.md',
      ruling: {ref: 'itchyshin/GLLVModels.jl#684 item 3', signed_by: 'Shinichi Nakagawa', signed_on: '2026-10-02'},
    };
    if (!a || before.has(expected.source_id) || canonicalDigest(a) !== canonicalDigest(expected)) fail(`invalid main-export census addition: ${name}`);
  }
  const addedIds = new Set(MAIN_C6_EXPORT_ADDITIONS.map(n => `julia-export/${n}`));
  for (const id of after.keys()) if (!before.has(id) && !addedIds.has(id)) fail(`unapproved C6 census addition: ${id}`);
  const extras = new Set(APPROVED_C6_EXTRAS), scoped = new Set(APPROVED_C6_SCOPED), held = new Set(HELD_C6_NAMES);
  for (const [id, b] of before) {
    const a = after.get(id);
    if (!a) fail(`C6 reverse-gap item missing: ${id}`);
    if (b.name !== a.name) fail(`C6 reverse-gap identity changed: ${id}`);
    const expectedDerived = b.derived_from.replace(/^names\(GLLVModels\) at [0-9a-f]{12};/, censusPrefix);
    if (a.derived_from !== expectedDerived) fail(`C6 census provenance changed: ${id}`);
    if (extras.has(b.name)) {
      if (b.decision !== null || a.decision !== 'KEPT_AS_JULIA_EXTRA' || a.status !== 'decided') fail(`approved C6 extra not applied exactly: ${b.name}`);
    } else if (scoped.has(b.name)) {
      if (b.decision !== null || a.decision !== 'SCOPED_R_TWIN' || a.status !== 'decided') fail(`approved C6 scope not applied exactly: ${b.name}`);
      const scopedContract = { counterpart: a.counterpart, receipt_ids: a.receipt_ids, scope: a.scope, ruling: a.ruling };
      if (canonicalDigest(scopedContract) !== APPROVED_C6_SCOPED_SHA256[b.name]) fail(`approved C6 scope contents changed: ${b.name}`);
    } else if (held.has(b.name)) {
      if (b.decision !== null || a.decision !== 'HOLD_PENDING_EVIDENCE' || a.status !== 'held'
          || a.ruling?.ref !== 'docs/dev-log/decisions/2026-10-17-p1-c6-and-c3-c4-contract-approval.md'
          || a.ruling?.signed_by !== 'Shinichi Nakagawa' || a.ruling?.signed_on !== '2026-10-17'
          || !a.basis?.includes('no twin or parity claim') || !a.basis?.includes('no scoreboard row')
          || 'counterpart' in a || 'scope' in a || 'receipt_ids' in a) {
        fail(`approved C6 hold not applied exactly: ${b.name}`);
      }
    } else if (JSON.stringify({...b, derived_from: expectedDerived}) !== JSON.stringify(a)) {
      fail(`unapproved C6 decision changed: ${b.name}`);
    }
  }
  const names = new Set([...before.values()].map(x => x.name));
  for (const name of [...extras, ...scoped, ...held]) if (!names.has(name)) fail(`expected C6 name missing: ${name}`);
  return { extras: extras.size, scoped: scoped.size, held: held.size, export_additions: addedIds.size };
}
export function provenanceProblem(rec, read, sourceId) {
  if (rec.pin !== 'P1' || rec.reference_commit !== P1) return 'missing P1 receipt provenance';
  const groups = [];
  const hashes = value => value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length;
  // Flat measurement hashes remain mandatory when supplied. Paired public-call
  // receipts may instead carry their hashes in the separately derived block.
  if (rec.read_from !== undefined) {
    if (!hashes(rec.read_from)) return 'missing raw hashes';
    groups.push(rec.read_from);
  }
  if (rec.behaviour !== undefined) {
    const behaviour = rec.behaviour;
    if (!behaviour || behaviour.pin !== 'P1') return 'missing behaviour P1 provenance';
    if (behaviour.ruling !== 'itchyshin/GLLVModels.jl#684 item 2') return 'invalid behaviour ruling';
    if (!Array.isArray(behaviour.cases) || !behaviour.cases.length) return 'missing behaviour cases';
    for (const entry of behaviour.cases) {
      if (!entry || !rec.case_id || entry.case_id !== rec.case_id) return 'behaviour case identity mismatch';
      if (!REQUIRED.slice(1).includes(entry.source_id) || (sourceId && entry.source_id !== sourceId)) return 'behaviour source identity mismatch';
    }
    if (!hashes(behaviour.read_from)) return 'missing behaviour raw hashes';
    groups.push(behaviour.read_from);
  }
  if (!groups.length) return 'missing raw hashes';
  for (const group of groups) for (const [path, sha] of Object.entries(group)) {
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
      const why = provenanceProblem(JSON.parse(blob(target,path)), p => blob(target,p), r.source_id);
      if (why) fail(`${r.source_id}: ${why}`);
    }
  }
  const c6Delta = compareC6Decisions(
    JSON.parse(blob(BASELINE, LEDGER+'reverse-gap.json')),
    JSON.parse(blob(target, LEDGER+'reverse-gap.json')),
  );
  for (const mode of MODES) {
    const out = canonical(source,target,mode);
    process.stdout.write(out);
    if (['C0','C1','C5','C7','C8'].includes(mode) && !out.includes(`${mode}_MET\n`)) fail(`baseline gate regressed: ${mode}`);
    const counts = { C2: [297,290], C3: [8,6], C4: [8,4], X2: [317,304] }[mode];
    if (counts && !out.includes(`rows=${counts[0]} done=${counts[1]} `)) fail(`unexpected canonical ${mode} counts`);
    if (mode === 'C6' && !out.includes('C6_NOT_MET\n')) fail('frozen baseline checker unexpectedly accepts the new C6 vocabulary');
  }
  // Frozen baseline measurements preserve the comparison contract, but cannot
  // authenticate the new vocabulary. Also validate decisions with the checker
  // committed at the target, including documented-extra bases and rulings.
  const currentC6 = canonical(blob(target, 'tools/true_parity_check.mjs'), target, 'C6');
  process.stdout.write(currentC6);
  if (!currentC6.includes('C6_MET\n')) fail('target C6 decision validation is not MET');
  process.stdout.write(`C6_APPROVED_DELTA_OK extras=${c6Delta.extras} scoped=${c6Delta.scoped} held=${c6Delta.held}\n`);
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

// Read exact regular-file bytes from a selected Git ref or a filesystem fixture.
// Returns null for unsafe paths, symlinks, missing files, and invalid revisions.
import { execFileSync } from 'node:child_process';
import { lstatSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

export const safeGitEnv = () => ({...Object.fromEntries(Object.entries(process.env).filter(([k]) => !k.startsWith('GIT_'))), GIT_NO_REPLACE_OBJECTS:'1'});

export function resolveGitRef(root, ref) {
  if (typeof root !== 'string' || typeof ref !== 'string' ||
      !/^[A-Za-z0-9_./~^-]+$/.test(ref) || ref.startsWith('-') || ref.includes('..')) return null;
  try {
    const commit = execFileSync('git',['rev-parse','--verify','--end-of-options',`${ref}^{commit}`],
      {cwd:resolve(root), stdio:['ignore','pipe','pipe'], maxBuffer:64 << 20, env:safeGitEnv()}).toString('utf8').trim();
    return /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(commit) ? commit : null;
  } catch { return null; }
}

export function readRefBytes({root, ref, path}) {
  if (typeof root !== 'string' || typeof ref !== 'string' || typeof path !== 'string' ||
      !/^[A-Za-z0-9_./-]+$/.test(path) || path.startsWith('/') ||
      path.split('/').some((part) => !part || part === '.' || part === '..')) return null;
  try {
    const base = resolve(root);
    if (ref === 'FS') {
      let current = base;
      for (const part of path.split('/')) {
        current = join(current, part);
        if (lstatSync(current).isSymbolicLink()) return null;
      }
      if (!lstatSync(current).isFile()) return null;
      return readFileSync(current);
    }
    const options = {cwd:base, stdio:['ignore','pipe','pipe'], maxBuffer:64 << 20, env:safeGitEnv()};
    const commit = resolveGitRef(base,ref);
    if (!commit) return null;
    const record = execFileSync('git',['ls-tree','-z',commit,'--',path],options).toString('utf8');
    if (!/^100(644|755) blob [0-9a-f]+\t/.test(record) || !record.endsWith('\0') ||
        record.slice(record.indexOf('\t')+1,-1) !== path) return null;
    return execFileSync('git',['show',`${commit}:${path}`],options);
  } catch { return null; }
}

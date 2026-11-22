#!/usr/bin/env node
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { readRefBytes, resolveGitRef } from './core070_ref_bytes.mjs';

const root = mkdtempSync(join(tmpdir(), 'p1-ref-bytes-'));
try {
  const git = (...args) => execFileSync('git', args, { cwd: root, stdio: 'pipe' });
  git('init', '-q'); mkdirSync(join(root,'artifacts'));
  const bytes = Buffer.from([0, 255, 128, 10, 65]);
  writeFileSync(join(root,'artifacts','binary'), bytes);
  symlinkSync('binary', join(root,'artifacts','link'));
  git('add', 'artifacts/binary', 'artifacts/link');
  git('-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture');
  writeFileSync(join(root,'artifacts','binary'), Buffer.from('changed'));
  assert.deepEqual(readRefBytes({root,ref:'HEAD',path:'artifacts/binary'}), bytes);
  assert.deepEqual(readRefBytes({root,ref:'FS',path:'artifacts/binary'}), Buffer.from('changed'));
  assert.equal(readRefBytes({root,ref:'HEAD',path:'artifacts/link'}), null);
  assert.equal(readRefBytes({root,ref:'FS',path:'artifacts/link'}), null);
  assert.equal(readRefBytes({root,ref:'HEAD',path:'../outside'}), null);
  assert.equal(readRefBytes({root,ref:'FS',path:'../outside'}), null);
  assert.equal(readRefBytes({root,ref:'HEAD',path:'artifacts/missing'}), null);
  assert.equal(readRefBytes({root,ref:'missing-ref',path:'artifacts/binary'}), null);
  assert.equal(readRefBytes({root,ref:'HEAD',path:'artifacts'}), null);
  assert.equal(readRefBytes({root,ref:'FS',path:'artifacts'}), null);
  symlinkSync('artifacts', join(root,'alias'));
  assert.equal(readRefBytes({root,ref:'FS',path:'alias/binary'}), null);
  const pinned = resolveGitRef(root,'HEAD');
  git('add', 'artifacts/binary');
  git('-c','user.name=fixture','-c','user.email=fixture@example.invalid','commit','-qm','second fixture');
  assert.deepEqual(readRefBytes({root,ref:'HEAD~1',path:'artifacts/binary'}), bytes);
  assert.deepEqual(readRefBytes({root,ref:'HEAD',path:'artifacts/binary'}), Buffer.from('changed'));
  assert.deepEqual(readRefBytes({root,ref:pinned,path:'artifacts/binary'}), bytes);
  assert.deepEqual(readFileSync(join(root,'artifacts','binary')), Buffer.from('changed'));
  git('replace', pinned, 'HEAD');
  assert.deepEqual(readRefBytes({root,ref:pinned,path:'artifacts/binary'}), bytes);
  console.log('CORE070_REF_BYTES_PASS');
} finally { rmSync(root,{recursive:true,force:true}); }

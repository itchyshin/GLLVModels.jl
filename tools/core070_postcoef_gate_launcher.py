#!/usr/bin/env python3
"""Run a committed POST-COEF gate from a fresh source-only checkout.

The launcher uses a separately approved gate commit; evidence-repository
arguments cannot select that source. The launcher identity also needs a
separate trusted binding before this prototype can be used for acceptance.
"""
import hashlib
import os
import re
from pathlib import Path
import subprocess
import sys
import tempfile

# This must be set to a separately reviewed gate-code commit before use.
TRUSTED_GATE_COMMIT = 'b53f8221a6fa78a3efaf57667b51b5acb32aff0d'

ENTRIES = {'live': 'tools/core070_postcoef_live_ref.py',
           'identity': 'tools/core070_postcoef_live_ref.py',
           'ancestry': 'tools/core070_postcoef_ref_compatible.py'}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ENTRIES:
        raise ValueError('unknown gate entry')
    args = sys.argv[2:]
    if sys.argv[1] == 'ancestry':
        if len(args) != 3 or not Path(args[0]).is_absolute():
            raise ValueError('ancestry requires an absolute repository path')
    else:
        if len(args) != 6 or args[0] != '--repo' or args[2] != '--head' or args[4] != '--runtime-root' or not Path(args[1]).is_absolute() or not Path(args[5]).is_absolute():
            raise ValueError('live gate requires absolute repository and runtime paths')
    gate_fd = globals().get('__gate_root_fd__')
    if not isinstance(gate_fd, int) or gate_fd < 3:
        raise ValueError('trusted source directory handle missing')
    os.fchdir(gate_fd)
    root = Path('.')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GIT_', 'PYTHON'))}
    env.update(GIT_NO_REPLACE_OBJECTS='1', GIT_CONFIG_NOSYSTEM='1',
               GIT_CONFIG_GLOBAL=os.devnull, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], env=env, capture_output=True,
                              check=True, timeout=30).stdout
    head = TRUSTED_GATE_COMMIT
    if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', head):
        raise ValueError('trusted gate commit is not configured')
    if git('cat-file', '-t', head).strip() != b'commit':
        raise ValueError('trusted gate source is not a commit')
    for rel in (ENTRIES[sys.argv[1]],):
        path = root / rel
        if path.is_symlink() or not path.is_file() or path.read_bytes() != git('show', f'{head}:{rel}'):
            raise ValueError('gate entry differs from committed source')
    entries = git('ls-tree', '-r', '-z', head, '--', 'tools').split(b'\0')
    for entry in filter(None, entries):
        meta, path = entry.split(b'\t', 1)
        mode, kind, object_id = meta.split(b' ')
        if mode == b'120000' or path.endswith(b'.pyc') or b'__pycache__' in path.split(b'/'):
            raise ValueError('gate source contains symlink or bytecode')
    with tempfile.TemporaryDirectory(prefix='postcoef-outer-gate-') as tmp:
        template = Path(tmp) / 'empty-template'; template.mkdir()
        checkout = Path(tmp) / 'checkout'
        subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', 'clone', '--no-local', '--no-hardlinks', '--no-checkout',
                        '--template', str(template), '--', str(root), str(checkout)],
                       check=True, capture_output=True, env=env, timeout=60)
        subprocess.run(['git', '-C', str(checkout), '-c', 'core.hooksPath=/dev/null',
                        'checkout', '--quiet', '--detach', head], check=True, capture_output=True,
                       env=env, timeout=60)
        if os.path.lexists(checkout / '.git/objects/info/alternates'):
            raise ValueError('private gate checkout uses object alternates')
        for item in filter(None, entries):
            meta, raw_path = item.split(b'\t', 1)
            mode, kind, object_id = meta.split(b' ')
            path = checkout / os.fsdecode(raw_path)
            if kind != b'blob' or path.is_symlink() or not path.is_file():
                raise ValueError('private gate input is not a regular blob')
            data = path.read_bytes()
            algorithm = hashlib.sha1 if len(object_id) == 40 else hashlib.sha256
            digest = algorithm(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest().encode()
            if digest != object_id:
                raise ValueError('private gate input differs from committed blob')
        entry = checkout / ENTRIES[sys.argv[1]]
        if entry.is_symlink() or entry.read_bytes() != git('show', f'{head}:{ENTRIES[sys.argv[1]]}'):
            raise ValueError('private gate entry differs from committed source')
        entry_args = [*sys.argv[2:], *(['--identity-only'] if sys.argv[1] == 'identity' else [])]
        proc = subprocess.run([sys.executable, '-B', '-I', str(entry), *entry_args],
                              cwd=checkout, env=env, capture_output=True, timeout=240)
        sys.stdout.buffer.write(proc.stdout); sys.stderr.buffer.write(proc.stderr)
        return proc.returncode


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print('POSTCOEF_GATE_REJECTED: ' + str(exc), file=sys.stderr)
        sys.exit(1)

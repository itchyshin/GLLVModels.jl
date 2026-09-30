#!/usr/bin/env python3
"""Create an exact Core070 R archive and build it into an isolated R library.

Preparation is local/read-only against Git. Building runs only when explicitly
requested, never installs dependencies, and refuses to reuse a build directory.
A source or build receipt is provenance, not evidence of statistical parity.

Pin selection (D-294/D-295): the git commit this tool builds is named by
tools/parity_oracle.py's GLLVM_PARITY_PIN switch (default "P0"; set to "P1"
to target the re-pin target instead) -- that module is the single place that
reads the environment variable and validates it. The three companion byte
hashes for whichever pin is selected (NAMESPACE / source tree / archive) live
in tools/core070_oracle_pins.toml, keyed by the same pin name, so this file
no longer hardcodes its own copy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tarfile
import tempfile
import time
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parity_oracle import R_REF_PINS, SELECTED_PIN  # noqa: E402

MARKER = 'CORE070_SOURCE_PIN.toml'

_PINS_FILE = Path(__file__).resolve().parent / 'core070_oracle_pins.toml'


def _load_pin(pin_name, pins_file=_PINS_FILE):
    if pin_name not in R_REF_PINS:
        raise ValueError(f'unknown pin {pin_name!r}; parity_oracle.R_REF_PINS has {sorted(R_REF_PINS)}')
    table = tomllib.loads(pins_file.read_text())
    if pin_name not in table:
        raise ValueError(f'pin {pin_name!r} has no hash entry in {pins_file}')
    entry = table[pin_name]
    if entry['reference_commit'] != R_REF_PINS[pin_name]:
        raise ValueError(
            f'{pins_file} pin {pin_name!r} reference_commit {entry["reference_commit"]!r} '
            f'disagrees with tools/parity_oracle.py R_REF_PINS[{pin_name!r}] = {R_REF_PINS[pin_name]!r}'
        )
    return entry


_PIN = _load_pin(SELECTED_PIN)
REFERENCE = _PIN['reference_commit']
NAMESPACE = _PIN['namespace_sha256']
SOURCE_TREE = _PIN['source_tree_sha256']
ARCHIVE = _PIN['archive_sha256']


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def files(root, excluded=()):
    root = Path(root)
    entries = {}
    for folder, directories, names in os.walk(root, followlinks=False):
        for name in directories + names:
            path = Path(folder) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError('symlink not admitted in oracle tree: ' + rel)
            if path.is_file() and rel not in excluded:
                entries[rel] = sha(path)
    return entries


def tree_hash(entries):
    return hashlib.sha256('\n'.join(sorted(k + '\0' + v for k, v in entries.items())).encode()).hexdigest()


def extract(archive, destination, expected_sha):
    if sha(archive) != expected_sha:
        raise ValueError('archive checksum mismatch')
    with tarfile.open(archive, 'r:') as source:
        if source.pax_headers.get('comment') != REFERENCE:
            raise ValueError('archive is not the frozen Git commit')
        for entry in source.getmembers():
            parts = Path(entry.name).parts
            if Path(entry.name).is_absolute() or '..' in parts or not (entry.isfile() or entry.isdir()):
                raise ValueError('unsafe or unsupported archive member: ' + entry.name)
        # Members are validated above; do not accept symlinks, devices or paths
        # outside this newly created private destination on older Python.
        try:
            source.extractall(destination, filter='data')
        except TypeError:  # Python before extraction filters; validated above.
            source.extractall(destination)
    if sha(Path(destination) / 'NAMESPACE') != NAMESPACE:
        raise ValueError('frozen NAMESPACE checksum mismatch')


def prepare(repo, destination):
    if destination.exists():
        raise ValueError('prepare destination must not exist')
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', REFERENCE + '^{commit}'], text=True).strip()
    if actual != REFERENCE:
        raise ValueError('reference commit mismatch')
    destination.mkdir(parents=True)
    archive = destination / 'gllvmTMB-core070.tar'
    with archive.open('xb') as stream:
        subprocess.run(['git', '-C', str(repo), 'archive', '--format=tar', REFERENCE], stdout=stream, check=True,
                       env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'), timeout=120)
    with tempfile.TemporaryDirectory(dir=destination) as tmp:
        extract(archive, tmp, sha(archive))
        manifest = files(tmp)
    receipt = {'reference_commit': REFERENCE, 'archive_sha256': sha(archive),
               'source_tree_sha256': tree_hash(manifest), 'namespace_sha256': NAMESPACE,
               'source_files': manifest}
    (destination / 'source.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_files'}, sort_keys=True))
    print('CORE070_ORACLE_SOURCE_PASS')


def build(archive, source_receipt, destination, r_binary, timeout):
    pin = json.loads(source_receipt.read_text())
    if pin.get('reference_commit') != REFERENCE or pin.get('namespace_sha256') != NAMESPACE:
        raise ValueError('source receipt does not match frozen reference')
    if pin.get('source_tree_sha256') != SOURCE_TREE or pin.get('archive_sha256') != ARCHIVE:
        raise ValueError('source bytes do not match independently frozen hashes')
    if tree_hash(pin['source_files']) != pin['source_tree_sha256']:
        raise ValueError('source-file manifest checksum mismatch')
    if destination.exists():
        raise ValueError('build destination must not exist; preserve the previous attempt')
    destination.mkdir(parents=True)
    source = destination / 'source'; source.mkdir()
    library = destination / 'library'; library.mkdir()
    extract(archive, source, pin['archive_sha256'])
    if files(source) != pin['source_files']:
        raise ValueError('extracted source differs from frozen inventory')
    command = [str(r_binary), 'CMD', 'INSTALL', '--no-multiarch', '--library=' + str(library), str(source)]
    log = destination / 'install.log'
    start = time.monotonic()
    status = None
    try:
        with log.open('wb') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                       env=dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MAKEFLAGS='-j1'))
            try:
                status = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                status = 124
    except OSError:
        status = 127
    original_unchanged = all((source / rel).is_file() and sha(source / rel) == digest for rel, digest in pin['source_files'].items())
    result = {'reference_commit': REFERENCE, 'archive_sha256': pin['archive_sha256'],
              'source_tree_sha256': pin['source_tree_sha256'], 'command': command,
              'exit_code': status, 'elapsed_seconds': time.monotonic() - start,
              'tool_sha256': sha(__file__), 'hostname': os.uname().nodename,
              'log_sha256': sha(log), 'original_source_unchanged': original_unchanged,
              'r_version': subprocess.check_output([str(r_binary), '--version'], text=True).splitlines()[0]}
    installed = library / 'gllvmTMB'
    if status == 0 and original_unchanged and installed.is_dir():
        if sha(installed / 'NAMESPACE') != NAMESPACE:
            raise ValueError('installed NAMESPACE mismatch')
        result['installed_tree_sha256'] = tree_hash(files(installed, excluded=(MARKER,)))
        marker = {key: result[key] for key in ('reference_commit', 'source_tree_sha256', 'installed_tree_sha256', 'archive_sha256')}
        marker['namespace_sha256'] = NAMESPACE
        marker['install_log_sha256'] = result['log_sha256']
        (installed / MARKER).write_text(''.join(key + ' = ' + json.dumps(value) + '\n' for key, value in marker.items()))
        result['marker_sha256'] = sha(installed / MARKER)
    (destination / 'build.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, sort_keys=True))
    if status != 0 or not original_unchanged or 'marker_sha256' not in result:
        raise ValueError('build failed or source drifted; retained attempt, no qualified marker')
    print('CORE070_ORACLE_BUILD_PASS')


def verify(destination):
    result = json.loads((destination / 'build.json').read_text())
    if result.get('exit_code') != 0 or not result.get('original_source_unchanged'):
        raise ValueError('unsuccessful build receipt')
    if result.get('reference_commit') != REFERENCE or result.get('source_tree_sha256') != SOURCE_TREE or result.get('archive_sha256') != ARCHIVE:
        raise ValueError('stale source provenance')
    installed = destination / 'library' / 'gllvmTMB'
    if sha(installed / MARKER) != result.get('marker_sha256'):
        raise ValueError('missing or changed installed marker')
    if tree_hash(files(installed, excluded=(MARKER,))) != result.get('installed_tree_sha256'):
        raise ValueError('installed tree changed since build')
    if sha(destination / 'install.log') != result.get('log_sha256'):
        raise ValueError('build log changed since receipt')
    print('CORE070_ORACLE_VERIFY_PASS')


def self_test():
    import io
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        archive = root / 'wrong.tar'
        with tarfile.open(archive, 'w', format=tarfile.PAX_FORMAT, pax_headers={'comment': '0' * 40}) as out:
            data = b'wrong'; member = tarfile.TarInfo('NAMESPACE'); member.size = len(data)
            out.addfile(member, io.BytesIO(data))
        for expected in ('0' * 64, sha(archive)):
            try:
                extract(archive, root / 'extract', expected)
            except ValueError:
                pass
            else:
                raise AssertionError('wrong archive was accepted')
        before = {'a': '1' * 64}
        assert tree_hash(before) != tree_hash({'a': '2' * 64})
        assert tree_hash(before) != tree_hash({'b': '1' * 64})
        link = root / 'link'; link.symlink_to('/not/inside')
        try:
            files(root)
        except ValueError:
            pass
        else:
            raise AssertionError('external symlink was accepted')
    print('CORE070_ORACLE_SELFTEST_PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('self-test')
    v = sub.add_parser('verify'); v.add_argument('--destination', type=Path, required=True)
    p = sub.add_parser('prepare'); p.add_argument('--repo', type=Path, required=True); p.add_argument('--destination', type=Path, required=True)
    b = sub.add_parser('build'); b.add_argument('--archive', type=Path, required=True); b.add_argument('--source-receipt', type=Path, required=True)
    b.add_argument('--destination', type=Path, required=True); b.add_argument('--r-binary', type=Path, default=Path('/usr/bin/R'))
    b.add_argument('--timeout', type=int, default=1200)
    args = parser.parse_args()
    if args.action == 'self-test': self_test()
    elif args.action == 'prepare': prepare(args.repo.resolve(), args.destination.resolve())
    elif args.action == 'verify': verify(args.destination.resolve())
    else:
        if not 1 <= args.timeout <= 1800: parser.error('bounded builds require timeout <=1800 seconds')
        build(args.archive.resolve(), args.source_receipt.resolve(), args.destination.resolve(), args.r_binary.resolve(), args.timeout)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Verify a selected-ref POST-COEF capture on its explicitly supplied live runtime.

This command validates evidence; it does not write or promote any parity row.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core070_json_semantics import json_value_equal
from core070_postcoef_fit_health import classify as classify_fit_health, historical_r_prior_bound, julia_prior_bound
from core070_postfit_p1_receipts import postcoef_git_env, postcoef_ref_compatible

PREFIX = 'docs/dev-log/core070/true-parity-latest/receipts/postfit/postcoef-empty-p1'
CID = 'CORE070-POSTFIT-COEF-EMPTY-NATIVE'  # captured harness identity
RECEIPT_CID = 'CORE070-POSTFIT-COEF-THREE-FIT-NATIVE'
CASE = f'docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/{RECEIPT_CID}.json'
HISTORICAL_CASE = f'docs/dev-log/core070/true-parity-latest/receipts/postfit/cases/{CID}.json'
# Exact non-discriminating receipt on the independently checked main baseline.
HISTORICAL_SHA256 = '72fdb2739730d2ca763a8b5d291442bd5847403b3e30ac49c82b0ae1a31bab80'
SHA = re.compile(r'^(?:[0-9a-f]{40}|[0-9a-f]{64})$')
HASH = re.compile(r'^[0-9a-f]{64}$')


def check_capture_pointers(case):
    if (not isinstance(case, dict) or
            case.get('raw') != [f'{PREFIX}/julia-results.json', f'{PREFIX}/r-oracle.json'] or
            not isinstance(case.get('batch_verifier'), dict) or
            case['batch_verifier'].get('log') != f'{PREFIX}/verify.txt' or
            not isinstance(case['batch_verifier'].get('log_sha256'), str) or
            not HASH.fullmatch(case['batch_verifier']['log_sha256'])):
        raise ValueError('capture pointers differ')


def check_verifier_transcript(case, data):
    check_capture_pointers(case)
    if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != case['batch_verifier']['log_sha256']:
        raise ValueError('verifier transcript hash differs')
    markers = (b'CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10',
               b'CORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4',
               b'CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5',
               b'CORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2')
    if not all(marker in data.splitlines() for marker in markers):
        raise ValueError('verifier transcript markers missing')
FILES = ['receipt.json', 'results.tsv', 'diagnostics.log', 'julia-results.json', 'r-oracle.json',
         'julia-Manifest.toml', 'julia-stdout.log', 'julia-stderr.log', 'oracle-verify-preload.log', 'oracle-verify-before.log',
         'oracle-verify-after.log', 'runner-snapshots/core070_postfit_policy_batch.R',
         'runner-snapshots/core070_postfit_policy_batch.jl', 'runner-snapshots/julia-launcher'] + [
    f'julia-control-{mode}.{suffix}' for mode in ('always_empty', 'spurious_coefficient')
    for suffix in ('json', 'exit', 'stdout.log', 'stderr.log')]


def stage_capture_files(state, files, read_selected):
    """Stage the selected immutable bytes required by the source verifier."""
    state = Path(state)
    for name in files:
        dest = state / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(read_selected(name))


def git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, timeout=30, env=postcoef_git_env()).stdout


def blob(repo, commit, rel):
    if not SHA.fullmatch(commit) or not re.fullmatch(r'[A-Za-z0-9_./-]+', rel) or any(x in ('', '.', '..') for x in rel.split('/')):
        raise ValueError('unsafe commit or path')
    entry = git(repo, 'ls-tree', '-z', commit, '--', rel)
    match = re.fullmatch(rb'100(?:644|755) blob [0-9a-f]+\t([^\0]+)\0', entry)
    if not match or match.group(1).decode('utf-8') != rel:
        raise ValueError('selected artifact is not a regular tracked file: ' + rel)
    return git(repo, 'show', f'{commit}:{rel}')


def check_comparison(case, report, oracle):
    actual = report['cases'][CID]
    health = actual.get('three_fit', {}).get('fit_health')
    if classify_fit_health(health) != 'FIT_HEALTH_PASS':
        raise ValueError('POST-COEF fit health failed or is incomplete')
    fits = oracle.get('postcoef_three_fit', {}).get('fits')
    if (not isinstance(fits, list) or len(fits) != 3 or
            not all(isinstance(fit, dict) and 'fit_health' in fit for fit in fits) or
            not json_value_equal(health['R'], [fit['fit_health'] for fit in fits])):
        raise ValueError('POST-COEF R fit health differs from retained oracle')
    comparisons = case.get('comparison', {}).get('cases', [])
    if (case.get('case_id') != RECEIPT_CID or case.get('harness_case_id') != CID or case.get('source_ids') != ['postfit-policy/POST-COEF-EMPTY'] or
        case.get('verdict') != 'PASS' or case.get('harness_pass') is not True or
        case.get('evidence_kind') != 'numeric_r_vs_julia' or case.get('glvmodels_worktree_dirty') != [] or
        case.get('pin') != 'P1' or case.get('reference_commit') != '9539352f66f2db2cc26b1c393e67212a359b60c9' or
        case.get('comparison', {}).get('pin') != 'P1' or not json_value_equal(case.get('harness_fields'), actual) or len(comparisons) != 1):
        raise ValueError('case identity or harness differs from capture')
    verifier = case.get('batch_verifier', {})
    if (verifier.get('status') != 'PASS' or type(verifier.get('exit_code')) is not int or
        verifier.get('exit_code') != 0 or verifier.get('tool') != 'tools/core070_verify_postfit_policy_batch.py' or
        verifier.get('accept_marker') != 'CORE070_POSTFIT_POLICY_BATCH_VERIFIED'):
        raise ValueError('case verifier record is invalid')
    cmp = comparisons[0]
    r = [fit['length'] for fit in oracle['postcoef_three_fit']['fits']]
    j = actual['three_fit']['julia_lengths']
    if (cmp.get('case_id') != CID or cmp.get('kind') != 'integer_equality' or
        type(cmp.get('tolerance')) not in (int, float) or cmp.get('tolerance') != 0.5 or
        cmp.get('r_value') != r or cmp.get('julia_value') != j or
        not all(type(x) is int for x in cmp.get('r_value', []) + cmp.get('julia_value', [])) or
        not all(type(x) is int for x in r + j) or r != [0, 5, 6] or j != r or
        type(cmp.get('max_abs_diff')) not in (int, float) or cmp.get('max_abs_diff') != 0 or
        cmp.get('discriminating') is not True):
        raise ValueError('comparison differs from the captured exact lengths')


def check_historical_case_binding(case, source_bytes, selected_bytes):
    binding = case.get('historical_case')
    if (not isinstance(binding, dict) or binding.get('path') != HISTORICAL_CASE or
            binding.get('sha256') != HISTORICAL_SHA256 or
            not isinstance(source_bytes, bytes) or source_bytes != selected_bytes or
            hashlib.sha256(source_bytes).hexdigest() != binding['sha256']):
        raise ValueError('historical case binding differs')
    try:
        historical = json.loads(source_bytes)
        if not isinstance(historical, dict) or not isinstance(historical.get('comparison'), dict):
            raise ValueError('historical case malformed')
        comparisons = historical['comparison']['cases']
        if (historical.get('case_id') != CID or not isinstance(comparisons, list) or
                len(comparisons) != 1 or not isinstance(comparisons[0], dict) or
                comparisons[0].get('discriminating') is not False):
            raise ValueError('historical case is not the non-discriminating record')
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('historical case malformed') from exc


def check_case_source_binding(case, record, read_selected):
    identity = record.get('postcoef_capture_identity', {})
    source_pin = record.get('source_pin', {})
    if not isinstance(identity, dict) or case.get('glvmodels_src_tree') != identity.get('source_tree'):
        raise ValueError('case source tree differs from captured source')
    root = 'docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle'
    build_path = case.get('oracle_build_receipt')
    if not isinstance(build_path, str) or not re.fullmatch(re.escape(root) + r'/build(?:-[A-Za-z0-9_-]+)?\.json', build_path):
        raise ValueError('case oracle build path is invalid')
    build_bytes = read_selected(build_path)
    if hashlib.sha256(build_bytes).hexdigest() != record.get('oracle_build_sha256'):
        raise ValueError('case oracle build receipt differs from captured build')
    build = json.loads(build_bytes)
    for key in ('reference_commit', 'archive_sha256', 'source_tree_sha256', 'installed_tree_sha256', 'marker_sha256'):
        if not source_pin.get(key) or build.get(key) != source_pin[key]:
            raise ValueError('case oracle build identity mismatch: ' + key)
    if build.get('gllvmTMB_version') != source_pin.get('version') or case.get('gllvmtmb_version') != source_pin.get('version'):
        raise ValueError('case oracle build version mismatch')
    health = case.get('harness_fields', {}).get('three_fit', {}).get('fit_health', {})
    julia_fits = health.get('Julia') if isinstance(health, dict) else None
    if (not isinstance(julia_fits, list) or len(julia_fits) != 3 or
        not all(isinstance(fit, dict) and julia_prior_bound(
            fit.get('penalty_provenance'), identity.get('source_tree'), fit.get('case'))
            for fit in julia_fits)):
        raise ValueError('Julia penalty provenance differs from authenticated source tree')
    fits = health.get('R') if isinstance(health, dict) else None
    if (not isinstance(fits, list) or len(fits) != 3 or
        not all(isinstance(fit, dict) and historical_r_prior_bound(
            fit.get('prior_provenance'), record.get('oracle_build_sha256')) for fit in fits)):
        raise ValueError('R prior provenance differs from authenticated oracle build')
    source_path = case.get('oracle_source_receipt')
    if source_path != f'{root}/source.json':
        raise ValueError('case oracle source path is invalid')
    source = json.loads(read_selected(source_path))
    for key in ('reference_commit', 'archive_sha256', 'namespace_sha256', 'source_tree_sha256'):
        if not source_pin.get(key) or source.get(key) != source_pin[key]:
            raise ValueError('case oracle source identity mismatch: ' + key)


def verification_identity(repo, head, runtime_root):
    """Read a source-bound identity tuple; this is not verification or an attestation."""
    if not SHA.fullmatch(head):
        raise ValueError('selected commit must be a full object ID')
    if git(repo, 'cat-file', '-t', head).decode().strip() != 'commit':
        raise ValueError('selected object is not a commit')
    tree = git(repo, 'rev-parse', '--verify', f'{head}^{{tree}}').decode().strip()
    if not SHA.fullmatch(tree):
        raise ValueError('selected tree is invalid')
    case_bytes = blob(repo, head, CASE)
    case = json.loads(case_bytes)
    check_capture_pointers(case)
    check_verifier_transcript(case, blob(repo, head, f'{PREFIX}/verify.txt'))
    binding = case['postcoef_capture']
    if binding.get('receipt') != f'{PREFIX}/receipt.json':
        raise ValueError('capture path differs')
    receipt_bytes = blob(repo, head, binding['receipt'])
    receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_hash != binding.get('sha256'):
        raise ValueError('capture receipt differs')
    record = json.loads(receipt_bytes)
    source = record['postcoef_capture_identity']['source_commit']
    check_historical_case_binding(case, blob(repo, source, HISTORICAL_CASE), blob(repo, head, HISTORICAL_CASE))
    case_commit = case['glvmodels_commit']
    if not SHA.fullmatch(source) or not SHA.fullmatch(case_commit):
        raise ValueError('source or case commit is invalid')
    build = case['oracle_build_receipt']
    build_hash = hashlib.sha256(blob(repo, head, build)).hexdigest()
    if build_hash != record['oracle_build_sha256']:
        raise ValueError('oracle build differs')
    runtime = record['postcoef_runtime']
    return {'selected_commit': head, 'selected_tree': tree, 'source_commit': source,
            'case_commit': case_commit, 'case_sha256': hashlib.sha256(case_bytes).hexdigest(),
            'capture_sha256': receipt_hash, 'oracle_build_sha256': build_hash,
            'runtime_record': runtime,
            'runtime_root': str(Path(runtime_root).resolve()),
            'python_executable': str(Path(sys.executable).resolve())}


SOURCE_FILES = ('tools/core070_verify_postfit_policy_batch.py', 'tools/parity_oracle.py',
                'tools/core070_source_pin_check.py', 'tools/core070_oracle_pins.toml',
                'tools/core070_postcoef_depot_check.py', 'tools/core070_build_oracle.py',
                'tools/core070_postcoef_fit_health.py', 'tools/core070_json_semantics.py', 'tools/core070_postcoef_fit_health.R',
                'tools/core070_postcoef_fit_health.jl',
                'tools/core070_source_pin.R', 'Project.toml',
                'docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json')
# These inputs execute or define the captured health transaction. The gate
# checkout is independently pinned by the launcher; capture commits cannot
# grant trust to their own modified verifier or runner.
TRUSTED_CAPTURE_FILES = SOURCE_FILES + ('tools/core070_postfit_policy_batch.R',
                                        'tools/core070_postfit_policy_batch.jl')


def check_trusted_capture_sources(read_capture, read_trusted):
    for rel in TRUSTED_CAPTURE_FILES:
        if read_capture(rel) != read_trusted(rel):
            raise ValueError('capture source differs from trusted gate: ' + rel)


def check_capture_trust(repo, source, trusted_root):
    # Exact tools tree also rules out extra Python shadow modules. This is
    # anchored only when trusted_root is the separately pinned gate checkout.
    def trusted_inventory(root, revision):
        entries = git(root, 'ls-tree', '-r', '-z', revision, '--', 'tools').split(b'\0')
        # The launcher is rebound after the trusted gate commit is made.
        # Its identity is separately checked by the caller binding.
        return sorted(e for e in entries if e and not e.endswith(b'\ttools/core070_postcoef_gate_launcher.py'))
    if trusted_inventory(repo, source) != trusted_inventory(trusted_root, 'HEAD'):
        raise ValueError('capture tools tree differs from trusted gate')
    trusted_root = Path(trusted_root)
    check_trusted_capture_sources(lambda rel: blob(repo, source, rel),
                                  lambda rel: (trusted_root / rel).read_bytes())


SELFTEST_MARKERS = ('CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS 10',
                    'CORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS 4',
                    'CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS 5',
                    'CORE070_POSTFIT_POLICY_BATCH_VERIFIED 15 negative_controls 2')


def run_source_selftest(root, source, state, runtime):
    """Run the captured source verifier; returned text is not an attestation."""
    if not re.fullmatch(r'(?:[0-9a-f]{40}|[0-9a-f]{64})', source):
        raise ValueError('invalid source commit')
    root = Path(root).resolve()
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True,
                              env=postcoef_git_env(), timeout=30).stdout
    if git('cat-file', '-t', source).strip() != b'commit':
        raise ValueError('source is not a commit')
    python = runtime.get('python_executable_path') if isinstance(runtime, dict) else None
    if not isinstance(python, str) or not Path(python).is_absolute() or Path(python).resolve() != Path(sys.executable).resolve():
        raise ValueError('captured Python executable differs')
    for rel in SOURCE_FILES:
        entry = git('ls-tree', '-z', source, '--', rel)
        if not re.fullmatch(rb'100(?:644|755) blob [0-9a-f]+\t' + re.escape(rel.encode()) + rb'\0', entry):
            raise ValueError('source verifier input is not a regular tracked file')
        current = root
        for part in rel.split('/'):
            current = current / part
            if current.is_symlink(): raise ValueError('source verifier input has a symlink')
        if not current.is_file() or current.read_bytes() != git('show', f'{source}:{rel}'):
            raise ValueError('source verifier input differs from captured commit')
    tracked_tools = git('ls-tree', '-r', '-z', '--name-only', source, '--', 'tools').split(b'\0')
    if any(path.endswith(b'.pyc') or b'__pycache__' in path.split(b'/') for path in tracked_tools if path):
        raise ValueError('captured source contains Python bytecode')
    env = postcoef_git_env()
    for name in list(env):
        if name.startswith('PYTHON'):
            del env[name]
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONNOUSERSITE'] = '1'
    env['GLLVM_PARITY_PIN'] = 'P1'
    # Execute only in a fresh checkout of the captured commit. The user's
    # ignored bytecode cannot be read or created by this subprocess.
    with tempfile.TemporaryDirectory(prefix='postcoef-verifier-source-') as tmp:
        sandbox = Path(tmp) / 'checkout'; template = Path(tmp) / 'empty-template'; template.mkdir()
        subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', 'clone', '--no-local', '--no-hardlinks', '--no-checkout',
                        '--template', str(template), '--', str(root), str(sandbox)],
                       check=True, capture_output=True, env=env, timeout=60)
        subprocess.run(['git', '-C', str(sandbox), '-c', 'core.hooksPath=/dev/null',
                        'checkout', '--quiet', '--detach', source],
                       check=True, capture_output=True, env=env, timeout=60)
        if os.path.lexists(sandbox / '.git/objects/info/alternates'):
            raise ValueError('private verifier checkout uses object alternates')
        for rel in SOURCE_FILES:
            local = sandbox / rel
            if local.is_symlink() or not local.is_file() or local.read_bytes() != git('show', f'{source}:{rel}'):
                raise ValueError('private verifier source differs from captured commit')
        # The captured runtime manifest/preferences are data snapshots, not
        # source imports. Supply them only when recorded, without overwriting a
        # different tracked file in the private checkout.
        snapshots = [('manifest_sha256', 'julia-Manifest.toml', 'Manifest.toml'),
                     ('preferences_sha256', 'julia-LocalPreferences.toml', 'LocalPreferences.toml')]
        for key, src, dest in snapshots:
            expected = runtime.get(key)
            if expected is None or expected == 'absent':
                continue
            data = (Path(state) / src).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('captured runtime snapshot differs')
            path = sandbox / dest
            if path.is_symlink() or (path.exists() and path.read_bytes() != data):
                raise ValueError('private runtime snapshot conflicts')
            if not path.exists(): path.write_bytes(data)
        proc = subprocess.run([sys.executable, '-B', '-I', str(sandbox / SOURCE_FILES[0]), '--state', str(state), '--self-test'],
                              cwd=sandbox, env=env, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0 or proc.stderr or not all(line in proc.stdout.splitlines() for line in SELFTEST_MARKERS):
            raise ValueError('source verifier self-test rejected capture')
        return proc.stdout



def verify(repo, head, runtime_root):
    repo, runtime_root = Path(repo).resolve(), Path(runtime_root).resolve()
    if not SHA.fullmatch(head):
        raise ValueError('selected ref must be a full commit ID')
    identity_before = verification_identity(repo, head, runtime_root)
    case = json.loads(blob(repo, head, CASE))
    check_capture_pointers(case)
    transcript = blob(repo, head, f'{PREFIX}/verify.txt')
    check_verifier_transcript(case, transcript)
    binding = case.get('postcoef_capture', {})
    if binding.get('receipt') != f'{PREFIX}/receipt.json' or not HASH.fullmatch(binding.get('sha256', '')):
        raise ValueError('case capture binding missing or malformed')
    recbytes = blob(repo, head, binding['receipt'])
    if hashlib.sha256(recbytes).hexdigest() != binding['sha256']:
        raise ValueError('capture receipt hash mismatch')
    record = json.loads(recbytes)
    check_case_source_binding(case, record, lambda rel: blob(repo, head, rel))
    source = record.get('postcoef_capture_identity', {}).get('source_commit', '')
    case_commit = case.get('glvmodels_commit', '')
    trusted_root = Path(__file__).resolve().parent.parent
    check_capture_trust(repo, source, trusted_root)
    if not (postcoef_ref_compatible(repo, source, case_commit) and postcoef_ref_compatible(repo, case_commit, head)):
        raise ValueError('selected ref is not an approved evidence-only descendant')
    if git(runtime_root, 'rev-parse', 'HEAD').decode().strip() != source:
        raise ValueError('runtime checkout is not the capture commit')
    if git(runtime_root, 'status', '--porcelain=v1', '--untracked-files=all'):
        raise ValueError('runtime checkout has tracked or untracked changes')
    for rel in SOURCE_FILES:
        local = (runtime_root / rel).resolve()
        if runtime_root not in local.parents or not local.is_file() or local.read_bytes() != blob(runtime_root, source, rel):
            raise ValueError('runtime source differs from captured commit: ' + rel)
    python_path = record.get('postcoef_runtime', {}).get('python_executable_path', '')
    if not python_path or Path(python_path).resolve() != Path(sys.executable).resolve():
        raise ValueError('verifier Python does not match recorded runtime')
    files = list(FILES)
    if record.get('postcoef_runtime', {}).get('preferences_sha256') != 'absent':
        files.append('julia-LocalPreferences.toml')
    with tempfile.TemporaryDirectory(prefix='postcoef-live-ref-') as tmp:
        state = Path(tmp)
        stage_capture_files(state, files, lambda name: blob(repo, head, f'{PREFIX}/{name}'))
        report = json.loads((state / 'julia-results.json').read_text())
        oracle = json.loads((state / 'r-oracle.json').read_text())
        check_comparison(case, report, oracle)
        output = run_source_selftest(runtime_root, source, state, record['postcoef_runtime'])
        if output.encode('utf8') != transcript:
            raise ValueError('live verifier transcript differs')
    if git(runtime_root, 'rev-parse', 'HEAD').decode().strip() != source or git(runtime_root, 'status', '--porcelain=v1', '--untracked-files=all'):
        raise ValueError('runtime checkout changed during verification')
    identity_after = verification_identity(repo, head, runtime_root)
    if not json_value_equal(identity_before, identity_after):
        raise ValueError('selected verification identity changed during verification')
    return {'status': 'POSTCOEF_LIVE_REF_VERIFIED', **identity_after,
            'capture_commit': source, 'scope': 'verified capture; no row admission'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--identity-only', action='store_true')
    args = parser.parse_args()
    try:
        result = (verification_identity(args.repo, args.head, args.runtime_root)
                  if args.identity_only else verify(args.repo, args.head, args.runtime_root))
        print(json.dumps(result, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print('POSTCOEF_LIVE_REF_REJECTED: ' + str(exc), file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())

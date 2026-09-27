#!/usr/bin/env python3
"""Unit tests for the P1 pin plumbing added to tools/core070_build_oracle.py
and tools/core070_oracle_pins.toml (D-294/D-295).

Covers:
  1. tools/core070_build_oracle.py's REFERENCE/NAMESPACE/SOURCE_TREE/ARCHIVE
     follow GLLVM_PARITY_PIN (tools/parity_oracle.py's switch): P0 by
     default, P1 when set, and a hard failure on an unrecognized value.
  2. tools/core070_oracle_pins.toml's per-pin reference_commit agrees with
     tools/parity_oracle.py's R_REF_PINS for every pin (single source of the
     commit SHA; the TOML file only supplies the companion byte hashes).
  3. The P1 entry's three byte hashes actually match a fresh `git archive`
     of the P1 commit, computed independently of tools/core070_build_oracle.py
     itself. Skipped cleanly if the local gllvmTMB clone is absent (this
     machine only; not something CI can run).
"""
import hashlib
import os
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

os.environ.pop("GLLVM_PARITY_PIN", None)

import parity_oracle  # noqa: E402

PINS_FILE = TOOLS / "core070_oracle_pins.toml"
GLLVMTMB_CLONE = "/Users/z3437171/Dropbox/Github Local/gllvmTMB"


def _clean_env(**overrides):
    env = {k: v for k, v in os.environ.items() if k != "GLLVM_PARITY_PIN"}
    env.update(overrides)
    return env


def _build_oracle_constants(pin_value=None):
    env = _clean_env() if pin_value is None else _clean_env(GLLVM_PARITY_PIN=pin_value)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import core070_build_oracle as b; "
            "print(b.REFERENCE); print(b.NAMESPACE); print(b.SOURCE_TREE); print(b.ARCHIVE)",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=str(TOOLS),
    )


class PinSelectionFollowsGllvmParityPin(unittest.TestCase):
    def test_default_is_p0(self):
        out = _build_oracle_constants()
        self.assertEqual(out.returncode, 0, out.stderr)
        reference, namespace, source_tree, archive = out.stdout.splitlines()
        self.assertEqual(reference, parity_oracle.FROZEN_GLLVMTMB_ORACLE)
        pin = tomllib.loads(PINS_FILE.read_text())["P0"]
        self.assertEqual(namespace, pin["namespace_sha256"])
        self.assertEqual(source_tree, pin["source_tree_sha256"])
        self.assertEqual(archive, pin["archive_sha256"])

    def test_p1_selected_when_env_set(self):
        out = _build_oracle_constants("P1")
        self.assertEqual(out.returncode, 0, out.stderr)
        reference, namespace, source_tree, archive = out.stdout.splitlines()
        self.assertEqual(reference, parity_oracle.P1_GLLVMTMB_ORACLE)
        pin = tomllib.loads(PINS_FILE.read_text())["P1"]
        self.assertEqual(namespace, pin["namespace_sha256"])
        self.assertEqual(source_tree, pin["source_tree_sha256"])
        self.assertEqual(archive, pin["archive_sha256"])

    def test_pin_switch_is_case_and_whitespace_insensitive(self):
        for raw in ("p1", " P1 ", "P1\n"):
            with self.subTest(raw=raw):
                out = _build_oracle_constants(raw)
                self.assertEqual(out.returncode, 0, out.stderr)
                reference = out.stdout.splitlines()[0]
                self.assertEqual(reference, parity_oracle.P1_GLLVMTMB_ORACLE)

    def test_unrecognized_pin_fails_loud(self):
        for raw in ("not-a-real-pin", "P2", "1P"):
            with self.subTest(raw=raw):
                out = _build_oracle_constants(raw)
                self.assertNotEqual(out.returncode, 0, out.stdout)
                self.assertIn("GLLVM_PARITY_PIN", out.stderr)


class PinsFileAgreesWithParityOracle(unittest.TestCase):
    def test_every_pin_reference_commit_matches_parity_oracle(self):
        pins = tomllib.loads(PINS_FILE.read_text())
        self.assertEqual(set(pins), set(parity_oracle.R_REF_PINS))
        for name, entry in pins.items():
            self.assertEqual(
                entry["reference_commit"],
                parity_oracle.R_REF_PINS[name],
                f"pin {name!r}: core070_oracle_pins.toml disagrees with "
                "parity_oracle.R_REF_PINS on the commit SHA",
            )


def _sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _git_archive_hashes(repo, commit):
    """Independent re-derivation of core070_build_oracle.py's prepare() hashes."""
    import tarfile
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        archive = tmp / "archive.tar"
        with archive.open("xb") as stream:
            subprocess.run(
                ["git", "-C", repo, "archive", "--format=tar", commit],
                stdout=stream,
                check=True,
                env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
                timeout=120,
            )
        archive_sha256 = _sha256_file(archive)
        extract_dir = tmp / "extracted"
        extract_dir.mkdir()
        with tarfile.open(archive, "r:") as source:
            try:
                source.extractall(extract_dir, filter="data")
            except TypeError:
                source.extractall(extract_dir)
        namespace_sha256 = _sha256_file(extract_dir / "NAMESPACE")
        entries = {}
        for folder, directories, names in os.walk(extract_dir, followlinks=False):
            for name in directories + names:
                path = Path(folder) / name
                if path.is_symlink():
                    raise ValueError("symlink not admitted in oracle tree")
                if path.is_file():
                    entries[path.relative_to(extract_dir).as_posix()] = _sha256_file(path)
        source_tree_sha256 = hashlib.sha256(
            "\n".join(sorted(k + "\0" + v for k, v in entries.items())).encode()
        ).hexdigest()
    return {
        "archive_sha256": archive_sha256,
        "namespace_sha256": namespace_sha256,
        "source_tree_sha256": source_tree_sha256,
    }


@unittest.skipUnless(
    Path(GLLVMTMB_CLONE).is_dir(), f"local gllvmTMB clone not found at {GLLVMTMB_CLONE!r}"
)
class P1HashesMatchGitArchive(unittest.TestCase):
    def test_p1_hashes_match_fresh_archive(self):
        pin = tomllib.loads(PINS_FILE.read_text())["P1"]
        actual = _git_archive_hashes(GLLVMTMB_CLONE, pin["reference_commit"])
        self.assertEqual(actual["archive_sha256"], pin["archive_sha256"])
        self.assertEqual(actual["namespace_sha256"], pin["namespace_sha256"])
        self.assertEqual(actual["source_tree_sha256"], pin["source_tree_sha256"])


if __name__ == "__main__":
    raise SystemExit(unittest.main())

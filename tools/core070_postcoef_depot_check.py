"""Check Julia package source trees against their manifest Git tree hashes.

Generated Julia line-coverage files are excluded. This does not verify Julia
artifacts, compiled caches, system libraries, or external preferences.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import re
import stat
import tomllib
import uuid


_COVERAGE = re.compile(r"(.+\.jl)\.\d+\.cov\Z")
_COVERAGE_LINE = re.compile(rb"^\s*(?:-|\d+) (.*)$")


def source_tree_hash(root):
    """Return the Git tree SHA-1 and the sorted generated-coverage paths."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError("package source directory does not exist")
    excluded = []

    def git_hash(kind, data):
        return hashlib.sha1(kind + b" " + str(len(data)).encode() + b"\0" + data).digest()

    def visit(directory):
        entries = []
        with os.scandir(directory) as scan:
            ordered = sorted(scan, key=lambda item: os.fsencode(item.name) +
                             (b"/" if item.is_dir(follow_symlinks=False) else b""))
        for item in ordered:
            if item.name == ".git":
                continue
            coverage = _COVERAGE.fullmatch(item.name) if item.is_file(follow_symlinks=False) else None
            if coverage:
                original = Path(directory) / coverage.group(1)
                if original.is_symlink() or not original.is_file():
                    raise ValueError("coverage file has no regular source file")
                decoded = []
                for line in Path(item.path).read_bytes().splitlines():
                    match = _COVERAGE_LINE.fullmatch(line)
                    if match is None:
                        raise ValueError("coverage file contains malformed data")
                    decoded.append(match.group(1))
                if decoded != original.read_bytes().splitlines():
                    raise ValueError("coverage file does not match its source")
                excluded.append(Path(item.path).relative_to(root).as_posix())
                continue
            if item.is_symlink():
                mode, content = b"120000", os.fsencode(os.readlink(item.path))
                child_hash = git_hash(b"blob", content)
            elif item.is_dir(follow_symlinks=False):
                child_hash, nonempty = visit(item.path)
                if not nonempty:
                    continue
                mode = b"40000"
            elif item.is_file(follow_symlinks=False):
                mode = b"100755" if item.stat(follow_symlinks=False).st_mode & stat.S_IXUSR else b"100644"
                child_hash = git_hash(b"blob", Path(item.path).read_bytes())
            else:
                raise ValueError("unsupported package filesystem entry")
            entries.append(mode + b" " + os.fsencode(item.name) + b"\0" + child_hash)
        content = b"".join(entries)
        return git_hash(b"tree", content), bool(entries)

    digest, _ = visit(root)
    return digest.hex(), sorted(excluded)


def verify_manifest_sources(manifest, records, stdlib_root=None):
    """Check manifest package trees and identify hashless Julia stdlibs."""
    deps = manifest.get("deps") if isinstance(manifest, dict) else None
    if not isinstance(deps, dict) or not isinstance(records, list):
        raise ValueError("manifest or package inventory is malformed")
    expected = {}
    for name, tables in deps.items():
        if not isinstance(name, str) or not isinstance(tables, list):
            raise ValueError("manifest dependency entries are malformed")
        for table in tables:
            if not isinstance(table, dict):
                raise ValueError("manifest dependency entry is malformed")
            is_root = name == "GLLVModels" and table.get("uuid") == "2dc8e01c-4f48-4476-aaae-e919b4a30df7"
            if "path" in table and not is_root:
                raise ValueError("unverified path dependency in manifest")
            if is_root:
                continue
            key, digest = table.get("uuid"), table.get("git-tree-sha1")
            try:
                valid_uuid = isinstance(key, str) and str(uuid.UUID(key)) == key
            except ValueError:
                valid_uuid = False
            if not valid_uuid or (digest is not None and
                                  (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{40}", digest))):
                raise ValueError("manifest package UUID or tree hash is malformed")
            if key in expected:
                raise ValueError("duplicate manifest package UUID")
            expected[key] = (name, digest, table.get("version"))
    actual = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"name", "uuid", "tree_hash", "source"}:
            raise ValueError("package inventory entry is malformed")
        key = record["uuid"]
        if not isinstance(key, str) or key in actual:
            raise ValueError("duplicate or malformed package inventory UUID")
        actual[key] = record
    if set(actual) != set(expected):
        raise ValueError("package inventory does not cover the manifest")
    results = []
    stdlib_checked = 0
    for key in sorted(expected):
        name, digest, version = expected[key]
        record = actual[key]
        source = record["source"]
        if record["name"] != name or record["tree_hash"] != digest or not isinstance(source, str) or not Path(source).is_absolute():
            raise ValueError("package inventory differs from manifest")
        if digest is None:
            if (stdlib_root is None or Path(name).name != name or name in (".", "..") or
                    not isinstance(source, str)):
                raise ValueError("hashless dependency is not a verified Julia stdlib")
            root = Path(stdlib_root).resolve()
            expected_path = (root / name).resolve()
            if root not in expected_path.parents or Path(source).resolve() != expected_path:
                raise ValueError("hashless dependency is outside Julia stdlib")
            try:
                project = tomllib.loads((expected_path / "Project.toml").read_text())
            except (OSError, ValueError) as exc:
                raise ValueError("Julia stdlib project metadata is unavailable") from exc
            if (project.get("name") != name or project.get("uuid") != key or
                    (version is not None and project.get("version") != version)):
                raise ValueError("Julia stdlib project metadata differs from manifest")
            observed, excluded = None, []
            stdlib_checked += 1
        else:
            observed, excluded = source_tree_hash(source)
            if observed != digest:
                raise ValueError("package source tree differs from manifest: " + name)
        results.append({"name": name, "uuid": key, "tree_hash": observed,
                        "source": source, "excluded_coverage_files": excluded})
    return {"checked": len(results) - stdlib_checked, "stdlib_checked": stdlib_checked, "packages": results,
            "excluded_coverage_files": sum(len(item["excluded_coverage_files"]) for item in results)}


def source_summary(result):
    """Summarize verified trees and retain every excluded coverage path."""
    return {"status": "PACKAGE_SOURCE_TREES_MATCH", "checked": result["checked"],
            "stdlib_checked": result["stdlib_checked"],
            "excluded_coverage_files": result["excluded_coverage_files"],
            "exclusions": [{"uuid": item["uuid"], "paths": item["excluded_coverage_files"]}
                           for item in result["packages"] if item["excluded_coverage_files"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--stdlib-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = tomllib.loads(args.manifest.read_text())
    records = json.loads(args.inventory.read_text())
    result = verify_manifest_sources(manifest, records, args.stdlib_root)
    if result["checked"] == 0:
        raise ValueError("manifest has no verifiable package trees")
    print(json.dumps(source_summary(result), sort_keys=True))


if __name__ == "__main__":
    main()

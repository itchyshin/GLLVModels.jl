#!/usr/bin/env python3
"""Reject unmistakable internal-process language in public Markdown sources."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple


DEVELOPER_ONLY_DIRS = frozenset({"developer", "developer-notes"})
SKIPPED_DIRS = frozenset({"dev-log"})
RULES = (
    ("issue-reference", re.compile(r"(?:\bissues?\s*#\d+|/issues/\d+)", re.IGNORECASE)),
    ("pull-request-reference", re.compile(r"(?:\bpr\s*#\d+|\bpull\s+request\s*#?\d+|/pull/\d+)", re.IGNORECASE)),
    ("process-terminology", re.compile(r"\b(?:arc|lane|worktree|dev-log)\b", re.IGNORECASE)),
    (
        "agent-audit-receipt",
        re.compile(
            r"\b(?:ada|boole|noether|fisher|curie|rose|florence|pat|grace|karpinski|hopper|p[óo]lya)\b"
            r".{0,60}\b(?:audit|receipt)\b|\b(?:audit|receipt)\b.{0,60}"
            r"\b(?:ada|boole|noether|fisher|curie|rose|florence|pat|grace|karpinski|hopper|p[óo]lya)\b",
            re.IGNORECASE,
        ),
    ),
)


class Finding(NamedTuple):
    path: Path
    line: int
    rule: str
    text: str


def is_public_markdown(path: Path, root: Path) -> bool:
    """Return whether a Markdown file is part of the reader-facing surface."""
    parts = set(path.relative_to(root).parts[:-1])
    return not (parts & DEVELOPER_ONLY_DIRS or parts & SKIPPED_DIRS)


def scan(docs_root: Path) -> list[Finding]:
    """Return stable, path-sorted findings for public Markdown under *docs_root*."""
    root = docs_root.resolve()
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.md")):
        if not is_public_markdown(path, root):
            continue
        relative = path.relative_to(root)
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for rule, pattern in RULES:
                if pattern.search(text):
                    findings.append(Finding(relative, number, rule, text.strip()))
                    break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "docs_root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "src",
        help="public documentation source directory (default: docs/src)",
    )
    args = parser.parse_args()
    findings = scan(args.docs_root)
    if findings:
        for finding in findings:
            print(
                f"{finding.path.as_posix()}:{finding.line}: {finding.rule}: {finding.text}",
                file=sys.stderr,
            )
        print(f"READER_SURFACE_FAIL findings={len(findings)}", file=sys.stderr)
        return 1
    print("READER_SURFACE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

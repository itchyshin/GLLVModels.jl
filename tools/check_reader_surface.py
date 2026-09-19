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
    ("process-terminology", re.compile(r"\b(?:lane|worktree|dev-log)\b", re.IGNORECASE)),
    (
        "agent-process-language",
        re.compile(
            r"\b(?:agents?|personas?)[\s-]+(?:(?:has|have)\s+)?"
            r"(?:approved|audited|reviewed)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "agent-audit-receipt",
        re.compile(
            r"\b(?:ada|boole|noether|fisher|curie|rose|florence|pat|grace|karpinski|hopper|p[óo]lya)\b"
            r".{0,60}\b(?:audit|receipt)\b|\b(?:audit|receipt)\b.{0,60}"
            r"\b(?:ada|boole|noether|fisher|curie|rose|florence|pat|grace|karpinski|hopper|p[óo]lya)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "internal-validation-language",
        re.compile(
            r"\b(?:capability\s+ledger|scoreboards?|fixture\s+evidence|optimizer-health|"
            r"scalar-density\s+(?:bug|problem)|parity\s+fixture)\b",
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
    if not root.is_dir():
        raise ValueError(f"Documentation root must be an existing directory: {root}")
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.md")):
        if not is_public_markdown(path, root):
            continue
        relative = path.relative_to(root)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        by_line: dict[int, Finding] = {}
        # Match the complete source so ordinary prose wrapping cannot hide a
        # phrase; retain the first rule per starting line for concise diagnostics.
        for rule, pattern in RULES:
            for match in pattern.finditer(text):
                number = text.count("\n", 0, match.start()) + 1
                by_line.setdefault(number, Finding(relative, number, rule, lines[number - 1].strip()))
        findings.extend(by_line[number] for number in sorted(by_line))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Scans *.md recursively, including code/comments/link targets, except "
            "developer/, developer-notes/, and dev-log/ subtrees. This is a source-text "
            "pattern check only: it does not inspect rendered output or generated "
            "docstrings, validate scientific claims, or prove accessibility. "
            "Run tests with: python3 -m unittest tools.tests.test_reader_surface"
        ),
    )
    parser.add_argument(
        "docs_root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "src",
        help="public documentation source directory (default: docs/src)",
    )
    args = parser.parse_args()
    try:
        findings = scan(args.docs_root)
    except (ValueError, OSError) as error:
        parser.error(str(error))
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

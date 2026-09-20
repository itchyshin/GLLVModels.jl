#!/usr/bin/env python3
"""Reject unmistakable internal-process language on the public reader surface."""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from typing import Iterable, NamedTuple


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


def landing_contract_findings(docs_root: Path) -> list[str]:
    """Return missing essentials from a reader's first GLLVModels page."""
    landing = docs_root / "index.md"
    if not landing.is_file():
        return ["index.md is missing"]
    text = landing.read_text(encoding="utf-8")
    requirements = {
        "an expansion of GLLVM": r"\bGLLVM\*{0,2}\s+(?:means|stands for)\s+\*{0,2}(?:generalised|generalized) linear latent[ -]variable model",
        "a plain multi-response purpose": r"\b(?:several|many) responses\b",
        "a standalone Julia identity": r"\bstandalone Julia\b",
        "a link to the first runnable route": r"\]\(quickstart\.md\)",
    }
    return [label for label, pattern in requirements.items()
            if not re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)]


class Finding(NamedTuple):
    path: Path
    line: int
    rule: str
    text: str


def makedocs_route_paths(make_file: Path) -> list[Path]:
    """Return Markdown routes passed to the literal ``makedocs(pages = [...])``."""
    source = make_file.read_text(encoding="utf-8")
    match = re.search(r"pages\s*=\s*\[", source)
    if match is None:
        raise ValueError(f"No literal makedocs(pages = [...]) list found in: {make_file}")
    depth = 0
    end = None
    for offset, character in enumerate(source[match.end() - 1 :], start=match.end() - 1):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                end = offset + 1
                break
    if end is None:
        raise ValueError(f"Unterminated makedocs(pages = [...]) list in: {make_file}")
    listed = re.findall(r"=>\s*\"([^\"]+\.md)\"", source[match.start() : end])
    if not listed:
        raise ValueError(f"No Markdown routes found in makedocs(pages = [...]) in: {make_file}")
    return [Path(route) for route in listed]


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


def scan_paths(paths: Iterable[Path], display_root: Path) -> list[Finding]:
    """Return stable findings for explicit public reader-source files."""
    findings: list[Finding] = []
    for path in sorted(paths):
        try:
            relative = path.relative_to(display_root)
        except ValueError:
            relative = Path(path.name)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        by_line: dict[int, Finding] = {}
        for rule, pattern in RULES:
            for match in pattern.finditer(text):
                number = text.count("\n", 0, match.start()) + 1
                by_line.setdefault(number, Finding(relative, number, rule, lines[number - 1].strip()))
        findings.extend(by_line[number] for number in sorted(by_line))
    return findings


def markdown_fence_findings(paths: Iterable[Path], display_root: Path) -> list[Finding]:
    """Reject public Markdown pages with an unclosed fenced code block.

    An unmatched fence can turn an otherwise readable tutorial into literal
    code in the generated site, so source process-language checks alone are
    not enough to protect the reader's route.
    """
    findings: list[Finding] = []
    fence = re.compile(r"^\s*(`{3,}|~{3,})")
    for path in sorted(paths):
        try:
            relative = path.relative_to(display_root)
        except ValueError:
            relative = Path(path.name)
        opening_line: int | None = None
        marker: str | None = None
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = fence.match(line)
            if match is None:
                continue
            current = match.group(1)
            if opening_line is None:
                opening_line, marker = line_number, current[0]
            elif current[0] == marker:
                opening_line, marker = None, None
        if opening_line is not None:
            findings.append(Finding(relative, opening_line, "unclosed-markdown-fence", ""))
    return findings


def source_surface_paths(docs_root: Path, make_file: Path, readme: Path) -> list[Path]:
    """Return the README and every existing Documenter navigation route.

    CI uses this rather than a recursive source scan: a Markdown file is public
    only when the actual ``makedocs(pages=...)`` navigation publishes it.
    """
    root = docs_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Documentation root must be an existing directory: {root}")
    routes = [root / route for route in makedocs_route_paths(make_file)]
    missing = [path for path in routes if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path.relative_to(root)) for path in missing)
        raise ValueError(f"makedocs navigation references missing route(s): {rendered}")
    if not readme.is_file():
        raise ValueError(f"Public README must be an existing file: {readme}")
    return [readme.resolve(), *routes]


def rendered_text(html_source: str) -> str:
    """Extract conservative visible text from a generated HTML page."""
    without_noncontent = re.sub(
        r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>", "", html_source,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return html.unescape(re.sub(r"<[^>]+>", " ", without_noncontent))


def scan_rendered(rendered_root: Path) -> list[Finding]:
    """Return findings from generated HTML, including Documenter docstrings."""
    root = rendered_root.resolve()
    if not root.is_dir():
        raise ValueError(f"Rendered documentation root must be an existing directory: {root}")
    pages = sorted(root.rglob("*.html"))
    if not pages:
        raise ValueError(f"Rendered documentation root contains no HTML pages: {root}")
    findings: list[Finding] = []
    for path in pages:
        text = rendered_text(path.read_text(encoding="utf-8"))
        lines = text.splitlines()
        for rule, pattern in RULES:
            for match in pattern.finditer(text):
                number = text.count("\n", 0, match.start()) + 1
                excerpt = lines[number - 1].strip() if number <= len(lines) else ""
                findings.append(Finding(path.relative_to(root), number, rule, excerpt))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "The default source check scans README.md and every route in the literal "
            "makedocs(pages = [...]) navigation list. --rendered scans visible text in "
            "generated HTML, including Documenter-expanded public docstrings. It does not "
            "validate scientific claims or prove accessibility. Run tests with: "
            "python3 -m unittest tools.tests.test_reader_surface"
        ),
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "src",
        help="Documenter source root (default: docs/src)",
    )
    parser.add_argument(
        "--make-file", type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "make.jl",
        help="Documenter entrypoint containing makedocs(pages = [...])",
    )
    parser.add_argument(
        "--readme", type=Path,
        default=Path(__file__).resolve().parents[1] / "README.md",
        help="public README to scan with the navigation routes",
    )
    parser.add_argument(
        "--rendered", type=Path,
        help="generated HTML root; use after the Documenter build",
    )
    parser.add_argument(
        "--landing-contract", action="store_true",
        help="require a plain GLLVM definition, standalone Julia identity, and first route",
    )
    args = parser.parse_args()
    try:
        if args.landing_contract:
            missing = landing_contract_findings(args.docs_root)
            if missing:
                print("LANDING_CONTRACT_FAIL missing=" + "; ".join(missing), file=sys.stderr)
                return 1
        if args.rendered is not None:
            findings = scan_rendered(args.rendered)
            checked = len(list(args.rendered.rglob("*.html")))
            scope = "rendered_pages"
        else:
            paths = source_surface_paths(args.docs_root, args.make_file, args.readme)
            findings = scan_paths(paths, args.docs_root.resolve().parent)
            findings.extend(markdown_fence_findings(paths, args.docs_root.resolve().parent))
            checked = len(paths)
            scope = "source_files"
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
    if args.landing_contract:
        print("LANDING_CONTRACT_PASS")
    print(f"READER_SURFACE_PASS {scope}={checked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

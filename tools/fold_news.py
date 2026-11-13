#!/usr/bin/env python3
"""Fold news/*.md fragments into the `## Unreleased` section of CHANGELOG.md.

    python3 tools/fold_news.py            dry run (prints the text to insert)
    python3 tools/fold_news.py --write    insert it and delete the fragment files

Fragments are inserted, in file-name order, directly under the `## Unreleased`
heading. Format of a fragment: see news/README.md.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
NEWS = ROOT / "news"
CHANGELOG = ROOT / "CHANGELOG.md"


def main(argv):
    write = "--write" in argv
    frags = sorted(p for p in NEWS.glob("*.md") if p.name != "README.md")
    if not frags:
        print("no fragments in news/")
        return 0
    block = ""
    for p in frags:
        text = p.read_text(encoding="utf-8")
        if not text.startswith("- **"):
            print(f"{p.name}: does not start with '- **'; not folding", file=sys.stderr)
            return 1
        block += text if text.endswith("\n") else text + "\n"
    lines = CHANGELOG.read_text(encoding="utf-8").split("\n")
    heads = [i for i, l in enumerate(lines) if l.rstrip() == "## Unreleased"]
    if len(heads) != 1:
        print("CHANGELOG.md must hold exactly one '## Unreleased' heading", file=sys.stderr)
        return 1
    at = heads[0] + 1
    while at < len(lines) and lines[at].strip() == "":
        at += 1
    new = lines[: heads[0] + 1] + [""] + block.rstrip("\n").split("\n") + lines[at:]
    if not write:
        print(block, end="")
        print(f"[dry run] {len(frags)} fragment(s); use --write to apply", file=sys.stderr)
        return 0
    CHANGELOG.write_text("\n".join(new), encoding="utf-8")
    for p in frags:
        p.unlink()
    print(f"folded {len(frags)} fragment(s) into CHANGELOG.md")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

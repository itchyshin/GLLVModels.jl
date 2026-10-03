#!/usr/bin/env python3
"""Wrong-close guard: list every issue GitHub could close from PR text.

GitHub closes an issue when the PR body, or a commit message on the default
branch, contains a closing keyword next to a reference. It does not read
negation ("Does not fix #142" closed #142 and gllvmTMB#897) and it can read a
conventional-commit prefix as the keyword ("fix(n<p): ... (refs #149)" closed
#149). This tool takes the issues you MEAN to close and the PR body and
commit messages, and fails if any reference that GitHub could read as a
closure is outside that set. It needs no network access.

What counts as a possible closure (case-insensitive):
  * strict form: keyword, optional colon, then a reference, e.g. `fixes #12`,
    `Closes: itchyshin/GLLVModels.jl#12`, `resolved https://github.com/o/r/issues/12`;
  * loose form (flagged on purpose, because the incidents above were loose or
    negated): any reference later on the same line as a keyword, e.g.
    `fix(scope): text (refs #149)` or `Does not fix #142`.
Keywords are GitHub's documented list: close, closes, closed, fix, fixes,
fixed, resolve, resolves, resolved. A keyword inside a longer word
(`prefix`, `fixture`, `unresolved`) is ignored. Fenced code blocks and HTML
comments are skipped (GitHub does not link inside code; HTML comments are
hidden), so quote a keyword there if you must write one.

Usage:
  wrong_close_guard.py --intended 701,149 --body-file pr_body.md \
      [--commits-file msgs.txt | --git-range origin/main..HEAD] [--text STR]
  wrong_close_guard.py --body-file pr_body.md          # intended = nothing
Exit status: 0 clean, 1 an unintended reference found, 2 usage error.
An intended entry is `N` (any repo) or `owner/repo#N`. With
`--intended-from-body`, an HTML comment `<!-- intended-closes: 12, 13 -->` in a body file adds to
the intended set; this
is how the CI wiring (.github/workflows/wrong-close-guard.yml) learns it.
"""
import argparse
import re
import subprocess
import sys

KEYWORD = r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)"
KW_RE = re.compile(r"(?<![A-Za-z0-9_])" + KEYWORD + r"(?![A-Za-z0-9_])", re.I)
# A reference: owner/repo#N, a GitHub issue URL, or a bare #N.
REF_RE = re.compile(
    r"(?:https?://github\.com/(?P<uo>[\w.-]+)/(?P<ur>[\w.-]+)/(?:issues|pull)/(?P<un>\d+))"
    r"|(?:(?<![\w/.-])(?P<ro>[\w.-]+)/(?P<rr>[\w.-]+)#(?P<rn>\d+))"
    r"|(?:(?<![\w/&])#(?P<bn>\d+))"
)
STRICT_RE = re.compile(
    r"(?<![A-Za-z0-9_])" + KEYWORD + r"(?![A-Za-z0-9_])\s*:?\s*(?=" + REF_RE.pattern + ")", re.I
)


def _ref_key(m):
    """Return (repo or None, number) for a REF_RE match."""
    if m.group("un"):
        return (f"{m.group('uo')}/{m.group('ur')}".lower(), int(m.group("un")))
    if m.group("rn"):
        return (f"{m.group('ro')}/{m.group('rr')}".lower(), int(m.group("rn")))
    return (None, int(m.group("bn")))


def _strip_hidden(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out, fenced = [], False
    for line in text.splitlines():
        if re.match(r"\s*(```|~~~)", line):
            fenced = not fenced
            continue
        if not fenced:
            out.append(line)
    return out


def find_refs(text):
    """Yield (repo, number, line, form) for every possible closing reference."""
    seen = set()
    for line in _strip_hidden(text):
        kws = [m.start() for m in KW_RE.finditer(line)]
        if not kws:
            continue
        first = kws[0]
        strict_ends = {m.end() for m in STRICT_RE.finditer(line)}
        for m in REF_RE.finditer(line):
            if m.start() < first:
                continue
            form = "strict" if m.start() in strict_ends else "loose"
            key = _ref_key(m) + (line,)
            if key in seen:
                continue
            seen.add(key)
            yield (*_ref_key(m), line.strip(), form)


def parse_intended(spec):
    out = set()
    for tok in (t.strip() for t in re.split(r"[,\s]+", spec or "") if t.strip()):
        m = re.fullmatch(r"(?:([\w.-]+/[\w.-]+))?#?(\d+)", tok)
        if not m:
            raise ValueError(f"cannot parse intended issue {tok!r}")
        out.add((m.group(1).lower() if m.group(1) else None, int(m.group(2))))
    return out


def check(texts, intended):
    """Return the list of (repo, num, line, form, source) outside `intended`."""
    bad = []
    for source, text in texts:
        for repo, num, line, form in find_refs(text):
            if (repo, num) in intended or (None, num) in intended:
                continue
            bad.append((repo, num, line, form, source))
    return bad


def _git_messages(rng):
    out = subprocess.run(["git", "log", "--format=%H%x1f%B%x1e", rng],
                         capture_output=True, text=True, check=True).stdout
    for rec in filter(None, (r.strip() for r in out.split("\x1e"))):
        sha, _, body = rec.partition("\x1f")
        yield f"commit {sha[:9]}", body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--intended", default="", help="comma-separated issues to close: N or owner/repo#N")
    ap.add_argument("--intended-from-body", action="store_true",
                    help="also read `intended-closes: N, N` lines from the body files")
    ap.add_argument("--body-file", action="append", default=[])
    ap.add_argument("--commits-file", action="append", default=[])
    ap.add_argument("--git-range", help="e.g. origin/main..HEAD; checks each commit message")
    ap.add_argument("--text", action="append", default=[])
    a = ap.parse_args(argv)
    try:
        intended = parse_intended(a.intended)
    except ValueError as e:
        print(f"WRONG_CLOSE_GUARD usage error: {e}", file=sys.stderr)
        return 2
    texts = []
    for f in a.body_file:
        texts.append((f"body {f}", open(f, encoding="utf-8").read()))
    if a.intended_from_body:
        for src, text in texts:
            for m in re.finditer(r"<!--[ \t]*intended-closes:[ \t]*([^\n>]*)", text, re.I):
                try:
                    intended |= parse_intended(m.group(1).rstrip(" -\t"))
                except ValueError as e:
                    print(f"WRONG_CLOSE_GUARD usage error: {e}", file=sys.stderr)
                    return 2
    for f in a.commits_file:
        texts.append((f"commits {f}", open(f, encoding="utf-8").read()))
    for t in a.text:
        texts.append(("text", t))
    if a.git_range:
        texts.extend(_git_messages(a.git_range))
    if not texts:
        print("WRONG_CLOSE_GUARD usage error: nothing to check", file=sys.stderr)
        return 2
    bad = check(texts, intended)
    for repo, num, line, form, source in bad:
        ref = f"{repo}#{num}" if repo else f"#{num}"
        print(f"WRONG_CLOSE_GUARD {ref} ({form}) in {source}: {line}")
    if bad:
        print("GitHub could close the issue(s) above. Reword to 'refs #N' without a "
              "closing keyword on the same line, or list them in --intended.")
        return 1
    print(f"WRONG_CLOSE_GUARD_OK intended={sorted(intended, key=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

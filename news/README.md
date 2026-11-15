# News fragments

One file per change replaces an edit to the top of `CHANGELOG.md`. Two PRs that
each add a fragment never touch the same line, so they merge without conflict on
GitHub. (GitHub's merge button does not honour `merge=union` in `.gitattributes`,
so the shared top of `CHANGELOG.md` cannot be made conflict-free that way.)

## Add a fragment

Create `news/<slug>.md`, where `<slug>` is lowercase letters, digits and hyphens
and names the change (for example `news/nb1-fisher-weight-65.md`; include the
issue or PR number so two PRs never pick the same name). The file holds exactly the
entry you would have put under `## Unreleased` in `CHANGELOG.md`:

```
- **`fit_foo`: one-line statement of the change (refs #123).** Detail sentences.
  Continuation lines are indented by two spaces.
```

Rules (checked by `test/auto/test_news_fragments.jl`):

- the first line starts with `- **`;
- every later non-blank line is indented by two spaces or starts with `- ` (so one
  fragment may hold several bullets);
- no heading lines (`#`), no tab characters, no trailing spaces;
- the file ends with one newline; `README.md` is the only other file allowed here.

Do not edit `CHANGELOG.md` for new changes. Existing entries stay where they are.

## Fold fragments into the changelog (release time, or any time)

```
python3 tools/fold_news.py            # dry run: prints what would be inserted
python3 tools/fold_news.py --write    # inserts under "## Unreleased", deletes the fragments
```

Fragments are inserted at the top of `## Unreleased` in file-name order. Review the
diff, then commit `CHANGELOG.md` and the deleted fragments together.

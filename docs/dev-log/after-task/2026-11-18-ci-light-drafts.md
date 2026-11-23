# After Task: Light CI on draft pull requests

## Goal

Cut GitHub Actions load from draft pull requests, and leave ready pull requests, pushes to `main`, tags, and `workflow_dispatch` running the same work they run now.

## Implemented

Draft `pull_request` runs create one Julia smoke job: stable Julia, `ubuntu-latest`, shard 1/8. The other seven shards are left out of that matrix. `ready_for_review` is on every `pull_request` workflow, with `opened`, `synchronize`, and `reopened` kept, so marking a draft ready starts the full pull-request set.

Documenter does not use a workflow `paths` filter. A `changes` job lists the pull-request files and the build jobs run only when that list hits docs, `src/`, `news/`, `README`, `Project.toml`, `CHANGELOG.md`, `tools/check_reader_surface.py`, or `Documenter.yml`. On a draft those jobs are skipped. On a push to `main` or a tag they still build and deploy. The parity P1 pair is skipped on drafts and still runs both Julia versions on a ready pull request and on manual dispatch. The true-parity negative-control job is skipped on drafts that already match its existing path list.

The wrong-close guard, its script, and every other guard were not edited.

## Mathematical Contract

N/A. No likelihood, packing, or link change.

## Files Changed

- `.github/workflows/CI.yml` — `ready_for_review`; draft matrix is shard 1 only.
- `.github/workflows/Documenter.yml` — `ready_for_review`; job-level docs file check; draft skip.
- `.github/workflows/parity-p1-twin.yml` — `ready_for_review`; both versions skipped on drafts.
- `.github/workflows/true-parity-check.yml` — `ready_for_review`; job skipped on drafts. Existing workflow `paths` left as they were.
- `docs/dev-log/check-log.md`, `AGENTS.md`, this report.

## Tests Added

None. No package behavior changed. The path-matcher arms were checked with the same `case` list as the workflow: 13/13.

## Benchmark Numbers

N/A — no hot-path change. Runner counts for one draft `synchronize` that touches `src/` (CI not ignored, true-parity paths do not match, wrong-close guard included):

- Before: 13 jobs that take a runner (CI 8, Documenter 2, P1 twin 2, guard 1).
- After: 2 (CI shard 1, guard 1).

The guard is unchanged, so the cut in the workflows this PR edits is 12 to 1.

## R-Parity Verdict

Parity: N/A — change does not touch the parity surface.

## JET / Allocs / Aqua Verdicts

- JET: not run — no Julia source change.
- Allocs: not run — no Julia source change.
- Aqua: not run — no package or dependency change.

## Checks Run

```sh
/tmp/actionlint 1.7.7 .github/workflows/{CI,Documenter,parity-p1-twin,true-parity-check,wrong-close-guard,TagBot}.yml
```

Exit 0, no findings. A job-level `if` that referenced `matrix.shard` was rejected by actionlint (`matrix` is not a legal context there) and was replaced with the shard-list matrix trick.

```sh
python3 -c 'import yaml'  # yaml.safe_load of the same six workflow files
```

All six parsed. Wrong-close queued runs at this count: 12 queued, 0 in progress. Rulesets for the repo: 0. Branch rules for `main`: 0. Classic branch protection returned HTTP 403, so it was not re-read.

## Consistency Audit

No README or likelihood claim changed. `rg` of family-status phrases was not useful here; the edited prose is the workflow comments plus this log.

## GitHub Issue Maintenance

No issue action. This follows an approved plan and does not close an issue. The pull request body must not use a closing keyword the wrong-close guard would reject.

## What Did Not Go Smoothly

A job-level `if: matrix.shard == 1` looks like the right draft gate and is illegal. actionlint caught it before push. Shrinking the matrix makes shards 2–8 absent on the draft run instead of skipped. That is the safer of the two for a required check: a skipped required check counts as success, so a draft could otherwise merge after only shard 1.

## Team Learning

When a check must exist on ready pull requests that do not touch a path, gate the job with `if` and a changes job. A workflow `paths` filter leaves the check pending.

## Remaining Risks

- On a draft run, CI shards 2–8 are missing, not skipped. They are created when the pull request is marked ready. Until that run finishes, a required shard check has nothing to report.
- `CI.yml` still `paths-ignore`s `*.md`. A markdown-only pull request, draft or ready, still does not start CI. That is the previous behavior. Those shard checks stay missing.
- `true-parity-check.yml` still has its workflow `paths` filter. A pull request that does not touch those files still does not start that workflow, so `negative-controls` is missing. The workflow comment says it is not a required check. This PR did not add or remove those paths. A draft that does match them skips the job, which reports skipped.
- Pushes to `main` and tags gain one skipped check, `Docs path filter`. `Documenter` and `Documenter deploy` still run the same steps.
- A ready pull request that touches `src/` still builds the docs. The draft saving is the skip; the ready-PR saving is limited to pull requests that avoid the docs-relevant list.
- No readable ruleset requires Documenter. If it is later required, a skipped Documenter job is a successful status. A workflow-level path skip would stay pending and block the merge. That is why the filter is a job.

## Known Limitations

Drafts do not run shards 2–8, the docs build, or the P1 pair. Marking the pull request ready starts them. `workflow_dispatch` coverage and the macOS/Windows matrix are unchanged. Schedules: this repo has none.

## Next Command

None for the package. Review the draft pull request. Do not merge it from this lane.

## Rose Verdict

Rose verdict: PASS WITH NOTES — draft shards 2–8 are missing until ready_for_review; the pre-existing markdown `paths-ignore` and true-parity `paths` can still omit those workflows; Documenter itself reports skipped or success and is not in the readable rulesets.

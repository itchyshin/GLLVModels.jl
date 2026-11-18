# Check-log: first-wave twin documentation remediation

This closeout records Julia documentation findings D011, D012, and D036, and reconciles them with the paired R-side D010 remediation.

Checks run:

- Local Documenter/VitePress build with deployment disabled: exited 0. Doctests, cross references, and page rendering passed; VitePress completed. Warnings were large bundle size and default `package.json` template substitution.
- Verified generated HTML and Documenter Markdown for the score uncertainty wording and the explicit Poisson import.
- Executed the exact documented Poisson CV examples for random-cell and site-block splits; both returned finite metrics.
- `git diff --check` passed. Prose style checks returned zero findings.

Shinichi authorized completion of D010 as package maintainer. Paired gllvmTMB commit `f16c4d015` / PR #1704 now reports non-finite Wald bounds as unavailable. The exact original negative control is unavailable with a diagnostic note; the replacement example returned 14/14 finite bounds. The paired R after-task report records focused tests and the completed full pkgdown article build. PR #1017's Julia shards, twin tests, and Documenter passed; its deploy job was cancelled. At 2026-10-27 01:31 UTC, all four R-CMD-check shards on #1704 were in progress. No deploy or merge.

The Julia lane changed documentation and an example import. The paired R lane changed bridge CI-status handling and its example. There was no deployment, parity claim, or interval-calibration claim.

PR #1017 is open on `codex/gllvm-doc-remediation` at `75fd1f80722cbefa26443e777e3bbb2eb80ff5f6`. The superseded audit worktree could not be archived because the app reports it is protected by a pinned task or workspace; its clean checkpoint and evidence remain preserved. The planning conversation is archived.

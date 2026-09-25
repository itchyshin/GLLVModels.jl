# G0: GLLVM.jl to GLLVModels.jl rename design

**Status 2026-09-25:** landed. G0 was approved and the rename executed in PR
#423 (commit `69a69b0a0`, "chore: rename Julia package to GLLVModels"). The
repository and `Project.toml` (`name = "GLLVModels"`) both reflect the
completed rename. The "STOP AT G0 pending Shinichi's approval" line below no
longer applies.

Proposed in [PR #422](https://github.com/itchyshin/GLLVM.jl/pull/422). STOP AT G0 pending Shinichi's approval.
The programme authority is D-269 and the vault's 2026-09-15 working document
for this two-package rename.
The rehydrated base is `origin/main` at `8cc75587e`; this note extends the docs-only #422 handover.

## Decision carried into the implementation PR

The Julia package will become **GLLVModels.jl**, with package and module name
`GLLVModels`. Its UUID and current `Project.toml` version (`0.3.0`) stay
unchanged. The existing GitHub repository keeps its history; no new repository
is created. D-111 keeps General registration out of this programme.

The implementation PR will begin from then-current `origin/main` only after
G0.  It will rename the module/file, package metadata, extension registrations,
tests, Documentation, README, citation and CI references as one convention
change. Historical `docs/dev-log/` entries remain written as history.

## Migration contract

- GitHub and Pages: the green, unmerged code PR comes before Shinichi's
  GitHub Settings rename click. After that click, remotes and registered
  worktrees can be repaired and Documenter can deploy at
  `itchyshin.github.io/GLLVModels.jl/`. GitHub redirects repository URLs, but
  Pages does not; old Pages URLs must be rewritten and declared dead in NEWS.
- R bridge and RCall: gllvmTMB keeps its own name. Its later, separately
  coordinated bridge PR changes the Julia load path to `using GLLVModels` and
  re-runs the existing RCall parity surface. This documentation-only lane makes
  no bridge, likelihood, or Julia-to-R difference claim.
- Test and CI environments: the future implementation PR updates the
  package test environment in `test/Project.toml` and the names/references in
  `.github/workflows/CI.yml` and `.github/workflows/Documenter.yml`, then runs
  the full `Pkg.test()` suite and the local Documenter build. This G0 note runs
  neither a suite nor a release check.
- Old module spelling: if Julia permits it cleanly, the renamed module may
  export a soft-deprecated `const GLLVM = GLLVModels` alias plus an `__init__`
  notice. It can support qualified old spelling after `using GLLVModels`; it
  cannot preserve `using GLLVM`, because that requires a distinct old package
  identity/UUID. The implementation PR must test and document the actual
  behaviour before claiming compatibility.

## Explicitly out of scope

- Any `src/` rename, `Project.toml` edit, version bump, General registration,
  GitHub repository rename, Pages deployment, merge, or release before G0.
- The protected paste-gated DRAFT PRs #399, #409, #410 and #411 (and #363/
  #314), all true-parity ledger work, and any gllvmTMB engine or TMB change.
- A parity promotion, RCall delta claim, capability claim, or rewrite of
  historical development logs.

## G0 request

Approve this migration contract to permit a separate, isolated
`GLLVModels` code-rename branch and PR. Until that approval, #422 remains a
docs-only preparation PR and no source or package metadata will change.

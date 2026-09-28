# PR #547 CI-fix review — commits a6706acde + 23daa0868 (head 90c5caac9)

Reviewer: independent (Claude Fable 5.1), 2026-09-28. Detached worktree at 90c5caac9, Julia 1.10.12 (aarch64), `OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4`. Read-only: no edits, comments, pushes.

## Verdict: NON-BLOCKING

The fix does what it claims, the frozen-evidence guard is intact, every test I ran passes on Julia 1.10, and the Documenter pipeline passes as CI runs it. Findings below are notes, not defects.

## Findings

1. **Hashed files are byte-identical to `origin/main` and to the destination-B receipts.**
   `sha256` at HEAD / `origin/main` / receipt `invoked_source_sha256`
   (`docs/dev-log/core070/destination-b-tree/independent-attempt-01.json`, `.../destination-b-pedigree-fit/independent-attempt-01.json`, the two the test loads):
   - `src/precision_multivariate_fit.jl` = `1eae35fd…0efb` (all three agree)
   - `src/precision_multivariate.jl` = `3a0eecb5…a634` (all three agree)
   - `src/marginal_target_intervals.jl` = `a63b1f21…41eb` (all three agree)
   `git diff origin/main HEAD -- <file>` is empty for all three. (The s3b-pilot receipt carries an older `precision_multivariate.jl` hash `159dc8…`, but `test_destination_b_phylo_uncertainty.jl` does not load that receipt.)

2. **`Base.getproperty` override falls through correctly and costs nothing on real fields.**
   Evidence (`bench.jl`, Julia 1.10, `--project=.`):
   - `@inferred` on `f.loglik`, `f.loading`, `f.species_id` and `f.species_labels`: all pass.
   - `code_warntype` for `f.loglik`: `Body::Float64`, no `Any`/`Union`.
   - 1e7-iteration loop summing two Float64 fields: `getfield` 0.0100 s / 1 alloc vs `fit.x` 0.0104–0.0107 s / 1 alloc (noise; the `name === :species_labels` branch is folded away at compile time). No allocation regression.
   - `show(MIME"text/plain")`, `deepcopy`, `==`, `hash` all work; `fieldnames` still excludes the two derived names (so `fieldnames`-based code — e.g. the removed `_pmv_with_labels` copier pattern — never sees them); `propertynames` lists them (`hasproperty(fit, :species_labels) == true`).
   - Repo-wide grep: no other `getproperty`/`fieldnames`/`propertynames` consumer of `PrecisionMultivariateFit` in `src/` (the `propertynames(m)` at `src/postfit_tables.jl:400` is on a metrics NamedTuple, not the fit). `Base.summary`/`Base.show` in `src/destination_b_postfit.jl:143-166` touch only real fields.
   - Derived access allocates: `fit.species_labels` = 3 allocations / 448 B on a 10-tip, 30-obs fit (O(n_aug + n_obs) per call). Not on a hot path; only `_pmv_with_labels` (once per fit), `extract_phylo_signal`, and tests call it.

3. **Derived labels equal what the removed stored field held, on every constructor route.**
   - Tree route (`src/phylo_latent.jl:172-186`): `node_labels = vcat(internal node names, labels)`, `species_aug_id = index[1:n_tip]` → `node_labels[species_aug_id] == labels == tip_names`.
   - Dense route (`:212`): `PrecisionPhy(..., levels, ..., collect(1:p))` and `tip_names = levels` → identical.
   - Both routes build `species_id` from `tip_position` over `tip_names` (`:379-380`), so `tips[species_id] == obs_labels`. Empirically: `fit.species_labels == species` true, `fit.tip_labels == sort(unique(species))` true (dense route; levels order, as the twin documents and `test_phylo_latent_twin.jl:220` asserts with `sort`).
   - Other constructors: the fitter at `src/precision_multivariate_fit.jl:267` (passes `phy`, `collect(Int, species_id)` unchanged), the fixture ctor at `:173`, and direct fixtures in `test/test_destination_b_postfit.jl:20` / `test_destination_b_fixed_effects.jl:21`. All carry a `PrecisionPhy` with `node_labels::Vector{String}` and `species_aug_id`, so the derivation cannot throw for in-range `species_id` (a fitter precondition). Verified the non-twin `GLLVModels.fit_precision_multivariate(Y, phy; species_id=…)` route also yields the same labels (`non-twin route labels: true true`).
   - Consequence: the `error("internal: … labels disagree")` branch in `_pmv_with_labels` is unreachable by construction. Harmless as a tripwire.

4. **Convention.** This is the first `Base.getproperty` override in `src/` (`git log --all -S"Base.getproperty" -- src` returns only a6706acde; the only precedent is a JSON wrapper in `tools/phylo_latent/compare_phylo_latent_p1.jl:22`). Defining it in a separate file from the struct is unusual but forced by the hash guard, and the include order (`phylo_latent_precision.jl` after `precision_multivariate_fit.jl`, before `phylo_latent.jl`, `src/GLLVModels.jl:180`) is correct. Aqua's piracy check is unaffected (own type).

5. **Tests (Julia 1.10.12).**
   - As CI runs P1 twins (`julia --project=. <file>`): `test/test_phylo_latent_paired_p1.jl` 113/113 pass; `test/parity/p1_pin_sentinel.jl` 2/2.
   - Scratch env (dev + JSON3/StableRNGs): `test_phylo_latent_twin.jl` 89/89; `test_destination_b_phylo_uncertainty.jl` 24/24; `test_destination_b_phylo_independent_receipt.jl` 18/18 (+1 broken, opt-in by design); `test_precision_multivariate_fit.jl` 47/47; `test_destination_b_postfit.jl` 29/29; `test_destination_b_fixed_effects.jl` 25/25; `test_bridge_precision_multivariate.jl` 43+5; `test_precision_shared_residual.jl` 19/19. All exit 0.

6. **Docs (as `Documenter.yml` runs them, Julia 1.13).** `tools/check_reader_surface.py --landing-contract` exit 0; `julia --project=docs docs/make.jl` exit 0 (only the usual "could not auto-detect deploy env"/Vitepress default warnings); `check_reader_surface.py --rendered docs/build/1` → `READER_SURFACE_PASS rendered_pages=33`. The README change removes the `dev-log` token the `process-terminology` regex (`tools/check_reader_surface.py:19`) rejects; no other `dev-log`/`lane`/`worktree` tokens remain in `README.md` or `docs/src/*.md`.

7. **Out of scope of the two fix commits, noted for the merger:** `src/destination_b_postfit.jl` differs from `origin/main` (earlier PR commit 16b5fe9b5 adds a `:phy` alias for `:phylo` and a new error string). It is not one of the three hashed files, and its tests pass, so it is not a guard concern; just be aware the PR's diff against main is not limited to new files.

## Simpler alternative (judgement: current fix is acceptable)

Plain accessor functions `species_labels(fit)` / `tip_labels(fit)` in `phylo_latent_precision.jl` would give the same result with no `getproperty` override, no `propertynames` extension, and no "virtual field" surprise for future readers; cost is changing `test_phylo_latent_twin.jl:139,220,221` and the `phylo_latent_postfit.jl` consumers. A wrapper type (`PhyloLatentFit(fit, species_labels, tip_labels)`) is cleaner still but a bigger diff and would break `fit isa PrecisionMultivariateFit` dispatch in the postfit. The override is the least-invasive route that keeps the twin's `fit.species_labels` surface; type stability and cost are demonstrably unaffected. Recommend the accessor-function form as an optional follow-up, not a merge condition.

## What I did not check

- The full `Pkg.test()` suite (Aqua/JET) on this head; only the files listed above.
- Julia 1.13 (CI `version: '1'`) for the test files — only for the docs build.
- The R-backed P1 twin path (`test/parity/runparity.jl` with RCall); not part of the fixed jobs.
- Whether `extract_phylo_signal` (`src/phylo_latent_postfit.jl`) semantics are correct — only that it still compiles and its tests pass.
- Any merge-conflict content in 90c5caac9 beyond the three hashed files and the two fix commits.

Scratch artefacts: benchmark and job logs under the session scratchpad; review worktree `GLLVM.jl-review-547c` removed.

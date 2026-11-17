## Design note: convergence status at a Heywood boundary (P182 / #1284)

**Status of the names: PROPOSED, answer pending from Brian.** Every name below lives in one place (`src/fit.jl`, the `_HEYWOOD_*` block and `convergence_status`), so a rename is a one-line change.

### Rule being implemented
`converged = true` only at an interior optimum. Bound, Heywood and active-clamp each get their own status plus a warning naming the parameter (SHINICHI-0545 / BRIAN-1430-1117). This PR covers the Heywood case on the Normal path only.

### How gllvmTMB reports status today (read-only check)
- `gllvmTMB` has no status enum. It returns a boolean `converged` (optimiser code 0, finite objective, gradient below tolerance, no premature stop) plus separate flags: `pd_hessian`, `max_gradient`, and `boundary_flags` strings (`variance_near_zero...`, `correlation_near_boundary...`) in `gllvmTMB_diagnose` / `sanity_multi` (R/diagnose.R).
- GLLVModels.jl already mirrors that: `fit.converged::Bool`, and `fit_diagnostic_table` rows `converged`, `variance_boundary`, `correlation_boundary` (`:pass`/`:fail`/`:unavailable`). Precedent for boundaries (Student-t `nu_boundary`, Tweedie `:power_at_boundary`): the boundary forces `converged = false` and a separate field carries the reason.

### Proposal (identical on both sides)
| Item | Julia | R (`gllvmTMB`, for later alignment) |
|---|---|---|
| Boolean | `fit.converged` (false at a boundary) | `converged` (FALSE at a boundary) |
| Status field | `convergence_status(fit)::Symbol` | `convergence_status` (character), same values |
| Values | `:converged`, `:heywood`, `:not_converged`; reserved for later PRs: `:at_bound`, `:active_clamp` | `"converged"`, `"heywood"`, `"not_converged"`, reserved `"at_bound"`, `"active_clamp"` |
| Parameter list | `heywood_traits(fit)::Vector{Int}` | `heywood_traits` (integer vector) |
| Warning | `@warn` naming the trait index and the remedy | same text |

Why a derived accessor rather than a new `GllvmFit` field: `GllvmFit` is rebuilt at five call sites (intercept wrapper, AGHQ wrapper, loading profile, bridge); a stored field would need all of them changed and could go stale. `convergence_status` is computed from `converged`, `Λ` and `σ_eps`, so it stays correct on every rebuilt copy. If the answer is a stored field, the accessor becomes a field read.

### Heywood definition used
Communality `(ΛΛ')_tt / ((ΛΛ')_tt + σ_eps²) > 0.999` for any trait t (the cut quoted in #1284). Screened only on the plain shared-`σ_eps` model; fits with per-trait B/W variances or a phylogenetic block return no Heywood trait for now (their uniqueness has more parts). The `pervar = true` path (#736) is unchanged.

### Not covered by this PR
- The "gradient + Hessian interior check" half of the rule is not added here; `converged` is only tightened (never loosened).
- `:at_bound` and `:active_clamp` statuses, the other Normal sub-paths, and the R side.
- Warnings are not de-duplicated, so a bootstrap that refits many Heywood draws will warn each time.

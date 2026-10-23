# Post-fit twin of gllvmTMB's `extract_phylo_signal()` for the phylo_latent
# twin (`fit_phylo_latent_gllvm`), at gllvmTMB P1 (R/extract-omega.R:468-640).
# `extract_Sigma(fit; level = :phy)` is served by the existing
# PrecisionMultivariateFit method in `destination_b_postfit.jl`, where `:phy`
# is R's level name and `:phylo` the earlier Julia spelling.

"""
    extract_phylo_signal(fit::PrecisionMultivariateFit; ci = false) -> NamedTuple

Per-trait phylogenetic signal of a [`fit_phylo_latent_gllvm`](@ref) (or any
[`PrecisionMultivariateFit`](@ref)) fit, R's `extract_phylo_signal()` point
estimate: `H2[t] = Sigma_phy[t,t] / V_eta[t]` with
`V_eta[t] = Sigma_phy[t,t] + Sigma_non[t,t] + Psi_non[t,t]`, the
species-level latent variance. The observation residual is not part of this
estimand. This fit has no species-level non-phylogenetic component, so
`Sigma_non = Psi_non = 0`, `H2 = 1` for every trait, `C2_non = Psi = 0`, and
`V_eta = diag(Sigma_phy)` (including the phylogenetic unique variance when
`unique = true`), exactly as R reports for a bare `phylo_latent()` fit.

Returns a column table `(trait, H2, C2_non, Psi, V_eta)`. `ci = true` is
refused: phylogenetic-signal intervals are outside the A14/A15 first-order
scope, and R itself labels their coverage exploratory.
"""
function extract_phylo_signal(fit::PrecisionMultivariateFit; ci::Bool = false)
    ci && throw(ArgumentError("phylogenetic signal intervals are outside the " *
        "A14/A15 first-order scope (GJL-GATE-PHYLO-LATENT-SIGNAL-CI)"))
    total = extract_Sigma(fit; level = :phy, part = :total).Sigma
    v_eta = collect(Float64, diag(total))
    n = length(v_eta)
    h2 = [v > 0 ? v / v : NaN for v in v_eta]
    return (trait = ["trait$(t)" for t in 1:n], H2 = h2, C2_non = zeros(n),
            Psi = zeros(n), V_eta = v_eta)
end

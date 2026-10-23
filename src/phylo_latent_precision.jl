# Observation and tip labels on a `PrecisionMultivariateFit`, used by the
# `fit_phylo_latent_gllvm` twin (`phylo_latent.jl`).
#
# The labels are derived from the fit's own precision rather than stored as
# struct fields: `precision_multivariate_fit.jl` is hashed by the frozen
# destination-B receipts (`tools/destination_b/compare_phylo_uncertainty.jl`)
# and must stay byte-identical. `fit.tip_labels[t]` names tip `t` (the
# precision's node label for that tip) and `fit.species_labels[o]` names the
# tip observation `o` maps to.

function _pmv_default_labels(phy::PrecisionPhy, species_id::AbstractVector{<:Integer})
    tips = String[phy.node_labels[i] for i in phy.species_aug_id]
    return tips[species_id], tips
end

function Base.getproperty(fit::PrecisionMultivariateFit, name::Symbol)
    if name === :species_labels
        return first(_pmv_default_labels(getfield(fit, :phy), getfield(fit, :species_id)))
    elseif name === :tip_labels
        return last(_pmv_default_labels(getfield(fit, :phy), getfield(fit, :species_id)))
    end
    return getfield(fit, name)
end

Base.propertynames(::PrecisionMultivariateFit, private::Bool = false) =
    (fieldnames(PrecisionMultivariateFit)..., :species_labels, :tip_labels)

# Check that `fit`'s derived labels are the caller's observation and tip labels.
# The twin builds its precision so that they always agree; a mismatch is a bug.
function _pmv_with_labels(fit::PrecisionMultivariateFit,
        species_labels::Vector{String}, tip_labels::Vector{String})
    length(species_labels) == length(fit.species_id) ||
        throw(DimensionMismatch("species_labels must have one entry per observation"))
    length(tip_labels) == fit.phy.n_leaves ||
        throw(DimensionMismatch("tip_labels must have one entry per tip"))
    (fit.species_labels == species_labels && fit.tip_labels == tip_labels) ||
        error("internal: phylo_latent precision labels disagree with the caller's labels")
    return fit
end

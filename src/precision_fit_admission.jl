# One-shot admission at precision-fitting boundaries.  `PrecisionPhy` remains
# a permissive diagnostic/raw container; this validator establishes the
# stronger invariant required before a public fitting route consumes it.

# Match the existing flat-payload checksum tolerance in `bridge.jl` without
# depending on that late-included bridge file.
const _PRECISION_FIT_LOGDET_TOL = 1e-8

@noinline function _precision_fit_input_gate(tag::AbstractString, msg::AbstractString)
    throw(ArgumentError("$(tag): $(msg)"))
end

"""
    _validate_precision_fit_input(phy::PrecisionPhy) -> PrecisionPhy{Float64}

Validate one precision bundle at a fitting boundary and return an owned,
canonical snapshot. This is deliberately stricter than the raw
`PrecisionPhy` constructor: it verifies dimensions, maps, labels, finite
metadata, sparse symmetry, positive definiteness, and the shipped
`logdet(Q)` checksum. `scale` is validated as positive metadata only; it is
already reflected in `Q` and is never reapplied. No covariance is inverted.
"""
function _validate_precision_fit_input(phy::PrecisionPhy)
    n_leaves = phy.n_leaves
    n_aug = phy.n_aug
    n_leaves > 0 || _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-DIM",
        "n_leaves must be positive")
    n_aug >= n_leaves || _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-DIM",
        "n_aug ($(n_aug)) must be >= n_leaves ($(n_leaves))")

    Q = phy.Q
    size(Q) == (n_aug, n_aug) ||
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-DIM",
            "Q has size $(size(Q)); expected ($(n_aug), $(n_aug))")
    all(isfinite, nonzeros(Q)) ||
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-NONFINITE",
            "Q entries must be finite")

    tip_map = phy.species_aug_id
    length(tip_map) == n_leaves ||
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-TIPMAP",
            "species_aug_id length ($(length(tip_map))) must equal n_leaves ($(n_leaves))")
    (all(i -> 1 <= i <= n_aug, tip_map) && length(unique(tip_map)) == n_leaves) ||
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-TIPMAP",
            "species_aug_id must be a unique 1-based map into 1:n_aug")

    labels = phy.node_labels
    length(labels) == n_aug ||
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-LABEL",
            "node_labels length ($(length(labels))) must equal n_aug ($(n_aug))")
    all(!isempty, labels) ||
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-LABEL",
            "node_labels must be non-empty strings")

    isfinite(phy.log_det) ||
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-NONFINITE",
            "log_det must be finite")
    isfinite(phy.scale) && phy.scale > 0 ||
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-SCALE",
            "scale must be finite and positive metadata")

    # Check the stored sparse matrix itself before any `Symmetric` wrapper.
    # The latter would otherwise select one triangle and hide a malformed
    # payload that could still yield a positive-definite downstream system.
    try
        _symmetric_sparse_admit(Q, "precision Q")
    catch err
        err isa ArgumentError || rethrow()
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-SYMMETRY", err.msg)
    end
    factor = try
        cholesky(Symmetric(Q); check = false)
    catch err
        err isa InterruptException && rethrow()
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-PD",
            "Q could not be factored as a positive-definite sparse precision")
    end
    issuccess(factor) || _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-PD",
        "Q must be positive definite before augmented likelihood construction")

    recomputed, shipped, abs_diff = try
        precision_logdet_check(phy)
    catch err
        err isa InterruptException && rethrow()
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-LOGDET",
            "could not independently verify shipped log_det")
    end
    abs_diff <= _PRECISION_FIT_LOGDET_TOL ||
        _precision_fit_input_gate("GJL-GATE-PRECISION-FIT-LOGDET",
            "shipped log_det ($(shipped)) disagrees with recomputed ($(recomputed)) by $(abs_diff)")

    # Copy every mutable field.  This preserves the canonical Q and its
    # recorded scale exactly; no second scale multiplication or covariance
    # inversion is permitted in this admission step.
    return PrecisionPhy{Float64}(n_leaves, n_aug,
        SparseMatrixCSC{Float64,Int}(Q), Float64(phy.log_det), Float64(phy.scale),
        copy(tip_map), copy(labels))
end

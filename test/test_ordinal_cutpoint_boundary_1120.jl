# Collapsed ordinal cutpoint increment counts as boundary (twin of gllvmTMB #1833 / #1397).
using GLLVModels, Test

_ord_shared(τ; conv = true, ll = -10.0) =
    OrdinalFit(ones(3, 1), τ, length(τ) + 1, LogitLink(), ll, conv, 5)
_ord_pt(τ; conv = true) =
    OrdinalPerTraitFit(ones(2, 1), zeros(2), τ, fill(size(τ, 2) + 1, 2), LogitLink(), -10.0, conv, 5)

@testset "ordinal cutpoint boundary (#1120)" begin
    healthy = [0.0, 0.8, 1.9]
    collapsed = [0.0, 0.8, 0.8 + exp(-21.5)]   # log-increment -21.5, as in the R twin test
    @test isempty(boundary_params(_ord_shared(healthy)))
    @test convergence_status(_ord_shared(healthy)) === :converged
    @test boundary_params(_ord_shared(collapsed))[1].kind === :ordinal_cutpoint
    @test convergence_status(_ord_shared(collapsed)) === :boundary
    # exactly coinciding cutpoints
    @test convergence_status(_ord_shared([0.0, 0.5, 0.5])) === :boundary
    # just above the -15 floor is not a boundary
    @test convergence_status(_ord_shared([0.0, exp(-14.0)])) === :converged

    # per-trait with NaN padding: only the collapsed trait is named
    τ = [0.0 0.9 NaN; 0.0 0.7 0.7 + exp(-20.0)]
    @test [b.name for b in boundary_params(_ord_pt(τ))] == ["trait 2 ordinal cutpoints"]
    @test convergence_status(_ord_pt(τ)) === :boundary
    @test convergence_status(_ord_pt([0.0 0.9 NaN; 0.0 0.7 1.5])) === :converged

    # not_converged is decided before boundary
    @test convergence_status(_ord_shared(collapsed; conv = false)) === :not_converged
    @test convergence_status(_ord_shared(collapsed; ll = -Inf)) === :not_converged
end

# nobs(fit, y) for the fits that are made from a response VECTOR (#1164).
# `nobs(fit)` on these fits throws an ArgumentError that names `nobs(fit, Y)`; the fitters take
# a vector `y`, and `nobs(fit, y)` with that vector used to be a MethodError, so the error named a
# call that did not exist. The count must equal what the matrix method gives for the same data.

using GLLVModels, Test
using GLLVModels: StatsAPI

@testset "nobs(fit, y) with a response vector (#1164)" begin
    nw = "((((t1:0.6,t2:0.6):0.3,t3:0.9):0.4,(t4:0.5,t5:0.5):0.8):0.3,((t6:0.6,t7:0.6):0.5,t8:1.1):0.5);"
    y = [0.3, -0.4, 1.1, 0.2, -0.9, 0.5, 0.8, -0.1]
    fits = (
        phylo_gaussian = fit_phylo_gaussian(augmented_phy(nw), y),
        branch_re = fit_branch_re(edge_phy(nw), y),
        branch_re_dense = fit_branch_re_dense(edge_phy(nw), y),
        multinomial = fit_multinomial_gllvm(repeat([1, 2, 3, 2], 10)),
    )
    ys = (phylo_gaussian = y, branch_re = y, branch_re_dense = y, multinomial = repeat([1, 2, 3, 2], 10))
    for k in keys(fits)
        @testset "$k" begin
            fit, yy = fits[k], ys[k]
            # the error that promises the call must name a call that works
            err = try; StatsAPI.nobs(fit); nothing; catch e; e; end
            @test err isa ArgumentError
            @test occursin("nobs(fit, y)", sprint(showerror, err))
            @test StatsAPI.nobs(fit, yy) == length(yy)
            @test StatsAPI.nobs(fit, yy) == StatsAPI.nobs(fit, reshape(yy, :, 1))
            @test StatsAPI.nobs(fit, yy; mask = trues(length(yy), 1)) == length(yy)
            @test StatsAPI.nobs(fit, yy; mask = [trues(length(yy) - 2); false; false]) == length(yy) - 2
        end
    end
    # a vector with a missing cell: counted like the matrix method counts it
    @test StatsAPI.nobs(fits.phylo_gaussian, Union{Missing,Float64}[y[1:7]; missing]) == 7
    # the BIC form for these fits is then reachable with the count
    @test isfinite(StatsAPI.bic(fits.phylo_gaussian, StatsAPI.nobs(fits.phylo_gaussian, y)))
end

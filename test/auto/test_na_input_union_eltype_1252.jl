# `em_fa` and `fit_coevolution_gaussian` with a `Union{Missing,Float64}` input (refs #1147).
# (1) A matrix with a real `missing` cell is refused with a typed ArgumentError naming the fitter and
#     the first cell, never a raw MethodError from the eltype conversion that follows the check.
# (2) A Union-eltype matrix with no missing cell gives bit-identical results to the plain matrix
#     (it is converted to the plain eltype once, after the check; on Julia 1.10 a Union eltype would
#     otherwise take the generic matmul kernel and differ in the last digits).
using Test, GLLVModels, Random, LinearAlgebra

@testset "Union eltype input on em_fa and fit_coevolution_gaussian (#1147)" begin
    rng = MersenneTwister(1252)
    p, n = 6, 40
    Y = 0.8 .* randn(rng, p, 1) * randn(rng, 1, n) .+ 0.5 .* randn(rng, p, n)
    Kstar = let A = randn(rng, n, n); A * A' ./ n + I; end

    U(M) = Matrix{Union{Missing,eltype(M)}}(M)
    thrown(f) = try; f(); nothing; catch e; e; end
    quiet(f) = redirect_stderr(devnull) do; f(); end

    cases = (
        ("em_fa", y -> em_fa(y, 1), r -> (r[1], r[2], r[3], r[4], r[5])),
        ("fit_coevolution_gaussian", y -> fit_coevolution_gaussian(y, Kstar; d = 1),
            r -> (r.Λ, r.σ, r.logLik, r.converged)),
    )

    for (name, call, parts) in cases
        @testset "$name" begin
            for cell in ((1, 2), (4, 17))
                Z = U(Y); Z[cell...] = missing
                e = thrown(() -> quiet(() -> call(Z)))
                @test e isa ArgumentError
                @test e isa ArgumentError && occursin(name, e.msg) &&
                    occursin("(row, column) = ($(cell[1]), $(cell[2]))", e.msg) &&
                    occursin("1 missing cell(s)", e.msg)
            end
            Z = U(Y); Z[2, 3] = missing; Z[5, 9] = missing
            e = thrown(() -> quiet(() -> call(Z)))
            @test e isa ArgumentError && occursin("2 missing cell(s)", e.msg) &&
                occursin("(row, column) = (2, 3)", e.msg)

            # Union eltype, nothing missing: fits, and equals the plain Float64 fit exactly
            Zu = U(Y)
            @test eltype(Zu) === Union{Missing,Float64} && !any(ismissing, Zu)
            base = quiet(() -> call(Y))
            got = quiet(() -> call(Zu))
            @test parts(got) == parts(base)
            @test all(isfinite, parts(base)[3])
            @test Zu isa Matrix{Union{Missing,Float64}}      # the caller's matrix is not modified
        end
    end

    @testset "em_fa NaN and Inf are still refused after the check" begin
        for v in (NaN, Inf)
            Yb = copy(Y); Yb[1, 2] = v
            e = thrown(() -> em_fa(Yb, 1))
            @test e isa ArgumentError && occursin("(row, column) = (1, 2)", e.msg)
            Zb = U(Yb)
            e = thrown(() -> em_fa(Zb, 1))
            @test e isa ArgumentError && occursin("(row, column) = (1, 2)", e.msg)
        end
    end
end

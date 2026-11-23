# Native confint on per-trait ordinal fits (#1187 doc parity).

using GLLVModels, Random, Test

@testset "per-trait ordinal native confint (#1187)" begin
    Random.seed!(81)
    p, K, n, C = 4, 1, 120, 4
    β = 0.2 .* randn(p)
    Λ = 0.55 .* ones(p, K)
    τ_row = [0.0, 0.8, 1.7]
    Y = Matrix{Int}(undef, p, n)
    for s in 1:n
        η = β .+ Λ * randn(K)
        for t in 1:p
            u = rand()
            cum = 0.0
            cat = C
            for c in 1:C
                Fhi = c == C ? 1.0 : inv(1 + exp(-(τ_row[c] - η[t])))
                Flo = c == 1 ? 0.0 : inv(1 + exp(-(τ_row[c - 1] - η[t])))
                cum += Fhi - Flo
                if u <= cum
                    cat = c
                    break
                end
            end
            Y[t, s] = cat
        end
    end
    fit = fit_ordinal_gllvm_pertrait(Y; K = K, iterations = 80)
    @test fit.converged
    ci = confint(fit, Y; method = :wald)
    @test all(isfinite, ci.lower)
    @test all(isfinite, ci.upper)
    @test "tau[1,2]" in ci.term
end
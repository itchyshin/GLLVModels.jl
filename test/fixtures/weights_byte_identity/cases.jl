# Shared case list for the weights byte-identity guard. Included by gen_main.jl (run against a
# detached worktree of origin/main, which has no `weights` keyword) and by
# test/test_weights_byte_identity.jl (run on the branch). `kw` is splatted into every fit: `(;)` on
# main and on the branch, `(; weights = nothing)` on the branch only. Small designs, seconds each.
using GLLVModels
using Distributions: Poisson, NegativeBinomial, Binomial, Normal
using Random

function _wbi_data()
    rng = Xoshiro(20261005)
    p, n = 5, 40
    z = randn(rng, n)
    lam = [0.6, -0.4, 0.5, 0.3, -0.5]
    mu = [0.8, 0.4, 0.6, 1.0, 0.5]
    eta = mu .+ lam .* z'                                  # p x n
    Ypois = [rand(rng, Poisson(exp(eta[t, s]))) for t in 1:p, s in 1:n]
    Ynb = [rand(rng, NegativeBinomial(3.0, 3.0 / (3.0 + exp(eta[t, s])))) for t in 1:p, s in 1:n]
    N = fill(5, p, n)
    Ybin = [rand(rng, Binomial(5, 1 / (1 + exp(-(eta[t, s] - 0.7))))) for t in 1:p, s in 1:n]
    Ygau = eta .+ 0.5 .* randn(rng, p, n)
    Ydg = [rand(rng) < 0.3 ? 0.0 : exp(eta[t, s] + 0.3 * randn(rng)) for t in 1:p, s in 1:n]
    mask = trues(p, n); mask[2, 3] = false; mask[4, 17] = false; mask[1, 30] = false
    O = log.(0.5 .+ 2.5 .* rand(rng, p, n))
    return (; p, n, Ypois, Ynb, N, Ybin, Ygau, Ydg, mask, O)
end

# name => thunk(kw) returning a fit (or a Float64 for the marginal case)
function _wbi_cases()
    d = _wbi_data()
    Λ0 = reshape([0.5, -0.3, 0.4, 0.2, -0.4], 5, 1)
    β0 = [0.7, 0.3, 0.5, 0.9, 0.4]
    return [
        "poisson_analytic" => kw -> fit_gllvm(d.Ypois; family = Poisson(), K = 1, kw...),
        "poisson_masked"   => kw -> fit_gllvm(d.Ypois; family = Poisson(), K = 1, mask = d.mask, kw...),
        "poisson_offset"   => kw -> fit_gllvm(d.Ypois; family = Poisson(), K = 1, offset = d.O, kw...),
        "poisson_direct"   => kw -> fit_poisson_gllvm(d.Ypois; K = 1, kw...),
        "poisson_aghq"     => kw -> fit_poisson_gllvm(d.Ypois; K = 1, aghq = 3, kw...),
        "nb2_shared"       => kw -> fit_gllvm(d.Ynb; family = NegativeBinomial(1.0, 0.5), K = 1, kw...),
        "nb2_grouped"      => kw -> fit_gllvm(d.Ynb; family = NegativeBinomial(1.0, 0.5), K = 1, disp_group = :species, kw...),
        "binomial"         => kw -> fit_gllvm(d.Ybin; family = Binomial(), K = 1, N = d.N, kw...),
        "gaussian"         => kw -> fit_gllvm(d.Ygau; family = Normal(), K = 1, kw...),
        "delta_gamma"      => kw -> fit_gllvm(d.Ydg; family = DeltaGamma(), K = 1, kw...),
        "poisson_marginal" => kw -> GLLVModels.poisson_marginal_loglik_laplace(d.Ypois, Λ0, β0; kw...),
        "poisson_marginal_masked" => kw -> GLLVModels.poisson_marginal_loglik_laplace(d.Ypois, Λ0, β0; mask = d.mask, kw...),
    ]
end

# Every Float64 / Int / Bool scalar or Float64 array field of a fit, as Float64 bit patterns (hex).
# Wall-clock fields (the Gaussian fit's `cputime`) are not results and are skipped.
const _WBI_CLOCK_FIELDS = ("cputime", "elapsed", "time", "walltime")
_wbi_bits(x::Float64) = [string(reinterpret(UInt64, x); base = 16)]
_wbi_bits(x::AbstractArray{Float64}) = [string(reinterpret(UInt64, v); base = 16) for v in vec(x)]
function _wbi_record(f)
    f isa Float64 && return Dict("value" => _wbi_bits(f))
    out = Dict{String, Any}()
    for nm in fieldnames(typeof(f))
        string(nm) in _WBI_CLOCK_FIELDS && continue
        v = getfield(f, nm)
        if v isa Float64 || v isa AbstractArray{Float64}
            out[string(nm)] = _wbi_bits(v)
        elseif v isa Bool || v isa Int
            out[string(nm)] = [string(v)]
        end
    end
    return out
end

_wbi_fingerprint() = string(VERSION, " | ", Sys.MACHINE, " | ", Sys.CPU_NAME, " | ",
    GLLVModels.LinearAlgebra.BLAS.get_config(), " | blas_threads=", GLLVModels.LinearAlgebra.BLAS.get_num_threads())

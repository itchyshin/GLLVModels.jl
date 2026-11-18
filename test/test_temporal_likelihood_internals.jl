# Internals of src/temporal_likelihood.jl checked against independent dense
# linear algebra (no call into the code under test to build the reference):
#   * `_temporal_block_nll_lowrank` vs a dense Cholesky of the explicitly built
#     covariance, with and without masked (series, time, trait) rows;
#   * `_temporal_tier_block` / `_temporal_trait_block`: symmetric PSD blocks
#     with the stated rank structure, against a hand-packed loading matrix;
#   * `_temporal_parameter_names` length vs the vector `temporal_marginal_nll`
#     accepts;
#   * `_temporal_cov_pieces` and `_temporal_series_rows` against hand-built
#     expectations.
using Test, GLLVModels, LinearAlgebra, Random, StatsModels
const GMI = GLLVModels

# Long-format panel: ns series x nt occasions x p traits. `drop` lists
# (series, occasion) pairs removed whole (a missing occasion).
function _tli_panel(rng, ns, nt, p; drop=Tuple{Int,Int}[], times=1:nt)
    cols = (series=String[], trait=String[], occasion=Float64[], value=Float64[])
    for s in 1:ns, (ti, t) in enumerate(times), j in 1:p
        (s, ti) in drop && continue
        push!(cols.series, "s$s"); push!(cols.trait, "t$j")
        push!(cols.occasion, Float64(t)); push!(cols.value, randn(rng))
    end
    return cols
end

function _tli_spec(cols; structure=:ar1, mode=:latent, unique=false)
    term = mode === :latent ? temporal_latent(:(0 + trait | series), :occasion; d=1,
            structure=structure, unique=unique) :
        mode === :dep ? temporal_dep(:(0 + trait | series), :occasion; structure=structure) :
        temporal_indep(:(0 + trait | series), :occasion; structure=structure)
    s0 = GMI._parse_temporal_term(term, cols; trait=:trait, response=:value, structure=Expr[])
    return GMI._temporal_with_composition(s0,
        GMI._temporal_composition(s0, Expr[], cols, Float64.(cols.value); unit=nothing, unit_obs=nothing))
end

# Dense reference: V = (K .* (Lambda Lambda')[trait, trait']) + s2 I over `rows`,
# the temporal correlation K written out from the AR1 / OU definition.
function _tli_dense_nll(spec, a, s2, Lambda, rows, residual)
    S = Lambda * Lambda'
    m = length(rows)
    V = zeros(m, m)
    for j in 1:m, i in 1:m
        oi, oj = rows[i], rows[j]
        dt = abs(spec.row_time[oi] - spec.row_time[oj])
        k = spec.structure === :ar1 ? a^round(Int, dt) : exp(-a * dt)
        V[i, j] = k * S[spec.trait_id[oi], spec.trait_id[oj]] + (i == j ? s2 : 0.0)
    end
    F = cholesky(Symmetric(V))
    z = F.L \ residual[rows]
    return (m * log(2pi) + logdet(F) + dot(z, z)) / 2
end

@testset "temporal low-rank block nll equals dense Cholesky" begin
    rng = Xoshiro(20261020)
    p = 5
    for (structure, a) in ((:ar1, 0.55), (:ar1, -0.4), (:ou, 0.35)),
        r in (1, 2), masked in (false, true)
        # irregular occasions for OU, a whole dropped occasion in one series
        times = structure === :ar1 ? (1:6) : [1.0, 1.5, 2.5, 3.0, 4.5, 5.0]
        cols = _tli_panel(rng, 3, 6, p; drop=[(3, 4)], times=times)
        spec = _tli_spec(cols; structure=structure)
        res = randn(rng, length(cols.value))
        Lambda = 0.8 .* randn(rng, p, r)
        s2 = 0.3
        P = (a=a, sigma2=s2)
        for rows in GMI._temporal_series_rows(spec)
            if masked
                # mask individual rows (a time point missing for some traits only)
                rows = [o for (n, o) in enumerate(rows) if n % 7 != 3]
            end
            got = GMI._temporal_block_nll_lowrank(spec, P, Lambda, rows, res)
            @test got isa Float64
            @test got ≈ _tli_dense_nll(spec, a, s2, Lambda, rows, res) atol = 1e-10 rtol = 1e-12
        end
    end
    # not positive definite M is impossible for s2 > 0; a non-finite value returns Inf
    cols = _tli_panel(rng, 1, 5, 4)
    spec = _tli_spec(cols)
    res = randn(rng, length(cols.value))
    rows = GMI._temporal_series_rows(spec)[1]
    Lam = randn(rng, 4, 1)
    @test GMI._temporal_block_nll_lowrank(spec, (a=0.5, sigma2=NaN), Lam, rows, res) === Inf
    # no gain (nt * r >= m) falls back to the dense block
    @test GMI._temporal_block_nll_lowrank(spec, (a=0.5, sigma2=0.3), randn(rng, 4, 4), rows, res) === nothing
end

@testset "temporal_marginal_nll latent route equals dense reference" begin
    rng = Xoshiro(7)
    p = 4
    cols = _tli_panel(rng, 2, 5, p; drop=[(2, 3)])
    spec = _tli_spec(cols)
    y = Float64.(cols.value)
    _, X, _ = GMI._temporal_design(@formula(value ~ 0 + trait), cols)
    L = GMI.TemporalLayout(size(X, 2), spec)
    theta = GMI._temporal_start(y, X, L) .+ 0.1 .* randn(rng, L.total)
    Lambda = GMI.unpack_lambda(theta[L.rr], p, L.rank)
    a = (1 - 1e-6) * tanh(theta[L.time])
    s2 = exp(2 * theta[L.log_sigma])
    res = y .- X * theta[L.beta]
    want = sum(_tli_dense_nll(spec, a, s2, Lambda, rows, res) for rows in GMI._temporal_series_rows(spec))
    @test GMI.temporal_marginal_nll(theta, y, X, spec) ≈ want atol = 1e-10 rtol = 1e-12
end

# Hand-packed lower-triangular loadings: theta[1:K] = diag, then strict lower
# entries column by column (src/packing.jl layout).
function _tli_pack(theta, p, K)
    Lam = zeros(p, K)
    for k in 1:K
        Lam[k, k] = theta[k]
    end
    c = K
    for k in 1:K, i in (k + 1):p
        c += 1
        Lam[i, k] = theta[c]
    end
    return Lam
end

@testset "tier and trait blocks are symmetric PSD with the stated rank" begin
    rng = Xoshiro(11)
    for p in (3, 5), rk in (0, 1, 2), dg in (false, true)
        rk == 0 && !dg && continue
        nrr = rk > 0 ? GMI.rr_theta_len(p, rk) : 0
        theta = randn(rng, nrr + (dg ? p : 0))
        rr = 1:nrr
        dgr = dg ? ((nrr + 1):(nrr + p)) : (1:0)
        S = GMI._temporal_tier_block(theta, p, rk, rr, dgr)
        want = zeros(p, p)
        rk > 0 && (Lam = _tli_pack(theta[rr], p, rk); want += Lam * Lam')
        dg && (want += Diagonal(exp.(2 .* theta[dgr])))
        @test S ≈ want atol = 1e-12
        @test issymmetric(S)
        @test eigmin(Symmetric(S)) >= -1e-12
        if !dg
            @test rank(S; atol=1e-9) == rk   # pure Lambda Lambda'
        else
            @test eigmin(Symmetric(S)) > 0     # diag part makes it full rank
        end
        if rk == 0
            @test S == Diagonal(diag(S))
        end
    end
    # trait block of the temporal source, all modes
    cols = _tli_panel(rng, 2, 4, 4)
    for (mode, unique) in ((:latent, false), (:latent, true), (:dep, false), (:indep, false))
        spec = _tli_spec(cols; mode=mode, unique=unique)
        L = GMI.TemporalLayout(4, spec)
        theta = randn(rng, L.total)
        Sigma, Lambda, psi = GMI._temporal_trait_block(L, theta)
        @test issymmetric(Sigma)
        @test eigmin(Symmetric(Sigma)) >= -1e-12
        want = zeros(4, 4)
        if spec.rank > 0
            Lam = _tli_pack(theta[L.rr], 4, spec.rank)
            @test Lambda ≈ Lam
            want += Lam * Lam'
        else
            @test Lambda === nothing
        end
        if spec.unique
            @test psi ≈ exp.(2 .* theta[L.diag])
            want += Diagonal(psi)
        else
            @test psi === nothing
        end
        @test Sigma ≈ want atol = 1e-12
        spec.rank > 0 && !spec.unique && @test rank(Sigma; atol=1e-9) == spec.rank
    end
end

@testset "parameter names match the vector temporal_marginal_nll expects" begin
    rng = Xoshiro(3)
    for p in (3, 4, 6), (mode, unique) in ((:latent, false), (:latent, true), (:dep, false), (:indep, false))
        cols = _tli_panel(rng, 2, 4, p)
        spec = _tli_spec(cols; mode=mode, unique=unique)
        y = Float64.(cols.value)
        _, X, _ = GMI._temporal_design(@formula(value ~ 0 + trait), cols)
        L = GMI.TemporalLayout(size(X, 2), spec)
        nm = GMI._temporal_parameter_names(L)
        @test length(nm) == L.total
        theta = GMI._temporal_start(y, X, L)
        @test length(nm) == length(theta)
        @test isfinite(GMI.temporal_marginal_nll(theta, y, X, spec))
        @test_throws DimensionMismatch GMI.temporal_marginal_nll(vcat(theta, 0.0), y, X, spec)
        @test_throws DimensionMismatch GMI.temporal_marginal_nll(theta[1:end-1], y, X, spec)
        @test count(==("b_fix"), nm) == size(X, 2)
        @test count(==("theta_temporal_rr"), nm) == (spec.rank > 0 ? GMI.rr_theta_len(p, spec.rank) : 0)
        @test count(==("theta_temporal_diag"), nm) == (spec.unique || mode === :indep ? p : 0)
        @test count(==("theta_temporal_time"), nm) == 1
    end
    # composed layouts: every optional block adds its own coordinates
    p = 4
    tierB = GMI.TemporalOrdinaryTier(:B, :latent, 2, true, :unit)
    tierW = GMI.TemporalOrdinaryTier(:W, :indep, 0, true, :obs)
    for sigma_fixed in (nothing, 1.0)
        comp = GMI.TemporalComposition(tierB, tierW, :g, :unit, :obs, String[], Int[], String[], Int[], Int[],
            sigma_fixed, Any[])
        L = GMI.TemporalLayout(p, p, 1, false, comp)
        nm = GMI._temporal_parameter_names(L)
        @test length(nm) == L.total
        @test count(==("log_sigma_eps"), nm) == (sigma_fixed === nothing ? 1 : 0)
        @test count(==("theta_rr_B"), nm) == GMI.rr_theta_len(p, 2)
        @test count(==("theta_diag_B"), nm) == p
        @test count(==("theta_rr_W"), nm) == 0
        @test count(==("theta_diag_W"), nm) == p
        @test count(==("log_sigma_re_int"), nm) == 1
    end
end

@testset "cov pieces and series rows" begin
    rng = Xoshiro(5)
    p = 4
    for structure in (:ar1, :ou)
        cols = _tli_panel(rng, 3, 4, p)
        spec = _tli_spec(cols; structure=structure, unique=true)
        L = GMI.TemporalLayout(p, spec)
        theta = 0.3 .* randn(rng, L.total)
        pc = GMI._temporal_cov_pieces(theta, spec, L)
        @test pc.a ≈ (structure === :ar1 ? (1 - 1e-6) * tanh(theta[L.time]) : exp(theta[L.time]))
        @test pc.sigma2 ≈ exp(2 * theta[L.log_sigma])
        @test pc.Sigma ≈ GMI._temporal_trait_block(L, theta)[1]
        @test pc.SB === nothing && pc.SW === nothing && pc.re2 === nothing
        # uncomposed: one block per series, rows in data order
        blocks = GMI._temporal_series_rows(spec)
        @test length(blocks) == 3
        @test sort(reduce(vcat, blocks)) == 1:length(spec.row_series)
        for b in blocks
            @test length(unique(spec.row_series[b])) == 1
            @test issorted(b)
        end
        # dense covariance has exactly the block structure
        V = GMI._temporal_covariance(theta, spec, p)
        @test issymmetric(V)
        @test eigmin(Symmetric(V)) > 0
        for i in eachindex(spec.row_series), j in eachindex(spec.row_series)
            spec.row_series[i] == spec.row_series[j] || @test V[i, j] == 0
        end
    end
    # composed: a shared unit links series s1 and s2; s3 stays alone
    cols = _tli_panel(rng, 3, 4, p)
    spec0 = _tli_spec(cols)
    n = length(spec0.row_series)
    unit = [s == "s3" ? 2 : 1 for s in spec0.row_series]
    tierB = GMI.TemporalOrdinaryTier(:B, :latent, 2, true, :unit)
    comp = GMI.TemporalComposition(tierB, nothing, nothing, :unit, nothing, ["u1", "u2"], unit,
        String[], Int[], Int[], nothing, Any[])
    spec = GMI._temporal_with_composition(spec0, comp)
    blocks = GMI._temporal_series_rows(spec)
    @test length(blocks) == 2
    @test Set(Set.(blocks)) == Set([Set(findall(!=("s3"), spec.row_series)), Set(findall(==("s3"), spec.row_series))])
    L = GMI.TemporalLayout(p, spec)
    theta = 0.3 .* randn(rng, L.total)
    pc = GMI._temporal_cov_pieces(theta, spec, L)
    @test pc.SB ≈ GMI._temporal_tier_block(theta, p, 2, L.rr_B, L.diag_B)
    @test pc.SW === nothing && pc.re2 === nothing
    @test eigmin(Symmetric(pc.SB)) > 0
end

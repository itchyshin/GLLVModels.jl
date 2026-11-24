# Coverage: response-support and covariate finiteness helpers, plus the
# maskless Gaussian NA refusal. Direct calls only; no fit.
using Test
using GLLVModels
using Distributions
using Logging

grab(f) = try
    f()
    nothing
catch e
    e
end

warn_text(f) = begin
    buf = IOBuffer()
    with_logger(ConsoleLogger(buf, Logging.Warn)) do
        f()
    end
    String(take!(buf))
end

@testset "response support and NA helpers (1124e)" begin
    @testset "support predicate and text" begin
        @test GLLVModels._in_support(:beta, 0.5, 1)
        @test !GLLVModels._in_support(:beta, 0.0, 1)
        @test !GLLVModels._in_support(:beta, 1.0, 1)
        @test !GLLVModels._in_support(:beta, NaN, 1)
        @test GLLVModels._in_support(:positive, 1e-8, 1)
        @test !GLLVModels._in_support(:positive, 0.0, 1)
        @test GLLVModels._in_support(:nonneg_real, 0.0, 1)
        @test !GLLVModels._in_support(:nonneg_real, -1e-12, 1)
        @test GLLVModels._in_support(:unit_closed, 0.0, 1)
        @test GLLVModels._in_support(:unit_closed, 1.0, 1)
        @test !GLLVModels._in_support(:unit_closed, 1.0000001, 1)
        @test GLLVModels._in_support(:finite, -3.0, 1)
        @test !GLLVModels._in_support(:finite, Inf, 1)
        @test !GLLVModels._in_support(:finite, missing, 1)
        @test GLLVModels._in_support(:count, 0, 1)
        @test !GLLVModels._in_support(:count, 1.5, 1)
        @test !GLLVModels._in_support(:count, -1, 1)
        @test GLLVModels._in_support(:binomial, 2, 3)
        @test !GLLVModels._in_support(:binomial, 4, 3)
        @test GLLVModels._support_text(:beta) == "Beta support, the open interval (0, 1)"
        @test GLLVModels._support_text(:positive) == "positive support, y > 0"
        @test GLLVModels._support_text(:finite) == "real support, a finite number"
        @test GLLVModels._support_text(:nonneg_real) == "non-negative real support, y >= 0"
        @test GLLVModels._support_text(:unit_closed) == "closed unit interval [0, 1]"
        @test GLLVModels._support_text(:binomial) == "Binomial support, an integer 0 <= y <= N"
        @test GLLVModels._support_text(:count) == "Poisson support, a non-negative integer"
        @test GLLVModels._support_text(:other) == "Poisson support, a non-negative integer"
    end

    @testset "family kind table" begin
        @test GLLVModels._family_support_kind(Poisson()) === :count
        @test GLLVModels._family_support_kind(NegativeBinomial()) === :count
        @test GLLVModels._family_support_kind(GLLVModels.NB1()) === :count
        @test GLLVModels._family_support_kind(Binomial()) === :binomial
        @test GLLVModels._family_support_kind(Beta()) === :beta
        @test GLLVModels._family_support_kind(Gamma()) === :positive
        @test GLLVModels._family_support_kind(Exponential()) === :positive
        @test GLLVModels._family_support_kind(Normal()) === nothing
        @test GLLVModels._family_support_kind_normal(Normal()) === :finite
        @test GLLVModels._family_support_kind_normal(Gamma()) === :positive
    end

    @testset "check messages, mask, missing, row offset" begin
        chk = GLLVModels._check_response_support
        @test chk(:nonneg_real, [0.0 1.5; 2.0 0.0], nothing; caller = "c") === nothing
        e = grab(() -> chk(:nonneg_real, [0.0 -0.1; 1.0 2.0], nothing; caller = "tw"))
        @test e isa ArgumentError
        @test occursin("tw: Y[1, 2] = -0.1", e.msg)
        @test occursin("non-negative real support", e.msg)
        @test occursin("1 observed cell", e.msg)

        # Column-major scan: the first offending cell is the earliest (trait, site)
        # with the site index in the outer loop. Mask the only offender.
        bad_unit = [0.0 1.2; 0.4 0.5]
        @test chk(:unit_closed, [0.0 1.0; 0.5 0.2], nothing; caller = "ob") === nothing
        hidden = trues(2, 2)
        hidden[1, 2] = false
        @test chk(:unit_closed, bad_unit, hidden; caller = "ob") === nothing
        e = grab(() -> chk(:unit_closed, bad_unit, nothing; caller = "ob"))
        @test e isa ArgumentError
        @test occursin("1 observed cell(s)", e.msg)
        @test occursin("Y[1, 2] = 1.2", e.msg)

        Ym = Matrix{Union{Missing, Float64}}([1.0 missing; 2.0 3.0])
        @test chk(:finite, Ym, nothing; caller = "g") === nothing
        e = grab(() -> chk(:finite, [1.0 Inf; 0.0 NaN], nothing; caller = "g"))
        @test e isa ArgumentError
        @test occursin("real support, a finite number", e.msg)
        @test occursin("Y[1, 2] = Inf", e.msg)

        e = grab(() -> chk(:binomial, [0 2; 1 1], nothing; caller = "b"))
        @test e isa ArgumentError
        @test occursin("with N[1, 2] = 1", e.msg)
        N = [1 2; 3 4]
        e = grab(() -> chk(:binomial, [0 3; 1 0], nothing; caller = "b", N = N))
        @test e isa ArgumentError
        @test occursin("with N[1, 2] = 2", e.msg)
        @test occursin("Y[1, 2] = 3", e.msg)

        e = grab(() -> chk(:count, reshape([1.5], 1, 1), nothing; caller = "c", row0 = 4))
        @test e isa ArgumentError
        @test occursin("Y[5, 1] = 1.5", e.msg)

        @test GLLVModels._check_family_response_support(Normal(), [NaN 1.0; 0.0 1.0], nothing;
                                                        caller = "n") === nothing
        e = grab(() -> GLLVModels._check_family_response_support(Poisson(), [0 -1; 2 3], nothing;
                                                                caller = "pois"))
        @test e isa ArgumentError
        @test occursin("pois: Y[1, 2] = -1", e.msg)
        e = grab(() -> GLLVModels._check_family_response_support_normal(Normal(), [Inf 0.0; 1.0 1.0],
                                                                       nothing; caller = "gn"))
        @test e isa ArgumentError
        @test occursin("gn: Y[1, 1] = Inf", e.msg)

        e = grab(() -> GLLVModels._check_mixed_response_support([Normal(), Poisson()],
                                                               [0.0 1.0; 1.5 2.0], nothing;
                                                               caller = "mix"))
        @test e isa ArgumentError
        @test occursin("mix: Y[2, 1] = 1.5", e.msg)
        @test occursin("Poisson support", e.msg)
        @test GLLVModels._check_mixed_response_support([Normal(), Poisson()],
                                                      [0.0 1.0; 1.0 2.0], nothing;
                                                      caller = "mix") === nothing
    end

    @testset "sanitize unobserved cells" begin
        Y = [1.0 2.0; 3.0 4.0]
        @test GLLVModels._sanitize_unobserved(Y, nothing, 0.0) === Y
        Ym = Matrix{Union{Missing, Float64}}([1.0 missing; 2.0 3.0])
        filled = GLLVModels._sanitize_unobserved(Ym, nothing, 0.0)
        @test filled == [1.0 0.0; 2.0 3.0]
        @test eltype(filled) <: AbstractFloat
        mask = trues(2, 2)
        mask[2, 2] = false
        replaced = GLLVModels._sanitize_unobserved(Y, mask, -1.0)
        @test replaced[2, 2] == -1.0
        @test replaced[1, 1] == 1.0
        Yany = Matrix{Any}(undef, 1, 2)
        Yany[1, 1] = 1
        Yany[1, 2] = 2
        narrowed = GLLVModels._sanitize_unobserved(Yany, [true false], 0)
        @test narrowed == [1 0]
        @test eltype(narrowed) == Int
    end

    @testset "finite covariates: skip unobserved, plural, missing" begin
        chk = GLLVModels._check_finite_covariates
        obs = [false true; false true]
        @test chk([NaN 1.0; 2.0 3.0], "caller"; obs = obs, unit = :site, name = "X") === nothing
        e = grab(() -> chk([1.0 1.0; NaN 3.0], "caller"; obs = obs, unit = :site, name = "X"))
        @test e isa ArgumentError
        @test occursin("caller: X[2, 1] = NaN", e.msg)
        @test occursin("1 non-finite entry at observed cells", e.msg)
        e = grab(() -> chk([NaN Inf; 1.0 2.0], "caller"; obs = trues(2, 2), unit = :site, name = "X"))
        @test e isa ArgumentError
        @test occursin("2 non-finite entries at observed cells", e.msg)
        trait_obs = [false false; true true]
        TR = [NaN 0.0; 1.0 2.0]
        @test chk(TR, "caller"; obs = trait_obs, unit = :trait, name = "TR") === nothing
        Xm = Matrix{Union{Missing, Float64}}(undef, 1, 2)
        Xm[1, 1] = 1.0
        Xm[1, 2] = missing
        e = grab(() -> chk(Xm, "caller"; name = "X"))
        @test e isa ArgumentError
        @test occursin("X[1, 2] = missing", e.msg)
        @test occursin("1 non-finite entry", e.msg)
        @test !occursin("at observed cells", e.msg)
    end

    @testset "all-zero count columns" begin
        # Traits are rows. Row 2 is entirely unobserved, so it is not an all-zero species.
        quiet = [0 5; 0 0]
        hidden = [true true; false false]
        @test warn_text(() -> GLLVModels._warn_all_zero_count_columns(quiet, hidden;
                                                                     caller = "fit_cov_1124e_quiet")) == ""
        @test GLLVModels._warn_all_zero_count_columns(quiet, hidden;
                                                     caller = "fit_cov_1124e_quiet") === nothing
        text = warn_text(() -> GLLVModels._warn_all_zero_count_columns([0 0; 1 2], nothing;
                                                                      caller = "fit_cov_1124e_zero"))
        @test occursin("fit_cov_1124e_zero", text)
        @test occursin("[1]", text)
        @test !occursin("zero-inflation", text)
        zip = warn_text(() -> GLLVModels._warn_all_zero_count_columns([0 0; 1 0], nothing;
                                                                     caller = "fit_zip_gllvm"))
        @test occursin("fit_zip_gllvm", zip)
        @test occursin("[1]", zip)
        @test occursin("zero-inflation", zip)
    end

    @testset "maskless NA refusal" begin
        refuse = GLLVModels._refuse_na_response
        @test refuse("fit_demo", [1.0 2.0; 3.0 4.0]) === nothing
        @test refuse("fit_demo", [1.0 NaN; 3.0 4.0]) === nothing
        Ym = Matrix{Union{Missing, Float64}}([1.0 2.0; missing 4.0])
        Ym[2, 2] = missing
        e = grab(() -> refuse("fit_demo", Ym))
        @test e isa ArgumentError
        @test occursin("fit_demo does not accept `missing` in `Y`: 2 missing", e.msg)
        @test occursin("(2, 1)", e.msg)
        @test occursin("fit_gaussian_gllvm", e.msg)
        e = grab(() -> refuse("fit_demo", [1.0 NaN; Inf 4.0]; nan = true))
        @test e isa ArgumentError
        @test occursin("does not accept NaN in `Y`: 1 NaN", e.msg)
        @test occursin("(1, 2)", e.msg)
        e = grab(() -> refuse("fit_demo", [1.0 2.0; Inf 4.0]; nan = true))
        @test e isa ArgumentError
        @test occursin("does not accept Inf in `Y`: 1 Inf", e.msg)
        @test occursin("(2, 1)", e.msg)
        @test refuse("fit_demo", [1.0 2.0; 3.0 4.0]; nan = true) === nothing
        @test refuse("fit_demo", [1 2; 3 4]; nan = true) === nothing
    end
end

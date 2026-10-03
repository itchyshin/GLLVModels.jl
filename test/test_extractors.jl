using GLLVModels, Test, Random, LinearAlgebra, Statistics

# src/extractors.jl — post-fit extractor family (core070 missing-surface work
# order, Cluster 1). Verifies the `extract_*`/`get*` public names against
# closed-form formulas and against the existing internal generics they
# forward to (sigma_y_site, communality, correlation, proportions,
# phylo_signal, repeatability, communality_B/W, correlation_B/W, getLoadings,
# ordination).

@testset "extractors.jl — post-fit extractor family" begin

    @testset "extract_Sigma(::GllvmFit) — J1 (no W, no diag)" begin
        Random.seed!(101)
        p, K, n = 4, 2, 300
        Λ_true = 0.6 .* randn(p, K)
        y = Λ_true * randn(K, n) + 0.4 .* randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)

        out_unit = extract_Sigma(fit; level = :unit, part = :total)
        Σ_manual = fit.pars.Λ * fit.pars.Λ'
        @test out_unit.Sigma ≈ Σ_manual atol = 1e-10
        @test out_unit.R ≈ GLLVModels._cov2cor(Σ_manual) atol = 1e-10
        @test out_unit.level === :unit

        # J1 has no W tier: :unit_obs collapses to σ_eps² I.
        out_w = extract_Sigma(fit; level = :unit_obs, part = :total)
        @test out_w.Sigma ≈ (fit.pars.σ_eps^2) .* Matrix(I, p, p) atol = 1e-10

        # :B / :W legacy aliases match :unit / :unit_obs.
        @test extract_Sigma(fit; level = :B).Sigma ≈ out_unit.Sigma
        @test extract_Sigma(fit; level = :W).Sigma ≈ out_w.Sigma

        # part = :shared is the Λ Λᵀ term alone (no R field).
        shared = extract_Sigma(fit; level = :unit, part = :shared)
        @test shared.Sigma ≈ Σ_manual atol = 1e-10
        @test !haskey(shared, :R)

        # part = :unique is the diagonal alone; :unit has no σ²_B in J1 → zero.
        uniq = extract_Sigma(fit; level = :unit, part = :unique)
        @test uniq.s ≈ zeros(p) atol = 1e-12
        uniq_w = extract_Sigma(fit; level = :unit_obs, part = :unique)
        @test uniq_w.s ≈ fill(fit.pars.σ_eps^2, p) atol = 1e-10

        # :site tier matches the existing sigma_y_site generic exactly.
        @test extract_Sigma(fit; level = :site).Sigma ≈ GLLVModels.sigma_y_site(fit) atol = 1e-12

        @test_throws ArgumentError extract_Sigma(fit; level = :bogus)
        @test_throws ArgumentError extract_Sigma(fit; part = :bogus)
    end

    @testset "extract_Sigma(::GllvmFit) — J2 (W tier + diag)" begin
        Random.seed!(102)
        p, K_B, K_W, n = 4, 1, 1, 400
        Λ_B = 0.6 .* randn(p, K_B); Λ_W = 0.4 .* randn(p, K_W)
        σ_B = 0.2 .+ 0.1 .* rand(p); σ_W = 0.2 .+ 0.1 .* rand(p)
        Y = Λ_B * randn(K_B, n) .+ σ_B .* randn(p, n) .+
            Λ_W * randn(K_W, n) .+ σ_W .* randn(p, n) .+ 0.3 .* randn(p, n)
        fit = fit_gaussian_gllvm(Y; K = K_B, K_W = K_W, has_diag = true)

        out_unit = extract_Sigma(fit; level = :unit)
        Σ_B_manual = fit.pars.Λ * fit.pars.Λ' + Diagonal(fit.pars.σ²_B)
        @test out_unit.Sigma ≈ Σ_B_manual atol = 1e-8

        out_w = extract_Sigma(fit; level = :unit_obs)
        Σ_W_manual = fit.pars.Λ_W * fit.pars.Λ_W' + Diagonal(fit.pars.σ²_W) +
                     (fit.pars.σ_eps^2) .* Matrix(I, p, p)
        @test out_w.Sigma ≈ Σ_W_manual atol = 1e-8

        # extract_Omega: level = :total (legacy) sums the two extract_Sigma
        # tiers unconditionally, out_w.Sigma included (folds σ_eps² in
        # always, per extract_Sigma's own :unit_obs convention).
        Ω_total = extract_Omega(fit; level = :total)
        @test Ω_total ≈ out_unit.Sigma .+ out_w.Sigma atol = 1e-8

        # Default (level = :auto, R-tier-scoped): genuine W tier is present
        # (K_W > 0), so it sums in, but WITHOUT σ_eps² (R never folds
        # σ_eps² into any B/W/phy tier — see extract_communality docstring).
        Ω_auto = extract_Omega(fit)
        Σ_W_no_eps = fit.pars.Λ_W * fit.pars.Λ_W' + Diagonal(fit.pars.σ²_W)
        @test Ω_auto ≈ out_unit.Sigma .+ Σ_W_no_eps atol = 1e-8
        @test !(Ω_auto ≈ Ω_total)  # σ_eps² > 0 on this fixture: must differ

        # extract_ICC_site is v_B / (v_B + v_W) per trait, in [0, 1].
        icc = extract_ICC_site(fit)
        vB = diag(out_unit.Sigma); vW = diag(out_w.Sigma)
        @test icc ≈ vB ./ (vB .+ vW) atol = 1e-10
        @test all(0.0 .≤ icc .≤ 1.0)
    end

    @testset "extract_Sigma_table — tidy long format matches extract_Sigma" begin
        Random.seed!(103)
        p, K, n = 3, 1, 150
        y = (0.5 .* randn(p, K)) * randn(K, n) + 0.4 .* randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        Σ = extract_Sigma(fit; level = :unit).Sigma
        tbl = extract_Sigma_table(fit; level = :unit)
        @test length(tbl) == div(p * (p + 1), 2)
        for row in tbl
            @test row.value ≈ Σ[row.trait_i, row.trait_j] atol = 1e-10
            @test row.trait_i ≤ row.trait_j
        end
        @test_throws ArgumentError extract_Sigma_table(fit; part = :unique)
    end

    @testset "extract_loadings / extract_rotated_loadings forward to getLoadings" begin
        Random.seed!(104)
        p, K, n = 4, 2, 200
        y = (0.5 .* randn(p, K)) * randn(K, n) + 0.3 .* randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)

        @test extract_loadings(fit; rotate = true) ≈ getLoadings(fit; rotate = true)
        @test extract_loadings(fit; rotate = false) ≈ getLoadings(fit; rotate = false)
        @test extract_loadings(fit) ≈ getLoadings(fit; rotate = true)  # default rotate=true

        rl = extract_rotated_loadings(fit)
        @test rl.Λ ≈ getLoadings(fit; rotate = true)
        R = rl.R
        @test R' * R ≈ Matrix(I, K, K) atol = 1e-10
        @test getLoadings(fit; rotate = false) * R ≈ rl.Λ atol = 1e-10
    end

    @testset "extract_communality / extract_correlations (GllvmFit) forward correctly" begin
        Random.seed!(105)
        p, K, n = 4, 1, 300
        y = (0.7 .* randn(p, K)) * randn(K, n) + 0.4 .* randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)

        # Maintainer decision round 1, item 3 (docs/dev-log/decisions/
        # 2026-09-01-maintainer-decisions-round1.md; provenance ledger:
        # docs/dev-log/core070/estimand-alignment-notes.md): default is now
        # R's tier-scoped estimand (level = :unit), NOT the old
        # total-variance `communality(fit)`/`correlation(fit)` forward. This
        # fixture has K > 0, no diag (has_diag = false), no W tier -- the
        # `:unit` shared and total tiers coincide EXACTLY (Λ_B Λ_Bᵀ / Λ_B
        # Λ_Bᵀ), so the default degenerates to 1.0 for every trait, matching
        # R's own confirmed degenerate behaviour on such a fit (Repair 3,
        # surface-conversion-notes.md: "level='unit' degenerates to an
        # uninformative constant 1.0"). This is a structural (algebraic)
        # identity, not a fit-dependent numeric match -- exact tolerance.
        @test extract_communality(fit) ≈ ones(p) atol = 1e-10
        # NOTE: shared == total does NOT imply correlation == I here: with
        # K = 1 and no diag, the :unit tier total IS Λ Λᵀ, a rank-1 (p x p)
        # matrix, whose cov2cor is +-1 off the diagonal (perfectly
        # (anti)correlated), not the identity. Cross-check against a direct
        # closed-form cov2cor(Λ Λᵀ), independent of the extractor internals.
        Σ_unit_manual = fit.pars.Λ * fit.pars.Λ'
        @test extract_correlations(fit) ≈ GLLVModels._cov2cor(Σ_unit_manual) atol = 1e-10
        @test all(≈(1.0; atol = 1e-10), diag(extract_correlations(fit)))
        @test all(x -> isapprox(abs(x), 1.0; atol = 1e-8), extract_correlations(fit))

        # `level = :total` recovers the legacy total-variance forward.
        @test extract_communality(fit; level = :total) ≈ GLLVModels.communality(fit)
        @test extract_correlations(fit; level = :total) ≈ GLLVModels.correlation(fit)
        R_total = extract_correlations(fit; level = :total)
        @test all(≈(1.0; atol = 1e-10), diag(R_total))
        @test all(-1.0 - 1e-9 .≤ R_total .≤ 1.0 + 1e-9)

        # extract_cross_correlations is a positional submatrix slice of
        # extract_correlations's own (now tier-scoped) default.
        R = extract_correlations(fit)
        sub = extract_cross_correlations(fit; traits_i = [1, 2], traits_j = [3, 4])
        @test sub ≈ R[[1, 2], [3, 4]] atol = 1e-12

        # level = :site has no R tier-scoped analogue.
        @test_throws ArgumentError extract_communality(fit; level = :site)
        @test_throws ArgumentError extract_correlations(fit; level = :site)

        # Regression: `level` was accepted and silently ignored for a
        # GllvmFit (which computes one site-level correlation tier). Match
        # bootstrap_Sigma's validate-and-throw pattern instead.
        @test_throws ArgumentError extract_cross_correlations(fit; level = :unit_obs,
                                                                traits_i = [1, 2], traits_j = [3, 4])
        @test_throws ArgumentError extract_cross_correlations(fit; level = :bogus,
                                                                traits_i = [1, 2], traits_j = [3, 4])
    end

    @testset "extract_proportions / extract_phylo_signal forward to internals" begin
        Random.seed!(106)
        p, K, n = 4, 1, 200
        y = (0.6 .* randn(p, K)) * randn(K, n) + 0.5 .* randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)

        # Maintainer decision round 1, item 3: default `component = :shared`
        # is now R's tier-scoped `:shared_unit / (sum of tiers the fit
        # carries)` estimand, not the total-variance `proportions(fit;
        # component=:shared)` forward. Same degeneracy as
        # extract_communality above (K > 0, no diag, no W tier): shared ==
        # total tier sum exactly -- algebraic identity, exact tolerance.
        @test extract_proportions(fit) ≈ ones(p) atol = 1e-10
        @test extract_proportions(fit; level = :total) ≈
              GLLVModels.proportions(fit; component = :shared)
        # Any non-:shared component forwards unchanged regardless of level
        # (not part of this alignment slice).
        @test extract_proportions(fit; component = :residual) ≈
              GLLVModels.proportions(fit; component = :residual)

        # No phylo block on this fit: phylo signal is all-NaN.
        hs = extract_phylo_signal(fit)
        @test length(hs) == p
        @test all(isnan, hs)
    end

    @testset "extract_ordination forwards to ordination" begin
        Random.seed!(107)
        p, K, n = 3, 2, 100
        y = (0.5 .* randn(p, K)) * randn(K, n) + 0.4 .* randn(p, n)
        fit = fit_gaussian_gllvm(y; K = K)
        ord1 = extract_ordination(fit, y)
        ord2 = ordination(fit, y)
        @test ord1.sites ≈ ord2.sites
        @test ord1.species ≈ ord2.species
        @test ord1.rotation ≈ ord2.rotation
    end

    @testset "extract_cutpoints (OrdinalFit)" begin
        Random.seed!(108)
        p, K, n, C = 4, 1, 200, 4
        Λ_true = 0.7 .* randn(p, K)
        η = Λ_true * randn(K, n)
        τ_true = [-1.0, 0.0, 1.0]
        Y = Matrix{Int}(undef, p, n)
        for j in 1:n, i in 1:p
            c = 1
            while c < C && η[i, j] > τ_true[c]
                c += 1
            end
            Y[i, j] = c
        end
        fit = fit_ordinal_gllvm(Y; K = K)
        cp = extract_cutpoints(fit)
        @test cp.τ == fit.τ
        @test cp.C == fit.C
        @test issorted(cp.τ)
    end

    @testset "TwoLevelFit extractors forward to twolevel.jl generics" begin
        Random.seed!(109)
        p, K_B, K_W = 3, 1, 1
        Λ_B = 0.6 .* randn(p, K_B); Λ_W = 0.4 .* randn(p, K_W)
        σ²_B = 0.1 .+ 0.1 .* rand(p); σ²_W = 0.1 .+ 0.1 .* rand(p)
        nindiv, nobs = 40, 4
        indiv = repeat(1:nindiv, inner = nobs)
        Y = zeros(p, nindiv * nobs)
        col = 0
        for i in 1:nindiv
            b = Λ_B * randn(K_B) .+ sqrt.(σ²_B) .* randn(p)
            for _ in 1:nobs
                col += 1
                Y[:, col] = b .+ Λ_W * randn(K_W) .+ sqrt.(σ²_W) .* randn(p)
            end
        end
        fit = fit_twolevel_gaussian(Y, indiv; K_B = K_B, K_W = K_W)

        @test extract_Sigma(fit; level = :unit).Sigma ≈ fit.Σ_B atol = 1e-10
        @test extract_Sigma(fit; level = :unit_obs).Sigma ≈ fit.Σ_W atol = 1e-10
        @test extract_Sigma(fit; level = :unit, part = :shared).Sigma ≈ fit.Λ_B * fit.Λ_B' atol = 1e-10
        @test extract_Sigma(fit; level = :unit, part = :unique).s ≈ fit.σ²_B atol = 1e-10

        @test extract_communality(fit; level = :unit) ≈ communality_B(fit)
        @test extract_communality(fit; level = :unit_obs) ≈ communality_W(fit)
        @test extract_correlations(fit; level = :unit) ≈ correlation_B(fit)
        @test extract_correlations(fit; level = :unit_obs) ≈ correlation_W(fit)

        @test extract_repeatability(fit) ≈ repeatability(fit)
        @test extract_ICC_site(fit) ≈ repeatability(fit)

        @test extract_residual_cov(fit; level = :unit) ≈ fit.Σ_B atol = 1e-10
        @test extract_residual_cor(fit; level = :unit) ≈ correlation_B(fit) atol = 1e-10
        @test getResidualCov(fit; level = :unit) ≈ extract_residual_cov(fit; level = :unit)
        @test getResidualCor(fit; level = :unit) ≈ extract_residual_cor(fit; level = :unit)
        @test getResidualCov(fit; level = :unit_obs) ≈ fit.Σ_W atol = 1e-10

        sub = extract_cross_correlations(fit; level = :unit, traits_i = [1], traits_j = [2, 3])
        @test sub ≈ correlation_B(fit)[[1], [2, 3]] atol = 1e-12
    end

    @testset "non-Gaussian extract_communality / extract_correlations forward" begin
        Random.seed!(110)
        p, K, n = 3, 1, 300
        Λ_true = 0.5 .* randn(p, K)
        η = Λ_true * randn(K, n)
        μ = exp.(η)
        Y = [rand(GLLVModels.Distributions.Poisson(μ[i, j])) for i in 1:p, j in 1:n]
        fit = fit_poisson_gllvm(Y; K = K)

        @test extract_communality(fit, Y) ≈ GLLVModels.communality(fit, Y)
        @test extract_correlations(fit, Y) ≈ GLLVModels.correlation(fit, Y)
    end

    @testset "R-tier-scoped oracle pins (gaussian_small fixture, wave5-conversion7)" begin
        # Provenance: .unlazy/core070-aghq/wave5-batches/wave5-conversion7/
        # r-oracle.json, fixture `gaussian_small` (p=5, K=2, n=80, no
        # diag, no W tier), the R-oracle input `y` reproduced exactly (column-
        # major p x n, matching tools/core070_surface_conversion_batch.jl's
        # own `reshape(Float64.(gs["y"]), p, n)`). R and Julia fit
        # INDEPENDENTLY on this identical Y (R's TMB optimiser vs Julia's
        # LBFGS) -- per the paired-independent-fit precedent (Repair 3,
        # docs/dev-log/core070/surface-conversion-notes.md), comparisons use
        # atol = 1e-4, matching the batch contract's own recalibrated
        # tolerance for `sigma_unit_total`/`residual_cor`
        # (docs/dev-log/core070/surface-conversion-batch-contract.json).
        Random.seed!(2026090101)  # unused by the fit itself (LBFGS is
                                   # deterministic given Y and PPCA warm
                                   # start); seeded for reproducibility only.
        p_g, K_g, n_g = 5, 2, 80
        y_g = Float64[
        0.9337124373090624, -0.33651733642450493, 0.8554617668581053, -0.7563633179741794, 0.32354827250689033, 1.2530118672427812, -0.06653297738606401, 0.27809883615148506,
        0.357328609380757, 0.7959199258271403, 0.12238375001162956, 0.7911628115833256, -0.9537586531282902, 1.1027996703381833, 0.5567864580695723, 1.0634881944923447,
        -0.2482694036805193, 1.0454764510548096, 0.04269613406876425, 0.06115794534490551, 1.6805718848665474, 0.6297771300074926, 0.9928276130537521, -0.17468940640519443,
        -0.06960102404791162, 0.06834385371479733, 2.5820162473857176, -0.031209478793072176, 0.35102849863762947, -0.35687873455177666, -1.0078037657173111, -1.0361951834823282,
        -0.02426436482313324, -0.7133296209326417, -0.9510896156027865, 0.6131991172222209, 0.6646220156915025, 0.22021486198588303, 1.6693937267142993, 0.21048038142323766,
        -1.6682390147836987, -1.4352856633021154, 1.9012840275167464, 0.22439469153310335, -1.8463126415614917, -2.7981586500381725, -0.8551704214364422, -1.8939743178902653,
        0.7502238004732197, -0.03492487089213374, -1.1263525220195525, 0.27084617120625554, 0.8000242632168422, -0.8341078204354481, -0.27527024485871876, -0.15152946556390454,
        0.6174545274998988, 0.6003562565180434, 0.5438940217665147, -0.608458473068827, 1.745936081083571, 0.5103262519436853, 0.4795385721010827, -1.2878707504443483,
        0.30308713455123426, -0.36781448155282337, -0.7584951207024251, 0.5680788715615899, -1.23698571602608, 0.2633421546396878, 0.13765707204825622, 0.5766881606974885,
        -0.3430303978143256, -0.684203828316402, -0.3916823082922215, 0.05049810166258993, 1.6617923840973878, -0.05718504239954897, 0.4863810878313444, -0.4598115863920652,
        0.27744451278045257, 0.9179141834743155, 1.5390889183992549, 0.416815038162241, -1.1027610994729888, 1.798750372945069, 0.0011533468099459432, 0.9236379301504494,
        -0.4128755709447194, -1.2017243561785973, -1.4350627870335801, -0.8013687332940538, -0.6298716180051855, -0.08145434832777282, 0.5175283462522741, -2.6952639372714144,
        -1.6352687770418202, -0.6020664336967858, -0.15732010235005642, -0.5560303592274682, 0.7355414570913987, 0.6321164424952962, 0.5789653114270105, -1.4637229853688456,
        -0.18266264970297752, 1.3116856273576292, 0.5793652595372238, 0.4886095160959896, -2.3508415785800194, -0.15666608177232627, -0.7331848585357144, -0.32531718912356067,
        -0.3693725991296607, 0.8462553309291206, -0.34171511629054996, 0.227684005837964, 0.8571095605964631, -1.5868775005182219, 1.9055264336605973, 1.1380647124244985,
        0.19150860124785707, -0.9191049410499774, -0.6861573082402213, 0.9130672024192044, 0.127007545683741, 0.46096261348728906, 1.4794447640283255, -0.058702924737932705,
        -1.8814502558386088, -0.06852006942251472, 0.7390780157244103, 1.5529020921449714, -0.24906062657424977, 1.6088373332276211, 0.20670075921010478, -0.3025125481801352,
        0.16149773937934206, 0.6241524100302998, 0.24255593206054712, -1.103138032980589, 0.500010304126913, 0.99278366908381, 0.2991972756242441, 0.47770352730135873,
        -0.06479969570004424, -2.475179412528029, -0.9208866314396118, -0.21276951418662876, -0.07732300713374926, -0.29443299402810946, -0.3836474523065609, -0.5366534283175002,
        -0.8042640249974418, -0.22135580870581037, 0.3724635048145949, 0.13384691132506588, 0.648402505149923, -0.16690604263864967, -0.2281019483105435, 1.1530674599099373,
        -0.9612366501714518, 0.6173397294110067, -0.5994500921290142, 0.850655607789219, 1.5274534089957157, -0.3492372854140847, 1.6599270473994867, -0.1884966258376543,
        -0.07846629022354454, 0.239854482892848, 1.2191368243024605, 0.662098079534505, -0.44854011286646905, 0.3373225905915918, 0.9432064929341956, -0.926102196363589,
        -1.4795198769968836, -1.087484113560326, -0.32025178505357577, -0.5574911019822051, 0.4436293728317145, 0.27606372777904975, 0.36828301113661777, -1.3270868115123098,
        0.1441214009031141, -0.28162938478015465, 0.9543964516576838, -1.468933698651107, 0.5142817220747882, 1.0102217892268999, 1.0812784395072481, 0.14663295575173485,
        0.6617666731007021, -0.4780064142021199, -0.15857771189543268, 0.08246674587382762, -1.2387883944013214, 0.8582119181405456, -0.5225002378946466, -1.5791131870389457,
        0.5758084087409143, 0.21621247579693315, 0.11569027457514641, -0.3549041814089076, 0.06414900439686438, -0.13970708791350628, 1.3632936329217045, -0.7912580059077803,
        -0.31441897700414473, 0.1597845269489266, 0.14426025870357861, -0.1009309700112134, 0.36835821454458245, -0.5522302988825322, 1.0194681432809118, -0.6972902080016277,
        -0.3446556947448227, -0.6645822566922919, 0.09985248557672305, 1.1160487353577708, 2.402005200294258, 0.27302681085292857, 0.3955035780787193, 1.2502768547536605,
        0.14575918953032302, 0.920784531114286, 0.2640792776971065, 1.1306375353086628, 0.5157846989496037, 0.42054568916491225, 0.7042242473501389, 1.0156016178983576,
        0.09155461198095749, 0.2349040216007733, 0.22974083606347367, -0.16125790461888956, -0.6953908183801996, 0.28384192316354756, -0.3330695282054406, 0.2252587626511881,
        -0.011138346158968485, -1.9828179630958447, 0.3870550102389009, -0.414818076253337, -1.0129346821688106, -0.6753172278430573, 0.49728094615076057, 0.32562149451025224,
        1.3984896025527134, -1.1895949989566634, 0.4288784579213257, 1.0764021992368262, -1.1177705972262029, -0.18909659269767579, -0.1677719720969758, -1.3485519186123833,
        -0.8576293900739363, -0.7869122230919634, 0.7366309729187243, 1.4614199371202663, -0.016435773654479693, -1.2204791007529394, -0.0693106064771841, -0.17543375289084595,
        0.5296260165251125, -0.3408416258848014, -0.6073845704533825, 0.33319137597073734, 0.18419614953874242, -0.2513105513781591, 1.5242916513955975, -0.137907016627843,
        -0.2167072075086811, -1.006119626621299, -0.030948719837459317, 0.5629891618967676, -1.3306041225736445, -0.12175229405964527, -0.7140060764200623, 0.9821210444276175,
        0.5219924772990858, -0.24402589145233394, 1.1606430842298219, -0.7080496656969338, 0.4084999403091951, -1.7785735312138973, -0.7682094985684693, -0.4509232984618506,
        0.000167142333668914, 0.04432298039695855, -1.362981225092434, 1.2562393574835442, -1.0107961694702208, 1.2251426647174668, 0.7754162785205814, -1.3530957469043643,
        0.601206382403991, 0.14044481069017517, 2.4712514624932806, -0.7087366851229246, -0.6650301248158766, -0.09338239968367446, 0.4548229614151322, -1.1405511084498383,
        -1.3520665848050542, 0.03478550227649388, 0.44782477651075847, 1.065396856893922, -1.96602352591735, -0.8339169824322415, -0.04798383538763956, -1.2114900653338938,
        0.7609776672538238, -1.882605402389744, -0.08550469540998334, -0.8049789797873188, -1.798387358660778, 0.5312665859448604, -0.6151538900314014, -0.4484895140934763,
        -1.1779071692519443, -1.7850594000023348, -1.342641323952055, -0.6399582830569513, 0.40753694574364785, 0.2433797769901489, 0.2439275220175753, -0.6197619984921188,
        -0.6807800522561004, 0.2454676302176707, -0.1649641903804451, 0.6602400494640871, -0.39552987617833607, 1.4631634578180888, -0.15070225351935423, 1.5849869039624593,
        0.7577493460661753, 0.1423298442189736, -0.6975524898320792, -0.25015134089550256, 0.5863566440807078, 1.3696516795757485, -0.41346446869012615, 0.7735141472922585,
        -0.21253752158938177, -0.5546569067276752, -0.9197226337350523, -0.6927500608626209, -0.732763640506841, 0.4654839848429697, 0.6714082091183274, -0.79765244516608,
        -0.3569946144848338, -0.43407563345697026, 1.2372705457431112, 1.032985804378769, -1.3767991538550415, -0.2674117626872955, 1.2016050540391978, -0.13722033622219804,
        0.7469079791433754, 0.29465742592032923, -0.041974770895392255, 0.8505299970636255, 2.495829964560622, 0.21933631217879546, -1.277905177093286, -0.28969691144168885,
        0.19287309935863903, -1.5658980510014282, -0.7457831114713703, -1.0035955723298597, 0.492105560071098, 0.2537691302392748, -0.49631016607274436, 0.04092150953097988,
        -0.929430956066018, 0.21879256490457344, 0.2399997200749048, -1.294240130257807, 0.7336459651669383, 0.24298770911149148, 0.9361493254035516, 0.8089017811414178,
        0.3371458084128725, 0.5531987414245336, -0.37563611339763614, 0.02178829897743345, -0.41438209066560205, 0.48784754271216146, -1.139281888570573, -0.18964084949808258,
        0.31208118560940107, 0.2699543683864452, 0.2271081414831292, -0.2562426925091513, 1.6178967761813394, -1.6389133109092549, 1.4421327278153604, -0.8298666832000315,
        ]
        Y_g = reshape(y_g, p_g, n_g)
        fit_g = fit_gaussian_gllvm(Y_g; K = K_g)

        # R's `extract_Sigma(fit_g, level = "unit", part = "total")$Sigma`
        # (CORE070-SURFCONV-POSTFIT-POSTFIT-SURFACE-EXTRACT-SIGMA), column-
        # major p x p.
        Σ_R = reshape(Float64[
        0.571630145169022, 0.3383093094681535, 0.2021783665011631, -0.1355487148890337, 0.03989501699045239, 0.3383093094681535, 0.4570005404423524, -0.044333761068435826,
        0.20589634730231193, 0.1540146372402689, 0.2021783665011631, -0.044333761068435826, 0.17623865505242486, -0.23066939683768636, -0.06917088035113536, -0.1355487148890337,
        0.20589634730231193, -0.23066939683768636, 0.35095362691402987, 0.13584367214837187, 0.03989501699045239, 0.1540146372402689, -0.06917088035113536, 0.13584367214837187,
        0.06900908912818107,
        ], p_g, p_g)
        R_R = reshape(Float64[
        1, 0.6619086096943135, 0.6369810856284869, -0.30263069503900375, 0.2008667001993506, 0.6619086096943135, 1.0000000000000002, -0.15621604364816483,
        0.5141210878176539, 0.8672623498102088, 0.6369810856284869, -0.15621604364816483, 1.0000000000000002, -0.927501301262931, -0.6272195269602543, -0.30263069503900375,
        0.5141210878176539, -0.927501301262931, 1, 0.872893778237423, 0.2008667001993506, 0.8672623498102088, -0.6272195269602543, 0.872893778237423,
        1,
        ], p_g, p_g)  # R's getResidualCor(level="unit") == cov2cor(Σ_R)

        # extract_Sigma(level=:unit) itself is unchanged by this decision
        # (only the communality/correlations/proportions/Omega DEFAULTS
        # changed) -- pins the underlying tier total against R directly.
        @test extract_Sigma(fit_g; level = :unit).Sigma ≈ Σ_R atol = 1e-4

        # This fixture has no diag (has_diag = false) and no W tier: the
        # `:unit` shared and total tiers coincide EXACTLY in both engines
        # (an algebraic identity, not a numeric fit match), so R's own
        # extract_communality(level="unit") / extract_proportions()
        # `shared_unit` both degenerate to the uninformative constant 1.0
        # for every trait (Repair 3 / Repair 4, surface-conversion-notes.md)
        # -- and GLLVModels.jl's new tier-scoped default reproduces that
        # degeneracy exactly.
        @test extract_communality(fit_g) ≈ ones(p_g) atol = 1e-10
        @test extract_proportions(fit_g) ≈ ones(p_g) atol = 1e-10

        # extract_correlations default (level = :unit) pins against R's
        # traced GETRESIDUALCOR / EXTRACT-SIGMA-derived correlation matrix.
        @test extract_correlations(fit_g) ≈ R_R atol = 1e-4
        @test extract_correlations(fit_g) ≈ GLLVModels._cov2cor(Σ_R) atol = 1e-4

        # extract_Omega default (level = :auto): gaussian_small carries only
        # the :unit tier (K > 0; no diag on :unit_obs, no W-tier loadings,
        # so _r_tier_present(fit_g, :unit_obs) is false) and no phy block --
        # Omega reduces to exactly the :unit tier total, matching R's own
        # `tiers = "B"`-only composition (Repair 5, surface-conversion-
        # notes.md: "Omega = Lambda_B Lambda_B^T exactly").
        @test extract_Omega(fit_g) ≈ Σ_R atol = 1e-4
        @test GLLVModels._r_tier_present(fit_g, :unit)
        @test !GLLVModels._r_tier_present(fit_g, :unit_obs)

        # level = :total recovers the legacy (pre-decision) composition,
        # which on this single-tier fixture adds sigma_eps^2*I as a spurious
        # baseline (the confirmed bug from Repair 5) -- so it must now
        # differ from the R-aligned default whenever sigma_eps > 0.
        Ω_total = extract_Omega(fit_g; level = :total)
        @test Ω_total ≈ Σ_R .+ (fit_g.pars.σ_eps^2) .* Matrix(I, p_g, p_g) atol = 1e-4
        @test !(Ω_total ≈ extract_Omega(fit_g))
    end

    @testset "unit-tier extractors on a has_diag fit use the identified total (#701)" begin
        # Twin of the #701 fixture (p = 4, n = 250, one factor, centred, K = 1,
        # has_diag = true). gllvmTMB `latent(0 + trait | site, d = 1)` fits the
        # whole identified diagonal sum as the unit-tier unique part, so its
        # `extract_communality(level = "unit")` equals `communality()` here
        # (the issue's R values: 0.90672, 0.70182, 0.2196, 0.52292, at 5 dp).
        Random.seed!(99)
        p, n = 4, 250
        Λ0 = [0.9, 0.7, 0.4, 0.6]
        Y = Λ0 * randn(1, n) .+ sqrt.([0.1, 0.4, 1.5, 0.7]) .* randn(p, n)
        Y .-= sum(Y; dims = 2) ./ n
        fit = fit_gaussian_gllvm(Y; K = 1, has_diag = true)
        Σ = GLLVModels.sigma_y_site(fit)

        c = extract_communality(fit; level = :unit)
        @test c ≈ communality(fit) atol = 1e-10
        @test c ≈ diag(fit.pars.Λ * fit.pars.Λ') ./ diag(Σ) atol = 1e-10
        # The unit-tier unique variance is the identified diagonal sum, not
        # sigma2_B alone: communality must sit strictly below the old value.
        c_old = diag(fit.pars.Λ * fit.pars.Λ') ./
                (diag(fit.pars.Λ * fit.pars.Λ') .+ fit.pars.σ²_B)
        @test all(c .< c_old)

        @test extract_correlations(fit; level = :unit) ≈ correlation(fit) atol = 1e-10
        @test extract_proportions(fit) ≈ GLLVModels.proportions(fit; component = :shared) atol = 1e-10
        # Omega: the diagonal W tier is already inside the unit total.
        @test diag(extract_Omega(fit)) ≈ diag(Σ) atol = 1e-10
    end
end

using Test

# DRAFT test for extract_phylo_signal on a JointPhyloGroupedGaussianFit.
# Common-point algebra only: every fit below is built at a declared parameter
# vector with `iterations = 0`, so the expected values follow from the declared
# variance components and the estimand of gllvmTMB at the P1 pin
# (R/extract-omega.R:592-623). No R fit is run and no R number is compared.
if Base.find_package("StableRNGs") === nothing
    @testset "extract_phylo_signal joint fit (StableRNGs unavailable)" begin
        @test_skip false
    end
else
    using GLLVModels
    using LinearAlgebra
    using SparseArrays
    using StableRNGs

    const _EPS_NEWICK = "(((s1:2,s2:2):1,(s3:1,s4:1):2):1,((s5:1.5,s6:1.5):1,(s7:1,s8:1):1.5):1.5);"

    # 8 species crossed with 3 sites (24 observations), 3 traits. Observations are
    # ordered site-major: obs = (site - 1) * 8 + species.
    function _eps_fixture()
        phy, _ = GLLVModels._phylo_latent_tree_precision(_EPS_NEWICK)
        species_id = repeat(1:8, 3)
        site = repeat(1:3, inner = 8)
        rng = StableRNG(20261102)
        Y = randn(rng, 3, length(species_id))
        return (; phy, species_id, site, Y)
    end

    # Packed start for rank 1, p = 3, three trait intercepts. Coordinates are
    # [beta(3); loading(3); (unique log-sd(3) if explicitunique); residual log-sd(3);
    #  one log-sd(3) block per ordinary term].
    _eps_logsd(variance) = 0.5 .* log.(variance)

    function _eps_fit(fx; loading, species_var, residual_var = fill(0.35, 3),
            phylo_unique = nothing, site_var = nothing, residual_mode = :trait)
        terms = [GroupingTerm(:cluster; mode = :indep, common = false)]
        theta = vcat(zeros(3), loading)
        phylo_unique === nothing || append!(theta, _eps_logsd(phylo_unique))
        if residual_mode === :shared
            allequal(residual_var) || error("residual_mode = :shared needs equal residual variances")
            push!(theta, only(_eps_logsd(residual_var[1:1])))
        else
            append!(theta, _eps_logsd(residual_var))
        end
        append!(theta, _eps_logsd(species_var))
        if site_var !== nothing
            push!(terms, GroupingTerm(:unit; mode = :indep, common = false))
            append!(theta, _eps_logsd(site_var))
        end
        kw = site_var === nothing ? (; cluster = fx.species_id) :
            (; cluster = fx.species_id, unit = fx.site)
        return GLLVModels.fit_joint_phylo_grouped_gaussian(fx.Y, fx.phy;
            rank = 1, phylo_mode = phylo_unique === nothing ? :barelowrank : :explicitunique,
            terms = terms, species_id = fx.species_id, start = theta,
            iterations = 0, g_tol = 1e-5, residual_mode = residual_mode, kw...)
    end

    @testset "extract_phylo_signal on a joint phylo-plus-grouped fit (draft)" begin
        fx = _eps_fixture()
        lambda = [0.70, -0.45, 0.50]
        u = [0.12, 0.22, 0.18]

        @testset "Case A: bare phylogeny plus species diagonal" begin
            fit = _eps_fit(fx; loading = lambda, species_var = u)
            @test GLLVModels._joint_phylo_grouped_component_identification(fit).reason === :identified
            out = extract_phylo_signal(fit; species_level = :cluster)
            @test keys(out) == (:trait, :H2, :C2_non, :Psi, :V_eta)
            @test out.trait == ["trait1", "trait2", "trait3"]
            @test out.V_eta ≈ [0.61, 0.4225, 0.43] rtol = 1e-12
            @test out.H2 ≈ [49 / 61, 81 / 169, 25 / 43] rtol = 1e-12
            @test out.C2_non == zeros(3)
            @test out.Psi ≈ 1 .- out.H2 rtol = 1e-12
            @test out.H2 .+ out.C2_non .+ out.Psi ≈ ones(3) rtol = 1e-12
        end

        @testset "Case B: phylogenetic unique plus crossed site term" begin
            unq = [0.16, 0.09, 0.12]
            site_var = [0.10, 0.15, 0.20]
            fit = _eps_fit(fx; loading = lambda, species_var = u, phylo_unique = unq,
                site_var = site_var)
            @test GLLVModels._joint_phylo_grouped_component_identification(fit).reason === :identified
            out = extract_phylo_signal(fit; species_level = :cluster)
            @test out.V_eta ≈ [0.77, 0.5125, 0.55] rtol = 1e-12
            @test out.H2 ≈ [65 / 77, 117 / 205, 37 / 55] rtol = 1e-12
            @test out.C2_non == zeros(3)
            @test out.Psi ≈ 1 .- out.H2 rtol = 1e-12

            # Control 1, wrong denominator: adding the site term or the observation
            # residual to V_eta would give a different answer, and changing only those
            # excluded variances must leave every output unchanged.
            wrong = out.V_eta .+ site_var .+ fill(0.35, 3)
            @test !(out.V_eta ≈ wrong)
            moved = _eps_fit(fx; loading = lambda, species_var = u, phylo_unique = unq,
                site_var = [0.31, 0.05, 0.12], residual_var = [0.5, 0.2, 0.9])
            out2 = extract_phylo_signal(moved; species_level = :cluster)
            @test out2.V_eta ≈ out.V_eta rtol = 1e-12
            @test out2.H2 ≈ out.H2 rtol = 1e-12
            @test out2.Psi ≈ out.Psi rtol = 1e-12

            # Control 2, source allocation: move delta from the phylogenetic unique
            # variance to the species diagonal. V_eta is fixed, H2 falls by
            # delta / V_eta and Psi rises by the same amount.
            delta = [0.03, 0.02, 0.04]
            shifted = _eps_fit(fx; loading = lambda, species_var = u .+ delta,
                phylo_unique = unq .- delta, site_var = site_var)
            out3 = extract_phylo_signal(shifted; species_level = :cluster)
            @test out3.V_eta ≈ out.V_eta rtol = 1e-12
            @test out3.H2 ≈ out.H2 .- delta ./ out.V_eta rtol = 1e-10
            @test out3.Psi ≈ out.Psi .+ delta ./ out.V_eta rtol = 1e-10
            @test !(shifted.loglik ≈ fit.loglik)
        end

        @testset "residual_mode = :shared leaves the signal estimand unchanged" begin
            for (kwargs, expected_H2) in (
                    ((; loading = lambda, species_var = u), [49 / 61, 81 / 169, 25 / 43]),
                    ((; loading = lambda, species_var = u, phylo_unique = [0.16, 0.09, 0.12],
                        site_var = [0.10, 0.15, 0.20]), [65 / 77, 117 / 205, 37 / 55]))
                trait_fit = _eps_fit(fx; kwargs...)
                shared_fit = _eps_fit(fx; kwargs..., residual_mode = :shared)
                @test shared_fit.residual_mode === :shared
                @test length(shared_fit.parameters) == length(trait_fit.parameters) - 2
                @test GLLVModels._joint_phylo_grouped_component_identification(shared_fit).reason ===
                    :identified
                out = extract_phylo_signal(shared_fit; species_level = :cluster)
                ref = extract_phylo_signal(trait_fit; species_level = :cluster)
                @test out.H2 ≈ expected_H2 rtol = 1e-12
                @test out.V_eta ≈ ref.V_eta rtol = 1e-12
                @test out.Psi ≈ ref.Psi rtol = 1e-12
                @test out.C2_non == zeros(3)
                # equal residual SDs: same likelihood as the :trait fit at the same point
                @test shared_fit.loglik ≈ trait_fit.loglik rtol = 1e-12
            end
            # the residual level is excluded from V_eta whatever its size
            moved = _eps_fit(fx; loading = lambda, species_var = u,
                residual_var = fill(0.9, 3), residual_mode = :shared)
            @test extract_phylo_signal(moved; species_level = :cluster).H2 ≈ [49 / 61, 81 / 169, 25 / 43] rtol = 1e-12
        end

        @testset "refusals" begin
            fit = _eps_fit(fx; loading = lambda, species_var = u,
                phylo_unique = [0.16, 0.09, 0.12], site_var = [0.10, 0.15, 0.20])
            @test_throws r"GJL-GATE-PHYLO-LATENT-SIGNAL-CI" extract_phylo_signal(fit;
                species_level = :cluster, ci = true)
            @test_throws ArgumentError extract_phylo_signal(fit)
            @test_throws ArgumentError extract_phylo_signal(fit; species_level = :cluster2)
            @test_throws ArgumentError extract_phylo_signal(fit; species_level = "cluster")
            # The site term is not on the species grouping, so it cannot be selected.
            @test_throws r"same observation partition" extract_phylo_signal(fit;
                species_level = :unit)
        end

        @testset "aliased components are refused" begin
            Q = sparse(Matrix{Float64}(I, 4, 4))
            phy = PrecisionPhy(findnz(Q)..., 4, 4, ["tip_$i" for i in 1:4],
                0.0, 1.0, collect(1:4))
            fit = fit_gllvm(reshape([0.20, -0.15, 0.31, -0.24], 1, :);
                phylo = phy, phylo_rank = 1, phylo_mode = :barelowrank,
                species_id = collect(1:4),
                grouping = [GroupingTerm(:unit; mode = :indep, common = false)],
                unit = collect(1:4), iterations = 0, g_tol = 1e-4)
            @test GLLVModels._joint_phylo_grouped_component_identification(fit).reason ===
                :nonidentifiable
            @test_throws r"GJL-GATE-PHYLO-LATENT-SIGNAL-IDENTIFICATION" extract_phylo_signal(fit;
                species_level = :unit)
        end
    end
end

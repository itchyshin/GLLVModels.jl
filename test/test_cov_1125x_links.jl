# Lane 1125x round 3: families/links.jl defaults and link algebra.
using Test, GLLVModels, Distributions

const G = GLLVModels

@testset "1125x link functions" begin
    @testset "default_link markers" begin
        @test G.default_link(Normal()) isa G.IdentityLink
        @test G.default_link(Binomial()) isa G.LogitLink
        @test G.default_link(Poisson()) isa G.LogLink
        @test G.default_link(NegativeBinomial()) isa G.LogLink
        @test G.default_link(Beta()) isa G.LogitLink
        @test G.default_link(Gamma()) isa G.LogLink
    end

    links = (G.LogitLink(), G.ProbitLink(), G.CLogLogLink(), G.IdentityLink(), G.LogLink())
    @testset "linkfun ∘ linkinv round-trip" for link in links
        for η in (-3.0, -0.5, 0.0, 0.4, 2.5)
            μ = G.linkinv(link, η)
            if link isa G.LogitLink
                μ = clamp(μ, 1e-8, 1 - 1e-8)
            elseif link isa G.CLogLogLink
                μ = clamp(μ, 1e-8, 1 - 1e-8)
            elseif link isa G.LogLink
                μ = max(μ, 1e-12)
            end
            @test G.linkfun(link, μ) ≈ η atol = 1e-10
        end
    end

    @testset "mu_eta matches finite difference" begin
        h = 1e-6
        for (link, η) in ((G.LogitLink(), 0.7), (G.ProbitLink(), 0.2), (G.LogLink(), 1.3))
            μp = G.linkinv(link, η + h)
            μm = G.linkinv(link, η - h)
            fd = (μp - μm) / (2h)
            @test G.mu_eta(link, η) ≈ fd rtol = 1e-5
        end
    end

    @testset "logit numerical stability at large |η|" begin
        for η in (20.0, -20.0)
            μ = G.linkinv(G.LogitLink(), η)
            @test 0 < μ < 1
            @test G.mu_eta(G.LogitLink(), η) > 0
            @test isfinite(G.mu_eta(G.LogitLink(), η))
        end
    end
end
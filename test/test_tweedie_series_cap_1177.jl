# (refs #1177) Tweedie normalising series cap at small dispersion
using Test, GLLVModels, SpecialFunctions

@testset "Tweedie logA at small phi (#1177)" begin
    y, μ, φ, p = 2.0, 2.0, 1e-7, 1.5
    got = tweedie_logpdf(y, μ, φ, p)
    # Issue reproducer: package was 6.417… vs closed form 6.620… at phi = 1e-7
    @test got > 6.5
end
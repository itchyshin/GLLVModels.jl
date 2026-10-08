using Test

@testset "issue 737 extract_Omega :auto vs :total doc lock" begin
    path = joinpath(@__DIR__, "..", "docs", "src", "postfit-extractors.md")
    @test isfile(path)
    collapsed = replace(read(path, String), r"\s+" => " ")

    # Wording from docs/src/postfit-extractors.md after 9d8c883ae: :auto
    # includes σ_eps² on has_diag Gaussian fits with no W loadings, so
    # :total coincides with :auto.
    @test occursin(
        "`extract_Omega` defaults to `level = :auto`, combining the sources present in the fit. That combination excludes observation noise `σ_eps²` only when the fit has a genuine `:unit_obs` tier or no diagonal. For a Gaussian fit with `has_diag = true` and `K_W == 0` the identified total `sigma_y_site(fit)` (including `σ_eps²`) is used, so `level = :total` coincides with `:auto`.",
        collapsed,
    )
end

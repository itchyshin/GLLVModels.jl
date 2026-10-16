# Issue #719: the Documenter deploy job printed "Found no bases suitable for
# deployment" and still went green, so /dev/ went stale. Two things were wrong:
# the build job (which writes docs/build/bases.txt through
# DocumenterVitepress/Documenter deploy_folder) had no GITHUB_TOKEN, so the deploy
# decision was all_ok = false and bases.txt was blank; and nothing failed when
# the deploy step skipped. This file reads the workflow and checks both stay fixed.

using Test

const DOCUMENTER_YML = read(joinpath(@__DIR__, "..", ".github", "workflows", "Documenter.yml"), String)

# Text of one `- name: <name>` step, up to the next step or job.
function _step(yml::AbstractString, name::AbstractString)
    m = match(Regex("- name: " * name * "\\n(.*?)(?=\\n      - |\\n  [a-z]+:|\\z)", "s"), yml)
    m === nothing && return ""
    return m.captures[1]
end

@testset "issue 719 Documenter deploy guard" begin
    build = _step(DOCUMENTER_YML, "Build documentation")
    @test !isempty(build)
    # Without a token at build time deploy_folder returns all_ok = false and
    # bases.txt is blank, so the deploy job has nothing to publish.
    @test occursin("GITHUB_TOKEN: \${{ secrets.GITHUB_TOKEN }}", build)

    deploy = _step(DOCUMENTER_YML, "Deploy checked documentation")
    @test !isempty(deploy)
    # A deploy that finds no bases must fail, not exit 0.
    @test occursin("docs/build/bases.txt", deploy)
    @test occursin("Found no bases", deploy)
    @test occursin("pipefail", deploy)
end

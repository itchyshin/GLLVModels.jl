# Docs regression for #1020: gllvmtmb-parity.md must document the confint default-method
# difference between R (profile) and Julia (:wald).
using Test

const _PARITY_MD = joinpath(@__DIR__, "..", "docs", "src", "gllvmtmb-parity.md")
const _CI_MD = joinpath(@__DIR__, "..", "docs", "src", "confidence-intervals.md")

@testset "gllvmtmb-parity documents confint default (#1020)" begin
    parity = read(_PARITY_MD, String)
    @test occursin("confint", parity)
    @test occursin("method = \"profile\"", parity)
    @test occursin("method = :wald", parity)
    ci = read(_CI_MD, String)
    @test occursin("gllvmtmb-parity", ci)
    @test occursin("profile", ci)
end
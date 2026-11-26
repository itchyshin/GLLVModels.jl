using GLLVModels
using Test

# Issue #1461: communality_wald_ci / correlation_wald_ci must not claim that
# confint(..., method = :profile) works for the same quantity (profile refused).
const _PROFILE_CLAIM = r"offers `method = :profile`"

function _assert_withdrawn_profile_doc(doc::AbstractString)
    @test !occursin(_PROFILE_CLAIM, doc)
    @test occursin("method = :profile` is refused", doc)
    @test occursin("withdrawn", doc)
end

@testset "derived Wald CI docstrings: profile via confint is withdrawn (#1461)" begin
    _assert_withdrawn_profile_doc(string(@doc communality_wald_ci))
    _assert_withdrawn_profile_doc(string(@doc correlation_wald_ci))
end
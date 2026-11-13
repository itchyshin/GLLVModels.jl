# bridge_fit: when every interval bound is NaN and no note was given, ci_note says why (#1157).
# A bootstrap with fewer than 10 usable replicates forms no percentile bound (`_family_bootstrap`
# needs `length(col) >= 10`), so every bound is NaN; the payload used to carry ci_note == "" and a
# caller could not tell "not computed" from "computed". Text only: no bound or estimate changes.

using GLLVModels, Test, Random
const _G1157 = GLLVModels

@testset "bridge CI: all-NaN bounds carry a reason in ci_note (#1157)" begin
    Random.seed!(7)
    Y = Float64.(rand(0:6, 5, 60))
    bs(nb) = _G1157.bridge_fit(; y = Y, family = "poisson", d = 1,
        options = Dict("ci_method" => "bootstrap", "ci_nboot" => nb, "ci_seed" => 1))

    few = bs(5)
    @test all(isnan, few.ci_lower) && all(isnan, few.ci_upper)
    @test !isempty(few.ci_param_names)
    @test occursin("fewer than 10 usable bootstrap replicates", few.ci_note)
    @test all(isfinite, few.ci_estimate)                    # estimates are untouched

    enough = bs(12)
    @test all(isfinite, enough.ci_lower) && all(isfinite, enough.ci_upper)
    @test enough.ci_note == ""                              # finite bounds: payload unchanged

    wald = _G1157.bridge_fit(; y = Y, family = "poisson", d = 1, options = Dict("ci_method" => "wald"))
    @test wald.ci_note == ""

    # the helper itself: a note that was already given is never replaced; a partial NaN table
    # (some finite bound) gets no note; an empty table gets none
    nanci = (term = ["a", "b"], estimate = [1.0, 2.0], lower = [NaN, NaN], upper = [NaN, NaN])
    @test !isempty(_G1157._bridge_ci_all_nan_note("profile", nanci))
    @test occursin("profile", _G1157._bridge_ci_all_nan_note("profile", nanci))
    @test _G1157._bridge_ci_from_native("wald", 0.95, nanci; note = "kept").ci_note == "kept"
    partial = (term = ["a", "b"], estimate = [1.0, 2.0], lower = [NaN, 0.5], upper = [NaN, 3.0])
    @test _G1157._bridge_ci_all_nan_note("profile", partial) == ""
    @test _G1157._bridge_ci_all_nan_note("wald", (term = String[], estimate = Float64[], lower = Float64[], upper = Float64[])) == ""
end

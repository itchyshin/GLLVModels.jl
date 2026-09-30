# Isolate driver globals; keep Test.jl assertions in the enclosing testset.
# Tight R controls always. The retained-baseline comparison runs only where ./baseline
# exists (a machine holding the pinned receipts); CI has none, so it opts in to skipping
# that one comparison, and the skip is recorded rather than counted as a pass.
const _CORE070_HAVE_BASELINE = isdir("baseline")
const _CORE070_MODES_TOOL = joinpath(@__DIR__, "..", "..", "tools", "core070_covariance_mode_fits.jl")
_CORE070_HAVE_BASELINE ||
    @warn "Core070 covariance modes: no retained ./baseline, so the baseline data/map comparison is NOT checked in this run (tight R controls still apply)."
withenv("CORE070_BASELINE_OPTIONAL" => (_CORE070_HAVE_BASELINE ? nothing : "1")) do
    @eval module Core070CovarianceModesFixture
    const ARGS = [joinpath(Main._core070_receipt_dir(), "covariance-modes-raw"), "tight-control"]
    include($(_CORE070_MODES_TOOL))
    end
end
# Visible in the test summary: the retained-baseline comparison did not run here.
_CORE070_HAVE_BASELINE || @test_broken Core070CovarianceModesFixture.baseline_compared

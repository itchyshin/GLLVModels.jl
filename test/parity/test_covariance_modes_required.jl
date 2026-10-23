# Isolate driver globals; keep Test.jl assertions in the enclosing testset.
# Tight R controls always. The tight-control run compares each case's TMB data, map
# and outer-parameter names against a default-control run of the same tool
# (./baseline/<id>.rds), so tightening the optimizer controls is shown to change
# nothing else. Where no retained ./baseline exists (CI), the default-control
# baseline is produced here first (#614), in a SEPARATE Julia process: its own
# checks include the known default-control R gradient failures, which must not count
# in this run; only its saved .rds evidence is used. If that still yields no
# complete baseline, the comparison is skipped and recorded as not checked.
const _CORE070_MODES_TOOL = joinpath(@__DIR__, "..", "..", "tools", "core070_covariance_mode_fits.jl")
const _CORE070_RETAINED_BASELINE = isdir("baseline")
const _CORE070_MADE_BASELINE = Ref(false)
if !_CORE070_RETAINED_BASELINE
    let dir = joinpath(Main._core070_receipt_dir(), "covariance-modes-default-baseline")
        cmd = addenv(`$(Base.julia_cmd()) --project=$(dirname(Base.active_project())) $(_CORE070_MODES_TOOL) $dir`,
            "CORE070_BASELINE_BUILD" => "1")  # this subprocess only
        @info "Core070 covariance modes: building a default-control baseline in a separate process" dir
        run(ignorestatus(cmd))
        n_rds = isdir(dir) ? count(endswith(".rds"), readdir(dir)) : 0
        if n_rds == 7
            symlink(dir, "baseline")
            _CORE070_MADE_BASELINE[] = true
        else
            @warn "Core070 covariance modes: default-control baseline incomplete ($n_rds of 7 cases)"
        end
    end
end
const _CORE070_HAVE_BASELINE = isdir("baseline")
_CORE070_HAVE_BASELINE ||
    @warn "Core070 covariance modes: no ./baseline, so the baseline data/map comparison is NOT checked in this run (tight R controls still apply)."
# The marker belongs to the baseline subprocess alone; the tight-control gate run must
# never see it (it would skip the assertions).
get(ENV, "CORE070_BASELINE_BUILD", "") == "" ||
    error("CORE070_BASELINE_BUILD leaked into the tight-control gate run")
try
    withenv("CORE070_BASELINE_OPTIONAL" => (_CORE070_HAVE_BASELINE ? nothing : "1")) do
        @eval module Core070CovarianceModesFixture
        const ARGS = [joinpath(Main._core070_receipt_dir(), "covariance-modes-raw"), "tight-control"]
        include($(_CORE070_MODES_TOOL))
        end
    end
finally
    # Remove only the link this file created; a retained ./baseline is never touched.
    _CORE070_MADE_BASELINE[] && islink("baseline") && rm("baseline")
end
# Visible in the test summary if the comparison could not run here.
_CORE070_HAVE_BASELINE || @test_broken Core070CovarianceModesFixture.baseline_compared

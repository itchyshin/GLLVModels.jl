# gllvm-parity-tag: P1
#
# Julia-only wording contract (issue 1286, post-fit helpers). Refusals raised by
# the post-fit temporal helpers name the marker the user called
# (`temporal_indep()`, `temporal_dep()` or `temporal_latent()`), not always
# `temporal_latent()`. The `temporal_latent()` texts stay byte for byte what
# they were. Only wording is tested; no fit result, tolerance or rule changes.
# Marker-name rule: one function, `_temporal_fit_marker`, in src/temporal_methods.jl.
using Test, GLLVModels, Random
const GMN = GLLVModels

@testset "temporal post-fit helpers name the called marker (#1286)" begin
    rows = [(s, o, t) for t in ["t1", "t2", "t3"] for o in 1:4 for s in ["a", "b"]]
    d = (series=[r[1] for r in rows], occasion=Float64[r[2] for r in rows],
         trait=[r[3] for r in rows], value=Float64.(sin.(1:length(rows))))
    mk(mode) = mode === :indep ? temporal_indep(:(0 + trait | series), :occasion) :
               mode === :dep ? temporal_dep(:(0 + trait | series), :occasion) :
               temporal_latent(:(0 + trait | series), :occasion; d=1)
    # iterations = 0 evaluates at the start value: cheap, and enough for wording.
    fitm(mode) = fit_temporal_gllvm(d; formula=@formula(value ~ 0 + trait),
        temporal=mk(mode), iterations=0)
    msg(f) = (err = try; f(); nothing; catch e; e; end;
        err isa TemporalContractError ? err.message : nothing)
    action(name) = "See `$(name)()` for the admitted temporal workflow."

    for mode in (:indep, :dep, :latent)
        name = "temporal_$(mode)"
        fit = fitm(mode)
        @testset "$name fit" begin
            @test GMN._temporal_fit_marker(fit) == name
            @test GMN._temporal_fit_action(fit) == action(name)

            m = msg(() -> predict(fit, d))
            @test m !== nothing
            @test occursin("not yet available for `$(name)()` fits.", m)
            other = filter(x -> x != name, ["temporal_indep", "temporal_dep", "temporal_latent"])
            @test !any(o -> occursin("`$(o)()`", m), other)

            for (f, nm) in ((confint, "confint"), (bootstrap_ci, "bootstrap_ci"),
                            (ordination_uncertainty, "ordination_uncertainty"))
                m = msg(() -> f(fit))
                @test m !== nothing
                @test occursin("`$(nm)()` is not available for `$(name)()` fits.", m)
                @test !any(o -> occursin("`$(o)()`", m), other)
            end

            # Default action line follows the fit's own marker (message-only change).
            # (forecast of a dep/latent fit hits the mode refusal, which has its own action line.)
            if mode === :indep
                m = msg(() -> forecast_temporal(fit, (series=["a"],); se_fit=false))
                @test m !== nothing && endswith(m, action(name))
            end
            m = msg(() -> bootstrap_temporal(fit; n_boot=0))
            @test m !== nothing && endswith(m, action(name))
        end
    end

    @testset "temporal_latent() texts are unchanged" begin
        fit = fitm(:latent)
        @test msg(() -> confint(fit)) ==
              "`confint()` is not available for `temporal_latent()` fits.\n" *
              "Its existing algorithm assumes iid latent scores or an iid refit path.\n" *
              "Use `extract_temporal()` for fitted parameters. The bounded `profile_temporal()` and `bootstrap_temporal()` helpers have their own contracts; this generic iid route remains unavailable."
        @test GMN._temporal_fit_action(fit) == GMN._TEMPORAL_ACTION
    end
end

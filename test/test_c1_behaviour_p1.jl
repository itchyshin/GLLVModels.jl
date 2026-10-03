# gllvm-parity-tag: P1
#
# Behavioural twins for three C1 rows (itchyshin/GLLVModels.jl#684 item 2), against gllvmTMB at P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9, 0.7.1):
#   print.anova.gllvmTMB_multi  -- the printed comparison table (R/aghq-report.R)
#   extract_latent_scores.default -- the refusal on an object with no method (R/extract-latent-scores.R)
#   update.gllvmTMB_multi -- replay of the saved call, data override, and the refusals (R/methods-gllvmTMB.R)
# No R at test time. R's side is recorded in test/fixtures/c1_behaviour_p1/ by
# tools/core070_c1_behaviour_p1.R (which first checks the installed functions equal the pinned
# source). Every label compared here is derived from a raw artefact by
# test/fixtures/c1_behaviour_p1/helpers.jl, the same code the receipt writer uses.
#
# What this does not claim: that the two engines raise the same exception class (R's
# cli::cli_abort gives rlang_error, Julia's fallback throws ArgumentError), or that the two print
# the same numbers or the same title text. It compares the printed column labels, the section
# headings, and the three properties of the refusal named in c1b_refusal_labels. For update() it
# compares four behaviours (c1b_update_labels); it does not claim R's evaluate = FALSE, R's
# formula override, or R's "does not retain a public call" refusal have Julia counterparts.
using Test
using GLLVModels
using TOML

include(joinpath(@__DIR__, "fixtures", "c1_behaviour_p1", "helpers.jl"))

@testset "C1 behavioural twins: gllvmTMB P1 (9539352f6)" begin
    rec = c1b_r_record()

    @testset "R fixture provenance" begin
        @test rec["gllvmtmb_commit"] == "9539352f66f2db2cc26b1c393e67212a359b60c9"
        @test rec["gllvmtmb_version"] == "0.7.1"
        prov = rec["provenance"]
        @test !isempty(prov)
        @test all(p -> p["deparse_identical"] === true, prov)
        @test Set(p["name"] for p in prov) ⊇ Set(["print.anova.gllvmTMB_multi", "anova.gllvmTMB_multi",
                                                  "extract_latent_scores.default", "update.gllvmTMB_multi"])
        # The raw printed table is the file the hash was taken from.
        txt = c1b_r_anova_print(rec)
        @test occursin("Likelihood-ratio comparison", txt)
        # Same data as the anova numeric twin: R's three log-likelihoods equal that fixture's.
        fx = TOML.parsefile(C1B_ANOVA_FIXTURE)
        @test isapprox(Float64.(rec["loglik"]), Float64[m["loglik"] for m in fx["per_model"]]; atol = 1e-6, rtol = 0)
    end

    @testset "print.anova.gllvmTMB_multi: printed fields" begin
        r_txt = c1b_r_anova_print(rec)
        j_txt = c1b_julia_anova_print()
        r_fields = c1b_header_fields(r_txt)
        j_fields = c1b_header_fields(j_txt)
        @test length(r_fields) == 9
        @test j_fields == r_fields
        @test r_fields == ["model", "d", "npar", "logLik", "deviance", "df", "LRT", "test", "p.value"]
        @test c1b_section_headings(j_txt) == c1b_section_headings(r_txt) == ["Notes:"]
    end

    @testset "extract_latent_scores.default: refusal on an object with no method" begin
        r_recs = c1b_r_refusals(rec)
        j_recs, exc_types = c1b_julia_refusals()
        @test length(r_recs) == length(j_recs) == 4
        @test all(r -> r.raised, r_recs)
        @test all(r -> r.raised && !r.returned, j_recs)
        @test c1b_refusal_labels(j_recs) == c1b_refusal_labels(r_recs)
        # Recorded, not equated: the exception class names are engine idioms.
        @test all(==("ArgumentError"), exc_types)
        @test all(r -> r["condition_classes"] == ["rlang_error", "error", "condition"], rec["refusal"])
    end
    @testset "update.gllvmTMB_multi: replay, data override, refusals" begin
        r = c1b_update_r_observation(rec)
        j, j_types = c1b_julia_update_observation(rec)
        # R's own observations behave as the R source says (sanity on the recorded fixture).
        @test rec["update"]["original"]["temporal_active"] === true
        @test rec["update"]["replay"]["temporal_active"] === rec["update"]["override"]["temporal_active"] === true
        @test r.nocall.raised && r.unnamed.raised
        # Julia: update() on the temporal fit replays and overrides.
        @test j.replay.returned && j.override.returned
        @test abs(j.replay.loglik - j.original.loglik) <= 1e-10
        @test j.override.response == j.changed_response
        # Same data in, same fit: Julia and R fit the same panel to the same log-likelihood.
        @test j.original.response == r.original.response
        # The labels agree, case by case.
        rl, jl = c1b_update_labels(r), c1b_update_labels(j)
        @test jl.replay == rl.replay == "replays the saved call: returns a refit with the original response and log-likelihood"
        @test jl.override == rl.override
        @test jl.unnamed == rl.unnamed == "signals an error and returns no model"
        @test jl.nocall == rl.nocall == "signals an error and returns no model"
        # Recorded, not equated: the condition types are engine idioms.
        @test j_types.unnamed == "MethodError" && j_types.nocall == "MethodError"
        @test rec["update"]["unnamed"]["condition_classes"][1] == "rlang_error"
        @test rec["update"]["nocall"]["condition_classes"][1] == "simpleError"
    end
end

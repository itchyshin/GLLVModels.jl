# DRAFT, DIAGNOSTIC ONLY. Not wired into runtests.jl. Not a receipt and not evidence for any row status.
#
# POST-COEF-EMPTY ruling packet (docs/dev-log/plans/2026-11-02-c2-post-coef-empty-ruling-packet.md).
# Julia half of the discriminating pair the packet calls option B. The R half is
# tools/core070_coef_empty_pair_draft.R, whose output is committed as
# docs/dev-log/plans/2026-11-02-c2-post-coef-empty-r-diagnostic.json (public gllvmTMB() fits at the P1 pin).
#
# Every pair is compared through the public gllvm(@formula, Y, data) entry point:
#   R  value ~ 0 + latent(..)           Julia  y ~ 0      coef length 0
#   R  value ~ 0 + trait + latent(..)   Julia  y ~ 1      coef length p
#   R  value ~ 0 + trait + x + latent() Julia  y ~ 1 + x  coef length p + 1
# A constant-empty accessor fails the second and third pair. A spurious extra coefficient fails the first.
#
# Run:  julia --project=. test/draft_coef_empty_pair.jl
# Select the accessor under test (default real):  COEF_PAIR_ACCESSOR=always_empty | spurious | real
# With a mutant selected the file is expected to FAIL; that is the red run of the controls.
using GLLVModels, Test, Random
using GLLVModels: StatsAPI
import JSON3

const R_DIAG = joinpath(@__DIR__, "..", "docs", "dev-log", "plans", "2026-11-02-c2-post-coef-empty-r-diagnostic.json")
const PIN = "9539352f66f2db2cc26b1c393e67212a359b60c9"

const ACCESSORS = Dict{String, Function}(
    "real" => StatsAPI.coef,
    "always_empty" => fit -> Float64[],                      # control: constant-empty accessor
    "spurious" => fit -> vcat(StatsAPI.coef(fit), 0.0),      # control: one extra coefficient
)
const ACCESSOR = get(ENV, "COEF_PAIR_ACCESSOR", "real")
haskey(ACCESSORS, ACCESSOR) || error("COEF_PAIR_ACCESSOR must be one of $(sort(collect(keys(ACCESSORS))))")
const coef_fn = ACCESSORS[ACCESSOR]

rdiag = JSON3.read(read(R_DIAG, String))
@assert rdiag.binding == false
@assert rdiag.reference_commit == PIN
p, K, n = Int(rdiag.p), Int(rdiag.K), Int(rdiag.n)

Random.seed!(42)
Y = 0.6 .* randn(p, K) * randn(K, n) .+ 0.5 .* randn(p, n)
site_data = (x = randn(n),)

fit_for(label) = label == "no fixed effects" ? gllvm(@formula(y ~ 0), Y, site_data; K = K) :
                 label == "trait intercepts" ? gllvm(@formula(y ~ 1), Y, site_data; K = K) :
                 label == "trait intercepts plus one shared covariate" ? gllvm(@formula(y ~ 1 + x), Y, site_data; K = K) :
                 error("unknown pair label $label")

@testset "POST-COEF-EMPTY draft pair (accessor = $ACCESSOR)" begin
    for pr in rdiag.pairs
        @testset "$(pr.label): R $(pr.r_formula) vs Julia $(pr.julia_formula)" begin
            c = coef_fn(fit_for(String(pr.label)))
            @test c isa Vector{Float64}
            @test length(c) == Int(pr.coef_length)       # exact integer equality, tolerance 0
        end
    end
    # The pair is only worth having if the two lengths differ, otherwise it cannot discriminate.
    @test length(unique(Int(pr.coef_length) for pr in rdiag.pairs)) == length(rdiag.pairs)
end

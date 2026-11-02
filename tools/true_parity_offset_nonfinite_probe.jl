#!/usr/bin/env julia
# Probe receipt for data/DATA-OFF-PREDICT-NONFINITE-HELPER (maintainer ruling 2026-10-05, vault D-319, item N7).
#
# R's internal helper .gllvmTMB_offset_newdata evaluates the stored offset expression on newdata without a
# finiteness guard, so log(0) comes back as -Inf (receipt CORE070-DATA-OFF-PREDICT-NONFINITE-HELPER-POSTFIT-READBACK).
# GLLVModels takes the offset as a value and refuses a non-finite offset at an observed cell, at fit time and at
# predict time. This script records that refusal on a small deterministic Poisson fit, so the signed disposition
# cites an observed Julia behaviour rather than a reading of the source. It compares no number with R.
#
# Usage (from the repository root):
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=. tools/true_parity_offset_nonfinite_probe.jl
# Writes docs/dev-log/core070/true-parity-latest/receipts/data/n7-offset-nonfinite-probe.json.

using GLLVModels
const GMJ = GLLVModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUT = "docs/dev-log/core070/true-parity-latest/receipts/data/n7-offset-nonfinite-probe.json"
const P1_SHA = "9539352f66f2db2cc26b1c393e67212a359b60c9"
const R_RECEIPT = "docs/dev-log/core070/true-parity-latest/receipts/data/cases/CORE070-DATA-OFF-PREDICT-NONFINITE-HELPER-POSTFIT-READBACK.json"

json_escape(s::AbstractString) = replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n")
to_json(x::Bool) = x ? "true" : "false"
to_json(x::AbstractString) = "\"" * json_escape(x) * "\""
to_json(x::Integer) = string(x)
to_json(x::AbstractVector) = "[" * join(to_json.(x), ", ") * "]"
to_json(x::Pair) = to_json(x.first) * ": " * to_json(x.second)
to_json(x::Tuple) = "{" * join(to_json.(collect(x)), ", ") * "}"

# Deterministic data (no RNG): 4 traits x 12 units of counts, a unit-level exposure offset.
p, n = 4, 12
Y = [1 + (t * s) % 4 for t in 1:p, s in 1:n]
O = [log(1.0 + s % 3) for _ in 1:p, s in 1:n]
O_bad = copy(O)
O_bad[1, 1] = log(0.0)          # -Inf at an observed cell, as R's log(e) gives for e = 0

function attempt(f)
    try
        f()
        return (false, "", "")
    catch e
        return (true, string(nameof(typeof(e))), first(sprint(showerror, e), 240))
    end
end

fit = GMJ.fit_gllvm(Y; family = GMJ.Poisson(), K = 1, offset = O)
fit_ok = all(isfinite, GMJ.predict(fit, Y; type = :link))
probes = [
    ("predict(fit, Y; type = :link, offset = O with -Inf at trait 1, unit 1)",
     attempt(() -> GMJ.predict(fit, Y; type = :link, offset = O_bad))),
    ("fit_gllvm(Y; family = Poisson(), K = 1, offset = O with -Inf at trait 1, unit 1)",
     attempt(() -> GMJ.fit_gllvm(Y; family = GMJ.Poisson(), K = 1, offset = O_bad))),
]
all(pr[2][1] && pr[2][2] == "ArgumentError" for pr in probes) ||
    error("expected both probes to refuse with ArgumentError; got $(probes)")

head = strip(read(`git -C $ROOT rev-parse HEAD`, String))
dirty = strip(read(`git -C $ROOT status --porcelain --untracked-files=no`, String))
body = (
    "schema" => "true-parity-julia-probe-receipt/v1",
    "source_ids" => ["data/DATA-OFF-PREDICT-NONFINITE-HELPER"],
    "ruling" => "maintainer ruling 2026-10-05 (D-319), item N7",
    "pin" => "P1",
    "reference_commit" => P1_SHA,
    "generator" => "tools/true_parity_offset_nonfinite_probe.jl",
    "glvmodels_commit" => head,
    "glvmodels_worktree_dirty" => !isempty(dirty),
    "fixture" => "deterministic 4 x 12 Poisson counts Y[t, s] = 1 + (t*s) % 4, offset O[t, s] = log(1 + s % 3); no RNG",
    "control_fit_with_finite_offset_predicts_finite_link" => fit_ok,
    "probes" => [(("call" => c), ("refused" => r[1]), ("error_type" => r[2]), ("message" => r[3])) for (c, r) in probes],
    "r_side" => (("receipt" => R_RECEIPT),
                 ("behaviour" => ".gllvmTMB_offset_newdata evaluates log(e) on newdata e = c(0, 1, 2) and returns " *
                                 "c(-Inf, 0, log(2)) with no error (identical() TRUE at P1)")),
    "finding" => "By design the two engines differ at this internal helper: R propagates -Inf from the offset " *
                 "expression, Julia refuses a non-finite offset at an observed cell with an ArgumentError, at fit " *
                 "and at predict time. No R-vs-Julia number is compared.",
)
mkpath(dirname(joinpath(ROOT, OUT)))
write(joinpath(ROOT, OUT), to_json(body) * "\n")
println("wrote ", OUT)

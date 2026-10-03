# Shared by test/test_c1_behaviour_p1.jl and the behavioural receipt writer
# (tools/true_parity_julia_receipts.jl, section "c1-behaviour"), so the test and the receipt
# derive every label with the same code. A label is never typed: it is read off a raw artefact,
# the text one engine printed or the condition it raised. Raw R side: r_c1_behaviour.toml and
# r_anova_print.txt, written by tools/core070_c1_behaviour_p1.R at gllvmTMB P1
# (9539352f66f2db2cc26b1c393e67212a359b60c9).
#
# Needs `using GLLVModels, TOML` in the including scope.

const C1B_DIR = joinpath(@__DIR__)
const C1B_R_TOML = joinpath(C1B_DIR, "r_c1_behaviour.toml")
const C1B_ANOVA_FIXTURE = joinpath(@__DIR__, "..", "gllvmtmb_anova_fixture.toml")

c1b_r_record() = TOML.parsefile(C1B_R_TOML)
c1b_r_anova_print(rec) = read(joinpath(C1B_DIR, rec["anova_print_file"]), String)

# ---- printed fields ---------------------------------------------------------------------
# The two tables are a title line, a blank line, then one header line of column labels.
function c1b_header_fields(text::AbstractString)
    lines = split(text, '\n')
    length(lines) >= 3 || error("printed table too short to hold a header line")
    i = findfirst(l -> !isempty(strip(l)), lines[2:end])
    i === nothing && error("no header line after the title")
    return String.(split(strip(lines[1 + i])))
end

# Section headings: a line that starts in column 1 and is one capitalised word and a colon.
c1b_section_headings(text::AbstractString) =
    String[strip(l) for l in split(text, '\n') if occursin(r"^[A-Z][a-z]+:\s*$", l)]

# Julia side: the three nested Gaussian fits of the anova twin fixture, compared and printed.
function c1b_julia_anova_print()
    fx = TOML.parsefile(C1B_ANOVA_FIXTURE)
    Yw = fx["Y_wide"]
    p, n = length(Yw), length(Yw[1])
    Y = Matrix{Float64}(undef, p, n)
    for i in 1:p, j in 1:n
        Y[i, j] = Yw[i][j]
    end
    X = zeros(p, n, p)
    for t in 1:p, s in 1:n
        X[t, s, t] = 1.0
    end
    fits = [fit_gaussian_gllvm(Y; X = X, K = K) for K in 1:3]
    tab = gllvm_anova(fits...; test = :chibar)
    io = IOBuffer()
    show(io, MIME"text/plain"(), tab)
    return String(take!(io))
end

# ---- refusal ----------------------------------------------------------------------------
# One record per object the engine has no method for:
#   (object_class, raised, message, returned)
# R's records come from r_c1_behaviour.toml; Julia's from calling the package.
c1b_r_refusals(rec) = [(object_class = String(r["object_class"]), raised = Bool(r["raised"]),
                         message = String(r["message"]), returned = !isempty(r["returned_class"]))
                        for r in rec["refusal"]]

const C1B_JULIA_OBJECTS = Any["a string", [1, 2, 3], Dict("a" => 1), nothing]   # R: character, integer, list, NULL

function c1b_julia_refusals()
    out = NamedTuple[]
    exc_types = String[]
    for x in C1B_JULIA_OBJECTS
        raised, msg, returned = false, "", false
        try
            extract_latent_scores(x)
            returned = true
        catch e
            raised = true
            msg = e isa ArgumentError ? e.msg : sprint(showerror, e)
            push!(exc_types, string(typeof(e)))
        end
        push!(out, (object_class = string(typeof(x)), raised = raised, message = msg, returned = returned))
    end
    return out, exc_types
end

_c1b_collapse(s) = join(split(s), " ")

# Three labels, each true or false of EVERY object tried. They say the same thing about both
# engines: what happened, whether the message names the offending class, whether it says what
# is accepted. The exception class names (R: rlang_error; Julia: ArgumentError) are engine
# idioms and are recorded beside the labels, not compared.
function c1b_refusal_labels(records)
    all_ = f -> all(f, records)
    outcome = all_(r -> r.raised && !r.returned) ? "signals an error and returns no value" :
              "does not always signal an error"
    names_class = all_(r -> occursin(r.object_class, _c1b_collapse(r.message))) ?
        "message names the offending class" : "message does not always name the offending class"
    states_accepted = all_(r -> occursin(r"expected a fitted"i, _c1b_collapse(r.message))) ?
        "message states the accepted inputs" : "message does not always state the accepted inputs"
    return [outcome, names_class, states_accepted]
end

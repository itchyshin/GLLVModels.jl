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

# ---- update() -----------------------------------------------------------------------------
# model-comparison/update.gllvmTMB_multi. R's method replays the saved public call of a temporal
# fit with named overrides; on any fit that keeps no call it falls to stats::update.default,
# which errors. Julia's twin is update(::TemporalGaussianFit; ...) (src/temporal_methods.jl),
# and Julia has no update method for any other fit. Both sides are observed on the same panel,
# recorded literally in the [update] table of r_c1_behaviour.toml. An observation is
#   (original = (loglik, response),
#    replay = (returned, loglik, response), override = (returned, loglik, response),
#    changed_response,
#    unnamed = (raised, returned), nocall = (raised, returned)).
c1b_update_r_observation(rec) = begin
    u = rec["update"]
    f64(v) = Float64.(v)
    (original = (loglik = Float64(u["original"]["loglik"]), response = f64(u["original"]["response"])),
     replay = (returned = true, loglik = Float64(u["replay"]["loglik"]), response = f64(u["replay"]["response"])),
     override = (returned = true, loglik = Float64(u["override"]["loglik"]), response = f64(u["override"]["response"])),
     changed_response = f64(u["changed_value"]),
     unnamed = (raised = Bool(u["unnamed"]["raised"]), returned = !isempty(u["unnamed"]["returned_class"])),
     nocall = (raised = Bool(u["nocall"]["raised"]), returned = !isempty(u["nocall"]["returned_class"])))
end

# Julia side: fit the temporal model on R's panel, then call update() the four ways. The
# exception types are returned beside the observation (recorded, not compared).
function c1b_julia_update_observation(rec)
    u = rec["update"]
    tbl = (series = String.(u["series"]), occasion = Float64.(u["occasion"]), trait = String.(u["trait"]),
           value = Float64.(u["value"]))
    f = fit_temporal_gllvm(tbl; formula = @formula(value ~ 0 + trait),
        temporal = temporal_indep(:(0 + trait | series), :occasion), unit = :series)
    replay = update(f)
    changed = merge(tbl, (value = Float64.(u["changed_value"]),))
    override = update(f; data = changed)
    attempt(thunk) = try
        thunk(); (raised = false, returned = true, type = "")
    catch e
        (raised = true, returned = false, type = string(typeof(e)))
    end
    unnamed = attempt(() -> update(f, changed))
    # An ordinary fit: the same responses as a traits x units matrix (units = series x occasion).
    Y = Matrix{Float64}(reshape(tbl.value, 20, 3)')
    ordinary = fit_gaussian_gllvm(Y; K = 1)
    nocall = attempt(() -> update(ordinary))
    obs = (original = (loglik = f.loglik, response = copy(f.y)),
           replay = (returned = replay isa TemporalGaussianFit, loglik = replay.loglik, response = copy(replay.y)),
           override = (returned = override isa TemporalGaussianFit, loglik = override.loglik, response = copy(override.y)),
           changed_response = Float64.(u["changed_value"]),
           unnamed = (raised = unnamed.raised, returned = unnamed.returned),
           nocall = (raised = nocall.raised, returned = nocall.returned))
    return obs, (unnamed = unnamed.type, nocall = nocall.type)
end

# The labels. They say the same thing about both engines. The log-likelihood threshold is a
# behavioural predicate (a replay reproduces the fit), not a parity tolerance: R's own temporal
# update test uses 1e-6 on the objective.
const C1B_UPDATE_LL_TOL = 1e-6
const C1B_UPDATE_RESPONSE_TOL = 1e-12
function c1b_update_labels(o)
    same_y(a, b) = length(a) == length(b) && maximum(abs.(a .- b)) <= C1B_UPDATE_RESPONSE_TOL
    replay = (o.replay.returned && same_y(o.replay.response, o.original.response) &&
              abs(o.replay.loglik - o.original.loglik) <= C1B_UPDATE_LL_TOL) ?
        "replays the saved call: returns a refit with the original response and log-likelihood" :
        "does not replay the saved call"
    override = (o.override.returned && same_y(o.override.response, o.changed_response) &&
                !same_y(o.override.response, o.original.response) &&
                abs(o.override.loglik - o.original.loglik) > C1B_UPDATE_LL_TOL) ?
        "refits on the supplied data: the response is the supplied column and the log-likelihood moves" :
        "does not refit on the supplied data"
    refusal(x) = x.raised && !x.returned ? "signals an error and returns no model" : "does not refuse"
    return (replay = replay, override = override, unnamed = refusal(o.unnamed), nocall = refusal(o.nocall))
end

# ---- print.gllvmTMB_select_lv -------------------------------------------------------------
# select-lv/print.gllvmTMB_select_lv. R prints, for a select_lv() result, a title line, then
# print(df, row.names = FALSE) of a data frame whose first column is a one-character marker
# (header ""), then the columns d npar logLik AIC BIC AICc conv pdHess (R/select-lv.R:349-369
# at P1). Julia's show method for LVSelection (src/model_selection.jl) prints the same layout.
# Both engines are observed on the data of the numeric select_lv twin, test/fixtures/select_lv_p1_data.csv.
# R's text is stored verbatim in r_select_lv_print.txt; Julia's is produced here by calling select_lv with
# pd_hessian = true (without it Julia prints NA in the pdHess column, R prints TRUE or FALSE).
const C1B_SELECT_LV_FIXTURE = joinpath(@__DIR__, "..", "select_lv_p1.toml")

c1b_r_select_lv_print(rec) = read(joinpath(C1B_DIR, rec["select_lv"]["print_file"]), String)

# Y (traits x units) from the CSV that holds R's data, as the numeric select_lv twin loads it.
function c1b_select_lv_data(fx)
    path = joinpath(dirname(C1B_SELECT_LV_FIXTURE), fx["data_file"])
    trait_names = String.(fx["trait_names"])
    Y = zeros(Float64, length(trait_names), Int(fx["n_unit"]))
    open(path) do io
        readline(io)
        for line in eachline(io)
            isempty(line) && continue
            parts = split(line, ",")
            t = findfirst(==(strip(parts[2], '"')), trait_names)
            t === nothing && error("unrecognised trait in $path")
            Y[t, parse(Int, strip(parts[1], '"'))] = parse(Float64, parts[3])
        end
    end
    return Y
end

# Julia side: the sweep K = 1:3 on that data, printed. Returns the text and the LVSelection.
function c1b_julia_select_lv_print()
    fx = TOML.parsefile(C1B_SELECT_LV_FIXTURE)
    Y = c1b_select_lv_data(fx)
    sel = select_lv(Y; Kmax = 3, criterion = :bic, pd_hessian = true)   # family defaults to Normal()
    return sprint(show, MIME"text/plain"(), sel), sel
end

# Split a printed table into title, column labels, marker cells and body cells. The layout is
# print(df, row.names = FALSE): every line starts with a blank, the marker cell is the one
# character at position 2, and the other cells are the whitespace-separated tokens after it. The
# marker column's own label is empty, so the column labels are the tokens of the header line.
# A table row is a line of that shape with one token per label; the first line that is not one
# ends the table (notes and failed-fit lines follow it).
function c1b_select_lv_table(text::AbstractString)
    lines = String.(split(chomp(text), '\n'))
    length(lines) >= 3 || error("printed select_lv table too short")
    labels = String.(split(strip(lines[2])))
    markers = Char[]
    rows = Vector{String}[]
    for l in lines[3:end]
        (length(l) >= 3 && l[1] == ' ' && l[2] in (' ', '*') && l[3] == ' ') || break
        cells = String.(split(l[3:end]))
        length(cells) == length(labels) || break
        push!(markers, l[2])
        push!(rows, cells)
    end
    isempty(rows) && error("no table rows after the header line")
    return (title = lines[1], header = lines[2], labels = labels, markers = markers, rows = rows)
end

# Three statements about the marker, each derived from the printed table alone, so the same
# code reads both engines: which character marks a row, how many rows carry it, and whether the
# marked row has the smallest value in the column of the criterion the title names.
function c1b_select_lv_marker_labels(t)
    marked = findall(!=(' '), t.markers)
    chars = unique(t.markers[marked])
    count_label = length(chars) == 1 ?
        "marks " * (length(marked) == 1 ? "exactly one row" : "$(length(marked)) rows") * ", with " * only(chars) :
        "uses $(length(chars)) different marker characters"
    crit = match(r"criterion = (\w+)", t.title)
    crit === nothing && error("no 'criterion = ' in the title line: $(t.title)")
    ccol = findfirst(l -> lowercase(l) == lowercase(crit.captures[1]), t.labels)
    ccol === nothing && error("title names criterion $(crit.captures[1]) but no column of that name")
    val(c) = c == "NA" ? Inf : parse(Float64, c)
    best = argmin([val(r[ccol]) for r in t.rows])
    min_label = length(marked) == 1 && only(marked) == best ?
        "the marked row has the smallest value of the criterion column" :
        "the marked row is not the one with the smallest value of the criterion column"
    return [count_label, min_label]
end

#!/usr/bin/env julia
# Classify the C6 reverse-gap items (GLLVModels exports with no gllvmTMB counterpart) and write
# docs/dev-log/core070/true-parity-latest/reverse-gap-decisions.json.
#
# This script applies the maintainer's signed ruling itchyshin/GLLVModels.jl#684 item 3
# mechanically. It decides nothing by hand: an export is decided only when the criterion below
# gives a clear answer, otherwise it stays out of the file (undecided) and is listed on stdout.
#
# CRITERION (written verbatim into the output file's "criterion" field; the test in
# tools/test_true_parity_c6_classify.jl fails if this header and the constant drift apart):
#
# Applied mechanically to each reverse-gap item (a julia export with no gllvmTMB counterpart) under
# the maintainer's signed ruling itchyshin/GLLVModels.jl#684 item 3. A mechanical rule decides only
# the clear cases; every uncertain case is held undecided (left out of this file) and listed for the
# maintainer. EXCLUDED_INTERNAL_HELPER when tools/parity_ledger.py --ref 9539352f6 classes the Julia
# name as internal (the class begins with internal, as in "internal marginal-likelihood kernel ..."
# and "internal-but-exported helper; candidate to unexport", or it says "engine internal" or
# "internals;"), regardless of whether it has a docstring, unless a guard below holds it; the basis
# quotes the ledger class. KEPT_AS_JULIA_EXTRA when the name is not classed internal, has a
# docstring, and that docstring is rendered on a docs/src page, unless a guard below holds it; the
# basis cites the page path. Otherwise undecided. Mechanical details: a class says the Julia name is
# internal only in the forms above, so a class that mentions an internal knob of R (as in "an
# internal control knob"), or the word internally, does not count. Guards, each of which leaves a
# name undecided: (1) its binding is owned by another package (a re-export); (2) parity_ledger.py
# ALIASES maps an R export to it, or to another export bound to the same object (===), because the
# ledger counts it as the Julia twin of an R export, so it is not Julia-only; (3) its authored
# docstring was silently dropped by Julia because a comment or blank line separates the docstring
# from its definition; (4) it is bound to the same object (===) as another export and the per-name
# verdicts differ, in which case all of them are held, or as an export that matches an R name; (5)
# it would be EXCLUDED but user-facing prose presents it as a user tool, meaning it is named on a
# docs/src page other than api.md and low-level-reference.md, or in README.md; (6) it would be
# EXCLUDED but its internal class is marked PROPOSED (unsigned), which would make that class the
# only basis for the exclusion; (7) it would be KEPT but its own ledger class says R has it ("R has
# it under another name", "R has it after 0.7.0"), so it is not Julia-only; (8) it is on the named
# review list of this script (NAMED_HOLDS): em_fit_phylo, em_fit_phylo_squarem and
# em_observed_information, whose ledger class (^em_, "solver internals") contradicts their
# docstrings, which document user fitters that return the public EMPhyloFit, and nine low-level
# names (ZI_LAPLACE_EIGMIN_FLOOR, random_balanced_tree, estep_edge_moments, AnBSparseSolver,
# solve_AnB, build_AnB_sparse, Q_times_x, precision_logdet_check, shrinkage_factor) whose docstrings
# address developers, not users.
#
# Inputs : docs/dev-log/core070/true-parity-latest/reverse-gap.json (the assembler's item list;
#          only its "name" fields are read), reverse-gap-inputs.json (checked against
#          names(GLLVModels)), the docstrings and the object identity of the loaded GLLVModels
#          exports, docs/src/**/*.md (the @docs blocks, and where a name is mentioned), README.md
#          (mentions only), src/**/*.jl (scanned for dropped docstrings) and the class tables of
#          tools/parity_ledger.py, read through tools/true_parity_c6_ledger_classes.py.
# Output : reverse-gap-decisions.json (deterministic: names sorted, fixed key order).
#
# A docstring that Julia dropped is an error in every src file, exported or not: the run prints
# DROPPED_DOCSTRING lines and exits 1. Move the comment above the docstring.
#
# Usage (from the repo root):
#   export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4
#   julia --project=. tools/true_parity_c6_classify.jl            # write the file
#   julia --project=. tools/true_parity_c6_classify.jl --check    # fail if the file is stale
#   julia --project=. tools/test_true_parity_c6_classify.jl       # fixture tests for this script
#
# Refresh order after names(GLLVModels) changes (another branch adds or removes an export):
#   1. python3 tools/true_parity_assemble.py --refresh-reverse-gap-inputs --julia-names-tsv names.tsv
#   2. python3 tools/true_parity_assemble.py        (if it stops on "names X, which is not a
#      reverse-gap item (stale)", delete reverse-gap-decisions.json and run it again)
#   3. julia --project=. tools/true_parity_c6_classify.jl
#   4. python3 tools/true_parity_assemble.py        (copies the decisions into reverse-gap.json)
# Docstring lookup uses Base.Docs.meta(owner)[Base.Docs.Binding(owner, name)], which works on
# Julia 1.10.

# The pure functions below (decide, resolve_objects, dropped_docstrings, rendered_names_in, render) need no package.
# Only a live run loads GLLVModels; a test include()s this file without running main.
const LIVE = !isempty(PROGRAM_FILE) && realpath(PROGRAM_FILE) == realpath(@__FILE__)
if LIVE
    using GLLVModels
end

const ROOT = normpath(joinpath(@__DIR__, ".."))
const LEDGER = joinpath(ROOT, "docs/dev-log/core070/true-parity-latest")
const OUT = joinpath(LEDGER, "reverse-gap-decisions.json")
const IN_GAP = joinpath(LEDGER, "reverse-gap.json")
const IN_INPUTS = joinpath(LEDGER, "reverse-gap-inputs.json")
const LEDGER_PY = joinpath(@__DIR__, "true_parity_c6_ledger_classes.py")
const LEDGER_REF = "9539352f6"

const CRITERION = "Applied mechanically to each reverse-gap item (a julia export with no gllvmTMB counterpart) " *
    "under the maintainer's signed ruling itchyshin/GLLVModels.jl#684 item 3. A mechanical rule " *
    "decides only the clear cases; every uncertain case is held undecided (left out of this file) and " *
    "listed for the maintainer. EXCLUDED_INTERNAL_HELPER when tools/parity_ledger.py --ref 9539352f6 " *
    "classes the Julia name as internal (the class begins with internal, as in \"internal " *
    "marginal-likelihood kernel ...\" and \"internal-but-exported helper; candidate to unexport\", or it " *
    "says \"engine internal\" or \"internals;\"), regardless of whether it has a docstring, unless a " *
    "guard below holds it; the basis quotes the ledger class. KEPT_AS_JULIA_EXTRA when the name is " *
    "not classed internal, has a docstring, and that docstring is rendered on a docs/src page, unless " *
    "a guard below holds it; the basis cites the page path. Otherwise undecided. Mechanical details: " *
    "a class says the Julia name is internal only in the forms above, so a class that mentions an " *
    "internal knob of R (as in \"an internal control knob\"), or the word internally, does not count. " *
    "Guards, each of which leaves a name undecided: (1) its binding is owned by another package (a " *
    "re-export); (2) parity_ledger.py ALIASES maps an R export to it, or to another export bound to " *
    "the same object (===), because the ledger counts it as the Julia twin of an R export, so it is " *
    "not Julia-only; (3) its authored docstring was silently dropped by Julia because a comment or " *
    "blank line separates the docstring from its definition; (4) it is bound to the same object (===) " *
    "as another export and the per-name verdicts differ, in which case all of them are held, or as an " *
    "export that matches an R name; (5) it would be EXCLUDED but user-facing prose presents it as a " *
    "user tool, meaning it is named on a docs/src page other than api.md and low-level-reference.md, " *
    "or in README.md; (6) it would be EXCLUDED but its internal class is marked PROPOSED (unsigned), " *
    "which would make that class the only basis for the exclusion; (7) it would be KEPT but its own " *
    "ledger class says R has it (\"R has it under another name\", \"R has it after 0.7.0\"), so it is not " *
    "Julia-only; (8) it is on the named review list of this script (NAMED_HOLDS): em_fit_phylo, " *
    "em_fit_phylo_squarem and em_observed_information, whose ledger class (^em_, \"solver internals\") " *
    "contradicts their docstrings, which document user fitters that return the public EMPhyloFit, and " *
    "nine low-level names (ZI_LAPLACE_EIGMIN_FLOOR, random_balanced_tree, estep_edge_moments, " *
    "AnBSparseSolver, solve_AnB, build_AnB_sparse, Q_times_x, precision_logdet_check, " *
    "shrinkage_factor) whose docstrings address developers, not users."

const GENERATOR = "tools/true_parity_c6_classify.jl"
const RULING_REF = "itchyshin/GLLVModels.jl#684 item 3"

jstr(s) = "\"" * replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""

# ---------------------------------------------------------------------------------------------
# The decision rule. Pure: everything it needs is in Facts.
# ---------------------------------------------------------------------------------------------

struct Facts
    name::String
    ledger_class::String         # tools/parity_ledger.py class; "" when it has none
    alias_of::Vector{String}     # R exports whose parity_ledger.py ALIASES entry maps to this name
    foreign_owner::String        # owning module when it is not GLLVModels; "" otherwise
    has_docstring::Bool          # runtime Docs.meta holds a docstring for the binding
    dropped::String              # "src/file.jl:LINE" of an authored docstring Julia dropped; ""
    pages::Vector{String}        # docs/src pages whose @docs block names the binding
    mentions::Vector{String}     # docs/src pages and README.md that mention the name
    twin_aliases::Vector{String} # "Other (R x, y)": other exports bound to the same object (===) that ALIASES maps
end

# The ledger class says the JULIA name is internal only in these forms: the class begins with
# internal (after an optional [PROPOSED ...] tag), or it says internal-but-exported, engine internal
# or "internals;". A class that mentions an internal knob of R ("an internal control knob") or the
# word internally says nothing about the Julia name.
const INTERNAL_CLASS = r"^(?:\[PROPOSED[^\]]*\]\s*)?internal\b|\binternal-but-exported\b|\bengine internal\b|\binternals;"i
is_internal_class(cls::AbstractString) = occursin(INTERNAL_CLASS, cls)

# The agent-proposed classes of tools/parity_ledger.py carry a "[PROPOSED date, unsigned -- doc]" tag.
const PROPOSED_TAG = r"^\[PROPOSED[^\]]*\]\s*"
is_proposed_class(cls::AbstractString) = occursin(PROPOSED_TAG, cls)
# A class whose own text says R has the twin ("R has it under another name", "R has it after 0.7.0").
says_r_has_it(cls::AbstractString) = startswith(replace(cls, PROPOSED_TAG => ""), "R has it")

# User-facing prose: every docs/src page except these two reference pages, and README.md.
const REFERENCE_PAGES = ("docs/src/api.md", "docs/src/low-level-reference.md")
prose_pages(f::Facts) = [p for p in f.mentions if !(p in REFERENCE_PAGES)]

mention_note(f::Facts) = isempty(f.mentions) ? "" : "; named in " * join(f.mentions, ", ")

# Names held by the final review of the PR, each with the group it is reported under and why. The
# mechanical rule alone would decide these (the ^em_ ledger class excludes the three EM names, and the
# nine low-level names are KEPT because they have a rendered docstring), but the docstring says
# otherwise, so they stay undecided for the maintainer.
const EM_REASON = "the ledger's ^em_ class calls it an EM/SQUAREM solver internal, but its docstring " *
    "documents it as a user fitter or its post-fit standard errors, built on the public EMPhyloFit"
const LOW_REASON = "documented, but the docstring addresses developers (engine mechanics, a tuning " *
    "constant, a benchmark helper or a checksum), not a call a user is asked to make; the ledger gives " *
    "it no internal class, so the rule alone would KEEP it"
const NAMED_HOLDS = Dict{String,Tuple{String,String}}(
    "em_fit_phylo" => ("em_user_fitter", EM_REASON),
    "em_fit_phylo_squarem" => ("em_user_fitter", EM_REASON),
    "em_observed_information" => ("em_user_fitter", EM_REASON),
    "ZI_LAPLACE_EIGMIN_FLOOR" => ("low_level", LOW_REASON),
    "random_balanced_tree" => ("low_level", LOW_REASON),
    "estep_edge_moments" => ("low_level", LOW_REASON),
    "AnBSparseSolver" => ("low_level", LOW_REASON),
    "solve_AnB" => ("low_level", LOW_REASON),
    "build_AnB_sparse" => ("low_level", LOW_REASON),
    "Q_times_x" => ("low_level", LOW_REASON),
    "precision_logdet_check" => ("low_level", LOW_REASON),
    "shrinkage_factor" => ("low_level", LOW_REASON),
)

# decide returns (decision, text, kind): decision is "EXCLUDED_INTERNAL_HELPER", "KEPT_AS_JULIA_EXTRA" or
# nothing (undecided); text is the basis (decided) or the reason (undecided); kind is "" when decided and
# the reason group when undecided.
function decide(f::Facts)
    if haskey(NAMED_HOLDS, f.name)
        kind, why = NAMED_HOLDS[f.name]
        return (nothing, why, kind)
    elseif is_internal_class(f.ledger_class)
        holds = Tuple{String,String}[]
        prose = prose_pages(f)
        isempty(prose) || push!(holds, ("user_tool", "user-facing prose names it, so the internal class " *
            "does not settle it: " * join(prose, ", ")))
        is_proposed_class(f.ledger_class) && push!(holds, ("proposed_class", "its internal class is PROPOSED " *
            "(unsigned), so that class is the only basis for excluding it: \"" * f.ledger_class * "\""))
        isempty(holds) || return (nothing, join(last.(holds), "; also "), first(holds)[1])
        return ("EXCLUDED_INTERNAL_HELPER",
            "tools/parity_ledger.py --ref $LEDGER_REF classes it as internal: \"" * f.ledger_class * "\"", "")
    elseif !isempty(f.foreign_owner)
        return (nothing, "binding owned by $(f.foreign_owner), a re-export, not a GLLVModels definition" * mention_note(f), "reexport")
    elseif !isempty(f.alias_of)
        return (nothing, "tools/parity_ledger.py ALIASES maps R " * join(f.alias_of, ", ") *
            " to this name, so the ledger counts it as the Julia twin of an R export; not Julia-only", "alias_twin")
    elseif !isempty(f.twin_aliases)
        return (nothing, "bound to the same object as " * join(f.twin_aliases, "; ") * ", which tools/parity_ledger.py " *
            "ALIASES maps to an R export, so it is the same Julia twin of that R export; not Julia-only", "alias_twin")
    elseif !isempty(f.dropped)
        return (nothing, "an authored docstring at $(f.dropped) is silently dropped by Julia " *
            "(a comment or blank line separates it from its definition)", "dropped_docstring")
    elseif f.has_docstring && !isempty(f.pages)
        says_r_has_it(f.ledger_class) && return (nothing, "its own ledger class says R has it, so it is not " *
            "Julia-only: \"" * f.ledger_class * "\"", "r_twin_class")
        return ("KEPT_AS_JULIA_EXTRA",
            "not classed internal by tools/parity_ledger.py --ref $LEDGER_REF; has a docstring, " *
            "rendered in an @docs block on " * join(f.pages, ", "), "")
    elseif f.has_docstring
        return (nothing, "has a docstring but no docs/src @docs block renders it" * mention_note(f), "no_rendered_docstring")
    else
        return (nothing, "no docstring" * mention_note(f), "no_docstring")
    end
end

# Per-object resolution. Two exports bound to the same object (===) are one thing: when their per-name
# verdicts differ, every one of them is held (a decided name joins an undecided twin; the undecided one
# keeps its own reason); a name that shares its object with an export that is not a reverse-gap item (an
# R name matches that export) is held too. `results` maps each reverse-gap name to its
# decide() triple, `twins` maps each name to the other exports bound to the same object.
const Verdict = Tuple{Union{Nothing,String},String,String}
function resolve_objects(results::Dict{String,Verdict}, twins::Dict{String,Vector{String}})
    out = copy(results)
    for (n, (d, _, _)) in results
        d === nothing && continue                      # already undecided: keep its own, more specific reason
        tw = sort(get(twins, n, String[]))
        isempty(tw) && continue
        outside = [t for t in tw if !haskey(results, t)]
        if !isempty(outside)
            out[n] = (nothing, "bound to the same object as " * join(outside, ", ") * ", which matches an R name " *
                "(it is not a reverse-gap item), so it is not Julia-only", "same_object")
        elseif any(results[t][1] != d for t in tw)
            verdicts = join(("$m: " * something(results[m][1], "undecided") for m in sort([n; tw])), ", ")
            out[n] = (nothing, "bound to the same object as " * join(tw, ", ") * "; the per-name verdicts " *
                "differ ($verdicts), so the object is held as a whole", "same_object")
        end
    end
    out
end

# ---------------------------------------------------------------------------------------------
# Docstrings that Julia drops. A docstring attaches only when the definition follows it on the very
# next line; a comment line or a blank line in between leaves Docs.meta empty (Julia 1.10, checked).
# ---------------------------------------------------------------------------------------------

# Returns (line, name) for each top-level docstring ("""...""" or a one-line "...") whose next
# non-gap line is separated from it by blank or comment lines; `line` is the docstring's closing
# line. `name` is the first identifier of the docstring's header line (the signature line, 4-space
# indented) or "" when there is none. `@doc raw"""` and other prefixed strings are skipped, not judged.
function dropped_docstrings(text::AbstractString)
    lines = split(text, '\n')
    out = Tuple{Int,String}[]
    i = 1
    while i <= length(lines)
        s = lines[i]
        n3 = count("\"\"\"", s)
        close_at = 0
        header = ""
        if n3 % 2 == 1
            # a triple-quoted string opens here; find where it closes
            j = i + 1
            while j <= length(lines) && !occursin("\"\"\"", lines[j])
                j += 1
            end
            j > length(lines) && break
            if startswith(s, "\"\"\"")
                close_at = j
                for k in (i + 1):(j - 1)
                    if !isempty(strip(lines[k]))
                        header = lines[k]
                        break
                    end
                end
            end
            i = j + 1
        else
            if n3 == 2 && startswith(s, "\"\"\"")
                close_at = i
                header = s
            elseif n3 == 0 && occursin(r"^\"[^\"]", s) && endswith(rstrip(s), "\"")
                close_at = i
                header = s
            end
            i += 1
        end
        close_at == 0 && continue
        k = close_at + 1
        while k <= length(lines) && (isempty(strip(lines[k])) || startswith(lstrip(lines[k]), "#"))
            k += 1
        end
        gap = k - close_at - 1
        (gap > 0 && k <= length(lines)) || continue
        m = match(r"^\s*\"*\s*(@?[A-Za-z_][A-Za-z0-9_!]*)", header)
        push!(out, (close_at, m === nothing ? "" : String(m.captures[1])))
    end
    out
end

# ---------------------------------------------------------------------------------------------
# docs/src @docs blocks
# ---------------------------------------------------------------------------------------------

# Names an @docs block in `text` names (last dotted segment, before any "(").
function rendered_names_in(text::AbstractString, label::AbstractString = "page")
    out = Set{String}()
    inblock = false
    for line in split(text, '\n')
        s = strip(line)
        if startswith(s, "```@autodocs")
            error("$label uses @autodocs; extend this classifier before trusting it")
        elseif startswith(s, "```@docs")
            inblock = true
        elseif inblock && startswith(s, "```")
            inblock = false
        elseif inblock && !isempty(s) && !startswith(s, "#")
            head = first(split(s, '('; limit = 2))
            push!(out, String(last(split(head, r"\.(?=@?[A-Za-z_])"))))
        end
    end
    out
end

# ---------------------------------------------------------------------------------------------
# Live inputs
# ---------------------------------------------------------------------------------------------

relpage(p) = relpath(p, ROOT)

function walk_files(dir, ext)
    files = String[]
    for (d, _, fs) in walkdir(dir)
        for f in fs
            endswith(f, ext) && push!(files, joinpath(d, f))
        end
    end
    sort!(files)
end

function unescape_u(raw::AbstractString)
    # The assembler writes its files with ensure_ascii, so non-ASCII names arrive as \uXXXX.
    replace(String(raw), r"\\u([0-9a-fA-F]{4})" => h -> string(Char(parse(UInt16, h[3:end]; base = 16))))
end

function gap_names()
    names = String[]
    for line in eachline(IN_GAP)
        m = match(r"^\s*\"name\": \"(.*)\",?$", line)
        m === nothing || push!(names, unescape_u(m.captures[1]))
    end
    isempty(names) && error("no items read from $IN_GAP")
    sort!(unique!(names))
end

# names(GLLVModels) minus the module name, as the assembler's inputs file records them.
exports_now() = sort!(String[String(n) for n in names(GLLVModels) if n !== :GLLVModels])

# The "name" fields of the "julia_exports" objects in reverse-gap-inputs.json.
function exports_in_inputs()
    inarr = false
    out = String[]
    for line in eachline(IN_INPUTS)
        if !inarr
            occursin("\"julia_exports\": [", line) && (inarr = true)
            continue
        end
        m = match(r"^\s*\"name\": \"(.*)\",?$", line)
        m === nothing || push!(out, unescape_u(m.captures[1]))
    end
    isempty(out) && error("no julia_exports read from $IN_INPUTS")
    sort!(unique!(out))
end

# name => (class, alias_of) from tools/parity_ledger.py, through the read-only helper.
function ledger_classes(names::Vector{String})
    cmd = addenv(`python3 $LEDGER_PY $names`, "PYTHONUTF8" => "1")
    out = read(cmd, String)
    res = Dict{String,Tuple{String,Vector{String}}}()
    for line in split(out, '\n')
        isempty(line) && continue
        f = split(line, '\t'; keepempty = true)
        length(f) == 3 || error("unexpected line from $LEDGER_PY: $(repr(line))")
        res[String(f[1])] = (String(f[2]), isempty(f[3]) ? String[] : String.(split(f[3], ',')))
    end
    all(n -> haskey(res, n), names) || error("$LEDGER_PY did not answer for every name")
    res
end

# (file, line, name) for every dropped docstring in src/**/*.jl (also for non-exported names).
function dropped_in_src()
    out = Tuple{String,Int,String}[]
    for f in walk_files(joinpath(ROOT, "src"), ".jl")
        for (line, nm) in dropped_docstrings(read(f, String))
            push!(out, (relpage(f), line, nm))
        end
    end
    out
end

function owner_and_doc(sym::Symbol)
    owner = try
        Base.binding_module(GLLVModels, sym)
    catch
        GLLVModels
    end
    owner, haskey(Base.Docs.meta(owner), Base.Docs.Binding(owner, sym))
end

function mentioned_in(name::String, texts::Dict{String,String})
    re = Regex("(?<![A-Za-z0-9_!@])" * replace(name, r"([\\.^$|?*+()\[\]{}])" => s"\\\1") * "(?![A-Za-z0-9_!])")
    sort([p for (p, t) in texts if occursin(re, t)])
end

# Only functions, types and modules are compared by identity: two isbits constants with the same value are
# === without being one object in any sense that matters.
identity_groupable(x) = x isa Union{Function,Type,Module}

# name => the other exports bound to the same object (===); names with no twin are absent.
function object_twins(exported::Vector{String})
    groups = IdDict{Any,Vector{String}}()
    for n in exported
        x = getfield(GLLVModels, Symbol(n))
        identity_groupable(x) && push!(get!(groups, x, String[]), n)
    end
    twins = Dict{String,Vector{String}}()
    for g in values(groups)
        length(g) > 1 || continue
        for n in g
            twins[n] = sort([m for m in g if m != n])
        end
    end
    twins
end

function classify(names::Vector{String})
    exported = exports_now()
    in_inputs = exports_in_inputs()
    in_inputs == exported || error("reverse-gap-inputs.json is stale against names(GLLVModels): " *
        "only in inputs: $(first(setdiff(in_inputs, exported), 10)); only in names(): $(first(setdiff(exported, in_inputs), 10)). " *
        "Refresh in this order: the inputs (tools/true_parity_assemble.py --refresh-reverse-gap-inputs), " *
        "the assembler, this classifier, the assembler again (see the header of this script).")
    gapnames = Set(names)
    all(in(Set(exported)), gapnames) || error("reverse-gap.json names a non-export: $(setdiff(gapnames, Set(exported)))")
    all(in(gapnames), keys(NAMED_HOLDS)) || error("NAMED_HOLDS names a non-item: $(setdiff(Set(keys(NAMED_HOLDS)), gapnames))")

    pages = Dict{String,Set{String}}()
    texts = Dict{String,String}()
    for p in walk_files(joinpath(ROOT, "docs/src"), ".md")
        t = read(p, String)
        texts[relpage(p)] = t
        pages[relpage(p)] = rendered_names_in(t, relpage(p))
    end
    texts["README.md"] = read(joinpath(ROOT, "README.md"), String)     # prose, but it has no @docs block
    classes = ledger_classes(exported)
    dropped = Dict{String,String}(nm => "$f:$l" for (f, l, nm) in dropped_in_src() if !isempty(nm))
    twins = object_twins(exported)

    results = Dict{String,Verdict}()
    for n in names
        owner, doc = owner_and_doc(Symbol(n))
        cls, als = classes[n]
        twin_aliases = String["$t (R " * join(classes[t][2], ", ") * ")" for t in get(twins, n, String[]) if !isempty(classes[t][2])]
        facts = Facts(n, cls, als, owner === GLLVModels ? "" : String(nameof(owner)), doc,
            get(dropped, n, ""), sort([p for (p, s) in pages if n in s]), mentioned_in(n, texts), twin_aliases)
        results[n] = decide(facts)
    end
    results = resolve_objects(results, twins)

    decisions = Tuple{String,String,String}[]
    undecided = Tuple{String,String,String}[]
    for n in names
        d, text, kind = results[n]
        d === nothing ? push!(undecided, (n, kind, text)) : push!(decisions, (n, d, text))
    end
    decisions, undecided
end

# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------

function render(decisions)
    io = IOBuffer()
    println(io, "{")
    println(io, " \"schema\": 1,")
    println(io, " \"ruling\": {")
    println(io, "  \"ref\": ", jstr(RULING_REF), ",")
    println(io, "  \"signed_by\": ", jstr("Shinichi Nakagawa"), ",")
    println(io, "  \"signed_on\": ", jstr("2026-10-02"))
    println(io, " },")
    println(io, " \"criterion\": ", jstr(CRITERION), ",")
    println(io, " \"generator\": ", jstr(GENERATOR), ",")
    println(io, " \"decisions\": {")
    ds = sort(decisions; by = first)
    for (i, (n, d, b)) in enumerate(ds)
        println(io, "  ", jstr(n), ": {\"decision\": ", jstr(d), ", \"basis\": ", jstr(b), "}", i < length(ds) ? "," : "")
    end
    println(io, " }")
    println(io, "}")
    String(take!(io))
end

# The order undecided kinds are printed in (the reason groups of the maintainer list).
const KIND_ORDER = ["reexport", "alias_twin", "r_twin_class", "same_object", "em_user_fitter", "user_tool",
    "proposed_class", "low_level", "dropped_docstring", "no_rendered_docstring", "no_docstring"]

function main(args)
    dropped_src = dropped_in_src()
    if !isempty(dropped_src)
        for (f, l, nm) in dropped_src
            println("DROPPED_DOCSTRING\t$f:$l\t$(isempty(nm) ? "(no header)" : nm)")
        end
        println("A docstring separated from its definition by a comment or blank line is silently dropped by Julia.")
        println("Move the comment above the docstring so it attaches, then render it on a docs/src page.")
        exit(1)
    end
    decisions, undecided = classify(gap_names())
    text = render(decisions)
    if "--check" in args
        (isfile(OUT) && read(OUT, String) == text) || (println("C6_DECISIONS_STALE"); exit(1))
        println("C6_DECISIONS_OK $(length(decisions)) decided, $(length(undecided)) undecided")
    else
        write(OUT, text)
        println("C6_DECISIONS_WROTE $(length(decisions)) decided to $(relpath(OUT, ROOT))")
    end
    kept = count(d -> d[2] == "KEPT_AS_JULIA_EXTRA", decisions)
    println("KEPT_AS_JULIA_EXTRA=$kept EXCLUDED_INTERNAL_HELPER=$(length(decisions) - kept) UNDECIDED=$(length(undecided))")
    for kind in KIND_ORDER
        k = count(u -> u[2] == kind, undecided)
        k > 0 && println("UNDECIDED_KIND\t$kind\t$k")
    end
    for (n, kind, why) in sort(undecided; by = u -> (findfirst(==(u[2]), KIND_ORDER), u[1]))
        println("UNDECIDED\t$kind\t$n\t$why")
    end
end

LIVE && main(ARGS)

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
# the maintainer's signed ruling itchyshin/GLLVModels.jl#684 item 3. EXCLUDED_INTERNAL_HELPER when
# tools/parity_ledger.py --ref 9539352f6 classes the name as internal (classes such as "internal
# marginal-likelihood kernel ..." and "internal-but-exported helper; candidate to unexport", and
# any other class whose text says internal), regardless of whether it has a docstring; the basis
# quotes the ledger class. KEPT_AS_JULIA_EXTRA when the name is not classed internal, has a
# docstring, and that docstring is rendered on a docs/src page; the basis cites the page path.
# Otherwise undecided (left out of this file). Mechanical details: a class says internal when it
# contains the whole word internal or internals, so the word internally does not count. A name is
# left undecided, never kept, when its binding is owned by another package (a re-export), when
# parity_ledger.py ALIASES maps an R export to it (the ledger counts it as the Julia twin of an R
# export, so it is not Julia-only), or when its authored docstring was silently dropped by Julia
# because a comment or blank line separates the docstring from its definition.
#
# Inputs : docs/dev-log/core070/true-parity-latest/reverse-gap.json (the assembler's item list;
#          only its "name" fields are read), reverse-gap-inputs.json (checked against
#          names(GLLVModels)), the docstrings of the loaded GLLVModels module, docs/src/**/*.md
#          (the @docs blocks), src/**/*.jl (scanned for dropped docstrings) and the class tables of
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

# The pure functions below (decide, dropped_docstrings, rendered_names_in, render) need no package.
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

const CRITERION = "Applied mechanically to each reverse-gap item (a julia export with no gllvmTMB counterpart) under " *
    "the maintainer's signed ruling itchyshin/GLLVModels.jl#684 item 3. EXCLUDED_INTERNAL_HELPER when " *
    "tools/parity_ledger.py --ref $LEDGER_REF classes the name as internal (classes such as \"internal " *
    "marginal-likelihood kernel ...\" and \"internal-but-exported helper; candidate to unexport\", and " *
    "any other class whose text says internal), regardless of whether it has a docstring; the basis " *
    "quotes the ledger class. KEPT_AS_JULIA_EXTRA when the name is not classed internal, has a " *
    "docstring, and that docstring is rendered on a docs/src page; the basis cites the page path. " *
    "Otherwise undecided (left out of this file). Mechanical details: a class says internal when it " *
    "contains the whole word internal or internals, so the word internally does not count. A name is " *
    "left undecided, never kept, when its binding is owned by another package (a re-export), when " *
    "parity_ledger.py ALIASES maps an R export to it (the ledger counts it as the Julia twin of an R " *
    "export, so it is not Julia-only), or when its authored docstring was silently dropped by Julia " *
    "because a comment or blank line separates the docstring from its definition."

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
    mentions::Vector{String}     # docs/src pages that mention the name (context for undecided)
end

# A class says internal when it holds the whole word internal or internals ("internally" is a
# different word: "R refits ... internally" says nothing about the Julia name).
is_internal_class(cls::AbstractString) = occursin(r"\binternals?\b"i, cls)

mention_note(f::Facts) = isempty(f.mentions) ? "" : "; named in " * join(f.mentions, ", ")

function decide(f::Facts)
    if is_internal_class(f.ledger_class)
        return ("EXCLUDED_INTERNAL_HELPER",
            "tools/parity_ledger.py --ref $LEDGER_REF classes it as internal: \"" * f.ledger_class * "\"")
    elseif !isempty(f.foreign_owner)
        return (nothing, "binding owned by $(f.foreign_owner), a re-export, not a GLLVModels definition" * mention_note(f))
    elseif !isempty(f.alias_of)
        return (nothing, "tools/parity_ledger.py ALIASES maps R " * join(f.alias_of, ", ") *
            " to this name, so the ledger counts it as the Julia twin of an R export; not Julia-only")
    elseif !isempty(f.dropped)
        return (nothing, "an authored docstring at $(f.dropped) is silently dropped by Julia " *
            "(a comment or blank line separates it from its definition)")
    elseif f.has_docstring && !isempty(f.pages)
        return ("KEPT_AS_JULIA_EXTRA",
            "not classed internal by tools/parity_ledger.py --ref $LEDGER_REF; has a docstring, " *
            "rendered in an @docs block on " * join(f.pages, ", "))
    elseif f.has_docstring
        return (nothing, "has a docstring but no docs/src @docs block renders it" * mention_note(f))
    else
        return (nothing, "no docstring" * mention_note(f))
    end
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

function classify(names::Vector{String})
    exported = exports_now()
    in_inputs = exports_in_inputs()
    in_inputs == exported || error("reverse-gap-inputs.json is stale against names(GLLVModels): " *
        "only in inputs: $(first(setdiff(in_inputs, exported), 10)); only in names(): $(first(setdiff(exported, in_inputs), 10)). " *
        "Refresh in this order: the inputs (tools/true_parity_assemble.py --refresh-reverse-gap-inputs), " *
        "the assembler, this classifier, the assembler again (see the header of this script).")
    gapnames = Set(names)
    all(in(Set(exported)), gapnames) || error("reverse-gap.json names a non-export: $(setdiff(gapnames, Set(exported)))")

    pages = Dict{String,Set{String}}()
    texts = Dict{String,String}()
    for p in walk_files(joinpath(ROOT, "docs/src"), ".md")
        t = read(p, String)
        texts[relpage(p)] = t
        pages[relpage(p)] = rendered_names_in(t, relpage(p))
    end
    classes = ledger_classes(names)
    dropped = Dict{String,String}(nm => "$f:$l" for (f, l, nm) in dropped_in_src() if !isempty(nm))

    decisions = Tuple{String,String,String}[]
    undecided = Tuple{String,String}[]
    for n in names
        owner, doc = owner_and_doc(Symbol(n))
        cls, als = classes[n]
        facts = Facts(n, cls, als, owner === GLLVModels ? "" : String(nameof(owner)), doc,
            get(dropped, n, ""), sort([p for (p, s) in pages if n in s]), mentioned_in(n, texts))
        d, text = decide(facts)
        d === nothing ? push!(undecided, (n, text)) : push!(decisions, (n, d, text))
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
    for (n, why) in sort(undecided; by = first)
        println("UNDECIDED\t$n\t$why")
    end
end

LIVE && main(ARGS)

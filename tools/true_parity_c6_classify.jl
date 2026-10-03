#!/usr/bin/env julia
# Classify the C6 reverse-gap items (GLLVModels exports with no gllvmTMB counterpart) and write
# docs/dev-log/core070/true-parity-latest/reverse-gap-decisions.json.
#
# This script applies the maintainer's signed ruling itchyshin/GLLVModels.jl#684 item 3
# mechanically. It decides nothing by hand: an export is decided only when the criterion below
# gives a clear answer, otherwise it stays out of the file (undecided) and is listed on stdout.
#
# Criterion (also written verbatim into the output file's "criterion" field):
#   KEPT_AS_JULIA_EXTRA: the export has a docstring AND that docstring is rendered on a
#     docs/src page, that is, the page has an @docs block naming it.
#   EXCLUDED_INTERNAL_HELPER: the export has no docstring AND is not mentioned by name
#     (whole word) anywhere in docs/src/**/*.md or README.md.
#   Anything else (documented but not rendered; undocumented but mentioned in user docs; an
#   export whose binding is owned by another module) stays undecided.
#
# Inputs : docs/dev-log/core070/true-parity-latest/reverse-gap.json (the assembler's item list;
#          only its "name" fields are read), the docstrings of the loaded GLLVModels module,
#          docs/src/**/*.md and README.md.
# Output : reverse-gap-decisions.json (deterministic: names sorted, fixed key order).
#
# Usage (from the repo root):
#   export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4
#   julia --project=. tools/true_parity_c6_classify.jl            # write the file
#   julia --project=. tools/true_parity_c6_classify.jl --check    # fail if the file is stale
#
# Docstring lookup uses Base.Docs.meta(owner)[Base.Docs.Binding(owner, name)], which works on
# Julia 1.10. The input reverse-gap.json must be current (python3 tools/true_parity_assemble.py
# --check) and was built from names(GLLVModels) at the same commit.

using GLLVModels

const ROOT = normpath(joinpath(@__DIR__, ".."))
const LEDGER = joinpath(ROOT, "docs/dev-log/core070/true-parity-latest")
const OUT = joinpath(LEDGER, "reverse-gap-decisions.json")
const IN_GAP = joinpath(LEDGER, "reverse-gap.json")

const CRITERION = "Applied mechanically to each reverse-gap item, a julia export with no gllvmTMB counterpart. " *
    "KEPT_AS_JULIA_EXTRA: the export has a docstring AND that docstring is rendered on a docs/src page " *
    "(an @docs block there names it). " *
    "EXCLUDED_INTERNAL_HELPER: the export has no docstring AND is not mentioned by name (whole word) " *
    "anywhere in docs/src/**/*.md or README.md. " *
    "Anything else (documented but not rendered; undocumented but mentioned in user docs; an export whose " *
    "binding is owned by another module) is left undecided and is absent from this file."

const GENERATOR = "tools/true_parity_c6_classify.jl"

jstr(s) = "\"" * replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""

function gap_names()
    names = String[]
    for line in eachline(IN_GAP)
        m = match(r"^\s*\"name\": \"(.*)\",?$", line)
        m === nothing && continue
        # The assembler writes the file with ensure_ascii, so non-ASCII names arrive as \uXXXX.
        # Names are identifiers: no backslashes or quotes otherwise.
        raw = String(m.captures[1])
        push!(names, replace(raw, r"\\u([0-9a-fA-F]{4})" => h -> string(Char(parse(UInt16, h[3:end]; base = 16)))))
    end
    isempty(names) && error("no items read from $IN_GAP")
    sort!(unique!(names))
end

function md_files()
    files = String[]
    for (d, _, fs) in walkdir(joinpath(ROOT, "docs/src"))
        for f in fs
            endswith(f, ".md") && push!(files, joinpath(d, f))
        end
    end
    push!(files, joinpath(ROOT, "README.md"))
    sort!(files)
end

relpage(p) = relpath(p, ROOT)

# page => Set of names that an @docs block on that page names (last dotted segment, before any "(").
function rendered_names()
    out = Dict{String,Set{String}}()
    for p in md_files()
        inblock = false
        for line in eachline(p)
            s = strip(line)
            if startswith(s, "```@autodocs")
                error("$(relpage(p)) uses @autodocs; extend this classifier before trusting it")
            elseif startswith(s, "```@docs")
                inblock = true
            elseif inblock && startswith(s, "```")
                inblock = false
            elseif inblock && !isempty(s) && !startswith(s, "#")
                head = first(split(s, '('; limit = 2))
                nm = String(last(split(head, r"\.(?=@?[A-Za-z_])")))
                push!(get!(out, relpage(p), Set{String}()), nm)
            end
        end
    end
    out
end

function has_docstring(sym::Symbol)
    owner = try
        Base.binding_module(GLLVModels, sym)
    catch
        GLLVModels
    end
    md = Base.Docs.meta(owner)
    owner, haskey(md, Base.Docs.Binding(owner, sym))
end

function mentioned_in(name::String, texts::Dict{String,String})
    re = Regex("(?<![A-Za-z0-9_!@])" * replace(name, r"([\\.^$|?*+()\[\]{}])" => s"\\\1") * "(?![A-Za-z0-9_!])")
    sort([p for (p, t) in texts if occursin(re, t)])
end

function classify()
    names = gap_names()
    rendered = rendered_names()
    texts = Dict(relpage(p) => read(p, String) for p in md_files())
    decisions = Tuple{String,String,String}[]
    undecided = Tuple{String,String}[]
    for n in names
        owner, doc = has_docstring(Symbol(n))
        pages = sort([p for (p, s) in rendered if n in s])
        ments = mentioned_in(n, texts)
        if owner !== GLLVModels
            push!(undecided, (n, "binding owned by $(owner), not GLLVModels"))
        elseif doc && !isempty(pages)
            push!(decisions, (n, "KEPT_AS_JULIA_EXTRA",
                "has a docstring; rendered in an @docs block on " * join(pages, ", ")))
        elseif !doc && isempty(ments)
            push!(decisions, (n, "EXCLUDED_INTERNAL_HELPER",
                "no docstring and not mentioned by name in docs/src/**/*.md or README.md"))
        elseif doc
            push!(undecided, (n, "documented but not rendered in any @docs block; mentioned in " *
                (isempty(ments) ? "no docs page" : join(ments, ", "))))
        else
            push!(undecided, (n, "no docstring but mentioned in " * join(ments, ", ")))
        end
    end
    decisions, undecided
end

function render(decisions)
    io = IOBuffer()
    println(io, "{")
    println(io, " \"schema\": 1,")
    println(io, " \"ruling\": {")
    println(io, "  \"ref\": ", jstr("itchyshin/GLLVModels.jl#684 item 3"), ",")
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
    decisions, undecided = classify()
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

main(ARGS)

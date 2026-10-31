using GLLVModels, Test, Markdown
using Base.Docs: meta, DocStr

# Issue #1021: a docstring cross-reference such as ``[`f`](@ref)`` renders as a dead
# link (`./@ref`) when Julia's Markdown parser fails to see it as a link. The usual cause is
# a bare underscore next to a non-ASCII letter in the same paragraph (`Σ_y comparison via
# [...](@ref)`): `_y ... _` opens an emphasis span that swallows the link, and the literal
# text `](@ref)` (as plain text or inside a mis-paired code span) reaches the page. This test parses every docstring and fails if any
# `](@ref)` survives as plain text.

_leaked_ref(x::AbstractString) = occursin("](@ref)", x)
_leaked_ref(x::Markdown.Code) = occursin("](@ref)", x.code)   # a mis-paired backtick swallows it into code
_leaked_ref(x::Markdown.Paragraph) = any(_leaked_ref, x.content)
_leaked_ref(x::Markdown.Header) = any(_leaked_ref, x.text)
_leaked_ref(x::Markdown.List) = any(item -> any(_leaked_ref, item), x.items)
_leaked_ref(x::Markdown.BlockQuote) = any(_leaked_ref, x.content)
_leaked_ref(x::Markdown.Admonition) = any(_leaked_ref, x.content)
_leaked_ref(x::Markdown.Table) = any(row -> any(cell -> any(_leaked_ref, cell), row), x.rows)
_leaked_ref(x::Union{Markdown.Bold, Markdown.Italic}) = any(_leaked_ref, x.text)
_leaked_ref(x::Markdown.Link) = any(_leaked_ref, x.text)
_leaked_ref(x::Vector) = any(_leaked_ref, x)
_leaked_ref(x::Markdown.MD) = any(_leaked_ref, x.content)
_leaked_ref(::Any) = false

function _leaked_ref_docstrings(mod::Module)
    bad = String[]
    for (_, md) in meta(mod), (_, ds) in md.docs
        ds isa DocStr || continue
        text = join(string.(ds.text), "")
        occursin("](@ref)", text) || continue
        _leaked_ref(Markdown.parse(text)) && push!(bad, string(ds.data[:binding]))
    end
    return sort!(unique(bad))
end

# The hand-written pages in docs/src go through the same parser (e.g. `family_id 20`
# followed later in the paragraph by a link). Skipped when the docs tree is absent.
function _leaked_ref_pages(dir::AbstractString)
    isdir(dir) || return String[]
    bad = String[]
    for f in sort(readdir(dir)); endswith(f, ".md") || continue
        text = read(joinpath(dir, f), String)
        occursin("](@ref)", text) || continue
        # Parse block by block so that fenced @docs/@example blocks are plain Code nodes.
        _leaked_ref(Markdown.parse(text)) && push!(bad, f)
    end
    return bad
end

@testset "docs/src pages: @ref links survive Markdown parsing (#1021)" begin
    bad = _leaked_ref_pages(joinpath(pkgdir(GLLVModels), "docs", "src"))
    isempty(bad) || println("pages with a leaked `](@ref)`: ", join(bad, ", "))
    @test isempty(bad)
end

@testset "docstring @ref links survive Markdown parsing (#1021)" begin
    bad = _leaked_ref_docstrings(GLLVModels)
    isempty(bad) || println("docstrings with a leaked `](@ref)`: ", join(bad, ", "))
    @test isempty(bad)
end

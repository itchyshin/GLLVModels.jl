#!/usr/bin/env julia
# Fixture tests for tools/true_parity_c6_classify.jl (the C6 reverse-gap classifier).
#
#   julia --project=. tools/test_true_parity_c6_classify.jl
#
# The first groups need no package: they feed the classifier's pure functions fixture inputs and
# cover every branch of the criterion, including a dropped docstring and an internal class on a name
# that has a rendered docstring. The last group reads the committed decisions file and checks it
# against the classifier, the ledger helper and the tree.

using Test

include(joinpath(@__DIR__, "true_parity_c6_classify.jl"))   # LIVE is false here: main does not run

F(; name = "x", cls = "", alias = String[], owner = "", doc = false, dropped = "",
    pages = String[], mentions = String[]) = Facts(name, cls, alias, owner, doc, dropped, pages, mentions)

const q3 = "\"\"\""
src(lines...) = join(lines, "\n") * "\n"

# The Julia 1.10 behaviour the dropped-docstring scan relies on, and the scan's verdict on the same text.
@testset "C6 premise: a comment or blank line between docstring and definition drops it" begin
    text = src("module DropProbe",
        q3, "    WithComment", "", "doc", q3, "# a comment", "struct WithComment end", "",
        q3, "    WithBlank", "", "doc", q3, "", "struct WithBlank end", "",
        q3, "    Attached", "", "doc", q3, "struct Attached end",
        "end")
    m = include_string(Main, text)
    md = Base.Docs.meta(m)
    @test !haskey(md, Base.Docs.Binding(m, :WithComment))
    @test !haskey(md, Base.Docs.Binding(m, :WithBlank))
    @test haskey(md, Base.Docs.Binding(m, :Attached))
    # the scan flags exactly the two dropped ones
    @test sort([nm for (_, nm) in dropped_docstrings(text)]) == ["WithBlank", "WithComment"]
end

@testset "C6 dropped_docstrings scan" begin
    @test isempty(dropped_docstrings(src(q3, "    A", "", "doc", q3, "struct A end")))
    @test dropped_docstrings(src("# header", q3, "    A", "", "doc", q3, "# note", "struct A end")) == [(6, "A")]
    @test dropped_docstrings(src(q3, "    f(x)", "", "doc", q3, "", "f(x) = 1")) == [(5, "f")]
    # one-line forms
    @test dropped_docstrings(src(q3 * "    g(x) one line" * q3, "# c", "g(x) = 2")) == [(1, "g")]
    @test dropped_docstrings(src("\"short doc\"", "", "h() = 3")) == [(1, "short")]
    @test isempty(dropped_docstrings(src(q3 * "    g(x) one line" * q3, "g(x) = 2")))
    # several comment and blank lines still count as a gap
    @test length(dropped_docstrings(src(q3, "    B", q3, "# one", "", "# two", "struct B end"))) == 1
    # a docstring at end of file with nothing after it is not judged
    @test isempty(dropped_docstrings(src(q3, "    C", q3, "# trailing comment")))
    # `@doc raw` blocks and strings inside code are skipped, and do not confuse the next docstring
    mixed = src("@doc raw" * q3, "    D", q3, "struct D end", "",
        "function w()", "    msg = " * q3 * "text", "    more" * q3, "    msg", "end", "",
        q3, "    E", q3, "# dropped", "struct E end")
    @test [nm for (_, nm) in dropped_docstrings(mixed)] == ["E"]
end

@testset "C6 rendered_names_in" begin
    page = src("# Page", "```@docs", "GLLVModels.foo", "bar(x::Int)", "# a comment", "@baz", "", "```",
        "text mentioning notrendered", "```@example", "ignored", "```")
    @test rendered_names_in(page) == Set(["foo", "bar", "@baz"])
    @test_throws ErrorException rendered_names_in(src("```@autodocs", "Modules = [X]", "```"), "api.md")
end

@testset "C6 is_internal_class" begin
    @test is_internal_class("internal marginal-likelihood kernel; reached only from within the fit driver")
    @test is_internal_class("[PROPOSED 2026-09-25, unsigned -- x.md] internal-but-exported helper: spec; candidate to unexport")
    @test is_internal_class("hand-coded analytic-gradient kernel; engine internal, never reached by an R-facing name")
    @test is_internal_class("EM/SQUAREM alternative-solver internals; the TMB path never uses this solver family")
    @test is_internal_class("Internal helper")
    # "internally" is a different word and describes R, not the Julia name
    @test !is_internal_class("StatsAPI naming for a parameter count; gllvmTMB computes this internally for AIC/BIC")
    @test !is_internal_class("R refits an alternative model internally; Julia's is a generic two-fit bridge")
    @test !is_internal_class("struct suffix: backs a fitted model; R represents the same as an S3 class tag")
    @test !is_internal_class("")
end

@testset "C6 decide: every branch" begin
    internal = "internal marginal-likelihood kernel; reached only from within the fit driver, never an R-facing name"
    # EXCLUDED_INTERNAL_HELPER: internal ledger class, regardless of docstring
    d, b = decide(F(name = "k1", cls = internal, doc = true, pages = ["docs/src/api.md"]))
    @test d == "EXCLUDED_INTERNAL_HELPER"                                   # internal class WITH a rendered docstring
    @test occursin(internal, b) && occursin("tools/parity_ledger.py --ref 9539352f6", b)
    d, b = decide(F(name = "k2", cls = "[PROPOSED 2026-09-25, unsigned -- r.md] internal-but-exported helper: x; candidate to unexport"))
    @test d == "EXCLUDED_INTERNAL_HELPER"                                   # internal class, no docstring
    @test occursin("candidate to unexport", b)
    @test decide(F(name = "k3", cls = "distribution-kernel helper (log-density/normalizer/CDF); engine internal"))[1] ==
          "EXCLUDED_INTERNAL_HELPER"
    @test decide(F(name = "k4", cls = internal, owner = "StatsAPI", alias = ["r"], dropped = "src/a.jl:1"))[1] ==
          "EXCLUDED_INTERNAL_HELPER"                                        # internal wins over every guard
    # KEPT_AS_JULIA_EXTRA: not internal, docstring, rendered on a docs/src page
    d, b = decide(F(name = "e1", cls = "Julia link-function marker type; R uses a string", doc = true, pages = ["docs/src/api.md"]))
    @test d == "KEPT_AS_JULIA_EXTRA" && occursin("docs/src/api.md", b)
    d, b = decide(F(name = "e2", doc = true, pages = ["docs/src/api.md", "docs/src/low-level-reference.md"]))
    @test d == "KEPT_AS_JULIA_EXTRA" && occursin("docs/src/api.md, docs/src/low-level-reference.md", b)
    @test decide(F(name = "e3", cls = "StatsAPI naming; gllvmTMB computes this internally", doc = true,
                   pages = ["docs/src/api.md"]))[1] == "KEPT_AS_JULIA_EXTRA"   # "internally" is not "internal"
    # undecided, each with its reason
    d, why = decide(F(name = "u1"))
    @test d === nothing && why == "no docstring"
    d, why = decide(F(name = "u2", mentions = ["docs/src/tutorial.md"]))
    @test d === nothing && occursin("no docstring", why) && occursin("docs/src/tutorial.md", why)
    d, why = decide(F(name = "u3", doc = true))
    @test d === nothing && occursin("no docs/src @docs block renders it", why)
    d, why = decide(F(name = "u4", doc = false, dropped = "src/families/binomial.jl:133"))
    @test d === nothing && occursin("silently dropped by Julia", why) && occursin("binomial.jl:133", why)   # dropped docstring
    d, why = decide(F(name = "u5", doc = true, pages = ["docs/src/api.md"], dropped = "src/a.jl:7"))
    @test d === nothing && occursin("silently dropped", why)                # dropped beats kept
    d, why = decide(F(name = "u6", owner = "StatsAPI", doc = true, pages = ["docs/src/api.md"]))
    @test d === nothing && occursin("owned by StatsAPI", why)               # a re-export
    d, why = decide(F(name = "u7", alias = ["nbinom2"], doc = true, pages = ["docs/src/api.md"]))
    @test d === nothing && occursin("ALIASES maps R nbinom2", why) && occursin("not Julia-only", why)
end

@testset "C6 render and the assembler's basis rule" begin
    docs_src = r"(?<![A-Za-z0-9_])docs/src/(?:[A-Za-z0-9_-][A-Za-z0-9._-]*/)*[A-Za-z0-9_-][A-Za-z0-9._-]*\.(?:md|jl|toml|json|txt)(?![A-Za-z0-9_])"
    ds = [("b", "KEPT_AS_JULIA_EXTRA", decide(F(name = "b", doc = true, pages = ["docs/src/api.md"]))[2]),
          ("a", "EXCLUDED_INTERNAL_HELPER", decide(F(name = "a", cls = "internal kernel \"quoted\""))[2])]
    text = render(ds)
    @test text == render(reverse(ds))                                        # deterministic: names sorted
    @test occursin("\"ref\": \"itchyshin/GLLVModels.jl#684 item 3\"", text)
    @test occursin("\"signed_on\": \"2026-10-02\"", text)
    @test occursin("\\\"quoted\\\"", text)                                   # quotes in a class are escaped
    @test findfirst("\"a\":", text) < findfirst("\"b\":", text)
    @test match(docs_src, ds[1][3]) !== nothing                              # a KEPT basis cites a docs/src path
    @test isfile(joinpath(ROOT, match(docs_src, ds[1][3]).match))            # ... that resolves
end

@testset "C6 criterion text: header, constant and committed file agree" begin
    norm_ws(s) = join(split(s), " ")
    lines = readlines(joinpath(@__DIR__, "true_parity_c6_classify.jl"))
    i = findfirst(l -> startswith(l, "# Applied mechanically"), lines)
    j = findfirst(l -> startswith(l, "# Inputs :"), lines)
    @test i !== nothing && j !== nothing
    header = norm_ws(join((replace(l, r"^# ?" => "") for l in lines[i:j-2]), " "))
    @test header == norm_ws(CRITERION)
    for must in ("EXCLUDED_INTERNAL_HELPER when tools/parity_ledger.py --ref 9539352f6 classes the name as internal",
                 "\"internal marginal-likelihood kernel ...\"", "\"internal-but-exported helper; candidate to unexport\"",
                 "any other class whose text says internal", "regardless of whether it has a docstring",
                 "KEPT_AS_JULIA_EXTRA when the name is not classed internal, has a docstring, and that docstring is rendered on a docs/src page")
        @test occursin(must, CRITERION)
    end
    m = match(r"^ \"criterion\": \"(.*)\",$"m, read(OUT, String))
    @test m !== nothing
    @test replace(m.captures[1], "\\\"" => "\"") == CRITERION
end

@testset "C6 ledger helper (tools/parity_ledger.py class tables)" begin
    cls = ledger_classes(["rrr_marginal_loglik", "welch_t", "fit_em_phylo", "fit_gllvm", "dof", "totally_unclassed_name"])
    @test is_internal_class(cls["rrr_marginal_loglik"][1])
    @test is_internal_class(cls["welch_t"][1]) && occursin("candidate to unexport", cls["welch_t"][1])
    @test !is_internal_class(cls["fit_em_phylo"][1]) && !isempty(cls["fit_em_phylo"][1])
    @test cls["fit_gllvm"][2] == ["gllvmTMB"]                                # ALIASES: R gllvmTMB -> fit_gllvm
    @test !is_internal_class(cls["dof"][1]) && occursin("internally", cls["dof"][1])
    @test cls["totally_unclassed_name"] == ("", String[])
end

@testset "C6 committed decisions agree with the classifier, the ledger and the tree" begin
    @test isempty(dropped_in_src())                                          # no docstring dropped anywhere in src/
    entries = Dict{String,Tuple{String,String}}()
    for l in eachline(OUT)
        m = match(r"^  \"(.*)\": \{\"decision\": \"([A-Z_]+)\", \"basis\": \"(.*)\"\},?$", l)
        m === nothing || (entries[unescape_u(m.captures[1])] = (m.captures[2], replace(m.captures[3], "\\\"" => "\"")))
    end
    @test !isempty(entries)
    @test Set(d for (d, _) in values(entries)) <= Set(["KEPT_AS_JULIA_EXTRA", "EXCLUDED_INTERNAL_HELPER"])
    gap = Set(gap_names())
    @test all(in(gap), keys(entries))                                        # every decision names a reverse-gap item
    cls = ledger_classes(sort!(collect(keys(entries))))
    docs_src = r"docs/src/[A-Za-z0-9_./-]+\.md"
    for (n, (d, b)) in entries
        c, als = cls[n]
        @test isempty(als)                                                   # never an ALIASES twin
        if d == "EXCLUDED_INTERNAL_HELPER"
            @test is_internal_class(c)
            @test b == "tools/parity_ledger.py --ref 9539352f6 classes it as internal: \"" * c * "\""
        else
            @test !is_internal_class(c)
            pages = [m.match for m in eachmatch(docs_src, b)]
            @test !isempty(pages) && all(p -> isfile(joinpath(ROOT, p)), pages)
        end
    end
end

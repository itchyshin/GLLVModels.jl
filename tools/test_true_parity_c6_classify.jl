#!/usr/bin/env julia
# Fixture tests for tools/true_parity_c6_classify.jl (the C6 reverse-gap classifier).
#
#   julia --project=. tools/test_true_parity_c6_classify.jl
#
# The first groups need no package: they feed the classifier's pure functions fixture inputs and
# cover every branch and every guard of the criterion, including a dropped docstring and an internal
# class on a name that has a rendered docstring. The last groups load GLLVModels (for object identity),
# read the committed decisions file and check it against the classifier, the ledger helper and the tree.

using Test

include(joinpath(@__DIR__, "true_parity_c6_classify.jl"))   # LIVE is false here: main does not run

F(; name = "x", cls = "", alias = String[], owner = "", doc = false, dropped = "",
    pages = String[], mentions = String[], twin_aliases = String[]) =
    Facts(name, cls, alias, owner, doc, dropped, pages, mentions, twin_aliases)

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

const PROPOSED = "[PROPOSED 2026-09-25, unsigned -- docs/dev-log/core070/reverse-gap-classes-2026-09-25.md] "
const REML_CLASS = PROPOSED * "julia-only diagnostic or extractor: REML log-likelihood kernel (reml.jl); " *
    "gllvmTMB's own non-Gaussian REML route (allow_nongaussian_reml) is an internal control knob, not an " *
    "exported accessor of this shape"

@testset "C6 is_internal_class: only a class that calls the JULIA name internal" begin
    @test is_internal_class("internal marginal-likelihood kernel; reached only from within the fit driver")
    @test is_internal_class(PROPOSED * "internal-but-exported helper: spec; candidate to unexport")
    @test is_internal_class("hand-coded analytic-gradient kernel; engine internal, never reached by an R-facing name")
    @test is_internal_class("distribution-kernel helper (log-density/normalizer/CDF); engine internal")
    @test is_internal_class("EM/SQUAREM alternative-solver internals; the TMB path never uses this solver family")
    @test is_internal_class("Internal helper")
    # the class of gaussian_reml_loglik: "internal" describes R's knob, not the Julia export
    @test !is_internal_class(REML_CLASS)
    @test !is_internal_class("julia-only diagnostic: kernel; R keeps an internal fallback that is not exported")
    # "internally" is a different word and describes R, not the Julia name
    @test !is_internal_class("StatsAPI naming for a parameter count; gllvmTMB computes this internally for AIC/BIC")
    @test !is_internal_class("R refits an alternative model internally; Julia's is a generic two-fit bridge")
    @test !is_internal_class("struct suffix: backs a fitted model; R represents the same as an S3 class tag")
    @test !is_internal_class("")
    # the other class predicates
    @test is_proposed_class(PROPOSED * "internal-but-exported helper: x") && !is_proposed_class("internal kernel")
    @test says_r_has_it(PROPOSED * "R has it under another name (x): y")
    @test says_r_has_it(PROPOSED * "R has it after 0.7.0, FLAGGED: y") && says_r_has_it("R has it under another name")
    @test !says_r_has_it(PROPOSED * "julia-only family: gllvmTMB has no hurdle family") && !says_r_has_it("")
end

@testset "C6 decide: every branch" begin
    internal = "internal marginal-likelihood kernel; reached only from within the fit driver, never an R-facing name"
    # EXCLUDED_INTERNAL_HELPER: internal ledger class, regardless of docstring
    d, b = decide(F(name = "k1", cls = internal, doc = true, pages = ["docs/src/api.md"]))
    @test d == "EXCLUDED_INTERNAL_HELPER"                                   # internal class WITH a rendered docstring
    @test occursin(internal, b) && occursin("tools/parity_ledger.py --ref 9539352f6", b)
    @test decide(F(name = "k1", cls = internal))[3] == ""                   # a decided name has no kind
    @test decide(F(name = "k3", cls = "distribution-kernel helper (log-density/normalizer/CDF); engine internal"))[1] ==
          "EXCLUDED_INTERNAL_HELPER"
    # reading a basis page that is a reference page does not count as user prose
    @test decide(F(name = "k5", cls = internal, mentions = ["docs/src/api.md", "docs/src/low-level-reference.md"]))[1] ==
          "EXCLUDED_INTERNAL_HELPER"
    # KEPT_AS_JULIA_EXTRA: not internal, docstring, rendered on a docs/src page
    d, b = decide(F(name = "e1", cls = "Julia link-function marker type; R uses a string", doc = true, pages = ["docs/src/api.md"]))
    @test d == "KEPT_AS_JULIA_EXTRA" && occursin("docs/src/api.md", b)
    d, b = decide(F(name = "e2", doc = true, pages = ["docs/src/api.md", "docs/src/low-level-reference.md"]))
    @test d == "KEPT_AS_JULIA_EXTRA" && occursin("docs/src/api.md, docs/src/low-level-reference.md", b)
    @test decide(F(name = "e3", cls = "StatsAPI naming; gllvmTMB computes this internally", doc = true,
                   pages = ["docs/src/api.md"]))[1] == "KEPT_AS_JULIA_EXTRA"   # "internally" is not "internal"
    # the REML class: documented and rendered, so KEPT (this was the blocking false positive)
    d, b = decide(F(name = "gaussian_reml_loglik", cls = REML_CLASS, doc = true, pages = ["docs/src/api.md"]))
    @test d == "KEPT_AS_JULIA_EXTRA" && occursin("docs/src/api.md", b)
    # a prose mention does not change a KEPT name
    @test decide(F(name = "e4", doc = true, pages = ["docs/src/api.md"], mentions = ["docs/src/tutorial.md"]))[1] ==
          "KEPT_AS_JULIA_EXTRA"
    # undecided, each with its reason and kind
    d, why, kind = decide(F(name = "u1"))
    @test d === nothing && why == "no docstring" && kind == "no_docstring"
    d, why = decide(F(name = "u2", mentions = ["docs/src/tutorial.md"]))
    @test d === nothing && occursin("no docstring", why) && occursin("docs/src/tutorial.md", why)
    d, why, kind = decide(F(name = "u3", doc = true))
    @test d === nothing && occursin("no docs/src @docs block renders it", why) && kind == "no_rendered_docstring"
    d, why, kind = decide(F(name = "u4", doc = false, dropped = "src/families/binomial.jl:133"))
    @test d === nothing && occursin("silently dropped by Julia", why) && occursin("binomial.jl:133", why)   # dropped docstring
    @test kind == "dropped_docstring"
    d, why = decide(F(name = "u5", doc = true, pages = ["docs/src/api.md"], dropped = "src/a.jl:7"))
    @test d === nothing && occursin("silently dropped", why)                # dropped beats kept
    d, why, kind = decide(F(name = "u6", owner = "StatsAPI", doc = true, pages = ["docs/src/api.md"]))
    @test d === nothing && occursin("owned by StatsAPI", why) && kind == "reexport"   # a re-export
    d, why, kind = decide(F(name = "u7", alias = ["nbinom2"], doc = true, pages = ["docs/src/api.md"]))
    @test d === nothing && occursin("ALIASES maps R nbinom2", why) && occursin("not Julia-only", why) && kind == "alias_twin"
end

@testset "C6 guard: the ALIASES guard follows the object (twin_aliases)" begin
    # StudentTFamily is the same object as StudentT, which ALIASES maps to R student
    d, why, kind = decide(F(name = "StudentTFamily", doc = true, pages = ["docs/src/api.md"],
                            twin_aliases = ["StudentT (R student)"]))
    @test d === nothing && kind == "alias_twin" && occursin("same object as StudentT (R student)", why)
    # a name that is itself a twin of an ALIASES name is never kept, with or without a docstring
    @test decide(F(name = "t", twin_aliases = ["StudentT (R student)"]))[1] === nothing
end

@testset "C6 guard: user-facing prose keeps a would-be exclusion out of the file" begin
    internal = "internal marginal-likelihood kernel; reached only from within the fit driver, never an R-facing name"
    for page in ("docs/src/response-families.md", "docs/src/grouped-models.md", "README.md", "docs/src/changelog.md")
        d, why, kind = decide(F(name = "lognormal_marginal_loglik", cls = internal, doc = true,
                                pages = ["docs/src/api.md"], mentions = ["docs/src/api.md", page]))
        @test d === nothing && kind == "user_tool" && occursin(page, why) && !occursin("docs/src/api.md", why)
    end
    # a kernel that only the reference pages mention is still excluded (checked above); so is one nobody mentions
    @test decide(F(name = "k", cls = internal))[1] == "EXCLUDED_INTERNAL_HELPER"
end

@testset "C6 guard: a PROPOSED (unsigned) internal class is not a basis for exclusion" begin
    cls = PROPOSED * "internal-but-exported helper: spec; candidate to unexport"
    d, why, kind = decide(F(name = "GllvmModel", cls = cls, doc = true, pages = ["docs/src/api.md"]))
    @test d === nothing && kind == "proposed_class" && occursin("PROPOSED", why) && occursin("candidate to unexport", why)
    # no docstring, no mention: still held, not excluded
    @test decide(F(name = "p2", cls = cls))[1] === nothing
    # both reasons are reported, the user-tool reason first
    d, why, kind = decide(F(name = "some_grouping_type", cls = cls, mentions = ["docs/src/grouped-models.md"]))
    @test d === nothing && kind == "user_tool" && occursin("grouped-models.md", why) && occursin("PROPOSED", why)
    # an unmarked internal class is not held by this guard
    @test decide(F(name = "k", cls = "hand-coded analytic-gradient kernel; engine internal"))[1] == "EXCLUDED_INTERNAL_HELPER"
end

@testset "C6 guard: a KEPT name whose own class says R has it is not Julia-only" begin
    for cls in (PROPOSED * "R has it under another name (api-rename-notes.md): x",
                PROPOSED * "R has it after 0.7.0: result struct returned by select_lv()",
                PROPOSED * "R has it under another name, DEFERRED: y")
        d, why, kind = decide(F(name = "r", cls = cls, doc = true, pages = ["docs/src/api.md"]))
        @test d === nothing && kind == "r_twin_class" && occursin("R has it", why)
    end
    # a Julia-only class on the same name is KEPT
    @test decide(F(name = "j", cls = PROPOSED * "julia-only family: x; gllvmTMB has no x family", doc = true,
                   pages = ["docs/src/api.md"]))[1] == "KEPT_AS_JULIA_EXTRA"
end

@testset "C6 guard: the named review list (NAMED_HOLDS)" begin
    @test Set(keys(NAMED_HOLDS)) == Set(["random_balanced_tree", "shrinkage_factor"])
    em = "EM/SQUAREM alternative-solver internals; gllvmTMB's TMB path never uses this solver family"
    # em_fa has the ^em_ class and is neither held nor signed: it stays excluded
    @test decide(F(name = "em_fa", cls = em, doc = true, pages = ["docs/src/api.md"]))[1] == "EXCLUDED_INTERNAL_HELPER"
    for n in ("random_balanced_tree", "shrinkage_factor")
        d, _, kind = decide(F(name = n, doc = true, pages = ["docs/src/api.md"]))
        @test d === nothing && kind == "low_level"                         # the rule alone would KEEP these
    end
    # the same facts under another name are KEPT
    @test decide(F(name = "some_other_helper", doc = true, pages = ["docs/src/api.md"]))[1] == "KEPT_AS_JULIA_EXTRA"
end

# Maintainer ruling 2026-10-05 (vault D-319): only the 23 high-confidence names of
# c6-proposed-decisions-2026-10-05.md are signed; they win over the rule and every guard.
@testset "C6 signed overrides (maintainer ruling 2026-10-05, D-319)" begin
    @test length(SIGNED_OVERRIDES) == 23
    @test isdisjoint(keys(SIGNED_OVERRIDES), keys(NAMED_HOLDS))
    @test count(v -> v[1] == "EXCLUDED_INTERNAL_HELPER", values(SIGNED_OVERRIDES)) == 7
    @test count(v -> v[1] == "KEPT_AS_JULIA_EXTRA", values(SIGNED_OVERRIDES)) == 16
    em = "EM/SQUAREM alternative-solver internals; gllvmTMB's TMB path never uses this solver family"
    # the ^em_ internal class no longer excludes the signed EM fitters
    for n in ("em_fit_phylo", "em_fit_phylo_squarem", "em_observed_information")
        d, b, kind = decide(F(name = n, cls = em, doc = true, pages = ["docs/src/api.md"]))
        @test d == "KEPT_AS_JULIA_EXTRA" && kind == "" && startswith(b, "maintainer ruling 2026-10-05 (D-319)")
    end
    # a re-export guard does not hold a signed StatsAPI generic
    @test decide(F(name = "coeftable", owner = "StatsAPI", doc = true))[1] == "KEPT_AS_JULIA_EXTRA"
    # a signed EXCLUDED name needs no internal class
    @test decide(F(name = "ZI_LAPLACE_EIGMIN_FLOOR", doc = true, pages = ["docs/src/api.md"]))[1] == "EXCLUDED_INTERNAL_HELPER"
    # medium- and low-confidence names of the proposal are not signed
    for n in ("observed_mask", "StatsAPI", "fit_gllvm", "ZIB", "welch_t", "random_balanced_tree")
        @test !haskey(SIGNED_OVERRIDES, n)
    end
    # every signed KEPT basis cites an existing docs/src page
    for (n, (d, b)) in SIGNED_OVERRIDES
        d == "KEPT_AS_JULIA_EXTRA" || continue
        pages = [m.match for m in eachmatch(r"docs/src/[A-Za-z0-9_./-]+\.md", b)]
        @test !isempty(pages) && all(p -> isfile(joinpath(ROOT, p)), pages)
    end
end

@testset "C6 resolve_objects: one object, one verdict" begin
    kept = ("KEPT_AS_JULIA_EXTRA", "basis", "")
    excl = ("EXCLUDED_INTERNAL_HELPER", "basis", "")
    held(k) = (nothing, "own reason", k)
    # agreement: nothing changes
    res = Dict{String,Verdict}("a" => kept, "b" => kept)
    @test resolve_objects(res, Dict("a" => ["b"], "b" => ["a"])) == res
    res = Dict{String,Verdict}("a" => excl, "b" => excl)
    @test resolve_objects(res, Dict("a" => ["b"], "b" => ["a"])) == res
    # KEPT vs EXCLUDED: both held
    res = Dict{String,Verdict}("a" => kept, "b" => excl, "c" => kept)
    out = resolve_objects(res, Dict("a" => ["b"], "b" => ["a"]))
    @test out["a"][1] === nothing && out["b"][1] === nothing && out["a"][3] == "same_object" == out["b"][3]
    @test occursin("a: KEPT_AS_JULIA_EXTRA, b: EXCLUDED_INTERNAL_HELPER", out["a"][2])
    @test out["c"] == kept                                                  # an unrelated name is untouched
    # KEPT vs undecided: the KEPT one joins, the undecided one keeps its own reason
    res = Dict{String,Verdict}("em_fit_phylo" => held("em_user_fitter"), "fit_em_phylo" => kept)
    out = resolve_objects(res, Dict("em_fit_phylo" => ["fit_em_phylo"], "fit_em_phylo" => ["em_fit_phylo"]))
    @test out["fit_em_phylo"][1] === nothing && out["fit_em_phylo"][3] == "same_object"
    @test out["em_fit_phylo"] == held("em_user_fitter")
    # both undecided: each keeps its own reason
    res = Dict{String,Verdict}("a" => held("reexport"), "b" => held("alias_twin"))
    @test resolve_objects(res, Dict("a" => ["b"], "b" => ["a"])) == res
    # three names on one object, one dissenter: all three held
    res = Dict{String,Verdict}("a" => kept, "b" => kept, "c" => excl)
    tw = Dict("a" => ["b", "c"], "b" => ["a", "c"], "c" => ["a", "b"])
    @test all(n -> resolve_objects(res, tw)[n][1] === nothing, ["a", "b", "c"])
    # a twin that is not a reverse-gap item matches an R name: held
    res = Dict{String,Verdict}("a" => kept)
    out = resolve_objects(res, Dict("a" => ["matches_r"]))
    @test out["a"][1] === nothing && out["a"][3] == "same_object" && occursin("matches an R name", out["a"][2])
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
    for must in ("EXCLUDED_INTERNAL_HELPER when tools/parity_ledger.py --ref 9539352f6 classes the Julia name as internal",
                 "the class begins with internal", "\"internal marginal-likelihood kernel ...\"",
                 "\"internal-but-exported helper; candidate to unexport\"", "\"engine internal\"", "\"internals;\"",
                 "regardless of whether it has a docstring",
                 "KEPT_AS_JULIA_EXTRA when the name is not classed internal, has a docstring, and that docstring is rendered on a docs/src page",
                 "does not count",                                                   # internal knob of R, internally
                 "(1) its binding is owned by another package (a re-export)",
                 "(2) parity_ledger.py ALIASES maps an R export to it, or to another export bound to the same object (===)",
                 "(3) its authored docstring was silently dropped by Julia",
                 "(4) it is bound to the same object (===) as another export and the per-name verdicts differ",
                 "all of them are held",
                 "(5) it would be EXCLUDED but user-facing prose presents it as a user tool",
                 "other than api.md and low-level-reference.md, or in README.md",
                 "(6) it would be EXCLUDED but its internal class is marked PROPOSED (unsigned)",
                 "(7) it would be KEPT but its own ledger class says R has it",
                 "(8) it is on the named review list of this script (NAMED_HOLDS)")
        @test occursin(must, CRITERION)
    end
    for n in keys(NAMED_HOLDS)                                                       # every named hold is in the text
        @test occursin(n, CRITERION)
    end
    m = match(r"^ \"criterion\": \"(.*)\",$"m, read(OUT, String))
    @test m !== nothing
    @test replace(m.captures[1], "\\\"" => "\"") == CRITERION
end

@testset "C6 ledger helper (tools/parity_ledger.py class tables)" begin
    cls = ledger_classes(["rrr_marginal_loglik", "welch_t", "fit_em_phylo", "fit_gllvm", "dof", "totally_unclassed_name",
                          "gaussian_reml_loglik", "em_fit_phylo", "link_residual", "gaussian_marginal_loglik_sparse_phy"])
    @test is_internal_class(cls["rrr_marginal_loglik"][1])
    @test is_internal_class(cls["welch_t"][1]) && occursin("candidate to unexport", cls["welch_t"][1])
    @test is_proposed_class(cls["welch_t"][1]) && !is_proposed_class(cls["rrr_marginal_loglik"][1])
    @test !is_internal_class(cls["fit_em_phylo"][1]) && !isempty(cls["fit_em_phylo"][1])
    @test cls["fit_gllvm"][2] == ["gllvmTMB"]                                # ALIASES: R gllvmTMB -> fit_gllvm
    @test !is_internal_class(cls["dof"][1]) && occursin("internally", cls["dof"][1])
    @test cls["totally_unclassed_name"] == ("", String[])
    # the REML class says "internal" about R's knob: not an internal class for the Julia name
    @test occursin("internal control knob", cls["gaussian_reml_loglik"][1]) && !is_internal_class(cls["gaussian_reml_loglik"][1])
    # the ledger's ^em_ class is internal; the signed override (D-319) is what overrides it
    @test is_internal_class(cls["em_fit_phylo"][1]) && haskey(SIGNED_OVERRIDES, "em_fit_phylo")
    @test says_r_has_it(cls["link_residual"][1])
    @test is_internal_class(cls["gaussian_marginal_loglik_sparse_phy"][1])
end

# Every class of the ledger that contains the word internal, read as the classifier reads it: the Julia
# name is internal only for these class shapes, and no other shape slips in.
@testset "C6 every reverse-gap class that contains \"internal\" is read as intended" begin
    cls = ledger_classes(gap_names())
    shapes = Dict{String,Int}()
    for (n, (c, _)) in cls
        occursin(r"internal"i, c) || continue
        shape = is_internal_class(c) ? "internal" : "not-internal"
        shape = (is_proposed_class(c) ? "PROPOSED " : "") * shape
        shapes[shape] = get(shapes, shape, 0) + 1
        if !is_internal_class(c)
            # the not-internal ones are exactly the classes that say "internally" or name R's own knob
            @test occursin(r"internally|internal control knob", c)
        end
    end
    @test Set(keys(shapes)) <= Set(["internal", "PROPOSED internal", "not-internal", "PROPOSED not-internal"])
    @test shapes["internal"] == 64 && shapes["PROPOSED internal"] == 10        # 50 + 6 + 4 + 4 unmarked, 10 proposed
end

import GLLVModels                                                            # for object identity; no exports pulled in

@testset "C6 object identity (===) over names(GLLVModels)" begin
    exported = exports_now()
    twins = object_twins(exported)
    # symmetric, and every twin is an export
    for (n, ts) in twins
        @test n in exported && all(t -> t in exported && n in get(twins, t, String[]), ts)
    end
    # the pairs that exist today
    @test twins["StudentT"] == ["StudentTFamily"] && twins["StudentTFamily"] == ["StudentT"]
    @test twins["em_fit_phylo"] == ["fit_em_phylo"] && twins["fit_phylo_squarem"] == ["em_fit_phylo_squarem"]
    # a numeric constant is never grouped by value
    @test !haskey(twins, "ZI_LAPLACE_EIGMIN_FLOOR")
    @test identity_groupable(sin) && identity_groupable(Float64) && identity_groupable(Base) && !identity_groupable(0.1)
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
    texts = Dict{String,String}(relpage(p) => read(p, String) for p in walk_files(joinpath(ROOT, "docs/src"), ".md"))
    texts["README.md"] = read(joinpath(ROOT, "README.md"), String)
    docs_src = r"docs/src/[A-Za-z0-9_./-]+\.md"
    for (n, (d, b)) in entries
        if haskey(SIGNED_OVERRIDES, n)                                       # signed one by one (D-319)
            @test (d, b) == (SIGNED_OVERRIDES[n][1], SIGNED_RULING * SIGNED_OVERRIDES[n][2])
            continue
        end
        c, als = cls[n]
        @test isempty(als)                                                   # never an ALIASES twin
        @test !haskey(NAMED_HOLDS, n)                                        # never a name held by the review
        @test !says_r_has_it(c)                                              # never a name whose own class says R has it
        if d == "EXCLUDED_INTERNAL_HELPER"
            @test is_internal_class(c)
            @test !is_proposed_class(c)                                      # never on an unsigned class alone
            @test b == "tools/parity_ledger.py --ref 9539352f6 classes it as internal: \"" * c * "\""
            # no user-facing prose presents an excluded name
            @test isempty([p for p in mentioned_in(n, texts) if !(p in REFERENCE_PAGES)])
        else
            @test !is_internal_class(c)
            pages = [m.match for m in eachmatch(docs_src, b)]
            @test !isempty(pages) && all(p -> isfile(joinpath(ROOT, p)), pages)
        end
    end
    # the REML kernel is KEPT, not excluded (its class names R's internal knob, not the Julia name)
    @test entries["gaussian_reml_loglik"][1] == "KEPT_AS_JULIA_EXTRA"
    # the names the final review held stay out of the file
    for n in ("StudentTFamily", "StudentT", "bridge_capabilities", "lognormal_marginal_loglik",
              "GllvmModel", "welch_t", "link_residual", "ZIB", "random_balanced_tree", "shrinkage_factor")
        @test !haskey(entries, n)
    end
    # the signed names are in the file with their signed decision
    @test all(n -> haskey(entries, n) && entries[n][1] == SIGNED_OVERRIDES[n][1], keys(SIGNED_OVERRIDES))
    @test length(entries) == 341
    # decided names that share an object agree on the decision (the guard held the others)
    twins = object_twins(exports_now())
    for (n, (d, _)) in entries, t in get(twins, n, String[])
        @test haskey(entries, t) && entries[t][1] == d
    end
end

using Test
using GLLVModels

# Issue #750: the docstring signature lines of the exported generics `getLV`,
# `predict`, `residuals` and `simulate` must list every keyword the methods
# accept (`mask`, `offset`, ...). Each method is matched to the docstring that
# documents it by its argument types, and every keyword name must appear in
# that docstring's signature block (up to the first blank line). The varargs
# slot `kwargs...` of the forwarding methods names no keyword and is skipped.
_docs_750(g) = Base.Docs.meta(GLLVModels)[Base.Docs.Binding(GLLVModels, g)].docs
_sig_block_750(doc) = first(split(join(doc.text), "\n\n"))
_kws_750(me) = [string(k) for k in Base.kwarg_decl(me) if !endswith(string(k), "...")]

@testset "postfit docstring signatures list all keywords (#750)" begin
    for g in (:getLV, :predict, :residuals, :simulate)
        f = getfield(GLLVModels, g)
        docs = _docs_750(g)
        for me in methods(f)
            me.module === GLLVModels || continue
            doc = get(docs, Base.tuple_type_tail(me.sig), nothing)
            doc === nothing && continue
            block = _sig_block_750(doc)
            for kw in _kws_750(me)
                @test occursin(kw, block)
            end
        end
    end
end

# The three methods the issue names must each be matched to their own docstring
# (so the loop above cannot pass by skipping them), and their prose must say
# what `mask` and `offset` do.
@testset "postfit docstrings explain mask and offset (#750)" begin
    for (g, T) in ((:getLV, GllvmFit), (:getLV, BinomialFit), (:getLV, PoissonFit),
                   (:predict, GllvmFit), (:predict, BinomialFit), (:predict, PoissonFit),
                   (:residuals, GllvmFit), (:residuals, BinomialFit), (:residuals, PoissonFit),
                   (:simulate, BinomialFit), (:simulate, PoissonFit))
        docs = _docs_750(g)
        keys_T = [k for k in keys(docs) if k <: Tuple && length(k.parameters) >= 1 &&
                  k.parameters[1] === T]
        @test length(keys_T) == 1
        prose = join(docs[only(keys_T)].text)
        body = join(split(prose, "\n\n")[2:end], "\n\n")
        kws = _kws_750(first(m for m in methods(getfield(GLLVModels, g))
                             if m.module === GLLVModels &&
                                Base.tuple_type_tail(m.sig) == only(keys_T)))
        for kw in ("mask", "offset")
            kw in kws && @test occursin(kw, body)
        end
        "offset" in kws && @test occursin("ArgumentError", body)
    end
end

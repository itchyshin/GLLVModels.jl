using Test
using GLLVModels

# Issue #751: the `confint_lv_effects` docstring signature lines must list every
# keyword the methods accept. Each method is matched to the docstring that
# documents it (the OrdinalFit method is documented by the generic docstring)
# and every keyword name must appear in that docstring's signature block (up to the first blank line).
@testset "confint_lv_effects docstring signatures list all keywords (#751)" begin
    binding = Base.Docs.Binding(GLLVModels, :confint_lv_effects)
    docs = Base.Docs.meta(GLLVModels)[binding].docs
    sig_block(key) = first(split(join(docs[key].text), "\n\n"))
    gauss_key = Tuple{GllvmFit, AbstractMatrix, AbstractMatrix}
    generic_key = first(k for k in keys(docs) if k != gauss_key)

    for me in methods(GLLVModels.confint_lv_effects)
        argtypes = Base.tuple_type_tail(me.sig)
        key = argtypes <: gauss_key ? gauss_key : generic_key
        line = sig_block(key)
        for kw in Base.kwarg_decl(me)
            @test occursin(string(kw), line)
        end
    end
end

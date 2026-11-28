using GLLVModels, Test

struct FormulaResponseRead <: Exception end
struct FormulaUnreadResponse <: AbstractMatrix{Float64} end
Base.size(::FormulaUnreadResponse) = (3, 8)
Base.getindex(::FormulaUnreadResponse, ::Int, ::Int) = throw(FormulaResponseRead())

@testset "Formula site rows checked before response access" begin
    Y = FormulaUnreadResponse()
    for family in (GLLVModels.Normal(),GLLVModels.Poisson(),GLLVModels.NegativeBinomial(),GLLVModels.Beta())
        for m in (7,9), formula in (@formula(y ~ 1),@formula(y ~ 1 + temp))
            @test_throws DimensionMismatch gllvm(formula,Y,(temp=zeros(m),);family=family,K=1)
        end
        @test_throws DimensionMismatch gllvm(@formula(y ~ 1),Y,(site=1:8,unused=zeros(7));family=family,K=1)
        @test_throws FormulaResponseRead gllvm(@formula(y ~ 1),Y,(site=1:8,);family=family,K=1)
        @test_throws FormulaResponseRead gllvm(@formula(y ~ 1),Y,NamedTuple();family=family,K=1)
    end
end

@testset "formula site_data () refused (refs #1414)" begin
    using Random
    Random.seed!(1126)
    p, n = 3, 20
    Y = randn(p, n)
    for family in (GLLVModels.Normal(), GLLVModels.Poisson())
        err = try
            gllvm(@formula(y ~ 1), Y, (); family = family, K = 1)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("empty tuple `()`", sprint(showerror, err))
        @test occursin("NamedTuple()", sprint(showerror, err))
    end
    @test gllvm(@formula(y ~ 1), Y, NamedTuple(); family = GLLVModels.Normal(), K = 1) isa GLLVModels.AnyGllvmFit
end

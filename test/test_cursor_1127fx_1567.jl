using GLLVModels, Test, Distributions, Random

@testset "gllvm(Y; family, K) matrix-first refusal (#1567)" begin
    Random.seed!(1127)
    Y = randn(5, 12)
    @test_throws ArgumentError gllvm(Y; family = Normal(), K = 2)
    @test begin
        try
            gllvm(Y; family = Normal(), K = 2)
            false
        catch e
            e isa ArgumentError || return false
            msg = sprint(showerror, e)
            occursin("fit_gllvm", msg) &&
                occursin("fit_gaussian_gllvm", msg) &&
                occursin("@formula", msg)
        end
    end
end
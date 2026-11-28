using GLLVModels, Test, Random
@testset "select_lv TweedieED (#1436)" begin
    rng = MersenneTwister(26015)
    Y = abs.(randn(rng, 5, 50))
    fit_tweedie_gllvm(Y; K = 2, iterations = 300)
    sel = select_lv(Y; family = GLLVModels.TweedieED(1.0, 1.5), Kmax = 2, iterations = 300)
    @test sel isa LVSelection && sel.best_k in 1:2
end

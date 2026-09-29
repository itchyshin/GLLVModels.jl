# Gate J3: guarded select_lv on the NB n300 p20 K_true3 fixture (seed as in heavy.csv).
using GLLVModels, Distributions, Random
include(joinpath(@__DIR__, "pilot_sim.jl"))
Y = simulate("nb", 300, 20, 3, MersenneTwister(hash(("nb", 300, 20, 3, 1))))
@time sel = select_lv(Y; family = NegativeBinomial(), Kmax = 5)
show(stdout, MIME"text/plain"(), sel); println()
foreach(println, sel.attempts)

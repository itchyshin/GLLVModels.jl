src = read("/tmp/claude-503/sibling-screen/beta.jl", String)
src = replace(src, r"\nmain\(\)\s*$" => "\n")
include_string(Main, src)
Y, φtrue, _ = simulate(5)
R = replicate(Y)
res = Optim.optimize(R.negll, R.θ0, R.ls, R.opts; autodiff = :finite)
show(stdout, MIME"text/plain"(), res); println()
println("x_converged=", Optim.x_converged(res), " f_converged=", Optim.f_converged(res),
        " g_converged=", Optim.g_converged(res), " g_residual=", Optim.g_residual(res))

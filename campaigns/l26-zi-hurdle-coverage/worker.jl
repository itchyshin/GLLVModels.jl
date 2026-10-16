# Long-running worker: reads "cell rep" lines on stdin, prints protocol lines on stdout.
#   READY                      — package loaded
#   START <cell> <rep>         — timed work begins (after any per-family JIT warm-up)
#   RESULT\t<tab-separated receipt fields>
# Parameter rows are appended to this worker's own CSV (ARGS[1]).
include("/home/user/l26/l26core.jl")

const PARAMS_PATH = ARGS[1]
const RECEIPT_FIELDS = (:cell, :family, :pi, :n, :p, :K, :rep, :seed, :status, :converged, :iterations,
                        :logLik, :pd_hessian, :n_nan_se, :n_boundary_terms, :grad_maxabs, :r_hat,
                        :prop_zero, :t_fit_s, :t_wald_s, :t_total_s, :error)

fmt(x::AbstractFloat) = isfinite(x) ? repr(x) : (isnan(x) ? "NaN" : (x > 0 ? "Inf" : "-Inf"))
fmt(x) = string(x)
clean(s) = replace(string(s), r"[\t\r\n,\"]" => " ")

const WARMED = Set{String}()
io_params = open(PARAMS_PATH, "a")

println("READY"); flush(stdout)
for line in eachline(stdin)
    parts = split(strip(line))
    length(parts) == 2 || continue
    cell = parse(Int, parts[1]); rep = parse(Int, parts[2])
    sp = cell_spec(cell)
    if !(sp.family in WARMED)
        # JIT warm-up on the n = 100 cell of this family with rep 0 (seed never used by a real rep).
        wcell = (findfirst(==(sp.family), FAMILIES) - 1) * 9 + 1
        try run_one(wcell, 0) catch end
        push!(WARMED, sp.family)
    end
    println("START $cell $rep"); flush(stdout)
    rc = try
        r, prm = run_one(cell, rep)
        for x in prm
            println(io_params, join((cell, rep, x.param, fmt(x.truth), fmt(x.est), fmt(x.se),
                                     fmt(x.lower), fmt(x.upper), x.covered), ","))
        end
        flush(io_params)
        r
    catch e
        e isa InterruptException && rethrow()
        msg = clean(sprint(showerror, e))[1:min(end, 300)]
        (cell = cell, family = sp.family, pi = sp.pi, n = sp.n, p = P, K = K, rep = rep,
         seed = seed_of(cell, rep), status = "error", converged = false, iterations = -1, logLik = NaN,
         pd_hessian = false, n_nan_se = -1, n_boundary_terms = -1, grad_maxabs = NaN, r_hat = NaN,
         prop_zero = NaN, t_fit_s = NaN, t_wald_s = NaN, t_total_s = NaN, error = msg)
    end
    println("RESULT\t", join((fmt(getfield(rc, f)) for f in RECEIPT_FIELDS), "\t")); flush(stdout)
end
close(io_params)

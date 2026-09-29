using GLLVModels, Distributions, Random, Printf

outdir = @__DIR__
csvpath = joinpath(outdir, "ridge_bic.csv")
header = "n,p,K_true,rep,ridge,sel_bic,sel_bic_sites,statuses,secs"

# Load existing rows (for resumability): key = (n,p,K,r,ridge)
done = Set{NTuple{5,Any}}()
if isfile(csvpath)
    open(csvpath, "r") do io
        first = true
        for line in eachline(io)
            if first
                first = false
                continue
            end
            isempty(strip(line)) && continue
            parts = split(line, ",")
            length(parts) < 5 && continue
            n = parse(Int, parts[1]); p = parse(Int, parts[2]); K = parse(Int, parts[3])
            r = parse(Int, parts[4]); ridge = parts[5]
            push!(done, (n, p, K, r, ridge))
        end
    end
else
    open(csvpath, "w") do io
        println(io, header)
    end
end

ridge_key(ridge) = isinf(ridge) ? "Inf" : string(ridge)

reps, L = 10, 1.5
cells = [(60, 10, 2), (120, 10, 2), (120, 20, 2), (120, 20, 3)]

io = open(csvpath, "a")
for (n, p, K) in cells, r in 1:reps
    rng = MersenneTwister(hash((n, p, K, r, L)))
    Λ = L .* randn(rng, p, K); η = Λ * randn(rng, K, n)
    Y = [rand(rng, Bernoulli(1 / (1 + exp(-x)))) for x in η]
    for ridge in (Inf, 2.0)
        key = (n, p, K, r, ridge_key(ridge))
        if key in done
            continue
        end
        local sel_bic, sel_bic_sites, statuses, secs
        t0 = time()
        try
            sel = select_lv(Y; family = Binomial(), Kmax = 4, criterion = :bic, binary_ridge = ridge)
            secs = time() - t0
            sel_bic = sel.best_k
            if !isempty(sel.K)
                sel_bic_sites = sel.K[argmin(sel.bic_sites)]
            else
                sel_bic_sites = "NA"
            end
            statuses = join(string.(getfield.(sel.attempts, :status)), "/")
        catch e
            secs = time() - t0
            if e isa InterruptException
                rethrow()
            end
            msg = sprint(showerror, e)
            msg = replace(msg, "\n" => " ")
            msg = msg[1:min(end, 80)]
            sel_bic = "NA"
            sel_bic_sites = "NA"
            statuses = "ERROR " * msg
        end
        statuses_esc = replace(statuses, "\"" => "\"\"")
        row = @sprintf("%d,%d,%d,%d,%s,%s,%s,\"%s\",%.3f", n, p, K, r, ridge_key(ridge), string(sel_bic), string(sel_bic_sites), statuses_esc, secs)
        println(io, row)
        flush(io)
        push!(done, key)
    end
end
close(io)
println("DONE")

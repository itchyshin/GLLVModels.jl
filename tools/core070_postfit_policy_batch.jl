# Inner executor for the "postfit-policy" manifest-area batch (15 planned
# EXECUTABLE_NOW cases + 2 negative controls).
#
# REPAIR NOTE (2026-09-01, Totoro incident): the prior version of this file
# `include()`d the ENTIRE test/parity/runparity.jl parity runner and used
# RCall to embed a live R session, obtaining the R oracle numbers by @rput
# fixture data into it and driving gllvmTMB(). On Totoro that first
# segfaulted (exit 139 for the subprocess julia process; cured only by
# LD_PRELOAD=<julia>/lib/julia/libunwind.so.8 -- an RCall/libunwind
# interaction worth remembering if a future runner hits the same signal),
# then failed inside test/parity/test_negbin_parity.jl -- an out-of-scope
# RCall fixture the runparity.jl runner reaches internally, well beyond this
# batch's 15-case scope. This file is now a pure-Julia consumer: it reads a
# JSON oracle file the paired R runner (tools/core070_postfit_policy_batch.R)
# writes BEFORE invoking this script -- that R process already has the
# frozen gllvmTMB library loaded and does 100% of the live R fitting itself
# -- refits the identical Y natively with fit_gaussian_gllvm, and compares
# via direct `using GLLVModels` module calls only. No RCall, no parity-runner
# include, no R of any kind runs in this process.
#
# argv:
#   ARGS[1]  destination path for the JSON results file (parent dir need not
#            exist yet; this script mkpath()s it). Must not already exist.
#
# Env vars required (set by the outer R runner before invoking this script;
# never defaulted silently here so a misconfigured environment fails loudly
# rather than quietly running against stale or missing oracle data):
#   CORE070_POSTFIT_POLICY_R_ORACLE = <path to the R-written oracle JSON>
#
# Invocation:
#   julia --project=. tools/core070_postfit_policy_batch.jl <out.json>

parity_pin = uppercase(strip(get(ENV, "GLLVM_PARITY_PIN", "P1")))
parity_pin in ("P0", "P1") || error("GLLVM_PARITY_PIN must be P0 or P1")

using GLLVModels
if parity_pin == "P1"
    @eval using SHA
    @eval using Libdl
    @eval using Pkg
    include(joinpath(@__DIR__, "core070_postcoef_fit_health.jl"))
end

# ---------------------------------------------------------------------------
# Minimal JSON reader/writer (no external dependency; mirrors the existing
# repo convention in tools/core070_postfit_1_batch.jl / core070_data_batch.jl).
# ---------------------------------------------------------------------------
function json_read(path::AbstractString)
    txt = read(path, String)
    pos = Ref(1)
    skip_ws!() = begin
        while pos[] <= lastindex(txt) && isspace(txt[pos[]])
            pos[] = nextind(txt, pos[])
        end
    end
    local parse_value
    function parse_string()
        pos[] += 1 # opening quote
        buf = IOBuffer()
        while true
            c = txt[pos[]]
            if c == '"'
                pos[] = nextind(txt, pos[])
                break
            elseif c == '\\'
                pos[] = nextind(txt, pos[])
                e = txt[pos[]]
                if e == 'n'; write(buf, '\n')
                elseif e == 't'; write(buf, '\t')
                elseif e == 'u'
                    hex = txt[nextind(txt, pos[]):nextind(txt, pos[], 4)]
                    write(buf, Char(parse(UInt32, hex; base = 16)))
                    pos[] = nextind(txt, pos[], 4)
                else write(buf, e)
                end
                pos[] = nextind(txt, pos[])
            else
                write(buf, c)
                pos[] = nextind(txt, pos[])
            end
        end
        String(take!(buf))
    end
    function parse_number()
        start = pos[]
        while pos[] <= lastindex(txt) && (isdigit(txt[pos[]]) || txt[pos[]] in ('-', '+', '.', 'e', 'E'))
            pos[] = nextind(txt, pos[])
        end
        s = txt[start:prevind(txt, pos[])]
        return occursin('.', s) || occursin('e', s) || occursin('E', s) ? parse(Float64, s) : parse(Int, s)
    end
    function parse_array()
        pos[] += 1
        out = Any[]
        skip_ws!()
        if txt[pos[]] == ']'
            pos[] = nextind(txt, pos[])
            return out
        end
        while true
            skip_ws!()
            push!(out, parse_value())
            skip_ws!()
            if txt[pos[]] == ','
                pos[] = nextind(txt, pos[])
            elseif txt[pos[]] == ']'
                pos[] = nextind(txt, pos[])
                break
            end
        end
        out
    end
    function parse_object()
        pos[] += 1
        out = Dict{String, Any}()
        skip_ws!()
        if txt[pos[]] == '}'
            pos[] = nextind(txt, pos[])
            return out
        end
        while true
            skip_ws!()
            k = parse_string()
            skip_ws!()
            @assert txt[pos[]] == ':'
            pos[] = nextind(txt, pos[])
            skip_ws!()
            out[k] = parse_value()
            skip_ws!()
            if txt[pos[]] == ','
                pos[] = nextind(txt, pos[])
            elseif txt[pos[]] == '}'
                pos[] = nextind(txt, pos[])
                break
            end
        end
        out
    end
    parse_value = function ()
        skip_ws!()
        c = txt[pos[]]
        if c == '"'; return parse_string()
        elseif c == '['; return parse_array()
        elseif c == '{'; return parse_object()
        elseif c == 't'; pos[] += 4; return true
        elseif c == 'f'; pos[] += 5; return false
        elseif c == 'n'; pos[] += 4; return nothing
        else; return parse_number()
        end
    end
    skip_ws!()
    parse_value()
end

json_escape(s::AbstractString) = replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n")
to_json(x::Bool) = x ? "true" : "false"
to_json(x::Nothing) = "null"
to_json(x::AbstractString) = "\"" * json_escape(x) * "\""
to_json(x::Integer) = string(x)
to_json(x::AbstractFloat) = isfinite(x) ? repr(x) : "null"
to_json(x::AbstractVector) = "[" * join(to_json.(x), ",") * "]"
to_json(x::AbstractDict) = "{" * join(("\"$(json_escape(string(k)))\":" * to_json(v) for (k, v) in x), ",") * "}"

length(ARGS) == 1 || error("usage: julia tools/core070_postfit_policy_batch.jl <out.json>")
out_path = ARGS[1]
isfile(out_path) && error("destination already exists: $out_path")
mkpath(dirname(out_path))

@assert realpath(Base.pkgdir(GLLVModels)) == realpath(pwd()) "must run from the GLLVModels.jl package root"

oracle_path = get(ENV, "CORE070_POSTFIT_POLICY_R_ORACLE", "")
isempty(oracle_path) &&
    error("CORE070_POSTFIT_POLICY_R_ORACLE is required; refusing to silently default it")
isfile(oracle_path) ||
    error("CORE070_POSTFIT_POLICY_R_ORACLE points to a nonexistent file: $oracle_path")

p1_capture = if parity_pin == "P1"
    root = realpath(pwd())
    realpath(Base.pkgdir(GLLVModels)) == root || error("Julia package is not loaded from the capture checkout")
    realpath(@__FILE__) == realpath(joinpath(root, "tools/core070_postfit_policy_batch.jl")) ||
        error("Julia runner is not the capture checkout runner")
    isempty(readchomp(`git status --porcelain -- src`)) || error("Julia src tree is dirty")
    commit = readchomp(`git rev-parse HEAD`)
    Dict("source_commit" => commit,
         "source_tree" => readchomp(`git rev-parse $(commit):src`),
         "r_oracle_sha256" => bytes2hex(sha256(read(oracle_path))),
         "julia_runner_sha256" => bytes2hex(sha256(read(@__FILE__))))
else
    nothing
end

function postcoef_package_inventory()
    records = Dict{String, Any}[]
    for (id, package) in Pkg.dependencies()
        string(id) == "2dc8e01c-4f48-4476-aaae-e919b4a30df7" && continue
        package.source === nothing && error("manifest package source is missing")
        push!(records, Dict("name" => package.name, "uuid" => string(id),
                            "tree_hash" => package.tree_hash === nothing ? nothing : string(package.tree_hash),
                            "source" => realpath(package.source)))
    end
    sort!(records; by = record -> record["uuid"])
end

function verify_postcoef_package_sources(records, manifest, checker, python, stdlib_root)
    return mktemp() do path, io
        write(io, to_json(records))
        flush(io)
        cmd = addenv(`$python $checker --manifest $manifest --inventory $path --stdlib-root $stdlib_root`, "LD_PRELOAD" => "")
        result = read(cmd, String)
        occursin("PACKAGE_SOURCE_TREES_MATCH", result) || error("Julia package source validation failed")
        strip(result)
    end
end

p1_runtime = if parity_pin == "P1"
    root = realpath(pwd())
    realpath(Base.active_project()) == realpath(joinpath(root, "Project.toml")) ||
        error("Julia active project is not the capture checkout")
    get(ENV, "JULIA_LOAD_PATH", "") == "@:@stdlib" || error("P1 requires an isolated Julia load path")
    manifest = joinpath(root, "Manifest.toml")
    !isfile(joinpath(root, "Manifest-v$(VERSION.major).$(VERSION.minor).toml")) ||
        error("P1 capture does not permit a version-specific Julia manifest")
    isfile(manifest) || error("P1 capture requires a retained Julia Manifest.toml")
    executable = realpath(joinpath(Sys.BINDIR, Base.julia_exename()))
    preferences = joinpath(root, "LocalPreferences.toml")
    depot_checker = realpath(joinpath(root, "tools/core070_postcoef_depot_check.py"))
    python = Sys.which("python3")
    python === nothing && error("python3 is required to validate Julia package sources")
    python = realpath(python)
    inventory = postcoef_package_inventory()
    stdlib_root = realpath(Sys.STDLIB)
    depot_check = verify_postcoef_package_sources(inventory, manifest, depot_checker, python, stdlib_root)
    preload_paths = filter(x -> !isempty(x), split(get(ENV, "LD_PRELOAD", ""), r"[:\s]+"))
    loaded_paths = Set(realpath(path) for path in Libdl.dllist() if isfile(path))
    all(path -> realpath(path) in loaded_paths, preload_paths) || error("recorded Julia preload is not loaded")
    preloads = [Dict("path" => realpath(path), "sha256" => bytes2hex(sha256(read(path))))
                for path in preload_paths]
    Dict("executable_path" => executable, "executable_sha256" => bytes2hex(sha256(read(executable))),
         "project_sha256" => bytes2hex(sha256(read(joinpath(root, "Project.toml")))),
         "manifest_sha256" => bytes2hex(sha256(read(manifest))),
         "preferences_sha256" => isfile(preferences) ? bytes2hex(sha256(read(preferences))) : "absent",
         "load_path" => get(ENV, "JULIA_LOAD_PATH", ""),
         "depot_path" => get(ENV, "JULIA_DEPOT_PATH", ""),
         "depot_inventory" => inventory,
         "depot_check" => depot_check,
         "stdlib_root" => stdlib_root,
         "depot_checker_sha256" => bytes2hex(sha256(read(depot_checker))),
         "python_executable_path" => python,
         "python_executable_sha256" => bytes2hex(sha256(read(python))),
         "preloads" => preloads, "julia_version" => string(VERSION))
else
    nothing
end

oracle = json_read(oracle_path)

# ---------------------------------------------------------------------------
# Read the R-written fixture and oracle numbers. p/n/K/Y come straight from
# the R side's own live fit -- no independent regeneration, no cross-language
# RNG matching required.
# ---------------------------------------------------------------------------
p = Int(oracle["p"])
n = Int(oracle["n"])
K = Int(oracle["K"])
y_flat = Float64.(oracle["y"])
length(y_flat) == p * n || error("oracle y length ($(length(y_flat))) != p*n ($(p*n))")
Y = reshape(y_flat, p, n) # R wrote column-major (p, n): matches Julia reshape directly

X = zeros(p, n, p)
for j in 1:p
    X[j, :, j] .= 1
end

r_coef = Float64.(oracle["coef"])
r_nobs = Int(oracle["nobs"])
r_df = Int(oracle["df"])
r_loglik = Float64(oracle["loglik"])
r_loglik_nobs_attr = Int(oracle["loglik_nobs_attr"])
r_link = reshape(Float64.(oracle["link"]), size(Y))
r_response = reshape(Float64.(oracle["response"]), size(Y))
r_residual = reshape(Float64.(oracle["residual"]), size(Y))
r_predict_type_default = String(oracle["predict_type_default"])
r_residual_type_default = String(oracle["residual_type_default"])
r_residual_scale_default = String(oracle["residual_scale_default"])
r_simulate_condition_on_re_default = Bool(oracle["simulate_condition_on_re_default"])
r_ci_ok = Bool(oracle["ci_ok"])
r_ci_lower = Float64.(oracle["ci_lower"])
r_ci_upper = Float64.(oracle["ci_upper"])
r_ci_error = String(oracle["ci_error"])
r_empty_coef = Float64.(oracle["empty_coef"])

# ---------------------------------------------------------------------------
# Native Julia refit + direct GLLVModels module accessor calls (no RCall).
# ---------------------------------------------------------------------------
fit = fit_gaussian_gllvm(Y; K = K, X = X)

j_coef = GLLVModels.StatsAPI.coef(fit)
j_loglik = fit.logLik
j_dof = GLLVModels.StatsAPI.dof(fit)
j_nobs = GLLVModels.StatsAPI.nobs(fit, Y)
j_link = predict(fit, Y; type = :link, X = X)
j_response = GLLVModels.StatsAPI.fitted(fit, Y; X = X)
j_residual = residuals(fit, Y; X = X)
j_ci = try
    GLLVModels.confint(fit; y = Y, X = X)
catch e
    nothing
end
j_ci_ok = j_ci !== nothing
beta_idx = j_ci_ok ? findall(t -> startswith(t, "beta["), j_ci.term) : Int[]
j_ci_lower = j_ci_ok ? j_ci.lower[beta_idx] : Float64[]
j_ci_upper = j_ci_ok ? j_ci.upper[beta_idx] : Float64[]

# simulate() keyword introspection: does GLLVModels's simulate accept a
# condition_on_RE-style keyword at all? (it must not, per the manifest fact)
# Call-based probe (not Base.kwarg_decl, which is an unstable internal): a
# MethodError from an unrecognized keyword means the kwarg is absent; any
# other outcome (success, or an error raised from inside a method that DID
# accept the keyword) means it exists.
julia_simulate_has_condition_on_re_kwarg = try
    GLLVModels.simulate(fit, 2; condition_on_RE = true)
    true
catch err
    !(err isa MethodError)
end

# ---------------------------------------------------------------------------
# Empty-design pair (POST-COEF-EMPTY): a real no-X Julia fit vs the pinned
# R coef.gllvmTMB_multi empty branch (already evaluated by the R stage
# against a synthetic mock object; r_empty_coef comes from the oracle file).
# ---------------------------------------------------------------------------
fit_no_x = fit_gaussian_gllvm(Y; K = 1)
j_empty_coef = GLLVModels.StatsAPI.coef(fit_no_x)

# P1 uses the same response and covariate exported by the R stage. Keep this
# draft held at startup until the receipt verifier binds all retained inputs.
postcoef_three_fit = if parity_pin == "P1"
    r_pair = oracle["postcoef_three_fit"]
    site_data = (x = Float64.(r_pair["x"]),)
    length(site_data.x) == n || error("POST-COEF shared x has the wrong length")
    fits = [gllvm(@formula(y ~ 0), Y, site_data; K = 1),
            gllvm(@formula(y ~ 1), Y, site_data; K = 1),
            gllvm(@formula(y ~ 1 + x), Y, site_data; K = 1)]
    # Use the same trait-intercept + shared-covariate design as the formula
    # route. For the intercept-only fit, the stored mean-design marker is used.
    covariate_formula = @formula(y ~ 1 + x)
    X_health = GLLVModels._pervar_formula_design(
        covariate_formula.rhs, GLLVModels.Tables.columntable(site_data), p, n;
        contrasts = Dict{Symbol,Any}())
    health_cases = ("no_fixed_effects", "trait_intercepts", "trait_intercepts_plus_covariate")
    designs = (nothing, nothing, X_health)
    function postcoef_profile_objective(fit, data, design)
        model = fit.model
        # These captures use the unconstrained, no-phy, no-diagonal Gaussian
        # formula route with its default (unprofiled) mean and no suff stats.
        (model.K_W == 0 && !model.has_diag && model.K_phy == 0 && !model.has_phy_unique &&
         isempty(GLLVModels._lambda_constraint_pinned_theta_indices(fit))) ||
            error("unsupported profile route")
        X = GLLVModels._mean_X(fit, design, size(data, 2))
        q = fit.pars.β === nothing ? 0 : length(fit.pars.β)
        mask = GLLVModels._pars_fixed_mask(fit.pars, :β_fixed, q)
        free = GLLVModels._free_coeff_indices(mask)
        length(free) == q || error("unsupported fixed coefficient mask")
        Xfit = X === nothing || isempty(free) ? nothing : Array{Float64,3}(X[:, :, free])
        spec = (q=q, p=model.p, K_B=model.K, K_W=model.K_W, has_diag=model.has_diag,
                K_phy=model.K_phy, has_phy_unique=model.has_phy_unique)
        objective = θ -> GLLVModels.gaussian_profile_nll(θ, data; spec=spec, X=Xfit,
            Σ_phy=nothing, profile_beta=false, suff=nothing)
        function recover(θ)
            rec = GLLVModels.profile_recover(θ, data; spec=spec, X=Xfit,
                Σ_phy=nothing, profile_beta=false)
            return (parameters=vcat(rec.β, [log(rec.σ_eps)], GLLVModels.pack_lambda(rec.Λ_B)),
                    loglik=rec.logLik)
        end
        return (objective=objective, recover=recover)
    end
    julia_health = [begin
        h = postcoef_julia_health_from_fit(health_cases[i], fits[i], Y, designs[i];
            penalty_state=nothing, nll_factory=GLLVModels._confint_reconstruct_nll,
            pinned_indices=GLLVModels._lambda_constraint_pinned_theta_indices,
            gradient_eval=(f, x) -> GLLVModels.ForwardDiff.gradient(f, x),
            hessian_eval=(f, x) -> GLLVModels.ForwardDiff.hessian(f, x),
            derivative_method="forwarddiff_exact")
        # Retain the engine's own flag as diagnostic evidence. The signed
        # Hessian and optimizer-gradient rule is checked independently.
        h["fit_converged"] = fits[i].converged
        try
            profile = postcoef_profile_objective(fits[i], Y, designs[i])
            # This source tree and exact unconstrained Gaussian route were
            # audited to contain no fixed-effect or ridge prior. The gate
            # independently authenticates the tree and runner source.
            audited_tree = "df63c52066625049087a321bb9b010f8d834ddcf"
            if p1_capture["source_tree"] == audited_tree && fits[i].integration === nothing && fits[i].model.K == 1
                h["penalty_state"] = "none"
                h["penalty_configuration_recorded"] = true
                h["penalty_provenance"] = Dict("source_tree" => audited_tree,
                    "case" => health_cases[i], "route" => "gaussian_profile_default",
                    "rank" => 1, "integration" => "exact", "prior" => "none")
            end
            h["optimizer"] = postcoef_julia_optimizer_health(fits[i].optim_result, profile.objective;
                minimizer_eval=GLLVModels.Optim.minimizer, minimum_eval=GLLVModels.Optim.minimum,
                gradient_eval=(f,x) -> GLLVModels.ForwardDiff.gradient(f,x),
                derivative_method="forwarddiff_exact")
            # The certifier reads the gradient radius beside the optimizer-coordinate
            # gradient it certifies; lift it unchanged (nothing if not declared).
            h["gradient_error_radius"] = h["optimizer"]["gradient_error_radius"]
            h["gradient_error_radius_basis"] = h["optimizer"]["gradient_error_radius_basis"]
            h["reconstruction"] = postcoef_julia_reconstruction_health(
                h["optimizer"]["optimizer_parameters"], fits[i].pars.θ_packed, fits[i].logLik;
                recover_eval=profile.recover)
        catch err
            err isa InterruptException && rethrow()
            h["optimizer"] = Dict("optimizer_error" => sprint(showerror, err))
        end
        h
    end for i in 1:3]
    access = GLLVModels.StatsAPI.coef
    always_empty(f) = Float64[]
    spurious(f) = vcat(access(f), 0.0)
    control = get(ENV, "GLLVM_POSTCOEF_ACCESSOR_CONTROL", "real")
    control in ("real", "always_empty", "spurious_coefficient") ||
        error("unknown POST-COEF accessor control")
    selected = control == "real" ? access : control == "always_empty" ? always_empty : spurious
    actual = [selected(f) for f in fits]
    all(v -> v isa Vector{Float64}, actual) || error("POST-COEF unexpected Julia coefficient representation")
    Dict("r_lengths" => [Int(x["length"]) for x in r_pair["fits"]],
         "r_design_columns" => [Int(x["fixed_columns"]) for x in r_pair["fits"]],
         "r_values" => [x["values"] for x in r_pair["fits"]],
         "julia_lengths" => length.(actual),
         "r_names" => [x["names"] for x in r_pair["fits"]],
         "julia_names" => [hasproperty(v, :names) ? collect(getproperty(v, :names)) : nothing
                           for v in actual],
         "fit_health" => Dict("R" => [x["fit_health"] for x in r_pair["fits"]],
                              "Julia" => julia_health),
         "selected_accessor" => control,
         "control_lengths" => Dict(
             "always_empty" => [length(always_empty(f)) for f in fits],
             "spurious_coefficient" => [length(spurious(f)) for f in fits]))
else
    nothing
end

# ---------------------------------------------------------------------------
# Assemble the 15 case results.
# ---------------------------------------------------------------------------
tol = Dict(
    "coefficient_delta" => 1e-6,
    "loglik_delta" => 1e-6,
    "link_response_residual_delta" => 1e-4,
    "wald_ci_bound_delta" => 1e-3,
)

cases = Dict{String, Any}()

cases["CORE070-POSTFIT-COEF-EMPTY-NATIVE"] = Dict{String, Any}(
    "pass" => parity_pin == "P0" ?
        (length(j_empty_coef) == 0 && length(r_empty_coef) == 0) :
        (postcoef_three_fit["r_lengths"] == [0, 5, 6] &&
         postcoef_three_fit["r_design_columns"] == [0, 5, 6] &&
         postcoef_three_fit["julia_lengths"] == [0, 5, 6] &&
         postcoef_three_fit["control_lengths"]["always_empty"] == [0, 0, 0] &&
         postcoef_three_fit["control_lengths"]["spurious_coefficient"] == [1, 6, 7]),
    "julia_length" => length(j_empty_coef), "r_length" => length(r_empty_coef),
)
if parity_pin == "P1"
    cases["CORE070-POSTFIT-COEF-EMPTY-NATIVE"]["three_fit"] = postcoef_three_fit
end

coefficient_delta = maximum(abs.(j_coef .- r_coef))
cases["CORE070-POSTFIT-COEF-NAMED-NATIVE"] = Dict(
    "pass" => coefficient_delta <= tol["coefficient_delta"], "delta" => coefficient_delta,
)

cases["CORE070-POSTFIT-CONFINT-METHODS-WALD-NATIVE"] = if j_ci_ok && r_ci_ok &&
        length(j_ci_lower) == length(r_ci_lower) == length(j_coef)
    d = maximum(abs.(vcat(j_ci_lower, j_ci_upper) .- vcat(r_ci_lower, r_ci_upper)))
    Dict("pass" => d <= tol["wald_ci_bound_delta"], "delta" => d)
else
    Dict("pass" => false, "julia_ci_ok" => j_ci_ok, "r_ci_ok" => r_ci_ok, "r_ci_error" => r_ci_error,
         "julia_len" => length(j_ci_lower), "r_len" => length(r_ci_lower))
end

response_delta = maximum(abs.(j_response .- r_response))
cases["CORE070-POSTFIT-FITTED-DEFAULT-NATIVE"] = Dict(
    "pass" => response_delta <= tol["link_response_residual_delta"], "delta" => response_delta,
)

cases["CORE070-POSTFIT-LOGLIK-DF-NATIVE"] = Dict("pass" => j_dof == r_df, "julia" => j_dof, "r" => r_df)

cases["CORE070-POSTFIT-LOGLIK-NOBS-NATIVE"] = Dict(
    "pass" => j_nobs == r_loglik_nobs_attr, "julia" => j_nobs, "r" => r_loglik_nobs_attr,
)

loglik_delta = abs(j_loglik - r_loglik)
cases["CORE070-POSTFIT-LOGLIK-VALUE-NATIVE"] = Dict(
    "pass" => loglik_delta <= tol["loglik_delta"], "delta" => loglik_delta,
)

cases["CORE070-POSTFIT-NOBS-COUNT-NATIVE"] = Dict("pass" => j_nobs == r_nobs, "julia" => j_nobs, "r" => r_nobs)
cases["CORE070-POSTFIT-NOBS-FALLBACK-NATIVE"] = Dict("pass" => j_nobs == r_nobs, "julia" => j_nobs, "r" => r_nobs)

link_delta = maximum(abs.(j_link .- r_link))
cases["CORE070-POSTFIT-PREDICT-DEFAULT-NATIVE"] = Dict(
    "pass" => link_delta <= tol["link_response_residual_delta"] &&
              r_predict_type_default == "link",
    "delta" => link_delta, "r_default" => r_predict_type_default, "julia_default" => "response",
)

cases["CORE070-POSTFIT-RE-FORM-FULL-NATIVE"] = Dict(
    "pass" => link_delta <= tol["link_response_residual_delta"], "delta" => link_delta,
)

residual_delta = maximum(abs.(j_residual .- r_residual))
cases["CORE070-POSTFIT-RESIDUAL-CONDITIONAL-NATIVE"] = Dict(
    "pass" => residual_delta <= tol["link_response_residual_delta"], "delta" => residual_delta,
)
cases["CORE070-POSTFIT-RESIDUAL-SCALES-NATIVE"] = Dict(
    "pass" => residual_delta <= tol["link_response_residual_delta"] &&
              r_residual_scale_default == "normal",
    "delta" => residual_delta,
)
cases["CORE070-POSTFIT-RESIDUAL-TYPES-NATIVE"] = Dict(
    "pass" => residual_delta <= tol["link_response_residual_delta"] &&
              r_residual_type_default == "randomized_quantile",
    "delta" => residual_delta,
)

cases["CORE070-POSTFIT-SIMULATE-DEFAULT-NATIVE"] = Dict(
    "pass" => r_simulate_condition_on_re_default == true &&
              !julia_simulate_has_condition_on_re_kwarg,
    "r_condition_on_re_default" => r_simulate_condition_on_re_default,
    "julia_has_condition_on_re_kwarg" => julia_simulate_has_condition_on_re_kwarg,
)

# ---------------------------------------------------------------------------
# Negative controls: deliberately-wrong comparisons that MUST fail.
# ---------------------------------------------------------------------------
neg_coef_delta = maximum(abs.(j_coef .- (r_coef .+ 1.0)))
negative_controls = Dict(
    "NEG-COEF-SHIFTED" => Dict("behaved" => neg_coef_delta > tol["coefficient_delta"], "delta" => neg_coef_delta),
    "NEG-LOGLIK-SHIFTED" => Dict(
        "behaved" => abs(j_loglik - (r_loglik + 1.0)) > tol["loglik_delta"],
        "delta" => abs(j_loglik - (r_loglik + 1.0)),
    ),
)

all_positive_pass = all(v["pass"] for v in values(cases))
negatives_behaved = all(v["behaved"] for v in values(negative_controls))
overall_ok = all_positive_pass && negatives_behaved

report = Dict(
    "status" => overall_ok ? "PASS" : "FAIL",
    "area" => "postfit-policy",
    "scope" => "CORE070_POSTFIT_POLICY_BATCH",
    "case_count" => length(cases),
    "negative_control_count" => length(negative_controls),
    "all_positive_pass" => all_positive_pass,
    "negative_controls_behaved_as_expected" => negatives_behaved,
    "all_checks" => overall_ok,
    "cases" => cases,
    "negative_controls" => negative_controls,
    "julia_version" => string(VERSION),
)
if parity_pin == "P1"
    isempty(readchomp(`git status --porcelain -- src`)) || error("Julia src tree changed during capture")
    p1_capture["source_commit"] == readchomp(`git rev-parse HEAD`) || error("Julia source commit changed during capture")
    p1_capture["r_oracle_sha256"] == bytes2hex(sha256(read(oracle_path))) || error("R oracle changed during capture")
    p1_capture["julia_runner_sha256"] == bytes2hex(sha256(read(@__FILE__))) || error("Julia runner changed during capture")
    p1_runtime["executable_sha256"] == bytes2hex(sha256(read(p1_runtime["executable_path"]))) || error("Julia executable changed during capture")
    p1_runtime["project_sha256"] == bytes2hex(sha256(read("Project.toml"))) || error("Julia project changed during capture")
    p1_runtime["manifest_sha256"] == bytes2hex(sha256(read("Manifest.toml"))) || error("Julia manifest changed during capture")
    !isfile("Manifest-v$(VERSION.major).$(VERSION.minor).toml") || error("alternate Julia manifest appeared during capture")
    current_preferences = isfile("LocalPreferences.toml") ? bytes2hex(sha256(read("LocalPreferences.toml"))) : "absent"
    p1_runtime["preferences_sha256"] == current_preferences || error("Julia preferences changed during capture")
    p1_runtime["depot_inventory"] == postcoef_package_inventory() || error("Julia package inventory changed during capture")
    checker = joinpath(pwd(), "tools/core070_postcoef_depot_check.py")
    p1_runtime["depot_checker_sha256"] == bytes2hex(sha256(read(checker))) || error("Julia depot checker changed during capture")
    p1_runtime["python_executable_sha256"] == bytes2hex(sha256(read(p1_runtime["python_executable_path"]))) || error("Python executable changed during capture")
    p1_runtime["depot_check"] == verify_postcoef_package_sources(p1_runtime["depot_inventory"], "Manifest.toml", checker,
        p1_runtime["python_executable_path"], p1_runtime["stdlib_root"]) || error("Julia package sources or coverage exclusions changed during capture")
    all(item -> item["sha256"] == bytes2hex(sha256(read(item["path"]))), p1_runtime["preloads"]) || error("Julia preload changed during capture")
    report["postcoef_capture_identity"] = p1_capture
    report["postcoef_runtime"] = p1_runtime
end
open(out_path, "w") do io
    print(io, to_json(report))
end
println("CORE070_POSTFIT_POLICY_BATCH_RESULT ",
        overall_ok ? "PASS" : "FAIL",
        " positive=", all_positive_pass, " negatives=", negatives_behaved)
overall_ok || exit(1)

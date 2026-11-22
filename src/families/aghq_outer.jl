# Unpenalized fixed-k outer adaptation, frozen R b4d5fee fit-multi.R:6540–6960.
# Symbolic/stopping contract: docs/dev-log/core070/aghq-outer-contract.md.

function _aghq_outer_caches(adapt, theta)
    caches = adapt(copy(theta))
    caches isa AbstractVector && !isempty(caches) && all(a -> a isa AGHQAdaptation, caches) ||
        throw(ArgumentError("AGHQ adaptation callback must return nonempty site caches"))
    for a in caches
        d = length(a.mode)
        d > 0 && size(a.inverse_root) == (d,d) && all(isfinite,a.mode) &&
            all(isfinite,a.inverse_root) && isfinite(a.logjac) ||
            throw(ArgumentError("AGHQ adaptation callback returned an invalid site cache"))
    end
    return caches
end

function _aghq_outer_step(f, theta, cap, optimizer_iterations, grad_tol)
    g! = (g,x) -> ForwardDiff.gradient!(g,f,x)
    method = Optim.LBFGS(linesearch=Optim.LineSearches.BackTracking(order=3))
    options = Optim.Options(iterations=cap === nothing ? optimizer_iterations : cap,
                            g_tol=grad_tol)
    return Optim.minimizer(Optim.optimize(f,g!,copy(theta),method,options))
end

function _aghq_outer_gradient(f, theta)
    try
        return maximum(abs,ForwardDiff.gradient(f,theta))
    catch e
        e isa InterruptException && rethrow()
        return NaN
    end
end

"""
    aghq_outer_optimize(start, adapt, objective; kwargs...)

Internal unpenalized, fixed-node-count outer driver. `adapt(theta)` supplies a
nonempty vector of observed-curvature [`AGHQAdaptation`](@ref) caches, after
solving and checking conditional modes. `objective(theta, caches)` is the
negative normalized log integral, differentiable with caches held fixed.

Short LBFGS surrogate steps are evaluated after re-adaptation. Rejected steps
are backtracked before permanently lowering their continuation cap. Stagnation
without a satisfactory frozen gradient never certifies convergence. Every
returned usable point has freshly evaluated adaptation, objective and gradient.
A nonfinite final gradient can leave a usable point but not a convergence claim.

The result retains all evaluated attempts, stopping reason, error, parameter
movement, repair count and `gradient_kind = :frozen_surrogate`. It makes no
claim about total derivatives through adaptation, conditional-mode validity,
public eligibility, multistart selection, or fitted-object integration. The
`step(f, theta, cap)` callback is injectable for deterministic transition tests;
`cap = nothing` means the ordinary `optimizer_iterations` budget, not infinity.
"""
function aghq_outer_optimize(start::AbstractVector, adapt, objective;
        n_adapt=400, iter_cap=1, continuation=true, escalate_patience=3,
        shift_tol=1e-4, grad_tol=1e-4, grad_tol_rel=1e-6, f_tol=1e-9,
        rho_min=1/64, optimizer_iterations=1000, step=nothing)
    for (name,value) in ((:n_adapt,n_adapt),(:iter_cap,iter_cap),
            (:escalate_patience,escalate_patience),(:optimizer_iterations,optimizer_iterations))
        value isa Integer && !(value isa Bool) && value > 0 ||
            throw(ArgumentError("$name must be a positive integer"))
    end
    continuation isa Bool || throw(ArgumentError("continuation must be boolean"))
    for (name,value) in ((:shift_tol,shift_tol),(:grad_tol,grad_tol),
            (:grad_tol_rel,grad_tol_rel),(:f_tol,f_tol))
        value isa Real && isfinite(value) && value >= 0 ||
            throw(ArgumentError("$name must be finite and nonnegative"))
    end
    rho_min isa Real && isfinite(rho_min) && 0 < rho_min <= 1 ||
        throw(ArgumentError("rho_min must be finite and in (0,1]"))
    !isempty(start) && all(isfinite,start) ||
        throw(ArgumentError("AGHQ start must be a nonempty finite vector"))
    initial = Vector{Float64}(start)
    all(isfinite,initial) || throw(ArgumentError("AGHQ start is not finite in Float64"))
    propose = step === nothing ? (f,t,c)->_aghq_outer_step(f,t,c,optimizer_iterations,grad_tol) : step
    caps = continuation && iter_cap == 1 ? Union{Nothing,Int}[1,2,5,25,nothing] : [Int(iter_cap)]
    stage=1; ceiling=length(caps); n_ok=0; rho=1.0
    theta=copy(initial); best=copy(initial); best_f=Inf; previous_f=Inf
    previous_modes=nothing; best_modes=nothing; dimensions=nothing; direction=nothing
    trace=NamedTuple[]; stop_reason=:adaptation_cap; message=nothing; converged=false
    evaluations=0
    good_gradient(g,f) = isfinite(g) && (g < grad_tol ||
        (grad_tol_rel > 0 && g/max(1,abs(f)) < grad_tol_rel))
    for pass in 1:n_adapt
        evaluations=pass
        local caches,modes,frozen,value
        try
            caches=_aghq_outer_caches(adapt,theta)
            shape=length.(getfield.(caches,:mode))
            dimensions === nothing ? (dimensions=shape) : shape == dimensions ||
                throw(DimensionMismatch("AGHQ site adaptation dimensions changed"))
            modes=reduce(vcat,(copy(a.mode) for a in caches))
            frozen=t->objective(t,caches)
            value=Float64(frozen(theta))
            isfinite(value) || throw(ArgumentError("AGHQ re-adapted objective is not finite"))
        catch e
            e isa InterruptException && rethrow()
            stop_reason=:adaptation_failed; message=sprint(showerror,e); break
        end
        gradient=_aghq_outer_gradient(frozen,theta)
        shift=previous_modes === nothing ? Inf : maximum(abs.(modes-previous_modes))
        previous_modes=copy(modes)
        accepted=value <= best_f+1e-10
        push!(trace,(pass=pass,cap=caps[stage],parameters=copy(theta),objective=value,
            frozen_gradient_max=gradient,mode_shift=shift,rho=rho,accepted=accepted,
            curvature_repairs=count(a->a.curvature_repaired,caches)))
        if accepted
            improvement=previous_f-value
            best=copy(theta);best_f=value;previous_f=value;best_modes=copy(modes)
            direction=nothing;rho=1.0;n_ok+=1
        else
            n_ok=0
            if direction !== nothing && rho > rho_min
                rho=rho/2
                theta=best+rho*direction
                continue
            end
            theta=copy(best);direction=nothing;rho=1.0
            if stage==1
                stop_reason=:no_merit_descent;break
            end
            ceiling=max(1,stage-1);stage=ceiling
            continue
        end
        g_ok=good_gradient(gradient,value)
        if n_ok>=2 && isfinite(shift) && shift < shift_tol &&
                (g_ok || (isfinite(improvement) && abs(improvement)<f_tol))
            converged=g_ok
            stop_reason=g_ok ? :converged : theta==initial ? :warm_start_stagnation : :objective_stagnation
            break
        end
        # The next proposal cannot be evaluated if the adaptation budget is spent.
        pass==n_adapt && break
        if stage<ceiling && n_ok>=escalate_patience
            stage+=1;n_ok=0
        end
        try
            trial=Vector{Float64}(propose(frozen,copy(theta),caps[stage]))
            length(trial)==length(theta) && all(isfinite,trial) ||
                throw(ArgumentError("AGHQ optimizer returned an invalid parameter vector"))
            direction=trial-theta;rho=1.0;theta=trial
        catch e
            e isa InterruptException && rethrow()
            stop_reason=:optimizer_failed;message=sprint(showerror,e);break
        end
    end
    # Never expose the rejected trial's cache or gradient as returned-fit metadata.
    final_caches=nothing;final_value=Inf;final_gradient=NaN;final_shift=Inf;usable=false
    if isfinite(best_f)
        try
            final_caches=_aghq_outer_caches(adapt,best)
            length.(getfield.(final_caches,:mode))==dimensions ||
                throw(DimensionMismatch("AGHQ final adaptation dimensions changed"))
            final_value=Float64(objective(best,final_caches))
            isfinite(final_value) || throw(ArgumentError("AGHQ final objective is not finite"))
            final_gradient=_aghq_outer_gradient(t->objective(t,final_caches),best)
            final_modes=reduce(vcat,(a.mode for a in final_caches))
            final_shift=maximum(abs.(final_modes-best_modes))
            usable=true
            if converged && !(good_gradient(final_gradient,final_value) && final_shift<shift_tol)
                converged=false;stop_reason=:final_stationarity_lost
            end
        catch e
            e isa InterruptException && rethrow()
            converged=false;stop_reason=:finalization_failed;message=sprint(showerror,e)
            final_caches=nothing;final_value=Inf
        end
    end
    return (parameters=best,objective=final_value,adaptation=final_caches,usable=usable,
        converged=converged && usable,stop_reason=stop_reason,error=message,passes=evaluations,
        trace=trace,frozen_gradient_max=final_gradient,
        relative_gradient=final_gradient/max(1,abs(final_value)),gradient_kind=:frozen_surrogate,
        parameter_shift=maximum(abs.(best-initial)),final_mode_shift=final_shift,
        curvature_repairs=final_caches===nothing ? 0 : count(a->a.curvature_repaired,final_caches))
end

# Frozen R fit-multi.R ranks converged finite runs before raw objective.
# Keep this shared by real-family adapters; no public fit fallback is made here.
function _aghq_select_run(runs)
    usable=findall(r->r.usable && isfinite(r.objective) && !isempty(r.parameters) &&
        all(isfinite,r.parameters),runs)
    isempty(usable) && return nothing
    converged=filter(i->runs[i].converged,usable)
    candidates=isempty(converged) ? usable : converged
    return candidates[argmin([runs[i].objective for i in candidates])]
end

# Laplace-retention policy for the selected usable run of an AGHQ fitter. Every AGHQ
# fitter calls _aghq_unmoved_unconverged; only the ordinal fitter also calls
# _aghq_failure_stop. A fitter that retains Laplace records actual=:laplace with the
# selected run's own stop_reason, so the record differs from the no-usable-run fallback
# (integration.result.usable is true) and from an AGHQ estimate (actual is :laplace).
#
# Rule 1, all AGHQ fitters. A usable run that never left its start (parameter_shift==0)
# and did not converge has no AGHQ estimate: its parameters are the Laplace warm start or
# the synthetic multistart guess, unchanged. aghq_outer_optimize produces this state only
# with these stop reasons: :adaptation_cap (budget spent before a step was accepted),
# :no_merit_descent (every first-stage step rejected), :optimizer_failed (the first
# proposal threw), :adaptation_failed (a later proposal could not be re-adapted, so best
# stayed at the start), :warm_start_stagnation, and :final_stationarity_lost (an
# unmoved stationary start lost its certificate at finalization). :finalization_failed
# runs are unusable and never selected; :objective_stagnation implies movement.
# Laplace convergence is not required: returning a start point unchanged is never an
# AGHQ fit, and the returned Laplace fit keeps its own convergence flag.
_aghq_unmoved_unconverged(run)=run.parameter_shift==0 && !run.converged

# Rule 2, ordinal only (evidence: separated trait 3 of the test_convergence_sweep seed-4
# fixture under OpenBLAS Haswell/Zen kernels; no sibling case has been examined). A run
# that moved but ended on a failure stop, :adaptation_failed or :optimizer_failed, before
# converging has no stationarity certificate. The caller retains Laplace only when Laplace
# converged, because here a moved AGHQ point exists and is discarded only for a certified
# alternative. Budget and stagnation stops (:adaptation_cap, :no_merit_descent,
# :objective_stagnation, :warm_start_stagnation, :final_stationarity_lost) keep a moved
# AGHQ point. :finalization_failed is listed for completeness; such a run is unusable.
_aghq_failure_stop(run)=!run.converged &&
    run.stop_reason in (:adaptation_failed,:optimizer_failed,:finalization_failed)

"""
    aghq_multistart_optimize(starts, adapt, objective; kwargs...)

Run every supplied start independently through [`aghq_outer_optimize`](@ref).
Retain all attempts. Select a usable converged run before any nonconverged run,
then the lowest objective (first start wins ties). If all attempts are unusable,
`winner` and `selected` are `nothing`; public callers must report their fallback.
Input errors and interrupts propagate. Ordinary mode/adaptation failures are
retained by the single-start driver. All start vectors are copied and validated
before fitting; `kwargs` are the single-start controls, not a new optimizer.
"""
function aghq_multistart_optimize(starts::AbstractVector,adapt,objective;kwargs...)
    !isempty(starts) && all(t->t isa AbstractVector && !isempty(t) &&
        all(x->x isa Real && isfinite(x),t),starts) ||
        throw(ArgumentError("AGHQ starts must be nonempty finite numeric vectors"))
    length(unique(length.(starts)))==1 ||
        throw(ArgumentError("AGHQ starts must have equal parameter dimensions"))
    owned=[Vector{Float64}(t) for t in starts]
    all(t->all(isfinite,t),owned) || throw(ArgumentError("AGHQ starts must be finite in Float64"))
    runs=[aghq_outer_optimize(t,adapt,objective;kwargs...) for t in owned]
    winner=_aghq_select_run(runs)
    return (runs=runs,winner=winner,selected=winner===nothing ? nothing : runs[winner],
        usable=winner!==nothing)
end

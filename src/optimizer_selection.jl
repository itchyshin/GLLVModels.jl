"""
    OptimizerStage

One outer optimization attempt retained in a fit's [`OptimizerRecord`](@ref).
The record captures the method, derivative route, stopping controls, raw Optim
result, and whether this attempt supplied the retained estimate.
"""
struct OptimizerStage
    name::Symbol
    method::Any
    derivative::Symbol
    controls::NamedTuple
    result::Any
    accepted::Bool
end

"""
    OptimizerRecord

Immutable per-fit receipt for outer optimizer selection. `requested` is the
user's keyword (`nothing` means the historical default); `actual` is the
concrete method type used for the retained estimate. `stages` keeps the ordered
raw results for every outer attempt, including rejected restarts.
"""
struct OptimizerRecord
    requested::Any
    actual::Type
    stages::Tuple{Vararg{OptimizerStage}}
    controls::NamedTuple
    derivative::Symbol
end

"""
    optimizer_info(fit) -> Union{Nothing, OptimizerRecord}

Return the immutable record of outer optimizer attempts stored on `fit`, or
`nothing` for legacy objects created without one. Inner latent-mode solvers,
curvature settings, and integration methods are not outer optimizer choices.
"""
function optimizer_info(fit)
    hasfield(typeof(fit), :optimizer_info) || return nothing
    return getfield(fit, :optimizer_info)
end

function _check_optimizer_refit(fit, label::AbstractString)
    info = optimizer_info(fit)
    if info !== nothing && info.requested !== nothing
        throw(ArgumentError(
            "$label is not available after an explicit optimizer selection: its internal " *
            "refits do not forward the selected outer optimizer. Refit with optimizer=nothing " *
            "before requesting profile or bootstrap intervals"))
    end
    return nothing
end

function _refuse_optimizer(requested, route::AbstractString, reason::AbstractString)
    requested === nothing && return nothing
    throw(ArgumentError("$route: an explicit outer optimizer is not supported for $reason"))
end

function _optimizer_method(requested, default; route::AbstractString)
    requested === nothing && return default
    requested === :lbfgs && return Optim.LBFGS()
    requested === :bfgs && return Optim.BFGS()
    (requested isa Optim.LBFGS || requested isa Optim.BFGS) && return requested
    throw(ArgumentError("$route: optimizer must be nothing, :lbfgs, :bfgs, " *
                        "Optim.LBFGS(), or Optim.BFGS(); got $(typeof(requested))"))
end

function _optimizer_stage!(stages::Vector{OptimizerStage}, name::Symbol, method,
                           derivative::Symbol, controls::NamedTuple, result;
                           accepted::Bool = true)
    push!(stages, OptimizerStage(name, method, derivative, controls, result, accepted))
    return result
end

function _optimizer_record(requested, stages::Vector{OptimizerStage};
                           controls::NamedTuple = (;), derivative::Symbol = :unknown,
                           actual = nothing)
    chosen = actual
    if chosen === nothing
        for stage in stages
            stage.accepted && (chosen = stage.method)
        end
    end
    chosen === nothing && throw(ArgumentError("cannot record an optimizer before an outer solve"))
    return OptimizerRecord(requested, typeof(chosen).name.wrapper, Tuple(stages), controls, derivative)
end

function _last_accepted_optimizer(stages::Vector{OptimizerStage})
    chosen = nothing
    for stage in stages
        stage.accepted && (chosen = stage.method)
    end
    return chosen
end

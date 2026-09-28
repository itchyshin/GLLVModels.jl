# Julia child for the "namespace-1" manifest-area batch (Tier 0: existence /
# registration parity -- see docs/dev-log/core070/namespace-1-batch-contract.json
# for the full 48-EXECUTABLE_NOW / 6-NEEDS_NEW_JULIA_SURFACE / 36-REUSED_OR_RECLASSIFY
# triage and the deferred Tier 1 numeric-parity follow-up).
#
# This process carries NO RCall dependency and reads NO R state: it loads
# GLLVModels.jl itself and answers, for every symbol named in the contract's
# `cases` + `needs_new_julia_surface` lists, "does this symbol exist at
# GLLVModels module scope?" via `isdefined`. That is the entirety of what this
# tier checks -- the outer R runner (tools/core070_namespace_1_batch.R)
# answers the matching R-side question by scanning the pinned readback R
# source text, with no installed gllvmTMB package required either.
#
# argv (matches the contract's runner.inner_argv and this file's own
# --self-test, which never reaches disk under that path):
#   julia --project=. tools/core070_namespace_1_batch.jl <destination>/julia-facts.json
#
# <destination> must already exist (the outer R runner mkpath()s it before
# invoking this child). This script also mkpath()s the output's parent
# directory defensively in case it is ever invoked standalone.
#
# No external JSON dependency (JSON3 is not a declared project dependency);
# reuses the hand-rolled minimal JSON reader/writer already established as
# repo convention in tools/core070_postfit_policy_batch.jl / core070_postfit_1_batch.jl.

using GLLVModels

# ---------------------------------------------------------------------------
# Minimal JSON reader/writer (no external dependency; mirrors the existing
# repo convention in tools/core070_postfit_policy_batch.jl).
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
        pos[] += 1
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

# Pin selection: GLLVM_PARITY_PIN via the shared pin source
# (test/parity/core070_pin.jl reads tools/core070_oracle_pins.toml; default P0).
# The contract is the one frozen at the selected pin's reference_commit, so P0
# resolves to the original contract exactly as before.
include(joinpath(@__DIR__, "..", "test", "parity", "core070_pin.jl"))

const CONTRACT_PATHS = [
    joinpath(@__DIR__, "..", "docs/dev-log/core070/namespace-1-batch-contract.json"),
    joinpath(@__DIR__, "..", "docs/dev-log/core070/true-parity-latest/namespace-1-batch-contract-p1.json"),
]

function load_contract()
    for path in CONTRACT_PATHS
        isfile(path) || continue
        contract = json_read(path)
        if get(contract, "reference_commit", nothing) == _CORE070_REFERENCE_COMMIT
            _core070_check_frozen_contract_pin(contract, path)
            return contract
        end
    end
    error("FATAL: no namespace-1 contract frozen at $(_CORE070_REFERENCE_COMMIT) " *
          "(GLLVM_PARITY_PIN=$(repr(get(ENV, "GLLVM_PARITY_PIN", nothing)))) -- refusing to run with no state.")
end

# The one check this tier performs, exposed as a function so --self-test can
# call it directly without touching disk.
symbol_exists(mod::Module, sym::AbstractString) = isdefined(mod, Symbol(sym))

# Measured facts about one symbol (review of #559: `isdefined` alone let an unexported helper or
# a type pass as a twin). `exported` is Base.isexported; `callable` is true only for a Function
# (a type is constructible but is not the R function's twin); `kind` names what the binding is;
# `own_methods` / `first_own_method` count and locate methods defined in GLLVModels itself, so a
# re-exported generic with no GLLVModels method is visible in the receipt.
function symbol_facts(mod::Module, sym::AbstractString)
    s = Symbol(sym)
    exists = isdefined(mod, s)
    d = Dict{String, Any}("exists" => exists)
    exists || return d
    val = getfield(mod, s)
    d["exported"] = Base.isexported(mod, s)
    d["callable"] = val isa Function
    d["kind"] = val isa Function ? "Function" : val isa DataType ? "DataType" :
                val isa UnionAll ? "UnionAll" : val isa Module ? "Module" : string(typeof(val))
    if val isa Function
        own = [m for m in methods(val) if m.module === mod]
        d["own_methods"] = length(own)
        if !isempty(own)
            m = first(sort(own; by = m -> (string(m.file), m.line)))
            file = string(m.file)
            root = normpath(joinpath(@__DIR__, ".."))
            startswith(file, root) && (file = relpath(file, root))
            d["first_own_method"] = "$(file):$(m.line)"
        end
    end
    return d
end

const SYNTHETIC_NEG_SYMBOL = "gllvmTMB_julia_bridge_nonexistent_surface_zzz"

function collect_facts(contract)
    symbols = Set{String}()
    for c in contract["cases"]
        push!(symbols, c["julia_symbol"])
    end
    for c in contract["needs_new_julia_surface"]
        push!(symbols, c["julia_symbol"])
    end
    push!(symbols, SYNTHETIC_NEG_SYMBOL)

    facts = Dict{String, Any}()
    for sym in symbols
        facts[sym] = symbol_facts(GLLVModels, sym)
    end
    return facts
end

function main()
    length(ARGS) == 1 ||
        error("FATAL: expected exactly 1 argv (output path), got $(length(ARGS)): $(ARGS)")
    out_path = ARGS[1]
    isfile(out_path) && error("FATAL: destination already exists: $out_path")
    out_dir = dirname(out_path)
    isempty(out_dir) || mkpath(out_dir)

    contract = load_contract()
    facts = collect_facts(contract)

    report = Dict{String, Any}(
        "schema" => "core070-namespace-1-julia-facts/v1",
        "status" => "OK",
        "julia_version" => string(VERSION),
        "reference_commit" => contract["reference_commit"],
        "symbol_count" => length(facts),
        "facts" => facts,
    )

    open(out_path, "w") do io
        write(io, to_json(report))
    end
    println("CORE070_NAMESPACE_1_JULIA_FACTS_OK symbols=", length(facts))
end

# --self-test exercises collect_facts() against the real contract and the
# real GLLVModels module (no file I/O to a destination), and FAILS LOUDLY (throws,
# nonzero exit) rather than silently reporting a partial/empty result if
# GLLVModels does not load or the contract is missing -- matching the hardened
# template's "verifier FAILS LOUDLY on missing state even under --self-test"
# rule, applied here to the runner side of that same discipline.
function self_test()
    contract = load_contract()
    facts = collect_facts(contract)

    n_expect_true = count(c -> c["expected_julia_symbol_exists"], contract["cases"])
    n_expect_false = count(c -> !c["expected_julia_symbol_exists"], contract["needs_new_julia_surface"])

    mismatches = String[]
    for c in contract["cases"]
        sym = c["julia_symbol"]
        expect = c["expected_julia_symbol_exists"]
        got = get(get(facts, sym, Dict()), "exists", nothing)
        if got !== expect
            push!(mismatches, "case $(c["case_id"]): expected exists=$expect for $sym, got $got")
        end
    end
    for c in contract["needs_new_julia_surface"]
        sym = c["julia_symbol"]
        expect = c["expected_julia_symbol_exists"]
        got = get(get(facts, sym, Dict()), "exists", nothing)
        if got !== expect
            push!(mismatches, "needs-row $(c["case_id"]): expected exists=$expect for $sym, got $got")
        end
    end
    got_neg = get(get(facts, SYNTHETIC_NEG_SYMBOL, Dict()), "exists", nothing)
    if got_neg !== false
        push!(mismatches, "synthetic true-negative symbol resolved exists=$got_neg, expected false")
    end

    if !isempty(mismatches)
        for m in mismatches
            println(stderr, "SELF_TEST_MISMATCH: ", m)
        end
        error("FATAL: self-test found $(length(mismatches)) mismatch(es) -- see stderr. " *
              "n_expect_true=$n_expect_true n_expect_false=$n_expect_false symbols_collected=$(length(facts))")
    end

    println("CORE070_NAMESPACE_1_JULIA_SELF_TEST_OK exec_true=", n_expect_true,
            " needs_false=", n_expect_false, " total_symbols=", length(facts))
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 1 && ARGS[1] == "--self-test"
        self_test()
    else
        main()
    end
end

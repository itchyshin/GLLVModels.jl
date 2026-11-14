# Pure-logic helpers for deterministic Julia-suite sharding
# (GLLVM_TEST_SHARD="k/N"). No package `using` needed — plain Base only.
# Included by runtests.jl, and standalone by test_shard_selection.jl.

"""
    _parse_shard_spec(spec::AbstractString) -> (k::Int, N::Int)

Parse a `"k/N"` shard spec (1-based `k` of `N` total shards). Throws
`ArgumentError` for a malformed spec, a non-positive `N`, or `k` outside
`1:N`.
"""
function _parse_shard_spec(spec::AbstractString)
    parts = split(spec, '/')
    length(parts) == 2 ||
        throw(ArgumentError("malformed GLLVM_TEST_SHARD spec $(repr(spec)); expected \"k/N\""))
    k = tryparse(Int, parts[1])
    N = tryparse(Int, parts[2])
    (k === nothing || N === nothing) &&
        throw(ArgumentError("malformed GLLVM_TEST_SHARD spec $(repr(spec)); k and N must be integers"))
    N >= 1 ||
        throw(ArgumentError("GLLVM_TEST_SHARD spec $(repr(spec)): N must be >= 1, got N=$N"))
    1 <= k <= N ||
        throw(ArgumentError("GLLVM_TEST_SHARD spec $(repr(spec)): k must satisfy 1 <= k <= N, got k=$k, N=$N"))
    return k, N
end

"""
    _shard_indices(n::Integer, k::Integer, N::Integer) -> Vector{Int}

1-based indices into `1:n` assigned to shard `k` of `N`, by
`(i - 1) % N == k - 1`. The `N` shards partition `1:n` exactly: every index
in `1:n` falls in exactly one shard's result.
"""
function _shard_indices(n::Integer, k::Integer, N::Integer)
    n >= 0 || throw(ArgumentError("n must be >= 0, got n=$n"))
    N >= 1 || throw(ArgumentError("N must be >= 1, got N=$N"))
    1 <= k <= N || throw(ArgumentError("k must satisfy 1 <= k <= N, got k=$k, N=$N"))
    return [i for i in 1:n if (i - 1) % N == k - 1]
end

"""
    _auto_test_files(testdir::AbstractString) -> Vector{String}

Sorted file names of every `test_*.jl` directly inside `testdir/auto`
(empty when the directory does not exist). These files are included by
`runtests.jl` without any entry in its include list or shard table.
"""
function _auto_test_files(testdir::AbstractString)
    dir = joinpath(testdir, "auto")
    isdir(dir) || return String[]
    return sort!([f for f in readdir(dir) if startswith(f, "test_") && endswith(f, ".jl")])
end

"""
    _auto_shard_of(name::AbstractString, N::Integer) -> Int

Shard (1-based, of `N`) of an auto-discovered test file, from a stable FNV-1a
hash of its file name. It depends on the name alone: adding or removing another
file never moves it, and it does not change between Julia versions (unlike
`Base.hash`). Used for every `N`.
"""
function _auto_shard_of(name::AbstractString, N::Integer)
    N >= 1 || throw(ArgumentError("N must be >= 1, got N=$N"))
    h = 0x811c9dc5
    for b in codeunits(name)
        h = (h ⊻ b) * 0x01000193
    end
    return Int(h % UInt32(N)) + 1
end

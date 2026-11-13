using Test

include(joinpath(@__DIR__, "..", "shard_util.jl"))

# Every test/test_*.jl is run by the suite or named here as deliberately not run.
# test_quality.jl, test_readme_quickstart.jl and test_quality_jet.jl are run by
# runtests.jl through a plain `include` or by test_quality.jl.
const _TESTDIR = joinpath(@__DIR__, "..")
const _UNWIRED_ON_PURPOSE = Set([
    "test_binomial_case_contract.jl",     # stale SHA pins
    "test_phylo_gamma_xlv.jl",            # optimiser-sensitive canary
    "test_phylo_ordinal_xlv.jl",          # optimiser-sensitive canary
])
const _RUN_BY_OTHER_FILE = Set([
    "test_quality.jl", "test_quality_jet.jl", "test_readme_quickstart.jl",
])

@testset "every test file is assigned or opted out" begin
    src = read(joinpath(_TESTDIR, "runtests.jl"), String)
    # active (non-comment) calls only
    calls = [m.captures[1] for m in eachmatch(
        r"(?m)^[ \t]*(?:_shard_include|_shard_include_if\([^,\n]+,)\(?\s*\"(test_[^\"]+\.jl)\"", src)]
    listed = Set(calls)
    @test length(calls) == length(listed)            # no file listed twice
    files = Set(f for f in readdir(_TESTDIR) if startswith(f, "test_") && endswith(f, ".jl"))
    for f in sort!(collect(files))
        @test (f in listed) || (f in _UNWIRED_ON_PURPOSE) || (f in _RUN_BY_OTHER_FILE)
    end
    for f in listed
        @test f in files                              # no include of a missing file
        @test !(f in _UNWIRED_ON_PURPOSE)             # opt-out list is not stale
    end
    for f in union(_UNWIRED_ON_PURPOSE, _RUN_BY_OTHER_FILE)
        @test f in files
    end
    # auto files live in test/auto, are found, and are not also listed
    for f in _auto_test_files(_TESTDIR)
        @test !(f in listed)
    end
    @test "test_test_file_assignment.jl" in _auto_test_files(_TESTDIR)
end

@testset "auto shard assignment" begin
    names = ["test_a.jl", "test_b.jl", "test_news_fragments.jl", "test_some_long_name_1234.jl"]
    for N in (1, 2, 3, 4, 7), n in names
        @test 1 <= _auto_shard_of(n, N) <= N
        @test _auto_shard_of(n, N) == _auto_shard_of(n, N)
    end
    # pinned values: the hash must not drift (FNV-1a, 32 bit)
    @test _auto_shard_of("test_a.jl", 4) == 4
    @test _auto_shard_of("test_b.jl", 4) == 1
    # every file lands in exactly one of N shards
    for N in (2, 4)
        @test sort(vcat([[n for n in names if _auto_shard_of(n, N) == k] for k in 1:N]...)) == sort(names)
    end
    @test_throws ArgumentError _auto_shard_of("x", 0)
end

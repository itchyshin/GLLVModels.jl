# Lane cursor-cloud-1125y8: grouped_profile_foundation.jl validation branches.
using Test, GLLVModels, LinearAlgebra, SparseArrays, Random

const G = GLLVModels

function _gp_incidence(values)
    G._grouped_incidence(values, length(values))
end

@testset "cov 1125y8 grouped profile validate" begin
  n = 8
  unit = repeat(1:4; inner = 2)
  terms = [GroupingTerm(:unit; mode = :indep, common = false)]
  inc = [_gp_incidence(unit)]
  D = G._trait_mean_design(2, n)

  @testset "_grouped_indep_profile_validate_terms" begin
    @test G._grouped_indep_profile_validate_terms(terms, inc) == n
    @test_throws ArgumentError G._grouped_indep_profile_validate_terms(terms, inc[1:0])
    bad = [GroupingTerm(:unit; mode = :indep, common = true)]
    @test_throws ArgumentError G._grouped_indep_profile_validate_terms(bad, inc)
    dup = [GroupingTerm(:unit; mode = :indep, common = false), GroupingTerm(:unit; mode = :indep, common = false)]
    @test_throws ArgumentError G._grouped_indep_profile_validate_terms(dup, [inc[1], inc[1]])
    bad_inc = sparse(ones(3, 2))
    @test_throws Exception G._grouped_indep_profile_validate_terms(terms, [bad_inc])
    zero_row = sparse([1.0 0.0; 0.0 0.0; 0.0 1.0])
    @test_throws ArgumentError G._grouped_indep_profile_validate_terms(terms, [zero_row])
  end

  @testset "_grouped_indep_profile_design_svd" begin
    svd = G._grouped_indep_profile_design_svd(D)
    @test svd.rank == size(D, 2)
    @test all(>=(0), svd.singular_values)
    @test_throws ArgumentError G._grouped_indep_profile_design_svd(fill(Inf, 4, 2))
    @test_throws ArgumentError G._grouped_indep_profile_design_svd(zeros(4, 0))
  end

  @testset "_grouped_indep_variance_basis_gram valid fixture" begin
    gram = G._grouped_indep_variance_basis_gram(D, terms, inc)
    @test gram.valid
    @test gram.rank == length(gram.labels)
    @test length(gram.labels) == length(terms) * 2 + 1
    @test all(>(0), gram.diagonal)
  end

  @testset "_grouped_indep_profile_expand" begin
    data = randn(2, n)
    layout = G._grouped_indep_profile_layout(data, D, terms, inc, (1, 1), 0.2, 0.0)
    full = G._grouped_indep_profile_expand(layout, zeros(layout.reduced_length))
    @test full !== nothing
    @test length(full) == layout.full_length
    @test full[layout.selected_full_index] == 0.0
    @test G._grouped_indep_profile_expand(layout, zeros(layout.reduced_length - 1)) === nothing
  end
end
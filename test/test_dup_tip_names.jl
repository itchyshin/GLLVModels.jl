# P61 DUP-TIP-NAMES (refs #1165 item 3): a Newick string with two tips of the same name
# is an error in `augmented_phy` and `edge_phy`, and the message names the duplicates.
# Before, both parsers accepted it and any name-keyed lookup merged the two tips.
using Test, GLLVModels

@testset "duplicate tip names are an error (augmented_phy, edge_phy)" begin
    msg(f) = try; f(); ""; catch e; sprint(showerror, e); end
    dup  = "((t1:1,t1:1):1,(t3:1,t4:1):1);"
    dup2 = "((a:1,b:1):1,(a:1,b:1):1);"
    for (label, f) in (("augmented_phy", augmented_phy), ("edge_phy", edge_phy))
        @testset "$label" begin
            m = msg(() -> f(dup))
            @test occursin("tip names must be unique", m)
            @test occursin("t1", m) && !occursin("t3", m)
            m2 = msg(() -> f(dup2))
            @test occursin("tip names must be unique", m2)
            @test occursin("duplicated: a, b", m2)
        end
    end
    @testset "valid trees are unchanged" begin
        ok = "((A:0.1,B:0.2):0.3,C:0.5);"
        @test augmented_phy(ok).n_leaves == 3
        @test edge_phy(ok) isa GLLVModels.EdgePhy
        # unnamed tips are not flagged
        @test augmented_phy("((:1,:1):1,(:1,:1):1);").n_leaves == 4
        @test edge_phy("((:1,:1):1,(:1,:1):1);") isa GLLVModels.EdgePhy
    end
end

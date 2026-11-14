using Test

# Format of news/*.md fragments (see news/README.md).
const _NEWS_DIR = joinpath(@__DIR__, "..", "..", "news")

@testset "news fragments" begin
    @test isdir(_NEWS_DIR)
    @test isfile(joinpath(_NEWS_DIR, "README.md"))
    for f in sort(readdir(_NEWS_DIR))
        f == "README.md" && continue
        @testset "$f" begin
            @test occursin(r"^[a-z0-9][a-z0-9-]*\.md$", f)
            txt = read(joinpath(_NEWS_DIR, f), String)
            @test endswith(txt, "\n") && !endswith(txt, "\n\n")
            lines = split(chopsuffix(txt, "\n"), "\n")
            @test startswith(lines[1], "- **")
            for l in lines
                @test !occursin('\t', l)
                @test l == rstrip(l)
                @test isempty(l) || startswith(l, "  ") || startswith(l, "- ")
                @test !startswith(l, "#")
            end
        end
    end
end

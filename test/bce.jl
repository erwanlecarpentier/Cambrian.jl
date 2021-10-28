using YAML
using Statistics
using Cambrian
using Test
using Clustering
# import Cambrian.selection
# import Cambrian.mutate

"""
    Ind

Example individual class using a floating point genotype in [0, 1], a float
fitness vector and a float behavior vector.
"""
struct Ind <: Cambrian.Individual
    genes::Vector{Float64}
    fitness::Vector{Float64}
    behavior::Vector{Float64}
end

function Ind(cfg::NamedTuple)
    Ind(rand(cfg.n_genes), -Inf*ones(cfg.d_fitness), zeros(cfg.d_behavior))
end

function cluster(e::BCEEvo)
    X = reduce(hcat, [e.population[i].behavior for i in eachindex(e.population)])
    R = kmeans(X, e.cfg.n_emitters; maxiter=200, display=:iter)
    # TODO complete from here
    # Find best representative for each category
    [population[1]] # TODO return best representatives
end

cfgpath = joinpath(dirname(@__DIR__), "cfg", "bce.yaml")
cfg = get_config(cfgpath)

fitness(ind::Ind) = [(cos(2.0*π*maximum(ind.genes))+1.0)/2.0]
behavior(ind::Ind) = [mean(ind.genes)]

function test_bce_evo(e::BCEEvo)
    @test all([e.population[i].fitness[1] for i in eachindex(e.population)] .> -Inf)
    @test e.fmax > -Inf
end

@testset "BCE Evolution" begin
    e = BCEEvo{Ind}(cfg, fitness, behavior, cluster)
    @test length(e.population[1].genes) == cfg.n_genes
    test_bce_evo(e)
end



@testset "BCE Evolution with custom initialization function" begin
    initgenes = [0.0, 0.0]
    init_function(cfg::NamedTuple) = Ind(initgenes, [-Inf], [0.0])
    e = BCEEvo{Ind}(cfg, fitness, behavior, cluster, init_function=init_function)
    @test length(e.population[1].genes) == length(initgenes)
    test_bce_evo(e)
end

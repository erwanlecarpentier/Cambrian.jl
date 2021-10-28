export BCEEvo, explore!

"""
Behavior Clustering and Emitters (BCE) implementation (TODO: find a good name).
A new subtype of AbstractEvolution is created, BCEEvo, and the following
functions are defined for this type:
- `populate`
- `evaluate`
"""

"""
    BCEEvo

BCE evolution object.
Evaluation fitness is only implemented for one unique criterion.

Arguments:
- `cfg::NamedTuple`
- `logger::CambrianLogger`
- `population::Vector{T}`: population of individuals
- `cluster_representatives::Vector{T}`: individuals representing each cluster of the behavior space
- `fitness::Function`: map an individual's parameters to its fitness
- `behavior::Function`: map an individual's parameters to its behavior representation
- `cluster::Function`: map the population to a sub-set of representative individuals
- `fmax::Float64`: maximum reached fitness
- `gen::Int64`: generation number
"""
mutable struct BCEEvo{T} <: AbstractEvolution
    cfg::NamedTuple
    logger::CambrianLogger
    population::Vector{T}
    cluster_representatives::Vector{T}
    fitness::Function
    behavior::Function
    cluster::Function
    fmax::Float64
    gen::Int64
end

function BCEEvo{T}(
    cfg::NamedTuple,
    fitness::Function,
    behavior::Function,
    cluster::Function;
    logfile=string("logs/", cfg.id, ".csv"),
    kwargs...
) where T
    logger = CambrianLogger(logfile)
    kwargs_dict = Dict(kwargs)
    if haskey(kwargs_dict, :init_function)
        population = initialize(T, cfg, init_function=kwargs_dict[:init_function])
    else
        population = initialize(T, cfg)
    end
    cluster_representatives = Vector{T}()
    e = BCEEvo{T}(
        cfg, logger, population, cluster_representatives, fitness, behavior,
        cluster, -Inf, 0
    )
    # Evaluate initial population to deduce fmax
    evaluate_fitness!(e)
    e.fmax = maximum([population[i].fitness[1] for i in eachindex(population)])
    e
end

function evaluate_fitness!(e::BCEEvo)
    for i in eachindex(e.population)
        e.population[i].fitness[:] = e.fitness(e.population[i])[:]
    end
end

function evaluate_behavior!(e::BCEEvo)
    for i in eachindex(e.population)
        e.population[i].behavior[:] = e.behavior(e.population[i])[:]
    end
end

function explore!(e::BCEEvo)
    evaluate_behavior!(e)
    e.cluster_representatives = e.cluster(e)
    while length(e.cluster_representatives) < e.cfg.n_emitters
        B = reduce(hcat, [e.population[i].behavior for i in eachindex(e.population)])
        most_novel_index = argmax(mean(pairwise(Euclidean(), B, dims=2), dims=1))[2]
        for i in 1:e.cfg.lambda_explore
            push!(e.population, mutate(e.population[most_novel_index]))
        end
        e.cluster_representatives = e.cluster(e)
    end
end

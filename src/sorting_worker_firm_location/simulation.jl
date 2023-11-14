#==========================================================================================
Title: Simulation
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2023-10-28
Description: Simulation of the model
==========================================================================================#

#! I start by assuming that the model has been solved and the solution is stored 
#! in the structure "res" while the parameters are stored in the structure "prim"
#! Equilibrium distributions are stored in the structure "dist"

n_agents = 100000

n_cities = prim.n_j

# Create a Worker structure

@everywhere mutable struct Worker
    id          ::  String              # Unique identifier #! Inmutable
    x           ::  Int64               # Worker type (index) #! Inmutable
    employers   ::  Array{Int64}        # History of employers (index) #* Mutable
    locations   ::  Array{Int64}        # History of locations #* Mutable
    #! Wages are not calculated in this version of the model
    #! Adding this for the future
    # w       ::  Array{Float64}    # History of wages #* Mutable
end # Worker

# Using equilibrium distributions generate cities (as collection of workers)


# Compute a vector of city sizes to use as the probability distribution
@unpack ℓ, u, h = dist

μ_s = sum(ℓ, dims=2)[:]
# Compute aggrate distribution of worker types
ℓ_agg = sum(ℓ, dims=1)[:]
# Compute unemployment rates by worker type (for each city)
u_rates = u ./ ℓ
# Compute employment probabilities by worker type (for each city)
h_rates = h ./ sum(h, dims=3)
sum(h_rates, dims=1)

# Create a vector of workers
workers = Vector{Worker}(undef, n_agents)

for agent ∈ 1:n_agents
    # Draw a city
    city = rand(Categorical(μ_s))
    # Draw a worker type
    x = rand(Categorical(ℓ_agg))
    # Draw an employment status
    emp = (u_rates[city, x] < rand()) ? 1 : 0
    # If employed draw an employer
    if emp == 1
        employer = rand(Categorical(h_rates[city, x, :]))
    else
        employer = -1
    end
    # Create a unique identifier (String of lenght decimals in n_agents leters and numbers)
    id_length = length(string(n_agents))
    chars = ['A':'Z'; '0':'9']
    id = join(rand(chars, id_length))# Create a worker
    w = Worker(
        id,             # Unique identifier
        x,              # Worker type (inmutable)
        [employer],     # History of employers (mutable)
        [city]          # History of locations (mutable)
    )
    # Store the worker in the vector of workers
    workers[agent] = w
end

# Define a function to convert the vector of workers into a DataFrame
# function workers_to_df(workers)
    # Unpack
    @unpack y_grid, x_grid = prim

    # Get workers history of employers
    employers = [w.employers for w ∈ workers]
    # Get length of  histories
    periods = length.(employers)
    employers = vcat(employers...)
    employers = [(employer == -1) ? -1 : y_grid[employer] for employer ∈ employers]
    # Get history of locations
    locations = vcat([w.locations for w ∈ workers]...)
    # Repeat the worker id for each period
    ids = vcat([repeat([workers[i].id], periods[i]) for i ∈ 1:n_agents]...)
    # Repeat the worker type for each period
    types = vcat([repeat([ x_grid[workers[i].x] ], periods[i]) for i ∈ 1:n_agents]...)
    # Create a DataFrame
    df = DataFrame(
        id = ids,
        x = types,
        employer = employers,
        location = locations
    )
    
# end

using StatsPlots

@df df scatter(:x, :employer, group = :location)
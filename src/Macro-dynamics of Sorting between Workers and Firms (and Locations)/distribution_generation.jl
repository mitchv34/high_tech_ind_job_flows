#==========================================================================================
Title: Macro-dynamics of Sorting between Workers and Firms (and Locations)
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2023-09-06
Description: This file contains the code to generate the distributions used in the paper.
==========================================================================================#
#==========================================================================================
# * Packages 
==========================================================================================#
using Distributions
include("model.jl")
#?=========================================================================================
#? Functions
#?=========================================================================================
#==========================================================================================
# generate_component_parameters: Generates the parameters for each component
    of the mixture distribution. 
    In the model this represent different different skill distributions on each location.
==========================================================================================#
function generate_component_parameters(original_α::Float64, original_β::Float64, num_components::Int64)
    component_params = []
    while length(component_params) < num_components
        # Generate variations of the original parameters by adding random numbers
        α_i = original_α + rand() * 4.0 - 2.0  # Random number between -2 and 2
        # Check that the parameters are positive
        β_i = original_β + rand() * 12.0 - 6.0  # Random number between -2 and 2
        if α_i < 1.0 || β_i < 0.0
            continue
        end
        push!(component_params, (α_i, β_i))
    end
    return component_params
end
#==========================================================================================
# generate_decreasing_weights: Generates the weights for each component of the mixture
    distribution. The weights are decreasing in the order of the components.
    This will be used to generate the mixture distribution. 
    In the model this represent different initial sizes of locations.
==========================================================================================#
function generate_decreasing_weights(num_components::Int64)
    weights = [1.0 / i for i in 1:num_components]
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    return weights
end
#==========================================================================================
# split_beta: Generates a mixture distribution of beta distributions. 
    The mixture distribution is a weighted average of beta distributions. 
    The weights are decreasing in the order of the components.
    This will be used to generate the mixture distribution. 
    In the model this represent different initial sizes and skill distributions of locations.
==========================================================================================#
# ! For now I'm just using weights and same parammeters for all components
function split_skill_dist(prim::Primitives, unemployment_rate::Float64 = 0.1)
    # Generate the component parameters
    @unpack x_grid, dist_params, dist_name = prim
    α, β = dist_params
    num_components = prim.n_j
    component_params = generate_component_parameters(α, β, num_components)
    # Generate the weights
    weights = generate_decreasing_weights(num_components)
    # Check that the weights and component parameters have the same length
    if length(weights) != num_components || length(component_params) != num_components
        throw(ArgumentError("Length of weights and component_params must match num_components."))
    end

    # Initialize an empty array to store the component distributions
    location_dists = Vector{Beta}()
    
    for i in 1:num_components
        α_i, β_i = component_params[i]
        println("α = $α_i, β = $β_i")
        push!(location_dists, Beta(α_i, β_i))
        # push!(location_dists, Beta(α, β))
    end
    
    # Verify that the weights sum to 1
    if !(sum(weights) ≈ 1.0)
        throw(ArgumentError("Weights must sum to 1.0."))
    end
    
    # Compute the pdfs of each component over the grid of workers
    component_pdfs = [pdf.(dist, x_grid) for dist in location_dists]

    # Compute the pdf of the original distribution
    original_pdf = pdf.(Beta(α, β), x_grid)

    # Normalize all pdfs to have total mass 1
    component_pdfs = [pdf / sum(pdf) for pdf in component_pdfs]
    # Weight each component by its weight
    component_pdfs = [pdf .* weights[i] for (i, pdf) in enumerate(component_pdfs)]
    
    ℓ = Matrix(hcat(component_pdfs...)')
    # Aggregate all components into a single pdf
    ℓ_total = sum(ℓ, dims=1)[:]

    # Start with 10% unemployment rate in each location and skill and workers equally distributed across firms
    u = ℓ .* unemployment_rate # Initial unemployment for each skill level in each location
    # Distribution of workers across firms
    e = (1 - 0.1) .* ℓ # Distribution of employed workers
    h = zeros(prim.n_j, prim.n_x, prim.n_y) # Distribution of employed workers
    for j ∈ 1 : prim.n_j
        h[j, :, :] .= e[j, :] ./ prim.n_y
    end
    # Construct a distributions model with the mixture distribution
    dist = DistributionsModel(ℓ_total, ℓ, u, h);
    return dist
end


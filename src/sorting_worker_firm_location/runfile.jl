using Distributed
using YAML
using StatsBase
using BSON
# Load plot settings

# include("../plotconfig.jl")
@everywhere include("model.jl");
@everywhere include("distribution_generation.jl")
include("plots.jl")
path_params = "src/Sorting between Workers and Firms (and Locations)/parameters/params.yml";
figures_dir = "/figures/Sorting between Workers and Firms (and Locations)/"
results_dir = "./results/Sorting between Workers and Firms (and Locations)/"

# Load a solution to the model
results = BSON.load("/project/high_tech_ind/high_tech_ind_job_flows/results/sorting_worker_firm_location/results_5.bson")
prim, res, dist = results["prim"], results["res"], results["dist"]


using Plots

labor_alloc = dropdims(sum(dist.h, dims = 2), dims = 2)
labor_alloc = labor_alloc ./ sum(labor_alloc, dims = 2)

Plots.plot(cumsum(labor_alloc, dims=2)')


@unpack ω₁, ω₂, n_j, n_x, n_y, s, δ = prim
@unpack L, γ = res
    
new_h = zeros(n_j, n_x, n_y)
new_u = zeros(n_j, n_x)
h_u = zeros(n_j, n_x, n_y)
h_p = zeros(n_j, n_x, n_y)
h_r = zeros(n_j, n_x, n_y)
# Keep track of workers movement across locations (firms)
u_move = zeros(n_j, n_j, n_y) # Move j → j' (0 → y') unemployed 
h_move = zeros(n_j, n_j, n_y, n_y) # Move j → j' (y → y') employed
h_from = zeros(n_j, n_j, n_y, 1 + n_y) # Move j' → j (y' → y) y = 1 means unemployed
for j ∈ 1 : n_j, x ∈ 1 : n_x 
    # Split the mass of unemployed into their destinations
    splited_mass_u = dist.u_plus[j, x] .* res.ϕ_u[j, :, x] .* (res.S_move[j, :, x, :] .> 0) .* γ
    # Assign the mass to their destinations
    h_u[:, x, :] += splited_mass_u
    # Keep of unemployment movement
    u_move[j, :, :] += splited_mass_u
    # Remove the mass from the unemployed
    new_u[j, x] = dist.u_plus[j, x] - sum(splited_mass_u)
    h_from[:, j, :, 1] += splited_mass_u 
    for y ∈ 1 : n_y
        # Split the mass of employed into their destinations
        splited_mass_s = dist.h_plus[j, x, y] .* res.ϕ_s[j, :, x, y, :] .* (res.S_move[j, :, x, :] .- res.S_move[j, j, x, y] .> 0) .* γ
        # Keep track of employment movement
        h_move[j, :, y, :] += splited_mass_s
        # Assign the mass to their destinations
        h_p[:, x, :] += splited_mass_s
        # Remove the mass from the employed
        h_r[j, x, y] = dist.h_plus[j, x, y] - sum(splited_mass_s)
        h_from[:, j, :, 1 + y] += splited_mass_s
    end
end

# Compute the total hires for each firm
total_hires = dropdims(sum(h_p + h_u, dims = 2), dims = 2)
# Migration from 1 
h_from_1_poach = dropdims(sum(h_from[1, :, :, 2:end], dims = 3), dims = 3)
h_from_1_unemp = h_from[1, :, :, 1]
# Migration from 2
h_from_2_poach = dropdims(sum(h_from[2, :, :, 2:end], dims = 3), dims = 3)
h_from_2_unemp = h_from[2, :, :, 1]

Plots.plot(h_from_1')
Plots.plot(h_from_2')

Plots.plot(h_from_1_poach[1, :] ./ h_from_2_poach[1, :])
Plots.plot(h_from_1_poach[2, :] ./ h_from_2_poach[2, :])

Plots.plot(h_from_1_unemp[1, :] ./ h_from_2_unemp[1, :])
Plots.plot(h_from_1_unemp[2, :] ./ h_from_2_unemp[2, :])


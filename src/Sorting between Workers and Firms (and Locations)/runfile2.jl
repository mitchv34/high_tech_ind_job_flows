using Distributed
using DataFrames
using CSV
using YAML
using BenchmarkTools
using StatsBase
# using StatsPlots
using FixedEffectModels
using RegressionTables
using LaTeXStrings
# Load plot settings

include("../plotconfig.jl")
@everywhere include("model.jl");
@everywhere include("distribution_generation.jl")
include("plots.jl")
path_params = "src/Sorting between Workers and Firms (and Locations)/parameters/params.yml";
figures_dir = "/figures/Sorting between Workers and Firms (and Locations)/"

# n_z = 5
# addprocs(n_z) 
# @everywhere zs = range(0.8, 1.2, length = 5)
# @everywhere path_params = "src/Sorting between Workers and Firms (and Locations)/parameters/params.yml";
# @everywhere include("model.jl");
# @everywhere include("distribution_generation.jl")
# @everywhere prim, res = init_model(path_params);
# # Create SharedArrays arrays to store results of the model
# @everywhere dists_model = []# Array{DistributionsModel};
# @everywhere results_model = [] #Array{Results};
# @everywhere primitives_model = [] #Array{Primitives};

# results = pmap(z -> begin
#     μ = 0.8
#     println(@bold @blue "Solving with aggregate productivity z = $z")
#     prim, res = init_model(path_params, z)
#     dist = split_skill_dist(prim, weights=[μ, 1 - μ], verbose=false)
#     iterate_distributions!(prim, res, dist, verbose=false, max_iter=200)
#     sizes = round.(sum(dist.ℓ, dims=2), digits=4)[:]
#     println(@bold @green "z = $z ⟹ μ = $sizes")
#     # Store model results
#     return prim, res, dist
# end, zs)



# i = 1
# figs1 = plot_all_model( results[i][1],  results[i][2],  results[i][3]);
# figs1.poaching_rank + figs1.poaching_rank

# figs.dist_types
# figs.dist_unemp
# figs.dist_match
# figs.S_surv
# figs.vacancies
# figs.eq_vacancies

# @sync @distributed for j = eachindex(μ_s)
    # z = 1.1
#     μ = μ_s[j]
#     println(@bold @blue "Solving with initial condition μ₀ = $μ")
#     @everywhere prim, res = init_model(path_params)
#     dist = split_skill_dist(prim, weights=[μ, 1 - μ], verbose=false)
#     iterate_distributions!(prim, res, dist, verbose=false, max_iter=10000)
#     sizes = round.(sum(dist.ℓ, dims=2), digits=4)[:]
#     println(@bold @green "μ₀ = $([round(μ, digits=4), round(1-μ, digits=4)]) ⟹ μ = $sizes")
#     # Store model results
#     results_u[j, :, :] = copy(dist.u)
#     results_h[j, :, :, :] = copy(dist.h)
# end

# p1 = plot(title="City 1 (Unemployment) Different Initial Conditions", xlabel = "Worker Type")
# for j = eachindex(μ_s)
#     μ = μ_s[j]
#     if μ == 0.5
#         plot!(prim.x_grid, results_u[j, 1,:], lw = 5, c = 3, alpha = 0.7, label = latexstring("\\mu_0 = $μ"))
#     elseif μ < 0.5
#         plot!(prim.x_grid, results_u[j, 1,:], lw = 2, c = 1, alpha = 0.7, label = latexstring("\\mu_0 = $μ"))
#     else
#         plot!(prim.x_grid, results_u[j, 1,:], lw = 2, c = 5, alpha = 0.7, label = latexstring("\\mu_0 = $μ"))
#     end
# end
# p1

# p2 = plot(title="City 1 Different Initial Conditions", xlabel = "Worker Type")
# for j = eachindex(μ_s)
#     μ = μ_s[j]
#     ℓ =  results_u[j, 1,:] + dropdims(sum(results_h[j, 1,:, :], dims = 2), dims = 2)
#     if μ == 0.5
#         plot!(prim.x_grid, ℓ, lw = 5, c = 3, alpha = 0.7, label = latexstring("\\mu_0 = $μ"))
#     elseif μ < 0.5
#         plot!(prim.x_grid, ℓ, lw = 2, c = 1, alpha = 0.7, label = latexstring("\\mu_0 = $μ"))
#     else
#         plot!(prim.x_grid, ℓ, lw = 2, c = 5, alpha = 0.7, label = latexstring("\\mu_0 = $μ"))
        
#     end
# end
# p2

# μ = 0.8
# w = [μ ,  (1 - μ)]# .* μ , (1 - μ) .* (1 - μ)]

# dist.u = results_u[8, :, :]
# dist.h = results_h[8, :, :, :]
# dist.ℓ = dist.u .+ dropdims(sum(dist.h, dims = 3), dims = 3)
prim, res = init_model(path_params);
dist = split_skill_dist(prim, verbose=true)

#! Light testing only (tolerance should be lower)
μs, errs, urates = iterate_distributions!(prim, res, dist, verbose = true,  tol=1e-6)



figs = plot_all_model( prim,  res,  dist, cmap = :RdBu);

figs.agg_dist
figs.dist_types
figs.dist_unemp
figs.S_surv
figs.dist_match
figs.search_unemp
figs.search_emp
figs.vacancies
figs.eq_vacancies
figs.poaching_rank

# Save figures
save_all_figures(figs)


# # Unemployment hires
# #! Modify to be rates
# Plots.plot(res.u_move[1, :, :]' ./ sum(res.u_move[1, :, :]), lw = 2)
# Plots.plot!(res.u_move[2, :, :]' ./ sum(res.u_move[2, :, :]) , lw = 2)

# res.u_move[1, 1, :] ./ sum(res.u_move[1, :, :])

# # Hires from poaching
# Plots.plot(sum(res.h_move[1, 1, :, :], dims = 1)', lw = 2)
# Plots.plot!(sum(res.h_move[2, 1, :, :], dims = 1)', lw = 2)
# Plots.plot!(sum(res.h_move[1, 2, :, :], dims = 1)', lw = 2)
# Plots.plot!(sum(res.h_move[2, 2, :, :], dims = 1)', lw = 2)

# Plots.plot(sum(res.h_move[1, 1, :, :] .+ res.h_move[2, 1, :, :], dims = 1)', lw = 2)
# Plots.plot!(sum(res.h_move[1, 2, :, :], dims = 1)', lw = 2)
# Plots.plot!(sum(res.h_move[2, 2, :, :], dims = 1)', lw = 2)

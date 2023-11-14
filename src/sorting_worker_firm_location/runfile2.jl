using CairoMakie: linewidth
using Distributed
using SharedArrays
using YAML
using StatsBase

# include("../plotconfig.jl")
@everywhere include("model.jl");
@everywhere include("distribution_generation.jl")
include("plots.jl")
path_params = "src/sorting_worker_firm_location/parameters/params.yml";
figures_dir = "figures/sorting_worker_firm_location/"
results_dir = "./results/sorting_worker_firm_location/"

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

num_μ = 20
addprocs(num_μ)
@everywhere num_μ = $num_μ

@everywhere μ_s = range(0.0, 1.0, length = num_μ)
@everywhere path_params = $path_params
@everywhere include("model.jl");
@everywhere include("distribution_generation.jl")

@everywhere prim, res = init_model(path_params);

@everywhere results_u = SharedArray{Float64}(length(μ_s), 2, prim.n_x)
@everywhere results_h = SharedArray{Float64}(length(μ_s), 2, prim.n_x, prim.n_y)
@everywhere results_ℓ = SharedArray{Float64}(length(μ_s), 2, prim.n_x)

@sync @distributed for j = eachindex(μ_s)
    z = 1.1
    μ = μ_s[j]
    println(@bold @blue "Solving with initial condition μ₀ = $μ")
    @everywhere prim, res = init_model(path_params)
    dist = split_skill_dist(prim, weights=[μ, 1 - μ], verbose=false)
    iterate_distributions!(prim, res, dist, verbose=false, max_iter=100000)
    sizes = round.(sum(dist.ℓ, dims=2), digits=4)[:]
    println(@bold @green "μ₀ = $([round(μ, digits=4), round(1-μ, digits=4)]) ⟹ μ = $sizes")
    # Store model results
    results_u[j, :, :] = copy(dist.u)
    results_h[j, :, :, :] = copy(dist.h)
    results_ℓ[j, :, :] = copy(dist.ℓ)
end

# Solve the model with μ = 0.5 initial condition
@everywhere prim, res = init_model(path_params)
dist = split_skill_dist(prim, weights=[0.5, 0.5], verbose=false)
iterate_distributions!(prim, res, dist, verbose=false, max_iter=100000)

# Add it to the results
results_u = cat(results_u, reshape(dist.u, 1, 2, prim.n_x), dims = 1)
results_h = cat(results_h, reshape(dist.h, 1, 2, prim.n_x, prim.n_y), dims = 1)
results_ℓ = cat(results_ℓ, reshape(dist.ℓ, 1, 2, prim.n_x), dims = 1)

# Add the initial condition to the μ_s
μ_s = cat(μ_s, 0.5, dims = 1)

function plot_eqlibria(μ_s, results)

    fig = Figure(resolution = (600, 400),
                backgroundcolor = "#FAFAFA",
                fonts = (; regular= texfont(), bold = regular= texfont()))

    ax = Axis(fig[1,1], xlabel = "Worker Type", ylabel = "", title = "",  backgroundcolor = "#FAFAFA",
            ylabelsize = 18, xticks = 0:0.25:1,  titlesize = 22,
            xlabelsize = 18, xgridstyle = :dash, ygridstyle = :dash, xtickalign = 1,
            xticksize = 10, ytickalign = 1, yticksize = 10, xlabelpadding = 0)
    hidespines!(ax, :t, :r)

    labels = [true for i in 1:3]
    for j ∈ eachindex(μ_s)
        println(j)
        μ = μ_s[j]
        if μ < 0.5
            if labels[1]
                lines!(prim.x_grid, results[j, 1, :], linewidth = 2, color = "#D32F2F", label = "μ₀ ∈ [0, 0.5)")
                labels[1] = false
            else
                lines!(prim.x_grid, results[j, 1, :], linewidth = 2, color = "#D32F2F", alpha = 0.5)
            end
        elseif μ > 0.5
            if labels[2]
                lines!(prim.x_grid, results[j, 1, :], linewidth = 2, color = "#3F51B5", label = "μ₀ ∈ (0.5, 1]")
                labels[2] = false
            else
                lines!(prim.x_grid, results[j, 1, :], linewidth = 2, color = "#3F51B5", alpha = 0.5)
            end
        else 
            lines!(prim.x_grid, results[j, 1, :], linewidth = 4, color = "#000000", label = "μ₀ = 0.5", linestyle = :dash)
        end
    end

    axislegend(""; position = :rt, bgcolor = (:grey90, 0.25));
    return fig
end 

equilibria_ell = plot_eqlibria(μ_s, results_ℓ)
equilibria_u = plot_eqlibria(μ_s, results_u)
equilibria_e = plot_eqlibria(μ_s, sum(results_h, dims = 4)[:, :, :, 1])

# Save the figures

# Save the figures
save(joinpath(figures_dir, "equilibria_ell.pdf"), equilibria_ell)
save(joinpath(figures_dir, "equilibria_u.pdf"), equilibria_u)
save(joinpath(figures_dir, "equilibria_e.pdf"), equilibria_e)

# Solve a particular equilibrium

dist.u = results_u[8, :, :]
dist.h = results_h[8, :, :, :]
dist.ℓ = dist.u .+ dropdims(sum(dist.h, dims = 3), dims = 3)


prim, res = init_model(path_params);
dist = split_skill_dist(prim, verbose=true);

#! Light testing only (tolerance should be lower)
μs, errs, urates = iterate_distributions!(prim, res, dist, verbose = true,  tol=1e-2)

figs = plot_all_model( prim,  res,  dist, cmap = :managua, backgroundcolor =  "#FAFAFA")

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


xx = sum((max.( res.S_move[1,1,:,:] , 0) .* res.ϕ_u[1,1,:] .*dist.u_plus[1,:] ./ res.L[1]) + (max.( res.S_move[2,1,:,:] , 0) .* res.ϕ_u[2,1,:] .* dist.u_plus[2,:] ./ res.L[2]), dims = 1)

lines(xx[:])


ℓ_share = dist.ℓ ./ dist.ℓ_total'
u_share = dist.u ./ sum(dist.u, dims = 1)


lines(prim.x_grid, ℓ_share[1, :])
lines(prim.x_grid, u_share[1, :])



# mig = zeros(prim.n_j, prim.n_x)
# for j ∈ 1:prim.n_j
#     for j_prime ∈ 1:prim.n_j
#         if j == j_prime
#             continue
#         end
#     unemp_mig = res.ϕ_u[j, j_dest, :] 

    

# Unemplopyed workers

flow_form_1_u = sum( (res.S_move[1, 2, :, :] .> 0) .* res.ϕ_u[1, 2 , :] .* res.γ[2, :]' , dims = 2) .* dist.u_plus[1, :]
flow_form_2_u = sum( (res.S_move[2, 1, :, :] .> 0) .* res.ϕ_u[2, 1 , :] .* res.γ[1, :]' , dims = 2) .* dist.u_plus[2, :]

# Employed workers
flow_form_1_s = zeros(prim.n_x)
flow_form_2_s = zeros(prim.n_x)
for y ∈ 1:prim.n_y
    flow_form_1_s += sum( (res.S_move[1, 2, :, :] .- res.S_move[1, 1, :, y] .> 0) .* res.ϕ_s[1, 2 , :, y] .* res.γ[2, :]' , dims = 2) .* dist.h_plus[1, :, y]
    flow_form_2_s += sum( (res.S_move[2, 1, :, :] .- res.S_move[2, 2, :, y] .> 0) .* res.ϕ_s[2, 1 , :, y] .* res.γ[1, :]' , dims = 2) .* dist.h_plus[2, :, y]
end


lines( prim.x_grid, flow_form_1_u[:] ./ dist.u_plus[1, :], linewidth = 4)
lines(  prim.x_grid,flow_form_2_u[:] ./ dist.u_plus[2, :], linewidth = 4)


lines( prim.x_grid,  flow_form_1_s[:] ./ sum(dist.h_plus[1,:,:], dims = 2)[:], linewidth = 4)
lines( prim.x_grid, (flow_form_1_s[:] + flow_form_1_u[:]) ./ dist.ℓ[1, :], linewidth = 4)

lines( prim.x_grid, low_form_2_s[:] ./ sum(dist.h_plus[2,:,:], dims = 2)[:], linewidth = 4)
lines( prim.x_grid, (flow_form_2_s[:] + flow_form_2_u[:]) ./ dist.ℓ[2, :], linewidth = 4)

sum(flow_form_1_u) + sum(flow_form_1_s)
sum(flow_form_2_u) + sum(flow_form_2_s)

res.ϕ_s # [1, :, :]

res.S_move[1, :, :, :] .- res.S_move[1, 1, :, :]

# Employed workers
flow_form_1_h 



# Normalize outflows of unemployed wokres that move away from the initial location
plot(flow_form_1_u[2, :] ./ dist.u_plus[1, :])

plot(flow_form_2_u[2, :] ./ dist.u_plus[2, :])




res.S_move[1, :, :, :]

lines(prim.x_grid, flow_form_1[1, :] , linewidth = 2, color = "#D32F2F")
lines(prim.x_grid, flow_form_1[2, :] , linewidth = 2, color = "#3F51B5")

lines(prim.x_grid, flow_form_2[1, :] , linewidth = 2, color = "#D32F2F")
lines(prim.x_grid, flow_form_2[2, :] , linewidth = 2, color = "#3F51B5")

lines(prim.x_grid, flow_form_1[1, :] + flow_form_2[1, :] , linewidth = 2, color = "#D32F2F")
lines(prim.x_grid, flow_form_1[2, :] + flow_form_2[2, :] , linewidth = 2, color = "#3F51B5")

lines(prim.x_grid, dist.u_plus[1,:])
lines(prim.x_grid, dist.u_plus[2,:])

# Save figures
save_all_figures(figs)

using JLD2

# Save the variables
save( results_dir * "model_data_$(prim.n_j).jld2", "prim", prim, "res", res, "dist", dist)

# # Load the variables
# @load results_dir * "model_data.jld2" prim res dist

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


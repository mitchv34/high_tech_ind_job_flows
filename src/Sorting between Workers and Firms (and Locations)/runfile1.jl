include("model.jl");
include("distribution_generation.jl")
path_params = "src/Macro-dynamics of Sorting between Workers and Firms (and Locations)/parameters/params.yml";
prim, res = init_model(path_params);
# Create distributions
dist = split_skill_dist(prim);
plot(prim.x_grid, dist.ℓ', lw = 2, alpha = 0.5, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
# plot!(prim.x_grid, dist.ℓ_total, lw = 2, label = "Total")
# Set some parameters
z = 1.0; # Exogenous productivity shock I'm treating as a parameter for now
convergence_path = iterate_distributions!(prim, res, dist; verbose=true, store_path=true)
#?=========================================================================================
#? Solve the model  
#?=========================================================================================


plot(prim.x_grid, convergence_path[end], title ="Distribution of skills", lw = 2, label = reshape(["City $j Final" for j ∈ 1:prim.n_j], 1, prim.n_j), linestyle = :dash, c = reshape([j for j ∈ 1:prim.n_j], 1, prim.n_j))


# Plot the distribution of unemployment

plot(prim.x_grid, dist.u', title ="Distribution of skills", lw = 2, label = reshape(["City $j Final" for j ∈ 1:prim.n_j], 1, prim.n_j), linestyle = :dash, c = reshape([j for j ∈ 1:prim.n_j], 1, prim.n_j))

# Plot the distribution of employment
e = dist.ℓ .- dist.u

plot(prim.x_grid, e', title ="Distribution of skills", lw = 2, label = reshape(["City $j Final" for j ∈ 1:prim.n_j], 1, prim.n_j), linestyle = :dash, c = reshape([j for j ∈ 1:prim.n_j], 1, prim.n_j))

plot(prim.x_grid, convergence_path[1], title ="Distribution of skills", lw = 2, label = reshape(["City $j Initial" for j ∈ 1:prim.n_j], 1, prim.n_j))

# Create the animation
anim = @animate for it ∈ eachindex(convergence_path)
    dist = convergence_path[it]
    plot(prim.x_grid, dist, title ="Distribution of skills iteration $(it)", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
end

# Save the animation as a GIF file
gif(anim, "distributions.gif", fps = 24)

round.(sum(convergence_path[1], dims=1), digits = 3), sum(convergence_path[1])
round.(sum(convergence_path[end], dims=1), digits = 3), sum(convergence_path[end])



avg_skill1 = sum( convergence_path[1] .* prim.x_grid, dims = 1)
size1 = sum( convergence_path[1], dims = 1)

scatter(log.(avg_skill1'), log.(size1'), title = "Average skill vs. size", xlabel = "Average skill", ylabel = "Size",label= "Initial", legend = :topleft)

avg_skill2 = sum( convergence_path[end] .* prim.x_grid, dims = 1)
size2 = sum( convergence_path[end], dims = 1)

scatter!(log.(avg_skill2'), log.(size2'), title = "Average skill vs. size", xlabel = "Average skill", ylabel = "Size",label= "Final", legend = :topleft)
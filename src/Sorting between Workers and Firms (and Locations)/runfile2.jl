using Distributed
using YAML
using BenchmarkTools
z = 1
# Get number of processors from YAML file
path_params = "src/Macro-dynamics of Sorting between Workers and Firms (and Locations)/parameters/params.yml";
nprocs = YAML.load(open(path_params))["parallel"]["nprocs"];
# Start parallel workers
# addprocs(1);

# Load model
@everywhere path_params = "src/Macro-dynamics of Sorting between Workers and Firms (and Locations)/parameters/params.yml";
@everywhere include("model1.jl");
@everywhere prim, res = init_model(path_params);
include("distribution_generation.jl")
# Create distributions
dist = split_skill_dist(prim);

# Distribution of skills in each city
# plot(prim.x_grid, dist.ℓ', lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
# # Distriution of firm productivity
# plot(prim.y_grid, dist.Φ, lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))

bthread = @elapsed convergence_path = iterate_distributions!(prim, res, dist; verbose=true, store_path=true, tol=1e-4);


# Plot first and last distributions
# plot(prim.x_grid, convergence_path[1], lw = 2, 
#         label = reshape(["City $j (Initial)" for j ∈ 1:prim.n_j], 1, prim.n_j))
# plot!(prim.x_grid, convergence_path[end], lw = 2, c = reshape([j for j ∈ 1:prim.n_j], 1, prim.n_j), linestyle = :dash,
#         label = reshape(["City $j (Final)" for j ∈ 1:prim.n_j], 1, prim.n_j))
# # Plot cumulative distributions
# L_initial = cumsum(convergence_path[1], dims = 1)
# L_final = cumsum(convergence_path[end], dims = 1)
# plot(prim.x_grid, L_initial, lw = 2, 
#         label = reshape(["City $j (Initial)" for j ∈ 1:prim.n_j], 1, prim.n_j))
# plot!(prim.x_grid, L_final, lw = 2, c = reshape([j for j ∈ 1:prim.n_j], 1, prim.n_j), linestyle = :dash,
#         label = reshape(["City $j (Final)" for j ∈ 1:prim.n_j], 1, prim.n_j))

# sum(dist.u, dims = 2) ./ sum(dist.ℓ, dims = 2)

# iter = 0
# for i  = 1:50
# Update Distribution at interim stage
update_interim_distributions!(prim, res, dist);
# Update value of vacancy creation
get_vacancy_creation_value!(prim, res, dist);
# Update Market tightness and vacancies
update_market_tightness_and_vacancies!(prim, res, dist);
# Update surplus and unemployment
b_threads = @benchmark compute_surplus_and_unemployment!(prim, res, dist, verbose=false);
# Solve optimal strategies
optimal_strategy!(prim, res); 
# Store t - 1 distributions
u_initial = copy(dist.u);
h_initial = copy(dist.h);
# Update Distribution at next stage
update_distrutions!(prim, res, dist);
sum(dist.u .< 0)
# Update iteration counter
iter += 1
# Compute error
if iter % 1 == 0
    err = maximum(abs.(dist.u - u_initial)) + maximum(abs.(dist.h - h_initial));
    println(@bold @yellow "Error: $(round(err, digits=10))")
    # Print city sizes
    println(@bold @yellow "City sizes:  $(round.(sum(dist.ℓ, dims=2), digits=3))")
end 
# end



plot(prim.x_grid, dist.ℓ', lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))

plot(prim.y_grid, res.B', lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
plot(prim.y_grid, res.v', lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))


plot(prim.x_grid, res.U', lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))

# Plot distribution of unemployment
plot(prim.x_grid, dist.u', lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))

avgskill(ℓ) = sum(ℓ .* prim.x_grid', dims = 2) ./ sum(ℓ, dims = 2)
avgvacancy(v) = sum(v .* prim.y_grid', dims = 2) ./ sum(v, dims = 2)

scatter(avgskill(dist.ℓ)', avgvacancy(res.v)', markersize = 5, label = "", xlabel = "Average skill level", ylabel = "Average vacancy level")


# # Compute average skill level in each city
# #Scatter plot of average skill level in each city
# scatter(sum(convergence_path[1], dims =1)', avgskill(convergence_path[1])',markersize = 5,
#     label = "Average skill level (Initial)", legend = :topleft)
# scatter!(sum(convergence_path[end], dims =1)', avgskill(convergence_path[end])',markersize = 5,
#     label = "Average skill level (Final)", legend = :topleft)


j_dest = 1
plot(legend = :outerleft)
for j_orig ∈ 1:prim.n_j
    plot!(prim.x_grid, res.ϕ_u[j_orig, j_dest, :], lw = 2, label = "City $j_orig → $j_dest")
end
plot!()
j_dest = 2
plot(legend = :outerleft)
for j_orig ∈ 1:prim.n_j
    plot!(prim.x_grid, res.ϕ_u[j_orig, j_dest, :], lw = 2, label = "City $j_orig → $j_dest")
end
plot!()
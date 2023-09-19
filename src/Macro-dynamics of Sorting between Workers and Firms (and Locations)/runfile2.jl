z = 1
include("model1.jl");
include("distribution_generation.jl")
path_params = "src/Macro-dynamics of Sorting between Workers and Firms (and Locations)/parameters/params.yml";
prim, res = init_model(path_params);
# Create distributions
dist = split_skill_dist(prim);

# Distribution of skills in each city
plot(prim.x_grid, dist.ℓ', lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
# Distriution of firm productivity
plot(prim.y_grid, dist.Φ, lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))

# convergence_path = iterate_distributions!(prim, res, dist; verbose=true, store_path=true);

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

iter = 0
# for i  = 1:10
# Update Distribution at interim stage
update_interim_distributions!(prim, res, dist);
# Update value of vacancy creation
get_vacancy_creation_value!(prim, res, dist);
# Update Market tightness and vacancies
update_market_tightness_and_vacancies!(prim, res, dist);
# Update surplus and unemployment
compute_surplus_and_unemployment!(prim, res, dist, verbose=true);
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

scatter(avgskill(dist.ℓ), avgvacancy(res.v), markersize = 5, label = "", xlabel = "Average skill level", ylabel = "Average vacancy level")


# # Compute average skill level in each city
# #Scatter plot of average skill level in each city
# scatter(sum(convergence_path[1], dims =1)', avgskill(convergence_path[1])',markersize = 5,
#     label = "Average skill level (Initial)", legend = :topleft)
# scatter!(sum(convergence_path[end], dims =1)', avgskill(convergence_path[end])',markersize = 5,
#     label = "Average skill level (Final)", legend = :topleft)

# p_moveto1 = plot(prim.x_grid, res.ϕ_u[1, 1, :], lw = 2, label = "City 1 → 1")
# plot!(prim.x_grid, res.ϕ_u[2, 1, :], lw = 2, label = "City 2 → 1")
# plot!(prim.x_grid, res.ϕ_u[3, 1, :], lw = 2, label = "City 3 → 1")

# p_moveto2 = plot(prim.x_grid, res.ϕ_u[1, 2, :], lw = 2, label = "City 1 → 2")
# plot!(prim.x_grid, res.ϕ_u[2, 2, :], lw = 2, label = "City 2 → 2")
# plot!(prim.x_grid, res.ϕ_u[3, 2, :], lw = 2, label = "City 3 → 2")

# p_moveto3 = plot(prim.x_grid, res.ϕ_u[1, 3, :], lw = 2, label = "City 1 → 3")
# plot!(prim.x_grid, res.ϕ_u[2, 3, :], lw = 2, label = "City 2 → 3")
# plot!(prim.x_grid, res.ϕ_u[3, 3, :], lw = 2, label = "City 3 → 3")

# plot(p_moveto1, p_moveto2, p_moveto3, layout = (1, 3), size = (800, 600))
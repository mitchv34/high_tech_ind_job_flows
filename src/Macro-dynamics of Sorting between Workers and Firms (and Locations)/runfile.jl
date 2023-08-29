include("model.jl");
path_params = "src/Macro-dynamics of Sorting between Workers and Firms (and Locations)/parameters/params.yml";
prim, res = init_model(path_params);
# Set some parameters
z = 1; # Exogenous productivity shock I'm treating as a parameter for now

#?=========================================================================================
#? Create some Distribution to start with
#?=========================================================================================
#* Experiment 1: Uniform distribution of skills across locations
# begin
#     # We start with a uniform distribution of skills (overall)
#     ℓ_total = ones(prim.n_x) / prim.n_x # Vector (n_x) each element is a skill level
#     # We start with a uniform distribution of workers across locations and skills within each location
#     ℓ = ℓ_total' .* ones(prim.n_j, prim.n_x) / (prim.n_j ) # Matrix (n_j x n_x) each row is a location, each column is a skill level
#     # Start with 10% unemployment rate in each location and skill and workers equally distributed across firms
#     u = ℓ .* 0.1 # Initial unemployment for each skill level in each location
#     # Distribution of workers across firms
#     e = (1 - 0.1) .* ℓ # Distribution of employed workers
#     h = zeros(prim.n_j, prim.n_x, prim.n_y) # Distribution of employed workers
#     for j ∈ 1 : prim.n_j
#         h[j, :, :] .= e[j, :] ./ prim.n_y
#     end
#     dist = DistributionsModel(ℓ_total, ℓ, u, h);
# end
# * Experiment 2: Same distribution as in @liseMacrodynamicsSortingWorkers2017 equally distributed across locations
begin
    # Distribution of x-types is Beta(2.15, 12.0)
    ℓ_total = pdf.(Beta(2.15, 12.0), prim.x_grid)
    # Normalize to have total mass 1
    ℓ_total = ℓ_total / sum(ℓ_total)
    # We start with a uniform distribution of workers across locations and skills within each location
    ℓ = ℓ_total' .* ones(prim.n_j, prim.n_x) / (prim.n_j ) # Matrix (n_j x n_x) each row is a location, each column is a skill level
    # Start with 10% unemployment rate in each location and skill and workers equally distributed across firms
    u = ℓ .* 0.1 # Initial unemployment for each skill level in each location
    # Distribution of workers across firms
    e = (1 - 0.1) .* ℓ # Distribution of employed workers
    h = zeros(prim.n_j, prim.n_x, prim.n_y) # Distribution of employed workers
    for j ∈ 1 : prim.n_j
        h[j, :, :] .= e[j, :] ./ prim.n_y
    end
    dist = DistributionsModel(ℓ_total, ℓ, u, h);
end
plot(prim.x_grid, ℓ', title ="Initial distribution of skills", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
#?=========================================================================================
#? Solve the model  
#?=========================================================================================

C = congestion_cost(prim, dist)
f = output(prim, dist)
b = home_production(prim, dist)

b[1, :] .- C[1]

max_iter=50000
tol=1e-6
iter=0
err=Inf
while (err > tol) & (iter < max_iter)
    # Solve unemployment value
    compute_unemployment_value!(prim, res, dist; verbose = false);
    # Solve optimal strategies
    optimal_strategy!(prim, res);
    # Solve surplus
    compute_surplus!(prim, res, dist; verbose = false);
    # Update Distribution at interim stage
    update_interim_distributions!(prim, res, dist);
    # Update value of vacancy creation
    get_vacancy_creation_value!(prim, res, dist);
    # Update Market tightness and vacancies
    update_market_tightness_and_vacancies!(prim, res, dist);
    # Store t - 1 distributions
    u_initial = copy(dist.u);
    h_initial = copy(dist.h);
    # Update Distribution at next stage
    update_distrutions!(prim, res, dist);
    # Compute error
    err = maximum(abs.(dist.u - u_initial)) + maximum(abs.(dist.h - h_initial))
    if iter % 50 == 0
        println(@bold @yellow "Iteration:  $iter, Error: $(round(err, digits=10))")
    elseif err < tol
        println(@bold @green "Iteration:  $iter, Converged!")
    end
    iter += 1
end 

B = worker_productivity(prim, dist)

plot(prim.x_grid, B[1, :], label = "Worker Productivity", lw = 2, title = "Worker Productivity")
plot!(prim.x_grid, prim.x_grid, label = "45 degree line", lw = 2, ls = :dash)

# Compute the mass of employed workers at each location - skill 
e = dropdims(sum(dist.h, dims=3), dims=3);

plot(prim.x_grid[1:end], e[1, 1:end], label="Employed", lw = 2, title = "Employed workers at each skill level")

plot(prim.x_grid[1:end], dist.u[1, 1:end], label="Unemployed", lw = 2, title = "Unemployed workers at each skill level", legend = :right)

# Compute unemployment rate at each location - skill
u = dist.u ./ (dist.u .+ e);

plot(prim.x_grid[1:end], u[1, 1:end], label="Unemployment rate", lw = 2, title = "Unemployment rate at each skill level")

# Compute overall unemployment rate
sum(dist.u)

dist.u

μ = sum(dist.ℓ, dims = 2)
plot(μ, ylims=[0.0, 0.2], label="Initial")

# Value functions 
plot(prim.x_grid, res.U', title ="Value of unemployment", xlabel="Skill Level", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))

# Surplus heatmap
plot(prim.x_grid, prim.y_grid, res.S_move[1,1,:,:], st = :heatmap, title ="Surplus", xlabel="Skill Level", ylabel="Firm Type")

res.S_move[1,2,:,:]

# ! Building new functions :

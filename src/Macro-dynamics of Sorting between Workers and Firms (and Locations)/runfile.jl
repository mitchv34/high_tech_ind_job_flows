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
# * Experiment 3: I'm going to move half the mass of workers from city 1 to city 10 
#* (using Experiment 2 as a starting point)
begin
    ℓ[1, :] = ℓ[1, :] * 0.5
    ℓ[10, :] = ℓ[10, :] * 1.5
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

#* Experiment 4: City 1 with all the workers
# begin
#     # Distribution of x-types is Beta(2.15, 12.0)
#     ℓ_total = pdf.(Beta(2.15, 12.0), prim.x_grid)
#     # Normalize to have total mass 1
#     ℓ_total = ℓ_total / sum(ℓ_total)
#     ℓ = zeros(prim.n_j, prim.n_x) # Matrix (n_j x n_x) each row is a location, each column is a skill level
#     ℓ[1, :] = ℓ_total' # All workers in city 1
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

# * Experiment 5: Cities with different distributions of skills
begin
    # Distribution of x-types is Beta(2.15, 12.0)
    ℓ_total_low = pdf.(Beta(2.15, 12.0), prim.x_grid)
    ℓ_total_high = pdf.(Beta(3.15, 12.0), prim.x_grid)

    # Normalize to have total mass 1
    ℓ_total_low = ℓ_total_low / sum(ℓ_total_low)
    ℓ_total_high = ℓ_total_high / sum(ℓ_total_high)

    # Plot CDFs
    L_low = cumsum(ℓ_total_low)
    L_high = cumsum(ℓ_total_high)
    plot(prim.x_grid, L_low, label = "Low skill", lw = 2, title = "Distribution of skills")
    plot!(prim.x_grid, L_high, lw = 2, label = "High skill")

    # We start with a uniform distribution of workers across locations and skills within each location
    ℓ_low = ℓ_total_low' .* ones(Int64(prim.n_j/2), prim.n_x) / (prim.n_j ) # Matrix (n_j x n_x) each row is a location, each column is a skill level
    ℓ_high = ℓ_total_high' .* ones(Int64(prim.n_j/2), prim.n_x) / (prim.n_j ) # Matrix (n_j x n_x) each row is a location, each column is a skill level
    # 5 first cities have low skill workers, 5 last cities have high skill workers
    ℓ = [ℓ_low; ℓ_high]
    # Start with 10% unemployment rate in each location and skill and workers equally distributed across firms
    u = ℓ .* 0.1 # Initial unemployment for each skill level in each location
    # Distribution of workers across firms
    e = (1 - 0.1) .* ℓ # Distribution of employed workers
    h = zeros(prim.n_j, prim.n_x, prim.n_y) # Distribution of employed workers
    for j ∈ 1 : prim.n_j
        h[j, :, :] .= e[j, :] ./ prim.n_y
    end
    ℓ_total = ℓ_total_low + ℓ_total_high
    # Normalize to have total mass 1
    ℓ_total = ℓ_total / sum(ℓ_total)
    dist = DistributionsModel(ℓ_total, ℓ, u, h);
end 

sum(ℓ_low, dims=2)
sum(ℓ_high, dims=2)

plot(prim.x_grid, ℓ', title ="Initial distribution of skills", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))

plot(ℓ_total_low, label = "Low skill", lw = 2, title = "Distribution of skills")
plot!(ℓ_total_high, lw = 2, label = "High skill")
plot!(ℓ_total, lw = 2, label = "Overall")


# Unpack primitives
@unpack n_j, n_x, β, c  = prim
μ = sum(dist.ℓ, dims=2)
# Compute the average skill level in each location 
x̄ = [(μ[j] > 0) ? sum(ℓ[j, :] .* x_grid) ./ μ[j] : 0.0 for j ∈ 1:n_j]
# Compute the value of idea exchange in each location
X = (1  .- exp.(-prim.ν .* μ)) .* x̄
# Calcualte production in each location for each type of worker and firm
plot(prim.j_grid, X, title = "Idea exchange environment", lw = 2, label = "")
B = x_grid' .* (1 .+ prim.A .* X .* x_grid') 
plot(prim.x_grid, B', title = "Worker productivity", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
f = output(prim, dist);
plot(prim.x_grid, f[:, :, 10]', title = "Firm 10 productivity", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
# Compute the value of unemployment in each location for each type of worker
b = home_production(prim, dist);
plot(prim.x_grid, b', title = "Value of unemployment", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
# Compute the cost of living in each location
C = congestion_cost(prim, dist);
plot(prim.j_grid, C, title = "Cost of living", lw = 2, label = "")
compute_unemployment_value!(prim, res, dist; verbose = true);

# Compute the value of unemployment in each location for each type of worker
plot(prim.x_grid, (b .- C)')


U_new = b .- C  .+  β .* c .* ( log.( sum( exp.(res.U ./ c)  , dims = 1 )  ) .- log(n_j))
# Compute the error
err = maximum(abs.(U_new .- res.U))
# Update U
res.U = copy(U_new)


plot(prim.x_grid, res.U', title ="Value of unemployment", xlabel="Skill Level", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))

#?=========================================================================================
#? Solve the model  
#?=========================================================================================
max_iter=50000
tol=1e-6
iter=0
err=Inf
dists = []
while (err > tol) & (iter < max_iter)
    push!(dists, dist.ℓ')
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

# plot(prim.x_grid, dists[end], title ="Distribution of skills", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))

anim = @animate for i ∈ 1:length(dists)
    plot(prim.x_grid, dists[i], title ="Distribution of skills iteration = $(i)", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))#, ylims=[0.0, 0.25])
end
gif(anim, "distribution.gif", fps = 15)



B = worker_productivity(prim, dist)

plot(prim.x_grid, B[1, :], label = "City 1", lw = 2, title = "Worker Productivity")
plot!(prim.x_grid, B[5, :], label = "City 5", lw = 2, title = "Worker Productivity")
plot!(prim.x_grid, B[10, :], label = "City 10", lw = 2, title = "Worker Productivity")
plot!(prim.x_grid, prim.x_grid, label = "45 degree line", lw = 2, ls = :dash)

# Compute the mass of employed workers at each location - skill 
e = dropdims(sum(dist.h, dims=3), dims=3);

plot(prim.x_grid[1:end], e[1, 1:end], label="Employed", lw = 2, title = "Employed workers at each skill level")

plot(prim.x_grid[1:end], dist.u[1, 1:end], label="Unemployed", lw = 2, title = "Unemployed workers at each skill level", legend = :right)

# Compute unemployment rate at each location - skill
u = dist.u ./ (dist.u .+ e);

plot(prim.x_grid[1:end], u[1, 1:end], label="Unemployment rate", lw = 2, title = "Unemployment rate at each skill level")

# Compute overall unemployment rate
sum(dist.u) / (sum(dist.u) + sum(e))

dist.u

μ = sum(dist.ℓ, dims = 2)
plot(μ, ylims=[0.0, 0.2], label="Initial")

# Value functions 
plot(prim.x_grid, res.U', title ="Value of unemployment", xlabel="Skill Level", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
# Strategies
plot(prim.x_grid, res.ϕ_u', title ="Optimal strategy", xlabel="Skill Level", lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
# Surplus heatmap
plot(prim.x_grid, prim.y_grid, res.S_move[1,2,:,:] .≥ 0, st = :heatmap, title ="Surplus", xlabel="Skill Level", ylabel="Firm Type")

res.S_move[1,2,:,:]

# ! Building new functions :
# Unpack primitives

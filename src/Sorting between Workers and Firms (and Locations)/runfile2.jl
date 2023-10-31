using Distributed
using DataFrames
using CSV
using YAML
using BenchmarkTools
using StatsBase
using StatsPlots
using FixedEffectModels
using RegressionTables
# Load plot settings
include("../plotconfig.jl")
include("model.jl")
z = 1.0

# Load model
@everywhere path_params = "src/Sorting between Workers and Firms (and Locations)/parameters/params.yml";
@everywhere include("model.jl");
@everywhere prim, res = init_model(path_params);
# include("create_dist_form_data.jl")
include("distribution_generation.jl")
# # Create distributions
μ = 0.9    
dist = split_skill_dist(prim, weights = [μ ,  1 - μ]);

# bthread = @elapsed convergence_path = iterate_distributions!(prim, res, dist; verbose=true, store_path=true, tol=1e-6);

# begin
# Update surplus and unemployment
compute_surplus_and_unemployment!(prim, res, dist, verbose=true);

u_initial = copy(dist.u);
h_initial = copy(dist.h);
ℓ_initial = copy(dist.ℓ);
# # Update Distribution at next stage
# update_distrutions!(prim, res, dist);

# # Plot distribution of unemployment
plot(prim.x_grid, dist.ℓ', lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
# # Plot distribution of unemployment previous period
# plot!(prim.x_grid, ℓ_initial', lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j), linestyle = :dash, c = reshape([j for j ∈ 1:prim.n_j], 1, prim.n_j))

# err = maximum(abs.(dist.u - u_initial))/maximum(abs.(dist.u)) + maximum(abs.(dist.h - h_initial))/maximum(abs.(dist.h));
# println(@bold @yellow "Error: $(round(err, digits = 10))")
# println(@bold @yellow "Sizes: $(sum(dist.ℓ, dims = 2))")
# # end 
# plot!()



@unpack ω₁, ω₂, n_j, n_x, n_y, s, δ = prim
@unpack L = res
# Calculate total number of mathces in each location
M = min.(ω₁ .* L.^ω₂ .* res.V.^(1-ω₂), L, res.V)
p = M ./ L # Probability of a match being formed in each location
# Replace NaN with zero
p[isnan.(p)] .= 0
# Compute indicator of possitive surplus 
η = res.S_move .>= 0;
# Compute indicator of possitive surplus of moving to a different location
η_move = zeros(n_j, n_j, n_x, n_y, n_y);
for j ∈ 1:n_j # Loop over locations
    for x ∈ 1:n_x # Loop over skills
        for y ∈ 1:n_y # Loop over firms
            # η[j' → j, x, y'→y] = 1 if S[j' → j, x, y] > S[j' → j', x, y']
            #? To be clear n_move[j,j',x,y,y'] = 1 if firm [j',y'] can poach a worker with skill x from firm [j,y] 
            #? for example η_move[j, :, x, y, :] are all the firms that can poach a worker with skill x from firm [j,y]
            η_move[j, :, x, y, :] = res.S_move[j, :, x, :] .> res.S_move[j, j, x, y] 
        end # Loop over firms
    end # Loop over skills
end # Loop over locations


γ = p .* res.v./res.V
ϕ_hat_s = res.ϕ_s .* η_move
ϕ_hat_u = res.ϕ_u .* η

new_h = zeros(n_j, n_x, n_y)
new_u = zeros(n_j, n_x)
h_u = zeros(n_j, n_x, n_y)
h_p = zeros(n_j, n_x, n_y)
h_r = zeros(n_j, n_x, n_y)


for j ∈ 1 : n_j
    for y ∈ 1 : n_y
        h_u[j, :, y] = γ[j, y] .* sum( ϕ_hat_u[:,j,:,y] .* dist.u, dims = 1)[:]
        h_p[j, :, y] = sum(s .* γ[j, y] .* dist.h_plus .* ϕ_hat_s[:,j,:,:,y], dims = [1,3])[:]
        h_r[j, :, y] = dist.h_plus[j, :, y] .* prod(1 .- s .* sum(γ .* permutedims(dist.h_plus .* ϕ_hat_s[j,:,:,y,:], [1,3,2]), dims = 2), dims = 1)[:]
    end
    new_u[j, :] = dist.u_plus[j] .* (1 .- prod(sum(permutedims(ϕ_hat_u[j,:,:,:], [1,3,2]) .* γ, dims = 2), dims = 1)[:])
end

dist.u_plus[j] .* (1 .- prod(sum(permutedims(ϕ_hat_u[j,:,:,:], [1,3,2]) .* γ, dims = 2), dims = 1)[:])

new_h = h_u + h_p + h_r

sum(new_h) 
sum(new_u)

new_h

sum(dist.u)
sum(dist.h)

emp_dist_old = dropdims(sum(dist.h, dims = 3), dims = 3) 
emp_dist_new = dropdims(sum(new_h, dims = 3), dims = 3)

plot(prim.x_grid, emp_dist_old', lw = 2, label ="")
plot!(prim.x_grid, emp_dist_new', lw = 2, label ="", linestyle = :dash)
# for j ∈ 1:n_j
#         for x ∈ 1:n_x
#                 u_p = dist.u_plus[j, x]
#                 ϕ_hat_u_jx = ϕ_hat_u[j, :, x]
#                 for y ∈ 1:n_y
#                         γ_y = γ[j, y]
#                         ϕ_hat_s_jy = ϕ_hat_s[j, :, x, y, :]


[sum(res.ϕ_u[:,:,1], dims=2) for x  = 1:n_x]
                        
plot(prim.x_grid, res.ϕ_u[2, :, :]', lw = 2, label = reshape(["City 2 → $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
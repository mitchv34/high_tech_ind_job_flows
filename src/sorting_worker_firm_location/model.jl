#==========================================================================================
Title: Macro-dynamics of Sorting between Workers and Firms (and Locations)
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2023-08-26
Description:  This file contains the code to solve the model in the paper 
                    "Macro-dynamics of Sorting between Workers and Firms (and Locations)".
==========================================================================================#
#==========================================================================================
# * Packages 
==========================================================================================#
@everywhere using LinearAlgebra
@everywhere using Parameters
@everywhere using YAML
@everywhere using Term 
@everywhere using Distributions
@everywhere using Distributed
@everywhere using SharedArrays
#?=========================================================================================
#? Structures
#?=========================================================================================
#==========================================================================================
# Parameters: Strucutre to store all parameters of the model 
==========================================================================================#
# TODO: Speed up this function
@everywhere @with_kw mutable struct Primitives
    # Primitives
    z           ::Float64          # Aggregate productivity
    β           ::Float64          # Discount factor
    c           ::Float64          # Cost of search
    δ           ::Float64          # Separation rate
    s           ::Float64          # Search intensity of employed workers
    ξ           ::Float64          # Worker's bargaining power
    # Cost of living
    a₁          ::Float64          # Agglomeration (scale)
    b₁          ::Float64          # Agglomeration (curvature)
    # Cost of living
    a₂          ::Float64          # Cost of living (scale)
    b₂          ::Float64          # Cost of living (curvature)
    # Production
    output      ::Function         # Poduction function
    A           ::Float64          # Gains from idea exchange
    # α           ::Float64        #
    ν           ::Float64          # Idea exchange parameter
    b_hat       ::Float64          # Home production parameter
    # Vacancy creation cost function
    c₀          ::Float64          
    c₁          ::Float64          
    # Matching function
    ω₁          ::Float64          
    ω₂          ::Float64          
    # Fixed cost of moving locations
    F_bar       ::Float64          # Fixed cost of moving locations
    # Grids
    n_x         ::Int64            # Number of skill levels
    n_y         ::Int64            # Number of firm types
    n_z         ::Int64            # Number of shocks
    n_j         ::Int64            # Number of locations
    x_grid      ::Array{Float64,1} # Grid of worker's skills
    y_grid      ::Array{Float64,1} # Grid of firm's productivity
    z_grid      ::Array{Float64,1} # Grid of shocks
    j_grid      ::Array{Int64,1}   #! Grid of locations (Not sure if I need this)
    # Distribution parameters
    # Worker's skill distribution
    x_dist_name   ::String           # Name of the distribution of overal skills
    x_dist_params ::Array{Float64,1} # Parameters of the distribution
    # Firm's productivity distribution
    y_dist_name   ::String           # Name of the distribution of overal skills
    y_dist_params ::Array{Float64,1} # Parameters of the distribution
end # end of Parameters
#==========================================================================================
# Results: Strucutre to store all results of the model 
==========================================================================================#
# TODO: Speed up this function
@everywhere mutable struct Results
    # * Note that i'm ignoring the shocks since im leaving it fixed at 1
    # * In the general version of the model one extra dimension is needed in all results
    # Unemployment value function Uʲₜ(x) (n_j × n_x × n_z) 
    U          ::Array{Float64, 2} 
    # Optimal strategy of each unemployed worker in each location ϕᵤʲₜ(x, j') (n_j × n_j × n_x × n_z)
    ϕ_u        ::Array{Float64, 3}
    # Optimal strategy of each employed worker in each location ϕₛʲₜ(x, j') (n_j × n_j × n_x × n_y × n_z)
    ϕ_s        ::Array{Float64, 4}
    # Surplus of each move between locations for each worker - firm match Sₜ(j → j', x, y) (n_j × n_j × n_x × n_y × n_z)
    S_move     ::Array{Float64, 4}
    # Value of vacancy creation in each location for each type of firm (n_j × n_y × n_z)
    B          ::Array{Float64, 2}
    # Market tightness (n_j × n_y × n_z)
    θ          ::Array{Float64, 2} 
    # Number of vacancies (n_j × n_y × n_z)
    v          ::Array{Float64, 2}
    # Total number of vacancies in each location (n_j × n_z)
    V          ::Array{Float64, 1}
    # Total search effort in each location (n_j × n_z)
    L          ::Array{Float64, 1}
    ##* Static Objects in the inner fixed point
    # Home production for each type in each location
    b          ::Array{Float64, 2}
    # Market production for each match in each location
    f          ::Array{Float64, 3}
    # Cost of living in each location
    C          ::Array{Float64, 1}
    # Instant surplus of each move between locations for each worker - firm match s(j → j', x, y) = (f(x,y,j') - C(j')) - (b(x,j') - C(j))
    s_move     ::Array{Float64, 4}
    # Optimal firm for each type in each location
    y_star     ::Array{Float64, 2}
    ## * Equilibrium objects
    # Equilibrium distribution of vacacies in each location of each type of firm
    γ          ::Array{Float64, 2}
    # Adittional objects
    # Firm j’s poaching index πj measures the fraction of a firm’s hires that are poached from other firms,
    Π          ::Array{Float64, 2} 
    # Unemployed migration
    u_move     ::Array{Float64, 3}
    # Employed migration
    h_move     ::Array{Float64, 4}
    # Constructor
    function Results(prim::Primitives)
        @unpack n_j, n_x, n_y, n_z = prim
        # Pre-allocate unemployment value function
        U = zeros(n_j, n_x) #!, n_z)
        # Pre-allocate optimal strategy of each unemployed worker in each location (as random search)
        ϕ_u = ones(n_j, n_j, n_x) ./ n_j  #!, n_z)
        # Pre-allocate optimal strategy of each employed worker in each location (as random search)
        ϕ_s = ones(n_j, n_j, n_x, n_y) ./ n_j #!, n_z)
        # Pre-allocate surplus of each move between locations for each worker - firm match
        S_move = zeros(n_j, n_j, n_x, n_y) #!, n_z)
        # Pre-allocate value of vacancy creation in each location for each type of firm
        B = zeros(n_j, n_y) #!, n_z)
        # Pre-allocate market tightness
        θ = zeros(n_j, n_y) #!, n_z)
        # Pre-allocate number of vacancies
        v = zeros(n_j, n_y) #!, n_z)
        # Pre-allocate total number of vacancies in each location
        V = zeros(n_j) #!, n_z)
        # Pre-allocate total search effort in each location
        L = zeros(n_j) #!, n_z)
        # Pre-allocate static objects
        b = zeros(n_j, n_x) #!, n_z)
        f = zeros(n_j, n_x, n_y) #!, n_z)
        C = zeros(n_j) #!, n_z)
        s_move = zeros(n_j, n_j, n_x, n_y) #!, n_z)
        y_star = zeros(n_j, n_x) #!, n_z)
        # Pre-allocate equilibrium objects
        γ = zeros(n_j, n_y) #!, n_z)
        # Pre-allocate adittional objects
        Π = zeros(n_j, n_y) #!, n_z)
        new(U, ϕ_u, ϕ_s, S_move, B, θ, v, V, L, b, f, C, s_move, y_star, γ, Π)
    end # end of constructor of Results
end # end of Results
#==========================================================================================
# DistributionsModel: Strucutre to store the distribution of populations across locations,
        distribution of unemployed workers across locations and skills, and the distribution
        of employed workers across locations, skills and firms.
==========================================================================================#
# TODO: Speed up this function
@everywhere mutable struct DistributionsModel
    # Overal distribution of skills {ℓ(x)}
    ℓ_total    ::Array{Float64, 1} # ! This is inmutable 
    # Skill distribution on each location {ℓʲ(x)}
    ℓ          ::Array{Float64, 2} # * Initial distribution is provided, then it is updated
    # DistributionsModel of unemployed workers {uʲₜ(x)}
    u          ::Array{Float64, 2} # * Initial distribution is provided, then it is updated
    # Interim distribution of unemployed workers {uʲₜ₊(x)}
    u_plus     ::Array{Float64, 2} 
    # DistributionsModel of employed workers {hʲₜ(x, y)}
    h          ::Array{Float64, 3} # * Initial distribution is provided, then it is updated
    # Interim distribution of employed workers {hʲₜ₊(x, y)}
    h_plus     ::Array{Float64, 3}
    # Overal distribution of firm productivity {Φ(y)}
    Φ    ::Array{Float64, 1} # ! This is inmutable
    # Firm productivity distribution {Φʲₜ(y)}
    # TODO: I think I dont need this since firms will chose where to locate based on the free entry condition
    # Φ          ::Array{Float64, 2} # * Initial distribution is provided, then it is updated
    # Constructor
    function DistributionsModel(ℓ_total::Array{Float64, 1}, ℓ::Array{Float64, 2}, 
                                u::Array{Float64, 2}, h::Array{Float64, 3}, Φ::Array{Float64, 1})
        # Create interim DistributionsModel as zeros arrays size of the initial DistributionsModel
        u_plus = zeros(size(u))
        h_plus = zeros(size(h))
        new(ℓ_total, ℓ, u, u_plus, h, h_plus, Φ)
    end # end of constructor of DistributionsModel
end # end of DistributionsModel
#?=========================================================================================
#? Model Initialization 
#?=========================================================================================
#==========================================================================================
# read_primitives: A function that reads the primitives of the model from a YAML file
==========================================================================================#
# TODO: Speed up this function
@everywhere function read_primitives(path_to_params::AbstractString, z::Float64=1.0)
    # Read YAML file
    data = YAML.load_file(path_to_params)

    # Create x_grid, y_grid, z_grid 
    # x_grid = range(data["grids"]["x_min"], data["grids"]["x_max"], length=data["grids"]["n_x"])
    x_grid = exp.(range(log(data["grids"]["x_min"]), log(data["grids"]["x_max"]), length=data["grids"]["n_x"]))
    # y_grid = range(data["grids"]["x_min"], data["grids"]["x_max"], length=data["grids"]["n_y"])
    y_grid = exp.(range(log(data["grids"]["y_min"]), log(data["grids"]["y_max"]), length=data["grids"]["n_y"]))
    z_grid = [1.0] #! This is just a placeholder for now
    # Create j_grid
    j_grid = range(1, data["grids"]["n_j"], length=data["grids"]["n_j"])

    # Create production function
    # params = data["primitives"]["function"]["params"] # Get parameters of the function
    # functional = data["primitives"]["function"]["functional_form"] # Get functional in string form 

    # function fcnFromString(s) # Create function from string
    #     f = eval(Meta.parse("(x, y) -> " * s))
    #     return (x, y) -> Base.invokelatest(f, x, y)
    # end

    # f = fcnFromString(functional) # Create function from string

    # TODO: Figure out how to create a function from a string
    # ! I'm manually creating the function for now
    # f = (x,y) -> x.^params[1] .* y'.^(1-params[1])
    # f = (x,y) -> (params[1] .+ params[2] .* x .+ params[3] .* y' .+ params[4] .* x.^2 + params[5] .* (y').^2 .+ params[6] .* x .* y') # Functional form of production function 
    # f = (x,y) -> (params[1] .* x.^((1-params[2])/params[2]) .+ (1-params[1]) .* (y').^((1-params[2])/params[2])).^(params[2]/(1-params[2]))

    # Create Primitives struct
    Primitives(
        z = z;
        β = data["primitives"]["beta"],
        c = data["primitives"]["c"],
        δ = data["primitives"]["delta"],
        s = data["primitives"]["s"],
        ξ = data["primitives"]["xi"],
        a₁ = data["primitives"]["a_1"],
        b₁ = data["primitives"]["b_1"],
        a₂ = data["primitives"]["a_2"],
        b₂ = data["primitives"]["b_2"],
        # ! Temporal
        output = x -> x,
        A = data["primitives"]["A"],
        ν = data["primitives"]["nu"],
        b_hat = data["primitives"]["b_hat"],
        c₀ = data["primitives"]["c_0"],
        c₁ = data["primitives"]["c_1"],
        ω₁ = data["primitives"]["omega_1"],
        ω₂ = data["primitives"]["omega_2"],
        F_bar = data["primitives"]["F_bar"],
        n_x = data["grids"]["n_x"],
        n_y = data["grids"]["n_y"],
        n_z = data["grids"]["n_z"],
        n_j = data["grids"]["n_j"],
        x_grid = collect(x_grid),
        y_grid = collect(y_grid),
        z_grid = collect(z_grid),
        j_grid = collect(j_grid),
        x_dist_name = data["distributions"]["skill"]["name"],
        x_dist_params = data["distributions"]["skill"]["params"],
        y_dist_name = data["distributions"]["productivity"]["name"],
        y_dist_params = data["distributions"]["productivity"]["params"]
    )
end # end of read_primitives
#==========================================================================================
# init_model: A function that initializes the model 
==========================================================================================#
# TODO: Speed up this function
@everywhere function init_model(path_to_params::AbstractString, z::Float64=1.0)
    # Generate primitives
    prim = read_primitives(path_to_params, z)
    # Generate results
    res = Results(prim)
    # Return primitives and results
    return prim, res
end # end of init_model
#?=========================================================================================
#? Basic Functions
#?=========================================================================================
#==========================================================================================
# idea_exchange: A function that computes the the value of idea exchange in each 
        location given the distribution of workers.
==========================================================================================#
# TODO: Speed up this function
@everywhere function idea_exchange(prim::Primitives, dist::DistributionsModel)
    # Unpack DistributionsModel
    @unpack ℓ = dist
    # Unpack primitives
    @unpack x_grid, ν, n_j = prim
    # Compute each location's share of the population (μⱼ)
    μ = sum(ℓ, dims=2)
    # Compute the average skill level in each location 
    x̄ = sum(ℓ' .* x_grid, dims=1)[:]
    # Compute the value of idea exchange in each location
    X = (1  .- exp.(-ν .* μ)) .* x̄
    return X
end # end of idea_exchange
#==========================================================================================
# worker_productivity: A function that computes the producivity of each type of worker
        in each location
==========================================================================================#
#! REMOVE
# TODO: Speed up this function
@everywhere function worker_productivity(prim::Primitives, dist::DistributionsModel)
    # Unpack primitives
    @unpack A, x_grid = prim
    # Get value of idea exchange in each location
    X = idea_exchange(prim, dist)
    # Compute productivity of each type of worker in each location
    Ω = x_grid' .* (1 .+ A .* X .* x_grid') 
    return Ω
end # end of worker_productivity
#==========================================================================================
# output: A function that computes the value of output in each location for each type of
        worker and firm, aditionally it computes the home production of each type of worker
        in each location
==========================================================================================#
# TODO: Speed up this function
# begin
@everywhere function update_inner_static_objects!(prim::Primitives, res::Results, dist::DistributionsModel; 
                                                learning::Bool=false)
# begin
    # Unpack primitives
    @unpack  z, y_grid, x_grid, b_hat, A, n_j, n_x, n_y, b_hat, a₁, b₁, a₂, b₂ = prim
    # Compute each location's share of the population (μⱼ)
    μ = sum(dist.ℓ, dims=2)
    # Compute the cost of living in each location
    C =  a₂ .* (μ .^  b₂)
    if learning
        # Compute the productivity of each type of worker in each location
        # Get value of idea exchange in each location
        X = idea_exchange(prim, dist)
        # Compute productivity of each type of worker in each location
        Ω_grid = x_grid' .* (1 .+ A .* X .* x_grid') 
    else
        # Compute the TFP shock depending **only** on the size of the city
        Ω = (1 .+  a₁ .* μ .^ b₁)
    end
    # ! This needs to be fixed
    # TODO: Modify the model to allow for different functional forms of production
    f_joint(x, y, z, c, Ω) = Ω .* z .* (p1 .+ (p2 .* x) .+ ( p3 .* y)' .+ ( p4 * x.^2) .+ (p5 .* y.^2)' + p6 * (x .* y')) .- c
    # f_joint(x, y, z, c, Ω) = Ω .* z .* (x .* y') .- c
    # ρ = 0.5
    # f_joint(x, y, z, c, Ω) = Ω .* z .* (x.^ρ .* (y.*ρ)').^(1/ρ) .- c
    p1 = 0.000;
    p2 = 0.991;
    p3 = -0.126;
    p4 = 9.042;
    p5 = -0.425;
    p6 = 3.391;
    # p3 = -0.18 * p6;
    ystar(x) = max.(0, min.(1, ( -p3 .- (x .* p6) )./(2 * p5) ) )
    b_home(x, c, Ω) = b_hat .* f_joint(x, ystar(x)', 1, 0, Ω )  .- c
    b_home(x_grid,1,1)

    # Construct objects
    f = zeros(n_j, n_x, n_y)
    y_star = zeros(n_j, n_x)
    b = zeros(n_j, n_x)
    s_move = zeros(n_j, n_j, n_x, n_y)
    if learning
        # TODO: Generalize
        #! This needs to be fixed it wont work when learning environment is on
        for j ∈ 1 : n_j
            f[j, :, :] = f_joint(Ω_grid[j,:], y_grid, z, C[j])
            y_star[j, :] = ystar(Ω_grid[j,:])
            b[j, :] = b_home(Ω_grid[j,:], C[j])
            for j_prime ∈ 1 : n_j
                s_move[j, j_prime, :, :] = f_joint(Ω_grid[j_prime, :], y_grid, z, C[j_prime]) .- b_home(Ω_grid[j, :], C[j])
            end
        end
    else
        for j ∈ 1 : n_j
            f[j, :, :] = f_joint(x_grid, y_grid, z, C[j], Ω[j])
            y_star[j, :] = ystar(x_grid)
            b[j, :] = b_home(x_grid, C[j], Ω[j])
            for j_prime ∈ 1 : n_j
                s_move[j, j_prime, :, :] = f_joint(x_grid, y_grid, z, C[j_prime], Ω[j_prime]) .- b_home(x_grid, C[j], Ω[j])
            end
        end
    end

    # Update objects
    res.s_move = copy(s_move);
    res.b = copy(b);
    res.f = copy(f);
    res.C = copy(C[:]);
    res.y_star = copy(y_star);

    return nothing
end # end update_inner_static_objects!
#==========================================================================================
# congestion_cost: A function that computes the cost of living in each location
==========================================================================================#
# ! REMOVE
# TODO: Speed up this function
@everywhere function congestion_cost(prim::Primitives, dist::DistributionsModel)
    # Unpack DistributionsModel
    @unpack ℓ = dist
    # Unpack primitives
    @unpack θ, γ = prim
    # Compute each location's share of the population (μⱼ)
    μ = sum(ℓ, dims=2)
    # Compute the cost of living in each location
    C = θ .* (μ .^ γ)
    return C
    # return zeros(size(C)) # For now I'm setting the cost of living to zero
end # end of congestion_cost
#==========================================================================================
# instant_surplus: Compute instant surplus of each move between locations for each
        worker - firm match
==========================================================================================#
# ! REMOVE
# @everywhere function instant_surplus(prim::Primitives, f::Array{Float64, 3}, b::Array{Float64, 2}, C::Array{Float64, 2})
#     @unpack n_j, n_x, n_y = prim
#     # Pre allocate
#     s_move = zeros(n_j, n_j, n_x, n_y); # s(j➡j', x, y) 
#     for j in 1:n_j # Loop over locations
#         for j_prime in 1:n_j # Loop over locations
#             s_move[j, j_prime, :, :] = (f[j_prime, :, :] .- C[j_prime]) .- (b[j,:] .- C[j])
#         end # end of loop over destination locations
#     end # end of loop over origin locations
#     return s_move 
# end # end of instant_surplus
#==========================================================================================
# update_inner_static_object: Update objects that are static within the inner fixed point
        loop 
==========================================================================================#
# ! REMOVE
# function update_inner_static_objects!(prim::Primitives, res::Results, dist::DistributionsModel)
#     # Compute production in each location for each type of worker and firm
#     f, y_star, b = output(prim, dist);
#     # plot(prim.x_grid, f[:, :,10]', lw = 2, label = reshape(["City $j" for j ∈ 1:prim.n_j], 1, prim.n_j))
#     # Compute congestion cost in each location
#     C = congestion_cost(prim, dist);
#     # Calculate instant surplus of each move
#     s_move = instant_surplus(prim, f, b, C);
#     # Update results
#     res.s_move = copy(s_move)
#     res.b = copy(b)
#     res.f = copy(f)
#     res.C = copy(C[:])
#     res.y_star = copy(y_star)
# end
#?=========================================================================================
#? Dynamic Programming I : Solving Bellmans for Unemployemnt and Surplus
#?=========================================================================================
#==========================================================================================
# compute_Λ_0: Compute the continuation value of unemployed workers  given S and Γ(y|S)
==========================================================================================#
function compute_Λ_0(prim::Primitives, res::Results)
    @unpack n_j, ξ, c, n_y, n_x = prim
    Λ_0 = zeros(n_j, n_j, n_x)
    for j ∈ 1:n_j
        dest_value = zeros(n_j, n_x)
        for y_prime ∈ 1:n_y
            dest_value += max.(0, res.S_move[j, :, :, y_prime]) .* res.γ[:, y_prime]
        end
        Λ_0[j, :, :] = exp.( dest_value .* ξ / c )
    end
    # Compute the optimal strategy of unemployed workers
    ϕ_u = Λ_0 ./ sum(Λ_0, dims=2)
    # Take log
    Λ_0 = log.( dropdims( sum(Λ_0, dims=2), dims=2 ) ) .- log(n_j)
    # Substitue nan with 0
    Λ_0[isnan.(Λ_0)] .= 0
    # Update optimal strategy
    # ! Round to avoid numerical errors
    # ϕ_u = round.(ϕ_u, digits=4)
    res.ϕ_u = copy(ϕ_u)
    return Λ_0
end # end of compute_Λ_0
#==========================================================================================
# compute_Λ_1: Compute the continuation value of employed workers  given S and Γ(y|S)
==========================================================================================#
function compute_Λ_1(prim::Primitives, res::Results)
    @unpack n_j, ξ, c, n_y, n_x, s = prim
    Λ_1 = zeros(n_j, n_j, n_x, n_y)
    ϕ_s = zeros(n_j, n_j, n_x, n_y)
    for j ∈ 1:n_j
        for y ∈ 1:n_y
            dest_value = zeros(n_j, n_x)
            for y_prime ∈ 1:n_y
                dest_value += max.(0, res.S_move[j, :, :, y_prime] .- res.S_move[j, j, :, y]') .* res.γ[:, y_prime]
            end
            Λ_1[j,:, :, y] += exp.( s .* dest_value .* ξ / c )
        end
        ϕ_s[j,:,:,:] = Λ_1[j, :, :, :] ./ sum(Λ_1[j, :, :, :], dims=1)
    end 
    # Update optimal strategy
    # ! Round to avoid numerical errors
    # ϕ_s = round.(ϕ_s, digits=4)
    res.ϕ_s = copy(ϕ_s)
    # Compute the optimal strategy of employed workers
    Λ_1 = log.( dropdims(sum(Λ_1, dims=2), dims=2) ) .- log(n_j)
    # Substitue nan with 0
    Λ_1[isnan.(Λ_1)] .= 0
    return Λ_1
end # end of compute_Λ_1
#==========================================================================================
# compute_surplus: Compute the surplus Bellman equation 
==========================================================================================#
# TODO: Speed up this function
@everywhere function compute_surplus_and_unemployment!(prim::Primitives, res::Results, dist::DistributionsModel; 
                            max_iter::Int64=5000, tol::Float64=1e-6, verbose::Bool=true)
                            # max_iter=5000; tol=1e-6; verbose=true;
    if verbose
        println(@bold @blue "Solving the surplus Dynamic Programming problem")
    end
    # Unpack primitives
    @unpack n_j, n_x, n_y, β, F_bar, c, δ, ω₁, ω₂, ξ = prim
    @unpack f, y_star, b, C, s_move = res
    # Compute matrix of moving cost 
    F = F_bar .* ones(n_j, n_j)
    for j ∈ 1:n_j
        F[j, j] = 0
    end
    # Initialize surplus
    S_new = zeros( size( res.S_move ) );
    # Initiallize error
    err = Inf
    # Set iteration counter
    iter = 1
    # Iterate until convergence
    while (err > tol) & (iter < max_iter)
        # Update Distribution at interim stage
        update_interim_distributions!(prim, res, dist);
        # println(@bold @green "$(sum(dist.u_plus))")
        # Update value of vacancy creation
        get_vacancy_creation_value!(prim, res, dist);
        # Update Market tightness and vacancies
        update_market_tightness_and_vacancies!(prim, res, dist);

        Λ_0 = compute_Λ_0(prim, res)
        Λ_1 = compute_Λ_1(prim, res)

        U_new = (b .- C) .+ β .*( res.U .+ c .* Λ_0 )
        Threads.@threads for j ∈ 1:n_j
            for j_prime ∈ 1:n_j
                if j == j_prime
                    # Determine if S(j → j, x, y) ≥ 0)
                    ind_S = res.S_move[j, j, :, :] .≥ 0
                    S_new[j, j, :, :] = s_move[j,j,:,:].+β.*( (1-δ).* max.(0,res.S_move[j,j,:,:]) .+ c.*(1-δ) .* ind_S .* Λ_1[j,:,:].- Λ_0[j,:] )
                else
                    S_new[j, j_prime, :, :] = res.S_move[j_prime, j_prime, :, :] .- (res.U[j,:].-res.U[j_prime,:].+F[j,j_prime])
                end
            end
        end
        err = max(norm(res.S_move .- S_new) / norm(res.S_move), norm(res.U .- U_new) / norm(res.U))
        # Update U
        res.S_move = copy(S_new)
        res.U = copy(U_new)
        # Update iteration counter
        if verbose
            if iter % 100 == 0
                println(@bold @yellow "Iteration:  $iter, Error: $(round(err, digits=6))")
            elseif err < tol
                println(@bold @green "Iteration:  $iter, Converged!")
            end
        end
        iter += 1
    end # end of while loop
    # Update global objects before leaving function
    # Update Distribution at interim stage
    update_interim_distributions!(prim, res, dist);
    # Update value of vacancy creation
    get_vacancy_creation_value!(prim, res, dist);
    # Update Market tightness and vacancies
    update_market_tightness_and_vacancies!(prim, res, dist);
end # end of compute_surplus
#?=========================================================================================
#? Dynamic Programming II : Updating DistributionsModel
#?=========================================================================================
#==========================================================================================
# update_interim_DistributionsModel! : Updates the distribution of workers and vacancies 
        at the interim stage.
==========================================================================================#
# TODO: Speed up this function
@everywhere function update_interim_distributions!(prim::Primitives, res::Results, dist::DistributionsModel)
    # Unpack parameters
    @unpack δ, n_j, n_x, n_y = prim
    # Unpack results
    @unpack S_move = res
    # Pre-allocate
    u_plus = copy(dist.u);
    h_plus = zeros(size(dist.h));
    for j ∈ 1:n_j
        u_plus[j, :] += sum( ( ( δ .* ( S_move[j, j, :, :] .≥ 0 ) + (S_move[j, j, :, :] .< 0) ) .* dist.h[j, :, :] ), dims = 2)
        h_plus[j, :, :] = ((1 - δ) .* ( S_move[j, j, :, :] .≥ 0 ) ).* dist.h[j, :, :]
    end
    # Update the distribution of unemployed workers (at interim stage)
    dist.u_plus = copy(u_plus);
    # Update the distribution of employed workers (at interim stage)
    dist.h_plus = copy(h_plus);
end # end function update_interim_DistributionsModel!
#==========================================================================================
# get_total_effort : Calculate total search effort on each location as the sum of all 
        unemployed workers + s times the number of employed workers (at interim stage)
==========================================================================================#
# TODO: Speed up this function
@everywhere function get_total_effort(prim::Primitives, dist::DistributionsModel, res::Results)
    # Unpack parameters
    @unpack δ, s, n_j, n_y = prim
    # Preallocate
    L = zeros(n_j)
    # Compute total search effort in each location
    for j in 1:n_j # Loop over locations
        # Total number of unemployed searching for a job in location j
        unemp = sum( dist.u_plus .* res.ϕ_u[:, j, :] )
        # Total number of employed searching for a job in location j
        emp = 0
        for y ∈ 1:n_y
            emp += sum( dist.h_plus[:, :, y] .* res.ϕ_s[:, j, :, y] )
        end
        L[j] =unemp + s .* emp
    end
    return L
end
#==========================================================================================
# get_vacancy_creation_value!: Calculate the value of vacancy creation (for each location 
        - type of firm)
==========================================================================================#
# TODO: Speed up this function
@everywhere function get_vacancy_creation_value!(prim::Primitives, res::Results, dist::DistributionsModel)
    # Unpack parameters
    @unpack n_j, n_x, n_y, s, ξ = prim
    # Get total seach effort
    res.L = get_total_effort(prim, dist, res)
    B_tmp1 = zeros(n_j, n_y)
    B_tmp2 = zeros(n_j, n_y)
    for j ∈ 1:n_j
        for y in 1:n_y
            for x in 1:n_x
                for j_prime ∈ 1:n_j
                    B_tmp1[j, y] += max.(0.0, res.S_move[j_prime,j,x,y]) .* dist.u_plus[j_prime,x] .* res.ϕ_u[j_prime,j,x]
                    for y_prime ∈ 1:n_y
                        B_tmp2[j, y] += max.(0.0, (res.S_move[j_prime,j,x,y] .- res.S_move[j_prime,j_prime,x,y_prime])) .* dist.h_plus[j_prime,x,y_prime] .* res.ϕ_s[j_prime,j,x,y_prime]
                    end
                end
            end
        end
    end
    # Update the value of vacancy creation
    res.B  = (1 - ξ) .* ( B_tmp1 + s .* B_tmp2 ) ./ res.L 
end # end function get_vacancy_creation_value!
#==========================================================================================
# update_market_tightness_and_vacancies!:  Using the value of vacancy creation to compute
        each market tightness and number of vacancies
==========================================================================================#
#* Function is speeded up.
@everywhere function update_market_tightness_and_vacancies!(prim::Primitives, res::Results, dist::DistributionsModel)
    # Unpack parameters
    @unpack ω₁, ω₂, c₀, c₁, n_j, n_y = prim
    @unpack Φ = dist
    @unpack L = res
    IntB =  sum(( res.B ./ c₀ ).^(1/c₁), dims = 2)
    res.θ = ( ω₁^(1.0/(c₁+ω₂)) ) .* ( (IntB./L).^(c₁/(c₁+ω₂)) )
    V = res.θ .* L
    M = ω₁ .* (L.^ω₂) .* (V.^(1.0-ω₂))
    # TODO: Think about (c₀ .* res.C)
    v = ( (M./V)  .* ( res.B ./ c₀ ) ) .^ (1/c₁)
    res.v = v .* Φ'
    res.V = V[:]
    # Probability of a match being formed in each location
    p = M ./ res.L 
    # Distribution of vacancies in each location
    γ = p .* res.v ./ res.V
    # Substitue nan with 0
    γ[isnan.(γ)] .= 0
    # Update the distribution of vacancies
    res.γ = copy(γ)
end # end function update_market_tightness_and_vacancies
#==========================================================================================
# update_distrutions! : updates the distribution of workers across locations, skills and 
        firms at the next period  
==========================================================================================#
# TODO: Speed up this function
@everywhere function update_distrutions!(prim::Primitives, res::Results, dist::DistributionsModel)

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
    for j ∈ 1 : n_j, x ∈ 1 : n_x 
        # Split the mass of unemployed into their destinations
        splited_mass_u = dist.u_plus[j, x] .* res.ϕ_u[j, :, x] .* (res.S_move[j, :, x, :] .> 0) .* γ
        # Assign the mass to their destinations
        h_u[:, x, :] += splited_mass_u
        # Keep of unemployment movement
        u_move[j, :, :] += splited_mass_u
        # Remove the mass from the unemployed
        new_u[j, x] = dist.u_plus[j, x] - sum(splited_mass_u)
        for y ∈ 1 : n_y
            # Split the mass of employed into their destinations
            splited_mass_s = dist.h_plus[j, x, y] .* res.ϕ_s[j, :, x, y, :] .* (res.S_move[j, :, x, :] .- res.S_move[j, j, x, y] .> 0) .* γ
            # Keep track of employment movement
            h_move[j, :, y, :] += splited_mass_s
            # Assign the mass to their destinations
            h_p[:, x, :] += splited_mass_s
            # Remove the mass from the employed
            h_r[j, x, y] = dist.h_plus[j, x, y] - sum(splited_mass_s)
        end
    end

    # Store movement of workers
    res.u_move = copy(u_move)
    res.h_move = copy(h_move)

    # h_move ./ sum( h_p + h_u ,  dims = 2)
    # Add employment into a single distribution
    new_h = h_u + h_p + h_r
    
    # Compute poahing index Hires from Poaching / Total Hires
    res.Π = dropdims( sum(h_p, dims = 2)./ sum( h_p + h_u ,  dims = 2), dims = 2)

    # Update the distribution of unemployed workers
    dist.u = copy(new_u);
    # Update the distribution of employed workers
    dist.h = copy(new_h);
    # Update the distribution of types across locations
    # Update interim distributions
    update_interim_distributions!(prim, res, dist);
    # Update distributions to be be interim
    dist.u = copy(dist.u_plus);
    dist.h = copy(dist.h_plus);
    # Compute the distribution of types across locations
    new_ℓ = dropdims(sum(dist.h, dims = 3), dims = 3) + dist.u
    # Update the distribution of types across locations
    dist.ℓ = copy(new_ℓ);
end # End of function update_distrutions!
#?=========================================================================================
#? Putting it all together
#?=========================================================================================
#==========================================================================================
# iterate_distributions!: Iterates the model from the initial distribution until convergence
==========================================================================================#
# TODO: Speed up this function
@everywhere function iterate_distributions!(prim::Primitives, res::Results, dist::DistributionsModel; 
    max_iter::Int64=5000, tol::Float64=1e-6, verbose::Bool=true, store_path::Bool=false)
    # max_iter=5000; tol=1e-16; verbose=true; store_path=true;
    
    # Initialize error and iteration counter
    err = Inf
    iter = 1
    if store_path
        dists = []
    end
    if verbose
        println(@bold @blue "Solving the model")
    end
    
    # Collect error size of 1 and unemployment rate in convergence path
    μs = []
    errs = []
    urates = []

    n = 10
    results_u = zeros(n, prim.n_j, prim.n_x);
    results_h = zeros(n, prim.n_j, prim.n_x, prim.n_y);

    # begin
    while (err > tol) & (iter < max_iter)
        if store_path
            # push!(dists, dist.ℓ')
            push!(dists, dist.u')
        end
        update_inner_static_objects!(prim, res, dist);
        compute_surplus_and_unemployment!(prim, res, dist, verbose=false);
        # Save distributions
        u_initial = copy(dist.u);
        h_initial = copy(dist.h);
        ℓ_initial = copy(dist.ℓ);
        update_distrutions!(prim, res, dist)

        # Compute error
        err = max(norm(u_initial - dist.u) / norm(u_initial), norm(h_initial - dist.h) / norm(h_initial));
        sizes =  round.(sum(dist.ℓ, dims = 2), digits = 4)[:];
        if verbose
            if iter % 50 == 0
                println(@bold @yellow "Iter $(iter) -- Error: $(round(err, digits = 10))")
                println(@bold @yellow "Sizes: $sizes")
                println(@bold @red "Unemployment rate: $( round(100 * sum(dist.u), digits = 2))")
                # Print city sizes
            elseif err < tol
                println(@bold @green "Iteration:  $iter, Converged!")
            end
        end
        iter += 1
        # Save error and unemployment rate
        push!(errs, err)
        push!(urates, sum(dist.u))
        # Compute size of 1
        μ = sum(dist.ℓ, dims = 2)[1]
        push!(μs, μ);

        # # Save distributions
        results_u[ iter % n + 1 , :, :] = copy(dist.u);
        results_h[ iter % n + 1 , :, :, :] = copy(dist.h);

        

        # If enough iterations have passed, update distributions with the average of the last n iterations
        if iter > n
            dist.u = dropdims( mean(results_u, dims = 1), dims = 1)
            dist.h = dropdims( mean(results_h, dims = 1), dims = 1)
            dist.ℓ = dist.u .+ dropdims( sum(dist.h, dims = 3), dims = 3)
        end


    end
    if err > tol
        # if verbose
            println(@bold @red "Did not converge! Error: $(round(err, digits = 10)), Iteration: $(iter)")
        # end
    end
    if store_path
        return dists
    else 
        if verbose
            print(@bold @red "No path stored, use store_path = true to store the path of convergence")
        end
    end
    return μs, errs, urates
end # end of iterate_distributions!
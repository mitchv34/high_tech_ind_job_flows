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
@everywhere @with_kw struct Primitives
    # Primitives
    β           ::Float64          # Discount factor
    c           ::Float64          # Cost of search
    δ           ::Float64          # Separation rate
    s           ::Float64          # Search intensity of employed workers
    μ           ::Float64          # Worker's bargaining power
    # Cost of living
    θ           ::Float64          # Cost of living (scale)
    γ           ::Float64          # Cost of living (curvature)
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
        new(U, ϕ_u, ϕ_s, S_move, B, θ, v, V, L)
    end # end of constructor of Results
end # end of Results
#==========================================================================================
# DistributionsModel: Strucutre to store the distribution of populations across locations,
        distribution of unemployed workers across locations and skills, and the distribution
        of employed workers across locations, skills and firms.
==========================================================================================#
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
@everywhere function read_primitives(path_to_params::AbstractString)::Primitives
    # Read YAML file
    data = YAML.load_file(path_to_params)

    # Create x_grid, y_grid, z_grid 
    x_grid = range(data["grids"]["x_min"], data["grids"]["x_max"], length=data["grids"]["n_x"])
    y_grid = range(data["grids"]["x_min"], data["grids"]["x_max"], length=data["grids"]["n_y"])
    z_grid = [1.0] #! This is just a placeholder for now
    # Create j_grid
    j_grid = range(1, data["grids"]["n_j"], length=data["grids"]["n_j"])

    # Create production function
    params = data["primitives"]["function"]["params"] # Get parameters of the function
    functional = data["primitives"]["function"]["functional_form"] # Get functional in string form 

    # function fcnFromString(s) # Create function from string
    #     f = eval(Meta.parse("(x, y) -> " * s))
    #     return (x, y) -> Base.invokelatest(f, x, y)
    # end

    # f = fcnFromString(functional) # Create function from string

    # TODO: Figure out how to create a function from a string
    # ! I'm manually creating the function for now
    f = (x,y) -> x.^params[1] .* y'.^(1-params[1])
    # f = (x,y) -> (params[1] .+ params[2] .* x .+ params[3] .* y' .+ params[4] .* x.^2 + params[5] .* (y').^2 .+ params[6] .* x .* y') # Functional form of production function 
    # f = (x,y) -> (params[1] .* x.^((1-params[2])/params[2]) .+ (1-params[1]) .* (y').^((1-params[2])/params[2])).^(params[2]/(1-params[2]))

    # Create Primitives struct
    Primitives(
        β = data["primitives"]["beta"],
        c = data["primitives"]["c"],
        δ = data["primitives"]["delta"],
        s = data["primitives"]["s"],
        μ = data["primitives"]["mu"],
        θ = data["primitives"]["theta"],
        γ = data["primitives"]["gamma"],
        output = f,
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
@everywhere function init_model(path_to_params::AbstractString)
    # Generate primitives
    prim = read_primitives(path_to_params)
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
@everywhere function idea_exchange(prim::Primitives, dist::DistributionsModel)
    # Unpack DistributionsModel
    @unpack ℓ = dist
    # Unpack primitives
    @unpack x_grid, ν, n_j = prim
    # Compute each location's share of the population (μⱼ)
    μ = sum(ℓ, dims=2)
    # Compute the average skill level in each location 
    x̄ = [(μ[j] > 0) ? sum(ℓ[j, :] .* x_grid) ./ μ[j] : 0.0 for j ∈ 1:n_j]
    # Compute the value of idea exchange in each location
    X = (1  .- exp.(-ν .* μ)) .* x̄
    return X
end # end of idea_exchange
#==========================================================================================
# worker_productivity: A function that computes the producivity of each type of worker
        in each location
==========================================================================================#
@everywhere function worker_productivity(prim::Primitives, dist::DistributionsModel)
    # Unpack primitives
    @unpack A, x_grid = prim
    # Get value of idea exchange in each location
    X = idea_exchange(prim, dist)
    Ω = x_grid' .* (1 .+ A .* X .* x_grid') 
    return Ω
end # end of worker_productivity
#==========================================================================================
# output: A function that computes the value of output in each location for each type of
        worker and firm
==========================================================================================#
@everywhere function output(prim::Primitives, dist::DistributionsModel)
    # Unpack primitives
    @unpack  n_j, n_x, n_y, y_grid = prim
    output_funct = prim.output
    # Get productivity of each type of worker in each location
    Ω = worker_productivity(prim, dist)
    # Pre-allocate output
    Y = zeros(n_j, n_x, n_y)
    # Compute output
    for j in 1:n_j # Loop over locations
        # Compute output
        Y[j, :, :] = z .* output_funct(Ω[j, :],  y_grid)
    end # end of loop over locations
    return Y
end # end of output
#==========================================================================================
# home_production: Compute the value of home production in each location for each type of 
        worker
==========================================================================================#
@everywhere function home_production(prim::Primitives, dist::DistributionsModel)
    # Unpack primitives
    @unpack b_hat = prim
    f = output(prim, dist)
    f_y_star = dropdims(maximum(f, dims=3), dims=3)
    return b_hat .* f_y_star
    # return b_hat .* ones(prim.n_j, prim.n_x)
end # end of home_production
#==========================================================================================
# congestion_cost: A function that computes the cost of living in each location
==========================================================================================#
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
@everywhere function instant_surplus(prim::Primitives, f::Array{Float64, 3}, b::Array{Float64, 2}, C::Array{Float64, 2})
    @unpack n_j, n_x, n_y = prim
    # Pre allocate
    s_move = zeros(n_j, n_j, n_x, n_y); # s(j➡j', x, y) 
    for j in 1:n_j # Loop over locations
        for j_prime in 1:n_j # Loop over locations
            s_move[j, j_prime, :, :] = f[j_prime, :, :] .- (b[j,:] .- C[j])
        end # end of loop over destination locations
    end # end of loop over origin locations
    return s_move 
end # end of instant_surplus
#?=========================================================================================
#? Dynamic Programming I : Solving Bellmans for Unemployemnt and Surplus
#?=========================================================================================
#==========================================================================================
# optimal_strategy: Compute the optimal strategy of workers in each location 
==========================================================================================#
@everywhere function optimal_strategy!(prim::Primitives, res::Results)
    # Unpack primitives
    @unpack c, n_j, n_y, F_bar, ω₁, ω₂, c = prim
    @unpack L, V, v = res
    # Compute the optimal strategy of workers in each location
    # Note that ϕ_u is a (n_j x n_x) matrix where each row is a location and each 
    # column is a skill level, the elements of are the probability that a worker
    # of a given skill level will give to search for a job in each location
    # Calculate total number of mathces in each location
    M = min.(ω₁ .* L.^ω₂ .* V.^(1-ω₂), L, V)
    p = M ./ L # Probability of a match being formed in each location
    # Compute the contact rates of each location
    for j ∈ 1:n_j # Loop over locations
        # Compute the moving cost vector 
        exp_term = exp.(dropdims(sum(permutedims(p .* max.(0, res.S_move[j, :, :, :]), [1,3,2]) .* v ./ V, dims = 2) , dims = 2) ./ c )
        # Replace nan with 1
        exp_term[isnan.(exp_term)] .= 1
        res.ϕ_u[j, :, :] =  exp_term ./ sum(exp_term, dims = 1) 
        for y ∈ 1:n_y   
            exp_term = exp.(dropdims(sum(permutedims(p .* max.(0,res.S_move[j, :, :, :] .- res.S_move[j, j, :, y]' ), [1,3,2]) .* v ./ V, dims = 2) , dims = 2) ./ c )
            exp_term[isnan.(exp_term)] .= 1
            res.ϕ_s[j, :, :, y] = exp_term ./ sum(exp_term, dims = 1)
        end
    end
end # end of optimal_strategy
#==========================================================================================
# compute_surplus: Compute the surplus Bellman equation 
==========================================================================================#
@everywhere function compute_surplus_and_unemployment!(prim::Primitives, res::Results, dist::DistributionsModel; 
                            max_iter::Int64=5000, tol::Float64=1e-6, verbose::Bool=true)
    
    if verbose
        println(@bold @blue "Solving the surplus Dynamic Programming problem")
    end
    # Unpack primitives
    @unpack n_j, n_x, n_y, β, F_bar, c, δ, ω₁, ω₂, μ = prim
    @unpack L, V, v = res
    # Calcualte production in each location for each type of worker and firm
    f = output(prim, dist);
    # Calculate the value of unemployment in each location for each type of worker
    b = home_production(prim, dist);
    # Compute the cost of living in each location
    C = congestion_cost(prim, dist);
    # Calculate instant surplus of each move
    s_move = instant_surplus(prim, f, b, C);
    # Calculate total number of matches in each location
    M = min.(ω₁ .* L.^ω₂ .* V.^(1-ω₂), L, V)
    p = M ./ L # Probability of a match being formed in each location
    # Initiallize error
    err = Inf
    # Set iteration counter
    iter = 1
    # Iterate until convergence
    while (err > tol) & (iter < max_iter)
        U_new = SharedArray{Float64}(size( res.U ) );
        S_new = SharedArray{Float64}(size( res.S_move ) );
        # Pre-allocate new values
        # Compute continuation value
        @sync @distributed for j ∈ 1:n_j # Loop over locations (origin)
            exp_term = exp.(dropdims(sum( permutedims(p .* μ .* max.(0, res.S_move[j, :, :, :]), [1,3,2]) .* v ./ V, dims = 2) , dims = 2) ./ c )
            # Replace nan with 1
            exp_term[isnan.(exp_term)] .= 1
            cont_val = β .* (res.U[j, :]' + c .* log.( sum(exp_term, dims = 1) ) .- c .* log(n_j) )
            U_new[j, :] =( b[j, :] .- C[j])' .+ cont_val
            # @show U_new[j, :] 
            # Comppute the moving cost vector 
            F = F_bar .* ones(n_j) # Moving cost vector initialized at F_bar
            F[j] = 0 # Moving cost from location j to location j is zero
            for j_prime ∈ 1:n_j # Loop over locations (destination)
                if j == j_prime # Compute non movers value 
                    for y ∈ 1:n_y # Loop over types
                        exp_term_2 = exp.(dropdims(sum(permutedims(p .* μ .* max.(0, res.S_move[j, :, :, :] .- res.S_move[j, j, :, y]'), [1,3,2]) .* v ./ V, dims = 2) , dims = 2) ./ c )
                        # Replace nan with 1
                        exp_term_2[isnan.(exp_term_2)] .= 1
                        # Compute the continuation value
                        cont_val = β .* ( (1 - δ) .* max.(0, res.S_move[j, j, :, y] )' .+ c .* ( log.(sum(exp_term_2, dims=1)) .- log.(sum(exp_term, dims = 1)) ) )
                        S_new[j, j, :, y] = cont_val' .+ s_move[j, j, :, y]
                    end
                else # Compute movers value
                    S_new[j, j_prime, :, :] = res.S_move[j_prime, j_prime, :, :] .+ (res.U[j_prime, :] .- U_new[j, :] .- F[j_prime])
                end
            end # end of loop over destination locations
        end # end of loop over origin locations

        # Compute the error
        # err = maximum(abs.(S_non_move_new .- S_non_move))
        err = max(norm(res.S_move .- S_new), norm(res.U .- U_new)) 
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
end # end of compute_surplus
#?=========================================================================================
#? Dynamic Programming II : Updating DistributionsModel
#?=========================================================================================
#==========================================================================================
# update_interim_DistributionsModel! : Updates the distribution of workers and vacancies 
        at the interim stage.
==========================================================================================#
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
@everywhere function get_vacancy_creation_value!(prim::Primitives, res::Results, dist::DistributionsModel)
    # Unpack parameters
    @unpack n_j, n_x, n_y, s, μ = prim
    # Get total seach effort
    L = get_total_effort(prim, dist, res)
    B = zeros(n_j, n_y) # Pre-allocate
    for j in 1:n_j # Loop over locations (destination)
        for j_prime ∈ 1:n_j # Loop over locations (origin)
            # Add value of matches with unemployed workers 
            # Only positive surplus  moves between locations j and j_prime add value
            match_surv = (1 - μ) .* max.(0, res.S_move[j_prime, j, :, :]) 
            # Value of matches with unemployed workers
            s_from_unemp = sum(res.ϕ_u[j_prime, j, :] .* dist.u_plus[j_prime, :] .* match_surv, dims=1) / L[j]
            # added value of matches with unemployed workers
            B[j, :] += s_from_unemp' 
            # Add value of poaching other firms 
            for y ∈ 1:n_y # Loop over types of firms to find poaching opportunities
                # Poaching condition is surplus of moving is greater that surplus of staying
                worker_poached =  (1 - μ) .* max.(0, res.S_move[j_prime, j, :, y] .- res.S_move[j_prime, j_prime, :, :])
                # Value of matches with employed workers
                s_from_poaching = s * sum(res.ϕ_s[j_prime, j, :, y] .* dist.h_plus[j_prime, :, y] .* worker_poached) / L[j] 
                # added value of matches formed from poaching other firms' workers
                B[j, y] += s_from_poaching
            end # end loop over types of firms
        end # end loop over locations 
    end # end loop over locations
    # Update the value of vacancy creation
    res.B = copy(B);
end # end function get_vacancy_creation_value!
#==========================================================================================
# update_market_tightness_and_vacancies!:  Using the value of vacancy creation to compute
        each market tightness and number of vacancies
==========================================================================================#
@everywhere function update_market_tightness_and_vacancies!(prim::Primitives, res::Results, dist::DistributionsModel)
    # Unpack parameters
    @unpack ω₁, ω₂, c₀, c₁ = prim
    @unpack Φ = dist
    # Compute total search effort
    res.L = get_total_effort(prim, dist, res)
    # Market tightness
    res.θ = ( sum( (ω₁.* res.B ./ c₀).^(1/c₁) .* Φ', dims=2) ./ res.L).^(c₁/(ω₂ + c₁)) 
    # Number of vacancies 
    res.v = ((ω₁ .* (res.θ.^(-ω₂)) .* res.B )./ c₀).^(1/c₁) ./ res.L
    # Replace NaN with zero
    res.v[isnan.(res.v)] .= 0
    # Weight vacancies by the mass of firms of each type
    res.v = res.v .* Φ'
    # Compute total number of vacancies in each location
    res.V = sum(res.v, dims=2)[:]
    # Updae total search effort in each location
end # end function update_market_tightness_and_vacancies
#==========================================================================================
# update_distrutions! : updates the distribution of workers across locations, skills and 
        firms at the next period  
==========================================================================================#
@everywhere function update_distrutions!(prim::Primitives, res::Results, dist::DistributionsModel)
    @unpack ω₁, ω₂, n_j, n_x, n_y, s = prim
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
    # Update distribiutions:
    u_next = zeros(n_j, n_x);
    h_hired = zeros(n_j, n_x, n_y);
    # Compute the mass of workers that each firm in each location hires from unemployment
    for j ∈ 1:n_j # Loop over locations (destinations)
        for x ∈ 1:n_x # Loop over skills
            # Total mass of x workers that arrive to location j from all other locations
            arrive = dist.u_plus[:, x] .* res.ϕ_u[:, j, x]
            # Probability of that ech firm in location j offers a job to a worker of type x coming from any other location
            prob_hire = η[:, j, x, :] .* (res.v[j, :] / res.V[j])' .* p[j]
            # Replace NaN with zero
            prob_hire[isnan.(prob_hire)] .= 0
            # Compute the mass of x workers that each firm is able to hire from unemployment in location j
            h_hired[j, x, :]  = sum( prob_hire .* arrive, dims=1 )            
        end # Loop over skills
    end # Loop over locations (origins)

    # x = 10
    # arrive_1 = dist.u_plus[:, x] .* res.ϕ_u[:, 1, x]
    # arrive_2 = dist.u_plus[:, x] .* res.ϕ_u[:, 2, x]

    # prob_hire_1 = η[:, 1, x, :] .* (res.v[1, :] / res.V[1])' .* p[1]
    # prob_hire_2 = η[:, 2, x, :] .* (res.v[2, :] / res.V[2])' .* p[2]

    # h_hired_1 = sum( prob_hire_1 .* arrive_1, dims=1 )
    # h_hired_2 = sum( prob_hire_2 .* arrive_2, dims=1 )

    # sum(h_hired_1)
    # sum(h_hired_2)

    # sum(dist.h_plus[:, x, :], dims=2)

    # Compute the mass of workers that are not hired from unemployment for each location and skill
    for j ∈ 1:n_j # Loop over locations
        for x ∈ 1:n_x # Loop over skills
            # Mass of workers that migrate from location j
            migrate = dist.u_plus[j, x] .* res.ϕ_u[j, :, x]
            # Probability that each woker is hired from unemployment in each location
            prob_hire = η[j, :, x, :] .* (res.v ./ res.V) .* p
            # Replace NaN with zero
            prob_hire[isnan.(prob_hire)] .= 0
            # Compute the mass of workers that are not hired from unemployment for each location and skill
            u_next[j, x] = dist.u_plus[j,x] - sum(migrate .* prob_hire)
        end # Loop over skills
    end # Loop over locations
    # Check that the mass of workers in unemployement in interim stage is the same as the mass of workers in unemployment 
    # plus the mass of workers that are hired from unemployment in the next stage
    @assert sum(dist.u_plus) ≈ sum(u_next) + sum(h_hired) @bold @red "Error!: uₜ₊ ≂̸ uₜ₊₁ + hᵤ where hᵤ is the mass of workers hired from unemployment"
    sum(dist.u_plus, dims=2)
    sum(u_next, dims = 2) + sum(h_hired, dims=[2, 3] )
    # For each location, skill and firm compute the probability of not being poeached by any other firm in any other location
    # TODO : Optimize this loop
    h_retained = zeros(n_j, n_x, n_y); 
    for j ∈ 1:n_j # Loop over locations
        for x ∈ 1:n_x # Loop over skills
            for y ∈ 1:n_y # Loop over firms
                # Probability of a worker being poached by any other firm in any other location
                prob_poach = s .* p .* res.v ./ res.V .* η_move[j, :, x, y, :] .* res.ϕ_s[:, j, x, y]
                # Fill NaN with zero
                prob_poach[isnan.(prob_poach)] .= 0
                # Probability of a worker not being poached from any other location
                prob_no_poach = 1 .- sum(prob_poach)
                # Probability of a worker not being poached from any other location
                h_retained[j, x, y] = dist.h_plus[j, x, y] .* prob_no_poach
            end # Loop over firms
        end # Loop over skills
    end # Loop over locations
    # Compute the mass of workers that each firm in each location is able to poach from other firms in other locations
    h_poached = zeros(n_j, n_x, n_y)
    poached_from_self = 0
    # TODO : Optimize this loop
    for j ∈ 1:n_j # Loop over locations
        for x ∈ 1:n_x # Loop over skills
            for y ∈ 1:n_y # Loop over firms
                # Probability of a worker being poached by any other firm in any other location
                prob_poach = (s .* p[j] .* res.v[j, y] ./ res.V[j] ) .* η_move[:, j, x, :, y]
                # Fill NaN with zero
                prob_poach[isnan.(prob_poach)] .= 0
                # Mass that each firm is able to poach from other firms in other locations(sum over locations)
                mass_poach = sum( prob_poach .* dist.h_plus[:, x, :] .* res.ϕ_s[j, :, x, :])
                # Add mass to the corresponding location- skill -firm
                h_poached[j, x, y] = mass_poach
                # Compute the mass of workers that the firm poaches from itself
                # poached_from_self += prob_poach[j, y] .* dist.h_plus[j, x, y] .* res.ϕ_s[j, x, y]
            end # Loop over firms
        end # Loop over skills
    end # Loop over locations
    # Check that the mass of workers employed at interim stage is the same as the mass of workers that are retained or poached
    @assert sum(dist.h_plus) ≈ sum(h_poached) + sum(h_retained) @bold @red "Error!: hₜ₊ ≂̸ hₜ₊₁ + hₚ + hᵣ where hₚ is the mass of workers poached and hᵣ is the mass of workers retained"
    sum(h_retained, dims=[2, 3])
    sum(h_retained, dims=[2, 3]) + sum(h_poached, dims=[2, 3])
    # Update the distribution of unemployed and employed workers
    dist.u = copy(u_next)
    sum(dist.u, dims=2)
    dist.h = h_retained + h_poached + h_hired;
    sum(dist.h, dims=[2, 3])
    # Update the distribution of workers across locations
    dist.ℓ = dist.u  + dropdims(sum(dist.h, dims=3), dims=3)
end # End of function update_distrutions!
#?=========================================================================================
#? Putting it all together
#?=========================================================================================
#==========================================================================================
# iterate_distributions!: Iterates the model from the initial distribution until convergence
==========================================================================================#
@everywhere function iterate_distributions!(prim::Primitives, res::Results, dist::DistributionsModel; 
    max_iter::Int64=5000, tol::Float64=1e-6, verbose::Bool=true, store_path::Bool=false)
    
    # Initialize error and iteration counter
    err = Inf
    iter = 1
    if store_path
        dists = []
    end

    while (err > tol) & (iter < max_iter)
        if store_path
            push!(dists, dist.ℓ')
        end
        # Update Distribution at interim stage
        update_interim_distributions!(prim, res, dist);
        # Update value of vacancy creation
        get_vacancy_creation_value!(prim, res, dist);
        # Update Market tightness and vacancies
        update_market_tightness_and_vacancies!(prim, res, dist);
        # Update surplus and unemployment
        compute_surplus_and_unemployment!(prim, res, dist, verbose=false);
        # Solve optimal strategies
        optimal_strategy!(prim, res); 
        # Store t - 1 distributions
        u_initial = copy(dist.u);
        h_initial = copy(dist.h);
        # Update Distribution at next stage
        update_distrutions!(prim, res, dist);
        # Compute error
        err = maximum(abs.(dist.u - u_initial)) + maximum(abs.(dist.h - h_initial))
        if iter % 50 == 0
            println(@bold @yellow "Iteration:  $iter, Error: $(round(err, digits=10))")
            # Print city sizes
            println(@bold @yellow "City sizes:  $(round.(sum(dist.ℓ, dims=2), digits=3))")
        elseif err < tol
            println(@bold @green "Iteration:  $iter, Converged!")
        end
        iter += 1
    end
    if store_path
        return dists
    else 
        return "No path stored, use store_path = true to store the path of convergence"
    end
end

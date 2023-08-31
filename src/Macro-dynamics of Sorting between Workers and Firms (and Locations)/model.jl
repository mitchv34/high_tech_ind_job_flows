#==========================================================================================
Title: Macro-dynamics of Sorting between Workers and Firms (and Locations)
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2023-08-26
Description: 
==========================================================================================#
#==========================================================================================
# * Packages 
==========================================================================================#
using LinearAlgebra
using Parameters
using YAML
using Term 
using Distributions
#?=========================================================================================
#? Structures
#?=========================================================================================
#==========================================================================================
# Parameters: Strucutre to store all parameters of the model 
==========================================================================================#
@with_kw struct Primitives
    # Primitives
    # TODO: Add description of each parameter
    β           ::Float64          #
    c           ::Float64          #
    δ           ::Float64          #
    s           ::Float64          #
    # Cost of living
    θ           ::Float64          #
    γ           ::Float64          #
    # Production
    output      ::Function         # Poduction function
    A           ::Float64          #
    # α           ::Float64          #
    ν           ::Float64          #
    b_hat       ::Float64          #
    # Vacancy creation cost function
    c₀          ::Float64          #
    c₁          ::Float64          #
    # Matching function
    ω₁          ::Float64          #
    ω₂          ::Float64          #
    # Fixed cost of moving locations
    F_bar       ::Float64          #
    # Grids
    n_x         ::Int64            #
    n_y         ::Int64            #
    n_z         ::Int64            #
    n_j         ::Int64            #
    x_grid      ::Array{Float64,1} #
    y_grid      ::Array{Float64,1} #
    z_grid      ::Array{Float64,1} #
    j_grid      ::Array{Int64,1}   #
end # end of Parameters
#==========================================================================================
# Results: Strucutre to store all results of the model 
==========================================================================================#
mutable struct Results
    # * Note that i'm ignoring the shocks since im leaving it fixed at 1
    # * In the general version of the model one extra dimension is needed in all results
    # Unemployment value function Uʲₜ(x) (n_j × n_x × n_z) 
    U          ::Array{Float64,2} 
    # Optimal strategy of each unemployed worker in each location ϕᵤʲₜ(x) (n_j × n_x × n_z)
    ϕ_u        ::Array{Float64,2}
    # Optimal strategy of each employed worker in each location ϕₛʲₜ(x) (n_j × n_x × n_y × n_z)
    ϕ_s        ::Array{Float64,3}
    # Surplus of each move between locations for each worker - firm match Sₜ(j → j', x, y) (n_j × n_j × n_x × n_y × n_z)
    S_move     ::Array{Float64,4}
    # Value of vacancy creation in each location for each type of firm (n_j × n_y × n_z)
    B          ::Array{Float64,2}
    # Market tightness (n_j × n_y × n_z)
    θ          ::Array{Float64, 2} 
    # Number of vacancies (n_j × n_y × n_z)
    v          ::Array{Float64, 2}
    # Constructor
    function Results(prim::Primitives)
        @unpack n_j, n_x, n_y, n_z = prim
        # Pre-allocate unemployment value function
        U = zeros(n_j, n_x) #!, n_z)
        # Pre-allocate optimal strategy of each unemployed worker in each location
        ϕ_u = zeros(n_j, n_x) #!, n_z)
        # Pre-allocate optimal strategy of each employed worker in each location
        ϕ_s = zeros(n_j, n_x, n_y) #!, n_z)
        # Pre-allocate surplus of each move between locations for each worker - firm match
        S_move = zeros(n_j, n_j, n_x, n_y) #!, n_z)
        # Pre-allocate value of vacancy creation in each location for each type of firm
        B = zeros(n_j, n_y) #!, n_z)
        # Pre-allocate market tightness
        θ = zeros(n_j, n_y) #!, n_z)
        # Pre-allocate number of vacancies
        v = zeros(n_j, n_y) #!, n_z)
        new(U, ϕ_u, ϕ_s, S_move, B, v, θ)
    end # end of constructor of Results
end # end of Results
#==========================================================================================
# DistributionsModel: Strucutre to store the distribution of populations across locations,
        distribution of unemployed workers across locations and skills, and the distribution
        of employed workers across locations, skills and firms.
==========================================================================================#
mutable struct DistributionsModel
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
    # Constructor
    function DistributionsModel(ℓ_total::Array{Float64, 1}, ℓ::Array{Float64, 2}, u::Array{Float64, 2}, h::Array{Float64, 3})
        # Create interim DistributionsModel as zeros arrays size of the initial DistributionsModel
        u_plus = zeros(size(u))
        h_plus = zeros(size(h))
        new(ℓ_total, ℓ, u, u_plus, h, h_plus)
    end # end of constructor of DistributionsModel
end # end of DistributionsModel
#?=========================================================================================
#? Model Initialization 
#?=========================================================================================
#==========================================================================================
# read_primitives: A function that reads the primitives of the model from a YAML file
==========================================================================================#
function read_primitives(path_to_params::AbstractString)::Primitives
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


    # Create Primitives struct
    Primitives(
        β = data["primitives"]["beta"],
        c = data["primitives"]["c"],
        δ = data["primitives"]["delta"],
        s = data["primitives"]["s"],
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
        j_grid = collect(j_grid)
    )
end # end of read_primitives
#==========================================================================================
# init_model: A function that initializes the model 
==========================================================================================#
function init_model(path_to_params::AbstractString)
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

function idea_exchange(prim::Primitives, dist::DistributionsModel)
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
function worker_productivity(prim::Primitives, dist::DistributionsModel)
    # Unpack primitives
    @unpack A, x_grid = prim
    # Get value of idea exchange in each location
    X = idea_exchange(prim, dist)
    B = x_grid' .* (1 .+ A .* X .* x_grid') 
    return B
end # end of worker_productivity
#==========================================================================================
# output: A function that computes the value of output in each location for each type of
        worker and firm
==========================================================================================#
function output(prim::Primitives, dist::DistributionsModel)
    # Unpack primitives
    @unpack  n_j,n_x, n_y, x_grid, y_grid = prim
    output_funct = prim.output
    # Get productivity of each type of worker in each location
    B = worker_productivity(prim, dist)
    # Pre-allocate output
    Y = zeros(n_j, n_x, n_y)
    # Compute output
    for j in 1:n_j # Loop over locations
        x_j = B[j, :] # Get productivity of workers in location j
        # Compute output
        Y[j, :, :] = z .* output_funct(B[j, :],  y_grid)
    end # end of loop over locations
    return Y
end # end of output
#==========================================================================================
# home_production: Compute the value of home production in each location for each type of 
        worker
==========================================================================================#
function home_production(prim::Primitives, dist::DistributionsModel)
    # Unpack primitives
    @unpack b_hat = prim
    f = output(prim, dist)
    f_y_star = dropdims(maximum(f, dims=3), dims=3)
    return b_hat .* f_y_star
end # end of home_production
#==========================================================================================
# congestion_cost: A function that computes the cost of living in each location
==========================================================================================#
function congestion_cost(prim::Primitives, dist::DistributionsModel)
    # Unpack DistributionsModel
    @unpack ℓ = dist
    # Unpack primitives
    @unpack θ, γ = prim
    # Compute each location's share of the population (μⱼ)
    μ = sum(ℓ, dims=2)
    # Compute the cost of living in each location
    C = θ .* (μ .^ γ)
    # return C
    return zeros(size(C)) # For now I'm setting the cost of living to zero
end # end of congestion_cost
#==========================================================================================
# instant_surplus: Compute instant surplus of each move between locations for each
        worker - firm match
==========================================================================================#
function instant_surplus(prim::Primitives, f::Array{Float64, 3}, b::Array{Float64, 2}, C::Array{Float64, 2})
    @unpack n_j, n_x, n_y = prim
    # Pre allocate
    s_move = zeros(n_j, n_j, n_x, n_y); # s(j➡j', x, y) 
    # TODO: Fix those ulgy for loops
    for j in 1:n_j # Loop over locations
        for j_prime in 1:n_j # Loop over locations
            for x in 1:n_x # Loop over skills
                for y in 1:n_y # Loop over productivity
                    s_move[j, j_prime, x, y] = f[j_prime, x, y] - (b[j,x] - C[j])
                end # end of loop over productivity
            end # end of loop over skills
        end # end of loop over destination locations
    end # end of loop over origin locations
    return s_move 
end # end of instant_surplus
#==========================================================================================
# compute_Λ: Compute what I call Λʲ(x) in the notes 
==========================================================================================#
function compute_Λ(prim::Primitives, b::Array{Float64, 2})::Array{Float64, 2}
    # Unpack primitives
    @unpack n_j, n_x, β, c = prim
    # Pre-allocate Λ
    Λ = zeros(n_j, n_x)
    for j in 1:n_j # Loop over locations
        for x in 1:n_x # Loop over skills
            Λ[j, x] = β .* c .* (-log(n_j) .+ log(sum( (exp.(b[:, x] .- b[j, x]) ./ c))))
        end
    end
    return Λ
end 
#?=========================================================================================
#? Dynamic Programming I : Solving Bellmans for Unemployemnt and Surplus
#?=========================================================================================
#==========================================================================================
# compute_unemployment_value: A function that computes the value of unemployment in each 
        location to each type of worker by solving the Bellman equation using value 
        functio iteration
==========================================================================================#
function compute_unemployment_value!(prim::Primitives, res::Results, dist::DistributionsModel; 
                                        max_iter::Int64=5000, tol::Float64=1e-6, verbose::Bool=true)
    if verbose
        println(@bold @blue "Solving the unemployment value function")
    end
    # Unpack primitives
    @unpack n_j, n_x, β, c  = prim

    # Calcualte production in each location for each type of worker and firm
    f = output(prim, dist);
    # Compute the value of unemployment in each location for each type of worker
    b = home_production(prim, dist);
    # Compute the cost of living in each location
    C = congestion_cost(prim, dist);

    err = Inf
    # Set iteration counter
    iter = 1
    # Iterate until convergence
    while (err > tol) & (iter < max_iter)
        # Compute the value of unemployment in each location for each type of worker
        U_new = b .- C  .+  β .* c .* ( log.( sum( exp.(res.U ./ c)  , dims = 1 )  ) .- log(n_j))
        # Compute the error
        err = maximum(abs.(U_new .- res.U))
        # Update U
        res.U = copy(U_new)
        if verbose
            if iter % 100 == 0
                println(@bold @yellow "Iteration:  $iter, Error: $(round(err, digits=6))")
            elseif err < tol
                println(@bold @green "Iteration:  $iter, Converged!")
            end
        end
        # Update iteration counter
        iter += 1
    end # end of while loop
end # end of compute_unemployment_value
#==========================================================================================
# optimal_strategy: Compute the optimal strategy of workers in each location 
==========================================================================================#
function optimal_strategy!(prim::Primitives, res::Results)
    # Unpack primitives
    @unpack c, n_j = prim
    # Compute the optimal strategy of workers in each location
    # Note that ϕ_u is a (n_j x n_x) matrix where each row is a location and each 
    # column is a skill level, the elements of are the probability that a worker
    # of a given skill level will give to search for a job in each location
    res.ϕ_u = exp.(res.U ./ c) ./ sum(exp.(res.U ./ c), dims=1)
    #* I obtained that employed workers search randomly i.e. ϕ_s = 1/n_j
    res.ϕ_s .= 1 ./ n_j
end # end of optimal_strategy
#==========================================================================================
# compute_surplus: Compute the surplus Bellman equation using value function iteration
==========================================================================================#
function compute_surplus!(prim::Primitives, res::Results, dist::DistributionsModel; 
                            max_iter::Int64=5000, tol::Float64=1e-6, verbose::Bool=true)
    
    if verbose
        println(@bold @blue "Solving the surplus Bellman equation")
    end
    # Unpack primitives
    @unpack n_j, n_x, n_y, β, F_bar = prim
    # Calcualte production in each location for each type of worker and firm
    f = output(prim, dist);
    # Calculate the value of unemployment in each location for each type of worker
    b = home_production(prim, dist);
    # Compute the cost of living in each location
    C = congestion_cost(prim, dist);
    # Calculate instant surplus of each move
    s_move = instant_surplus(prim, f, b, C);
    # Calculate Λ
    Λ = compute_Λ(prim, b);
    ## Start by computing the surplus of non move matches
    # Initialize surplus of non move matches (intial guess is zero)
    S_non_move = zeros(n_j, n_x, n_y)
    # Initiallize error
    err = Inf
    # Set iteration counter
    iter = 1
    # Iterate until convergence
    while (err > tol) & (iter < max_iter)
        # Pre-allocate
        S_non_move_new = zeros(n_j, n_x, n_y)
        # Compute the surplus
        for j ∈ 1:n_j
            S_non_move_new[j, :, :] = s_move[j, j, :, :] .- Λ[j, :] + β .* max.(0, S_non_move[j, :, :] )
        end
        # Compute the error
        # err = maximum(abs.(S_non_move_new .- S_non_move))
        err = norm(S_non_move .- S_non_move_new)
        # Update U
        S_non_move = copy(S_non_move_new)
        # Update iteration counter
        if verbose
            if iter % 100 == 0
                println(@bold @yellow "Iteration:  $iter, Error: $(round(err, digits=6))")
            elseif err < tol
                println(@bold @green "Iteration:  $iter, Converged!")
            end
        end
        iter += 1
    end 

    # Compute the surplus of matches that involve a move using the non movers
    for j in 1:n_j # Loop over locations (origin)
        # Comppute the moving cost vector 
        F = F_bar .* ones(n_j) # Moving cost vector initialized at F_bar
        F[j] = 0 # Moving cost from location j to location j is zero
        for j_prime in 1:n_j # Loop over locations (destination)
            # If non move then fill with non move surplus
            if j == j_prime
                res.S_move[j, j_prime, :, :] = S_non_move[j, :, :]
            end
            for x in 1:n_x # Loop over skills
                for y in 1:n_y # Loop over productivity
                    res.S_move[j, j_prime, x, y] = s_move[j, j_prime, x, y] .- Λ[j_prime, x] .+ β .* max.(0, S_non_move[j_prime, x, y] ) 
                end
            end
            # Subtract moving cost from surplus
            res.S_move[j, j_prime, :, :] = res.S_move[j, j_prime, :, :] .- F[j_prime]
        end
    end
end # end of compute_surplus
#?=========================================================================================
#? Dynamic Programming II : Updating DistributionsModel
#?=========================================================================================
#==========================================================================================
# update_interim_DistributionsModel! : Updates the distribution of workers and vacancies 
        at the interim stage.
==========================================================================================#
function update_interim_distributions!(prim::Primitives, res::Results, dist::DistributionsModel)
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
function get_total_effort(prim::Primitives, dist::DistributionsModel, res::Results)
    # Unpack parameters
    @unpack δ, s, n_j, n_y = prim
    # Preallocate
    L = zeros(n_j)
    # Compute total search effort in each location
    for j in 1:n_j # Loop over locations
        # Total number of unemployed searching for a job in location j
        unemp = sum( dist.u_plus .* res.ϕ_u[j, :]' )
        # Total number of employed searching for a job in location j
        emp = 0
        for y ∈ 1:n_y
            emp += sum( dist.h_plus[:, :, y] .* res.ϕ_s[j, :, y]' )
        end
        L[j] =unemp + s .* emp
    end
    return L
end
#==========================================================================================
# get_vacancy_creation_value!: Calculate the value of vacancy creation (for each location 
        - type of firm)
==========================================================================================#
function get_vacancy_creation_value!(prim::Primitives, res::Results, dist::DistributionsModel)
    # Unpack parameters
    @unpack n_j, n_x, n_y, s = prim
    # Get total seach effort
    L = get_total_effort(prim, dist, res)
    B = zeros(n_j, n_y) # Pre-allocate
    for j in 1:n_j # Loop over locations (origin)
        for j_prime ∈ 1:n_j # Loop over locations (destination)
            # Add value of matches with unemployed workers 
            # Only positive surplus  moves between locations j and j_prime add value
            match_surv = max.(0, res.S_move[j_prime, j, :, :]) 
            # Value of matches with unemployed workers
            s_from_unemp = sum(res.ϕ_u[j, :] .* dist.u_plus[j_prime, :] .* match_surv, dims=1) / L[j]
            # added value of matches with unemployed workers
            B[j, :] += s_from_unemp' 
            # Add value of poaching other firms 
            for y ∈ 1:n_y # Loop over types of firms to find poaching opportunities
                # Poaching condition is surplus of moving is greater that surplus of staying
                worker_poached =  max.(0, res.S_move[j, j_prime, :, y] .- res.S_move[j_prime, j_prime, :, :])
                # Value of matches with employed workers
                s_from_poaching = s * sum(res.ϕ_s[j, :, y] .* dist.h_plus[j_prime, :, y] .* worker_poached) / L[j] 
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
function update_market_tightness_and_vacancies!(prim::Primitives, res::Results, dist::DistributionsModel)
    # Unpack parameters
    @unpack ω₁, ω₂, c₀, c₁ = prim
    # Compute total search effort
    L = get_total_effort(prim, dist, res)
    # Market tightness
    res.θ = (sum((ω₁.* res.B ./ c₀).^c₁, dims=2) ./ L).^(c₁/(ω₂ + c₁)) 
    # Number of vacancies 
    res.v = ((ω₁ .* (res.θ.^(-ω₂)) .* res.B )./ c₀).^(1/c₁) 
end # end function update_market_tightness_and_vacancies
#==========================================================================================
# update_distrutions! : updates the distribution of workers across locations, skills and 
        firms at the next period  
==========================================================================================#
function update_distrutions!(prim::Primitives, res::Results, dist::DistributionsModel)
    @unpack ω₁, ω₂, n_j, n_x, n_y, s = prim
    L = get_total_effort(prim, dist,res) 
    # Calculate total number of vacancies in each location
    V = sum(res.v, dims=2)
    # Calculate total number of mathces in each location
    M = min.(ω₁ .* L.^ω₂ .* V.^(1-ω₂), L, V)
    p = M ./ L # Probability of a match being formed in each location
    # Compute indicator of possitive surplus 
    η = res.S_move .>= 0;
    # Compute indicator of possitive surplus of moving to a different location
    η_move = zeros(n_j, n_j, n_x, n_y, n_y);
    for j ∈ 1:n_j # Loop over locations
        for x ∈ 1:n_x # Loop over skills
            for y ∈ 1:n_y # Loop over firms
                # η[j' → j, x, y'→y] = 1 if S[j' → j, x, y] > S[j' → j', x, y']
                #? To be clear n[j,j',x,y,y'] = 1 if firm [j',y'] can poach a worker with skill x from firm [j,y] 
                #? for example η_move[j, :, x, y, :] are all the firms that can poach a worker with skill x from firm [j,y]
                η_move[j, :, x, y, :] = res.S_move[j, :, x, :] .>= res.S_move[j, j, x, y] 
            end # Loop over firms
        end # Loop over skills
    end # Loop over locations
    # Update distribiutions:
    u_next = zeros(n_j, n_x);
    h_hired = zeros(n_j, n_x, n_y);
    for j ∈ 1:n_j # Loop over locations (origins)
        for x ∈ 1:n_x # Loop over skills
            # Total mass of x workers that arrive to location j from all other locations
            arrive = dist.u_plus[:, x] .* res.ϕ_u[j, x]
            # Probability of that ech firm in location j offers a job to a worker of type x coming from any other location
            prob_hire = η[:, j, x, :] .* (res.v[j, :] / V[j])' .* p[j]
            # Compute the mass of x workers that each firm is able to hire from unemployment in location j
            h_hired[j, x, :]  = sum( prob_hire .* arrive, dims=1 )
            # Compute the mass of workers that is hired by some firm in location j
            # Complemenatry probability of not being offered a job 
            prob_no_hire = 1 .- prob_hire
            # Probability of not being offered a job by any firm in location j coming from each location j'
            prob_no_hire = prod(prob_no_hire, dims=2)
            # Compute the mass of x workers that arrive to location j from all other locations and don't get a job offer
            should_be = sum(arrive) - sum(prob_hire .* arrive) # Mass of workers that should be in unemployment
            u_next[j, x] = should_be
        end # Loop over skills
    end # Loop over locations (origins)
    # Check that the mass of workers in unemployement in interim stage is the same as the mass of workers in unemployment 
    # plus the mass of workers that are hired from unemployment in the next stage
    @assert sum(dist.u_plus) ≈ sum(u_next) + sum(h_hired) @bold @red "Error!: uₜ₊ ≂̸ uₜ₊₁ + hᵤ where hᵤ is the mass of workers hired from unemployment"
    # For each location, skill and firm compute the probability of not being poeached by any other firm in any other location
    h_retained = zeros(n_j, n_x, n_y); 
    for j ∈ 1:n_j # Loop over locations
        for x ∈ 1:n_x # Loop over skills
            for y ∈ 1:n_y # Loop over firms
                # Probability of a worker being poached by any other firm in any other location
                prob_poach = s .* p .* res.v ./ V .* η_move[j, :, x, y, :] .* res.ϕ_s[j, x, y]
                # Probability of a worker not being poached from any other location
                prob_no_poach = 1 .- sum(prob_poach)
                # Probability of a worker not being poached from any other location
                h_retained[j, x, y] = dist.h_plus[j,x,y] .* prob_no_poach
            end # Loop over firms
        end # Loop over skills
    end # Loop over locations
    # Compute the mass of workers that each firm in each location is able to poach from other firms in other locations
    h_poached = zeros(n_j, n_x, n_y)
    poached_from_self = 0
    for j ∈ 1:n_j # Loop over locations
        for x ∈ 1:n_x # Loop over skills
            for y ∈ 1:n_y # Loop over firms
                # Probability of a worker being poached by any other firm in any other location
                prob_poach = (s .* p[j] .* res.v[j, y] ./ V[j] ) .* η_move[:, j, x, :, y]
                # Mass that each firm is able to poach from other firms in other locations(sum over locations)
                mass_poach = sum( prob_poach .* dist.h_plus[:, x, :] .* res.ϕ_s[j, x, :]')
                # Add mass to the corresponding location- skill -firm
                h_poached[j, x, y] = mass_poach
                # Compute the mass of workers that the firm poaches from itself
                poached_from_self += prob_poach[j, y] .* dist.h_plus[j, x, y] .* res.ϕ_s[j, x, y]
            end # Loop over firms
        end # Loop over skills
    end # Loop over locations
    # Check that the mass of workers employed at interim stage is the same as the mass of workers that are retained or poached
    @assert sum(dist.h_plus) ≈ sum(h_poached) + sum(h_retained) @bold @red "Error!: hₜ₊ ≂̸ hₜ₊₁ + hₚ + hᵣ where hₚ is the mass of workers poached and hᵣ is the mass of workers retained"
    
    # Update the distribution of unemployed and employed workers
    dist.u = u_next
    dist.h = h_retained + h_poached + h_hired;
    # Update the distribution of workers across locations
    dist.ℓ = dist.u  + dropdims(sum(dist.h, dims=3), dims=3)
end # End of function update_distrutions!


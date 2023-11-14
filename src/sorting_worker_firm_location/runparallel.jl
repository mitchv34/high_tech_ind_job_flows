println("Starting the script...")
using Dates: format
using Distributed
using YAML
using StatsBase 
using Dates
using Term.Progress

# Load the model to find find intial parameter values
include("model.jl");
include("distribution_generation.jl")
# include("plots.jl")
path_params = "src/sorting_worker_firm_location/parameters/params.yml";
figures_dir = "/figures/sorting_worker_firm_location/"
results_dir = "./results/sorting_worker_firm_location/"

prim, res = init_model(path_params);

# Range of values for the parameters of interest
## Cost of search
len = 30
# c_range = range(0.5 * prim.c, stop = 2 * prim.c, length = len) 
## Agglomeration parameters
a_1_range = range(0.5 * prim.a₁, stop = 2 * prim.a₁, length = len)
# b_1_range = range(0.5 * prim.b₁, stop = 2 * prim.b₁, length = len)
## Cost of living parameters
# a_2_range = range(0.5 * prim.a₂, stop = 2 * prim.a₂, length = len)
# b_2_range = range(0.5 * prim.b₂, stop = 2 * prim.b₂, length = len)
# Get all parameters in a matrix
# params_vary = hcat(c_range, a_1_range, b_1_range, a_2_range, b_2_range)
params_vary = a_1_range

addprocs(prod(size(params_vary)))

# Load everything on the workers
@everywhere include("model.jl");
@everywhere include("distribution_generation.jl")
@everywhere path_params = "src/sorting_worker_firm_location/parameters/params.yml";
@everywhere figures_dir = "/figures/sorting_worker_firm_location/test/"
@everywhere results_dir = "./results/sorting_worker_firm_location/"

@everywhere using BSON

@everywhere params_vary = $params_vary

@sync @distributed for i ∈ 1:prod(size(params_vary))
    prim, res = init_model(path_params);
    # If i ∈ [1. 10] modify c parameter and so on
    # if i ∈ 1:len
    #     println("Solving with c  = $(params_vary[ ( i%10 != 0) ? i%10 : 10, 1])")
    #     prim.c = params_vary[ ( i%10 != 0) ? i%10 : 10, 1]
    # elseif i ∈ (len + 1):(2 * len)
        # println("Solving with a₁ = $(params_vary[( i%10 != 0) ? i%10 : 10, 2])")
        # prim.a₁ = params_vary[( i%10 != 0) ? i%10 : 10, 2]
        println("Solving with a₁ = $(params_vary[i])")
        prim.a₁ = params_vary[i]
    # elseif i ∈ (2 * len + 1):(3 * len)
    #     println("Solving with b₁ = $(params_vary[( i%10 != 0) ? i%10 : 10, 3])")
    #     prim.b₁ = params_vary[( i%10 != 0) ? i%10 : 10, 3]
    # elseif i ∈ (3 * len + 1):(4 * len)
    #     println("Solving with a₂ = $(params_vary[( i%10 != 0) ? i%10 : 10, 4])")
    #     prim.a₂ = params_vary[( i%10 != 0) ? i%10 : 10, 4]
    # elseif i ∈ (4 * len + 1):(5 * len)
    #     println("Solving with b₂ = $(params_vary[( i%10 != 0) ? i%10 : 10, 5])")
    #     prim.b₂ = params_vary[( i%10 != 0) ? i%10 : 10, 5]
    # end
    # initialize disribution
    dist = split_skill_dist(prim, verbose=false);
    # solve the model
    μs, errs, urates = iterate_distributions!(prim, res, dist, verbose = false,  tol=1e-6, max_iter=100000000)
    # save the results
    bson(results_dir * "results_a1_$(params_vary[i]).bson", Dict("prim" => prim, "res" => res, "dist" => dist, "err" => errs[end]))
    println(@bold @green "Solved with a₁ = $(params_vary[i])")
end

# Plot the results 
@everywhere using BSON
@everywhere include("plots.jl")

# Create a function that takes a path to the results, loads them, plots them and saves the figures
@everywhere function plots_all_results(path_results, prefix, sufix; format = "png")
    results = BSON.load(path_results)
    prim, res, dist = results["prim"], results["res"], results["dist"]
    # Plot the results
    figs = plot_all_model( prim,  res,  dist, cmap = :berlin);
    save_all_figures(figs, prefix = prefix, sufix = sufix, format = format)
end

# List the contents of the results directory
results = readdir(results_dir)

# Get today's date
prefix = "$(Dates.today())/"
# Make a directory for today's results
mkdir("." * figures_dir * prefix)

@track for i ∈ eachindex(results)
    if occursin("a1", results[i])
        sufix = "_$(results[i][1:end-5])"
        plots_all_results(results_dir * results[i], prefix, sufix, format = "png")
    end
end

# dist = split_skill_dist(prim, verbose=false);

# # #! Light testing only (tolerance should be lower)
# println("Solving the model")
# μs, errs, urates = iterate_distributions!(prim, res, dist, verbose = false,  tol=1e-1)

# println("Saving the results")
# bson(results_dir * "results.bson", Dict("prim" => prim, "res" => res, "dist" => dist))


# results_model = load(results_dir * "results.bson")

# prim, res, dist = results_model["prim"], results_model["res"], results_model["dist"]
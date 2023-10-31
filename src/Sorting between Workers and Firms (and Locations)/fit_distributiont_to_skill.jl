using StatsBase
using CSV
using DataFrames
using StatsPlots
using StatsBase
using Distributions
using Term.Progress
using Term
using Distributed
using YAML

# Load plot settings
include("../plotconfig.jl")

# Include model files
include("model1.jl")
path_params = "/project/high_tech_ind/high_tech_ind_job_flows/src/Sorting between Workers and Firms (and Locations)/parameters/params.yml"
prim, res = init_model(path_params);
# Read parameters from YAML file 


# Sort df on SIZE
sort!(df, :SIZE, rev = true)

# Subset df to keep only MSA with at least 0.05% of total employment
df = subset(df, :SIZE => ByRow( >=(0.0005) ) )


# Scatter plot of mean skill level vs size
@df df scatter(log.(:SIZE), (:mean_skill), label = "", alpha = 0.5,
 xlabel= "Mean Skill", ylabel = "Size", markersize = 5)
title!("Mean Skill vs Size")


# Create a worker skill grid
x_grid = 0:0.01:1
# for each city evaluate the pdf in the grid weght using size and save it to a dataframe 
# Create a Dataframe with columns = MSA and one for each skill level in the grid
data = DataFrame(Dict("X_$i" => [] for i in eachindex(x_grid)))
# Add MSA column
data[!, :MSA] = []
# Add idea exchange environment value
data[!, :X_bar] = []
# Reorder columns so that MSA is the first one

@unpack ν, A = prim;

ℓ = Dict()
X_bar = Dict()
Ω = Dict()
for row in eachrow(df)
    msa = row.MSA
    ℓ[msa] = pdf.(Beta(row.alpha, row.beta), x_grid) .* row.SIZE
    # Compute idea exchange value on each MSA
    x̄ = (row.SIZE > 0) ? sum(ℓ[msa] .* x_grid)  : 0
    # Compute the value of idea exchange in each location
    X_bar[msa] = (1  .- exp.(-ν .* row.SIZE)) .*   ν  .* x̄
    # Compute Worker productivity in each location
    Ω[msa] = x_grid .* (1 .+ A .* X_bar[msa] .* x_grid)
    # Add to dataframe
    push!(data, [Ω[msa]..., msa, X_bar[msa]])
end

# Merge with df
df_merged = innerjoin(df, data, on = :MSA)

# scatter worker productivity vs size
@df df_merged scatter(log.(:SIZE), log.(:X_bar), label = "", alpha = 0.5, title = "Idea Exchange Value",
    ylabel= "", xlabel = "Size", markersize = 5)
@df df_merged scatter((:mean_skill), log.(:X_bar), label = "", alpha = 0.5, title = "Idea Exchange Value",
    ylabel= "Worker Productivity", xlabel = "Size", markersize = 5)

    
@df df_merged scatter(:SIZE, :X_10, label = "", alpha = 0.5,
    ylabel= "Worker Productivity", xlabel = "Size", markersize = 5)

@df df_merged scatter!(:SIZE, :X_50, label = "", alpha = 0.5,
    ylabel= "Worker Productivity", xlabel = "Size", markersize = 5)

@df df_merged scatter!(:SIZE, :X_90, label = "", alpha = 0.5,
    ylabel= "Worker Productivity", xlabel = "Size", markersize = 5)



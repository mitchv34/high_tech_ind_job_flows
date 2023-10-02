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

# ! This section should be self contained under data processing
# Load data
data_path_metro = "/project/high_tech_ind/high_tech_ind_job_flows/data/OEWS/estimated_skill_distributions/msa_skill/"
# List files 
files = readdir(data_path_metro)

function fit_distributiont_to_skill(data::DataFrame)
    # Sample SKILL data using EMP_PCT column as weights
    sample_size = 1000000
    data_sample = sample(data.SKILL, Weights(data.EMP), sample_size, replace = true)

    e_cdf = ecdf(data_sample)

    dist_fit = fit( Beta,  data_sample );

    return dist_fit
end


# for each file (year) load the data and fit the distribution for each MSA
# Create an empty dataframe columns = MSA (msa), alpha, beta (fitted parameters of the beta distribution)
df = DataFrame(:YEAR => Int64[], :MSA => Int64[], :alpha => Float64[], :beta => Float64[] , :SIZE => Float64[])
# for file in files[end-3:end-3]
file  = files[end-3]
println(@bold @blue "Processing file:  $file")
data_msa = CSV.read(data_path_metro * file, DataFrame)
# Get the year from the file name
year = parse(Int64, split(split(file, ".")[1], "_")[end])
# For each msa fitt the distribution and save the parameters
# Compute total employment by MSA
msa_tot_emp = combine(groupby(data_msa, :AREA), :TOT_EMP => sum => :TOT_EMP)
msa_tot_emp[!, :SIZE] = msa_tot_emp.TOT_EMP ./ sum(msa_tot_emp.TOT_EMP)
# Create dictionary MSA - SIZE
d_msa_size = Dict(zip(msa_tot_emp.AREA, msa_tot_emp.SIZE))
@track for msa in keys(d_msa_size)
    sub_Data = subset(data_msa, :AREA => ByRow( ==(msa) ) )
    fitted_dist, p = fit_distributiont_to_skill( sub_Data )
    push!(df, [year, msa, fitted_dist.α, fitted_dist.β, d_msa_size[msa]])
end
# end

# create dictionary AREA => AREA_NAME
d_msa_name = Dict(zip(data_msa.AREA, data_msa.AREA_NAME))

# Compute mean skill using the parameters of the fitted distribution
df[!, :mean_skill] = df.alpha ./ (df.alpha + df.beta)

# Add MSA name to the dataframe
df[!, :MSA_NAME] = [d_msa_name[msa] for msa in df.MSA]

# ! Until here

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



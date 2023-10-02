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

plotlyjs()

# Include model files
include("model1.jl")
# Read parameters from YAML file 
params = YAML.load_file("/project/high_tech_ind/high_tech_ind_job_flows/src/Macro-dynamics of Sorting between Workers and Firms (and Locations)/parameters/params.yml")

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

    # Ecaluate and plot the CDF
    x = 0:0.01:1;
    y = e_cdf.(x);
    p = plot(x, y, label = "Empirical CDF", lw = 2, legend = :topleft, xlabel = "Skill", ylabel = "CDF")
    plot!(x, cdf(dist_fit, x), label = "Normal Fit", lw = 2);
    
    return dist_fit, p 
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

# Add MSA name to the dataframe
df[!, :MSA_NAME] = [d_msa_name[msa] for msa in df.MSA]

# Load National data
data_path_national = "/project/high_tech_ind/high_tech_ind_job_flows/data/OEWS/estimated_skill_distributions/national_skill/"
# For each year fit the distribution and save the parameters
df_nat = DataFrame(:YEAR => Int64[], :alpha => Float64[], :beta => Float64[])
files = readdir(data_path_national)
for file in files[end-3:end]
    println(@bold @blue "Processing file:  $file")
    data_nat = CSV.read(data_path_national * file, DataFrame)
    # Get the year from the file name
    year = parse(Int64, split(split(file, ".")[1], "_")[end])
    fitted_dist, p = fit_distributiont_to_skill( data_nat )
    push!(df_nat, [year, fitted_dist.α, fitted_dist.β])
end

# Compute mean skill using the parameters of the fitted distribution
df[!, :mean_skill] = df.alpha ./ (df.alpha + df.beta)
df_nat[!, :mean_skill] = df_nat.alpha ./ (df_nat.alpha + df_nat.beta)

using PlotlyJS

plot(
    df, x=:SIZE, y=:mean_skill, mode="markers", text=:MSA_NAME,# marker_color=:Population,
    Layout(title="Populations of USA States")
)


using PlotlyJS, HTTP, CSV, DataFrames

read_remote_csv(url) = DataFrame(CSV.File(HTTP.get(url).body))

df_2 = read_remote_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")

plot(
    df_2, x=:Postal, y=:Population, mode="markers", text=:State, marker_color=:Population,
    Layout(title="Populations of USA States")
)

# Calculate productivity for each MSA
# Pick a year
year = 2019
# sub-sample
df_2019 = subset(df, :YEAR => ByRow( ==(year) ) )

# begin
# ν = params["primitives"]["nu"]
ν = 10
A = params["primitives"]["A"]

# Create a worker skill grid
x_grid = 0:0.01:1
# for each city evaluate the pdf in the grid weght using size and save it to a dataframe 
ℓ = Dict()
X_bar = Dict()
Ω = Dict()
for row in eachrow(df_2019)
    msa = row.MSA
    ℓ[msa] = pdf.(Beta(row.alpha, row.beta), x_grid) .* row.SIZE
    # Compute idea exchange value on each MSA
    x̄ = (row.SIZE > 0) ? sum(ℓ[msa] .* x_grid)  : 0
    # Compute the value of idea exchange in each location
    X_bar[msa] = (1  .- exp.(-ν .* row.SIZE)) .* x̄
    # Compute Worker productivity in each location
    Ω[msa] = x_grid .* (1 .+ A .* X_bar[msa] .* x_grid)
end




# 31540 Madison, WI
# 35620 New York-Newark-Jersey City, NY-NJ-PA
# 31080 Los Angeles-Long Beach-Anaheim, CA
# 16980 Chicago-Naperville-Elgin, IL-IN-WI
# 27500 Janesville-Beloit, WI
# 33100 Miami-Fort Lauderdale-Pompano Beach, FL

# Plot the productivity distribution for each MSA
 plot(x_grid, ℓ[31540], label = "Madison", lw = 2,  xlabel = "Skill", ylabel = "Worker Productivity")
plot!(x_grid, ℓ[35620], label = "New York", lw = 2)
plot!(x_grid, ℓ[31080], label = "Los Angeles", lw = 2)
plot!(x_grid, ℓ[16980], label = "Chicago", lw = 2)
plot!(x_grid, ℓ[27500], label = "Janesville", lw = 2)
plot!(x_grid, ℓ[33100], label = "Miami", lw = 2)

plot(x_grid, Ω[31540], label = "Madison", lw = 2, legend = :topleft, xlabel = "Skill", ylabel = "Worker Productivity")
plot!(x_grid, Ω[35620], label = "New York", lw = 2)
plot!(x_grid, Ω[31080], label = "Los Angeles", lw = 2)
plot!(x_grid, Ω[16980], label = "Chicago", lw = 2)
plot!(x_grid, Ω[27500], label = "Janesville", lw = 2)
plot!(x_grid, Ω[33100], label = "Miami", lw = 2)
# end
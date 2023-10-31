#==========================================================================================
Title: 
Author: Mitchell Valdes-Bobes @mitchv34
Date: 2023-10-03
Description:
==========================================================================================#
#?=========================================================================================
#? SECTION
#?=========================================================================================
using StatsBase
using CSV
using DataFrames
using StatsPlots
using StatsBase
using Distributions
using Term.Progress
using Term
include("../../plotconfig.jl")
#?=========================================================================================
#? SECTION
#?=========================================================================================
function fit_distributiont_to_skill(data::DataFrame)
    # Sample SKILL data using EMP_PCT column as weights
    sample_size = 10000000
    data_sample = sample(data.SKILL, Weights(data.EMP), sample_size, replace = true)
    e_cdf = ecdf(data_sample)

    dist_fit = fit( Normal, data_sample );

    # Ecaluate and plot the CDF
    x = 0:0.01:1;
    y = e_cdf.(x);
    p_cdf = plot(x, y, label = "Empirical CDF", lw = 2, legend = :topleft, xlabel = "Skill", ylabel = "CDF")
    plot!(x, cdf(dist_fit, x), label = "Beta Fit", lw = 2);
    
    p_pdf = density(data_sample, trim = true, bandwidth = 0.05, label = "Empirical PDF", lw = 2, xlabel = "Skill", ylabel = "PDF")
    plot!(x, exp.(pdf.(dist_fit, x)), label = "Beta Fit", lw = 2);

    return dist_fit, p_cdf, p_pdf
end
#?=========================================================================================
#? SECTION
#?=========================================================================================
# Load data
DATA_PATH_METRO = "/project/high_tech_ind/high_tech_ind_job_flows/data/OEWS/estimated_skill_distributions/msa_skill/"
DATA_PATH_NATIONAL = "/project/high_tech_ind/high_tech_ind_job_flows/data/OEWS/estimated_skill_distributions/national_skill/"
# List files 
files_metro = readdir(DATA_PATH_METRO)[6:6]#[1:end-1]
files_national = readdir(DATA_PATH_NATIONAL)[16:16]#[end-7: end]

# Path to save
PATH_SAVE_FIGURES = "/project/high_tech_ind/high_tech_ind_job_flows/figures/fit_skill_gamma/"
PATH_SAVE_DATA = "/project/high_tech_ind/high_tech_ind_job_flows/data/OEWS/estimated_skill_distributions/gamma_fit_skill/"

# for each file (year) load the data and fit the distribution for each MSA
# Create an empty dataframe columns = MSA (msa), alpha, beta (fitted parameters of the beta distribution)
df = DataFrame(:YEAR => Int64[], :MSA => Int64[], :alpha => Float64[], :theta => Float64[] , :SIZE => Float64[])
# for file in files_metro
file = files_metro[1]
    # println(@bold @blue "Processing file:  $file")
    data_msa = CSV.read(DATA_PATH_METRO * file, DataFrame)
    # Get the year from the file name
    year = parse(Int64, split(split(file, ".")[1], "_")[end])
    # For each msa fitt the distribution and save the parameters
    # Compute total employment by MSA
    msa_tot_emp = combine(groupby(data_msa, :AREA), :TOT_EMP => sum => :TOT_EMP)
    msa_tot_emp[!, :SIZE] = msa_tot_emp.TOT_EMP ./ sum(msa_tot_emp.TOT_EMP)
    # Create dictionary MSA - SIZE
    d_msa_size = Dict(zip(msa_tot_emp.AREA, msa_tot_emp.SIZE))
    @track for msa in keys(d_msa_size)
        println(@bold @green "Processing MSA:  $msa")
        sub_Data = subset(data_msa, :AREA => ByRow( ==(msa) ) )
        # fitted_dist, p_cdf, p_pdf = fit_distributiont_to_skill( sub_Data )
        fitted_dist = fit_mle(Gamma, sub_Data.SKILL, sub_Data.EMP)
        push!(df, [year, msa, fitted_dist.α, fitted_dist.θ, d_msa_size[msa]])
        # Save figures
        # savefig(p_cdf, PATH_SAVE_FIGURES * "$(year)_$(msa)_gamma_cdf.png")
        # savefig(p_pdf, PATH_SAVE_FIGURES * "$(year)_$(msa)_gamma_pdf.png")
    end
end

sort!(msa_tot_emp, :SIZE, rev = true)

# Save list of sizes to file
CSV.write(PATH_SAVE_DATA * "msa_size.csv", msa_tot_emp)

# Get list of 100 largest MSAs
msa_list = msa_tot_emp.SIZE[1:100]

fit(LogNormal, sub_Data.SKILL, sub_Data.EMP)

sub_Data = subset(data_msa, :AREA => ByRow( ==(31080) ) )
x = sub_Data.SKILL
w = sub_Data.EMP
n = length(x)
if length(w) != n
    throw(DimensionMismatch("Inconsistent argument dimensions."))
end

sx = zero(Float64)
slogx = zero(Float64)
tw = zero(Float64)
for i in eachindex(x, w)
    @inbounds xi = x[i]
    @inbounds wi = w[i]
    sx += wi * xi
    slogx += wi * log(xi)
    tw += wi
end
ss = Distributions.GammaStats(sx, slogx, tw)

mx = ss.sx / ss.tw
logmx = log(mx)
mlogx = ss.slogx / ss.tw

alpha0=3

a::Float64 = isnan(alpha0) ? (logmx - mlogx)/2 : alpha0
converged = false

maxiter = 10

t = 0
tol = 1e-16
while !converged && t < maxiter
    t += 1
    a_old = a
    a = Distributions.gamma_mle_update(logmx, mlogx, a)
    @show a, a_old

    converged = abs(a - a_old) <= tol
end


Gamma(a, mx / a)
Distributions.digamma(a)
log(a) - Distributions.digamma(a)
# create dictionary AREA => AREA_NAME
# d_msa_name = Dict(zip(data_msa.AREA, data_msa.AREA_NAME))

# Compute mean skill using the parameters of the fitted distribution
df[!, :mean_skill] = df.alpha .* df.theta

# Add MSA name to the dataframe
# df[!, :MSA_NAME] = [d_msa_name[msa] for msa in df.MSA]


# For each year fit the distribution and save the parameters

# for file in files_national
file = files_national[1]
    println(@bold @blue "Processing file:  $file")
    data_nat = CSV.read(DATA_PATH_NATIONAL * file, DataFrame)
    fitted_dist, p_cdf, p_pdf = fit_distributiont_to_skill( data_nat )
    # Save figures
    savefig(p_cdf, PATH_SAVE_FIGURES * "$(year)_00000_gamma_cdf.png")
    savefig(p_pdf, PATH_SAVE_FIGURES * "$(year)_00000_gamma_pdf.png")
    # Get year
    year = parse(Int64, split(split(file, ".")[1], "_")[end])
    # Get mean skill
    mean_skill = fitted_dist.α ./ (fitted_dist.α + fitted_dist.β)
    # Add to dataframe
    push!(df, [year, 0, fitted_dist.α, fitted_dist.β, 1, mean_skill])
# end

# Save dataframe
CSV.write(PATH_SAVE_DATA * "gamma_fit_skill.csv", df)

p_cdf
p_pdf
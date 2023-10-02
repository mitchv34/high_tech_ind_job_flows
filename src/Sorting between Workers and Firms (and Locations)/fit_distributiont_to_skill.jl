using StatsBase
using CSV
using DataFrames
using StatsPlots
using StatsBase
using Distributions
using Term.Progress
using Term

# Load data
data_path_metro = "/project/high_tech_ind/high_tech_ind_job_flows/data/OEWS/estimated_skill_distributions/msa_skill/"
# List files 
files = readdir(data_path_metro)

function fit_distributiont_to_skill(data::DataFrame)
    # Sample SKILL data using EMP_PCT column as weights
    sample_size = 1000000
    data_sample = sample(data.SKILL, Weights(data.EMP_PCT), sample_size, replace = true)

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
# for file in files
    file = files[1]
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
        print(msa)
        fitted_dist, p = fit_distributiont_to_skill(subset(data_metro, :AREA => ByRow( x -> x in [msa] ) ))
        push!(df, [year, msa, fitted_dist.α, fitted_dist.β, d_msa_size[msa]])
    end
# end


m = 48864
fitted_dist, p = fit_distributiont_to_skill(subset(data_metro, :AREA => ByRow( x -> x in [m] ) ))

subset(data_metro, :AREA => ByRow( x -> x in [m] ) )

# 31540 Madison, WI
# 35620 New York-Newark-Jersey City, NY-NJ-PA
# 31080 Los Angeles-Long Beach-Anaheim, CA
# 16980 Chicago-Naperville-Elgin, IL-IN-WI
# 27500 Janesville-Beloit, WI
# 33100 Miami-Fort Lauderdale-Pompano Beach, FL

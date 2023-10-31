unemployment_rate = 0.1

# Load Skill MSA data
skill_msa_data = CSV.read("/project/high_tech_ind/high_tech_ind_job_flows/data/OEWS/estimated_skill_distributions/msa_skill/msa_skill_2019.csv", DataFrame);

skill_msa_data[!, :wSKILL] = skill_msa_data.SKILL .* skill_msa_data.EMP

# Compute the SKILL for each MSA and the total number of workers
skill_msa_data = combine(
    groupby(skill_msa_data, :AREA), 
    :TOT_EMP => sum,
    :wSKILL => sum
)

# Sort by total number of workers
sort!(skill_msa_data, :TOT_EMP_sum, rev = true)

# Keep only the top 100 MSAs
skill_msa_data = skill_msa_data[1:prim.n_j, :]

# Compute the SHARE of employment in each MSA
skill_msa_data[!, :SHARE] = skill_msa_data.TOT_EMP_sum ./ sum(skill_msa_data.TOT_EMP_sum)

# Estimate a regression of log(wSKILL_sum) on log(TOT_EMP_sum)
regression = reg(skill_msa_data,
    @formula(log(wSKILL_sum) ~ log(SHARE) + log(SHARE)^2), 
    save = true
    )
# Predict the value of log(wSKILL_sum) for each MSA

predicted_wSKILL = exp.(predict(regression, skill_msa_data))

# Plot
@df skill_msa_data scatter(log.(:SHARE), log.(:wSKILL_sum), label = "", xlabel = "Total number of workers", ylabel = "Total skill level")
@df skill_msa_data scatter!(log.(:SHARE), log.(predicted_wSKILL), label = "Fit")

# Fix β as to match the national distribution (Beta)
β = prim.x_dist_params[2]
# Compute α = β * predicted_wSKILL / (1 - predicted_wSKILL )
α_S = β .* predicted_wSKILL ./ (1 .- predicted_wSKILL)

# Create each MSA distribution
location_dists = [Beta(α_S[i], β) for i in 1:size(skill_msa_data, 1)]

# Compute the pdfs of each MSA over the grid of workers
MSA_pdfs = [pdf.(dist, prim.x_grid) for dist in location_dists]

# Normalize all pdfs to have total mass 1
MSA_pdfs = [pdf / sum(pdf) for pdf in MSA_pdfs]
# Weight each MSA by its weight (share of employment)
weight = skill_msa_data.SHARE
MSA_pdfs = [pdf .* weight[i] for (i, pdf) in enumerate(MSA_pdfs)]

# Plot the pdfs of each MSA
plot_msa = plot()
for i in 1:size(skill_msa_data, 1)
    plot!(prim.x_grid, MSA_pdfs[i], lw = 2, label = "")
end
plot_msa

ℓ = Matrix(hcat(MSA_pdfs...)')
# Aggregate all MSAs into a single pdf
ℓ_total = sum(ℓ, dims=1)[:]

# Start with 10% unemployment rate in each location and skill and workers equally distributed across firms
u = ℓ .* unemployment_rate # Initial unemployment for each skill level in each location
# Distribution of workers across firms
e = (1 - 0.1) .* ℓ # Distribution of employed workers
h = zeros(prim.n_j, prim.n_x, prim.n_y) # Distribution of employed workers
for j ∈ 1 : prim.n_j
    h[j, :, :] .= e[j, :] ./ prim.n_y
end

# Construct the firm productivity distribution
α_y, β_y = prim.y_dist_params
Φ = pdf.(Beta(α_y, β_y), prim.y_grid)
# Generate the distribution object
dist = DistributionsModel(ℓ_total, ℓ, u, h, Φ);


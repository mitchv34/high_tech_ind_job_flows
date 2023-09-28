using DataFrames
using CSV
using Plots
using StatsBase
using StatsPlots
using LaTeXStrings
using FixedEffectModels
using VegaLite, VegaDatasets
using RegressionTables
using HPFilter
using CategoricalArrays


gr()

## Configure Plots
theme(:wong) 
default(fontfamily="Computer Modern", framestyle=:none) # LaTex-style

# Global variables
variable = true
b = "0.75"

begin
	# Folders
	flows_folder   = "/project/high_tech_ind/high_tech_job_flows//data/j2j_od/proc/"
	ht_data_folder = "/project/high_tech_ind/high_tech_job_flows//data/bds_high_tech/proc/"
	qwi_folder 	   = "/project/high_tech_ind/high_tech_job_flows//data/qwi/proc/"
	# Load data
	data_flows = CSV.read(flows_folder * "msa_dist_sector_dsimj2jod_sex1_age_lower_07.csv", DataFrame)
	data_flows_educ = CSV.read(flows_folder * "msa_dist_sector_dsimj2jod_sex1_educ.csv", DataFrame)
	data_bds_ht_msa = CSV.read(ht_data_folder * "data_bds_ht_msa.csv", DataFrame)

	data_qwi_avg = CSV.read(qwi_folder * "data_qwi_avg.csv", DataFrame)
	data_qwi = CSV.read(qwi_folder* "data_qwi.csv", DataFrame)
	
	threshold = parse(Float64, b)
	if variable
		# Tag MSA as ht if share_ht_variate is above threshold
		data_flows[!, :msa_ht_orig] = data_flows.share_ht_orig_variate .≥ threshold
		data_flows[!, :msa_ht_dest] = data_flows.share_ht_dest_variate .≥ threshold
		data_flows_educ[!, :msa_ht_orig] = data_flows_educ.share_ht_orig_variate .≥ threshold
		data_flows_educ[!, :msa_ht_dest] = data_flows_educ.share_ht_dest_variate .≥ threshold
	else
		# Tag MSA as ht if share_ht is above threshold
		data_flows[!, :msa_ht_orig] = data_flows.share_ht_orig .≥ threshold
		data_flows[!, :msa_ht_dest] = data_flows.share_ht_dest .≥ threshold
		data_flows_educ[!, :msa_ht_orig] = data_flows_educ.share_ht_rank_orig .≥ threshold
		data_flows_educ[!, :msa_ht_dest] = data_flows_educ.share_ht_rank_dest .≥ threshold
		
	end 
	
	data_flows[!, :NHT_NHT] = (1 .- data_flows.msa_ht_orig) .* (1 .- data_flows.msa_ht_dest)
	data_flows[!, :NHT_HT]  = (1 .- data_flows.msa_ht_orig) .* data_flows.msa_ht_dest
	data_flows[!, :HT_NHT]  = data_flows.msa_ht_orig .* (1 .- data_flows.msa_ht_dest)
	data_flows[!, :HT_HT]   = data_flows.msa_ht_orig .*  data_flows.msa_ht_dest
	
	data_flows_educ[!, :NHT_NHT] = (1 .- data_flows_educ.msa_ht_orig) .* (1 .- data_flows_educ.msa_ht_dest)
	data_flows_educ[!, :NHT_HT]  = (1 .- data_flows_educ.msa_ht_orig) .* data_flows_educ.msa_ht_dest
	data_flows_educ[!, :HT_NHT]  = data_flows_educ.msa_ht_orig .* (1 .- data_flows_educ.msa_ht_dest)
	data_flows_educ[!, :HT_HT]   = data_flows_educ.msa_ht_orig .*  data_flows_educ.msa_ht_dest

	# Multiply shares by 100 to get better coeficients
	data_flows[:, :OFLOW_share] = 100 * data_flows[:, :OFLOW_share] 
	data_flows_educ[:, :OFLOW_share] = 100 * data_flows_educ[:, :OFLOW_share] 
	
	data_flows_educ[!, :HighEduc] = data_flows_educ[!, :education] .== "H"

	# # Split data_flows_educ into two samples (only college degree and rest)
	data_flows_heduc = data_flows_educ[data_flows_educ[!, :education] .== "H", :]
	data_flows_leduc = data_flows_educ[data_flows_educ[!, :education] .!= "H", :]

# md"""
# ## Data
# 1. Business Dynamics Statistics of U.S.: High Tech Industries (BDS-HT)
# - Variables include geography, firm-size, year, employment, job creation and destruction.
# 2. LEHD Origin-Destination Employment Statistics (LODES)
# - Tracks job mobility in the United States.
# - Contains information of the origin and destination of workers.
# - Aggreagated at the MSA level.
# 3. Quarterly Workforce Indicators (QWI).
# """
end 
begin
	## Regressions
	function regression_wo_educ(data)
		
		### Regression 4: shares of migration on distance and city fixed effects and sectoral dissimilarity (dissimilarity^2 and distance^2)
		reg_4_1 = reg(data, @formula(OFLOW_share ~  dissimilarity + distance + distance^2 + dissimilarity^2 + fe(msa_orig) + fe(msa_dest) +fe(year) ), Vcov.robust());
		
		### Regression 5: shares of migration on distance and sectoral dissimilarity (dissimilarity^2 and distance^2) and a dummy if msa is High Tech or not
		reg_5_1 = reg(data, @formula(OFLOW_share ~  dissimilarity + distance + distance^2 + dissimilarity^2 + fe(msa_orig) + fe(msa_dest)+ NHT_HT + NHT_NHT + HT_HT + fe(year) ), Vcov.robust());
		
		### Regression with continous variables 
		reg_6_1 = reg(data, @formula(OFLOW_share ~  dissimilarity + distance + distance^2 + dissimilarity^2 + (share_ht_dest - share_ht_orig) +
		+ fe(msa_orig) + fe(msa_dest)+ fe(year) ), Vcov.robust());

		
		reg_table_1 = regtable(
			reg_4_1,
			reg_5_1; 
			renderSettings = latexOutput(), 
			print_result = true
		)
		return reg_table_1
	end

	## Regressions
	function regressions_w_education(data)
		
		reg_4_3 = reg(data, @formula(OFLOW_share ~  dissimilarity + distance + fe(HighEduc) + distance^2 + dissimilarity^2 + fe(msa_orig) + fe(msa_dest) +fe(year) ), Vcov.robust());
		
		### Regression 5: shares of migration on distance and sectoral dissimilarity (dissimilarity^2 and distance^2) and a dummy if msa is High Tech or not
		
		reg_5_31 = reg(data, @formula(OFLOW_share ~  dissimilarity + distance + fe(HighEduc) + distance^2 + dissimilarity^2 + NHT_HT + NHT_NHT + HT_HT + fe(msa_orig) + fe(msa_dest)+ fe(year) ), Vcov.robust());
		
		reg_5_32 = reg(data, @formula(OFLOW_share ~  dissimilarity + distance + fe(HighEduc) + distance^2 + dissimilarity^2 + NHT_HT + NHT_NHT + HT_HT + (HT_NHT & HighEduc) + (NHT_HT & HighEduc) + (HT_HT & HighEduc) + fe(msa_orig) + fe(msa_dest)+ fe(year) ), Vcov.robust());
		
		reg_table_1 = regtable(
			reg_4_3,
			reg_5_31, 
			reg_5_32; 
			renderSettings = latexOutput(), 
			print_result = true
		)

		return reg_table_1
	end 

	## Regressions
	function regressions_w_education_extra_controls(data)
		
		reg_4_3 = reg(data, @formula(
			OFLOW_share ~  dissimilarity + distance 
						+ distance^2 + dissimilarity^2 
						+ (dissimilarity & HighEduc)  + (distance & HighEduc)
						+ ((distance^2) & HighEduc) + ((dissimilarity^2) & HighEduc)
						+ fe(msa_orig) + fe(msa_dest) +fe(year) + fe(HighEduc)  	+ + fe(year) & fe(HighEduc)),
			Vcov.robust());
		
		### Regression 5: shares of migration on distance and sectoral dissimilarity 
		
		reg_5_31 = reg(data, @formula(
					OFLOW_share ~  
						dissimilarity
						+ distance 
						+ distance^2
						+ dissimilarity^2 
						+ (dissimilarity & HighEduc)
						+ (distance & HighEduc)
						+ ((distance^2) & HighEduc)
						+ ((dissimilarity^2) & HighEduc)
						+ NHT_HT
						+ NHT_NHT
						+ HT_HT 
						+ fe(msa_orig)
						+ fe(msa_dest)
						+ fe(year)
						+ (HighEduc) 
						+ fe(year) & fe(HighEduc)
		),Vcov.robust());

		reg_5_32 = reg(data, @formula(
					OFLOW_share ~  
						dissimilarity
						+ distance 
						+ distance^2
						+ dissimilarity^2 
						+ (dissimilarity & HighEduc)
						+ (distance & HighEduc)
						+ ((distance^2) & HighEduc)
						+ ((dissimilarity^2) & HighEduc)
						+ NHT_HT
						+ NHT_NHT
						+ HT_HT 
						+ (NHT_HT & HighEduc)
						+ (NHT_NHT & HighEduc)
						+ (HT_HT & HighEduc)
						+ fe(msa_orig)
						+ fe(msa_dest)
						+ fe(year)
						+ (HighEduc)
						+ fe(year) & fe(HighEduc)
		),Vcov.robust());
		
		reg_table_1 = regtable(
			reg_4_3,
			reg_5_31, 
			reg_5_32; 
			renderSettings = latexOutput(), 
			print_result = true
		)

		return reg_table_1
	end 

end


reg_table_1 = regression_wo_educ(data_flows)
reg_table_1 = regression_wo_educ(data_flows_heduc)
reg_table_3 = regressions_w_education(data_flows_educ)
reg_table_5 = regressions_w_education_extra_controls(data_flows_educ)


begin
	conf_int = confint(reg_time_1, level = 0.95)
	coefs = coef(reg_time_1)[12:end]
	conf_int = conf_int[12:end, :]
	# # Plot coefficients and confidence intervals
	plot(coefs, yerr=(coefs.-conf_int[:, 1], conf_int[:, 2].-coefs), 
	markerstrokecolor =1, size = (800, 600), 
	    xtick=(1:2:length(coefs), 2001:2:2020), c = 2, label = "",
	    xlabel="Coefficient", ylabel="Value", title="Outflows of skilled workers over time", framestyle = :origin)
	plot!(coefs, seriestype = :scatter, markersize = 5, markerstrokewidth = 0, label = "")
end

# Load unemployment data
msa_unemp = CSV.read("/project/high_tech_ind/high_tech_job_flows/data/BLS/proc/bls_msa_unemployment_rate_yearly.csv", DataFrame)
data_msa = CSV.read("/project/high_tech_ind/high_tech_job_flows/data/bds_all/bds2020_msa.csv", DataFrame)
# Rename geography column to msa
rename!(msa_unemp, :geography => :msa)
# Merge with data_bds_ht_msa
data_msa = innerjoin(data_msa, msa_unemp, on = [:msa, :year])

# Split the data into two samples 2000-209 and 2010-2019
data_msa_2000 = filter(row -> row.year <= 2009, data_msa)
data_msa_2010 = filter(row -> row.year >= 2010, data_msa)
# Calculate the mean unemployment rate for each MSA and job creation and job destruction rates
data_msa_2000 = combine(groupby(data_msa_2000, [:msa]), [:unemployment_rate => mean => :unemployment_rate_2000,
																		:job_creation_rate => mean => :job_creation_rate_2000,
																		:job_destruction_rate => mean => :job_destruction_rate_2000])
data_msa_2010 = combine(groupby(data_msa_2010, [:msa]), [:unemployment_rate => mean => :unemployment_rate_2010,
															:job_creation_rate => mean => :job_creation_rate_2010,
															:job_destruction_rate => mean => :job_destruction_rate_2010])




# Merge the two dataframes
data_two_samples = innerjoin(data_msa_2000, data_msa_2010, on = :msa)
# Scatter plot persistance of unemployment rate
@df data_two_samples scatter(
	log.(:unemployment_rate_2000./100), 
	log.(:unemployment_rate_2010./100), 
	label = "", 
	title = "Persistence of Unemployment Rate",
	xlabel = "(log) Unemployment Rate 2000-2009", 
	ylabel = "(log) Unemployment Rate 2010-2019", 
	legend = true, size = (600, 600),
	c = 2, alpha = 0.5)
# Add 45 degree line
@df data_two_samples_unemp plot!(log.([0.02, 0.25]), log.([0.02, 0.25]), label = L"45^{o}", c = 4, lw = 2, linestyle = :dash)
savefig("unemployment_persistence.png")



# Average the same variables for the whole sample
data_msa = combine(groupby(data_msa, [:msa]), [:unemployment_rate => mean => :unemployment_rate,
												:job_creation_rate => mean => :job_creation_rate,
												:job_destruction_rate => mean => :job_destruction_rate])

# Scatter plot persistance of job creation rate
p1 = @df data_msa scatter(
			log.(:unemployment_rate/100),
			- log.(:job_creation_rate),
			label = "",
			c = 2,
			title = "(b) Job Creation Rate",
			xlabel = "(log) Unemployment Rate 2000-2019",
			ylabel = "-(log) Job Creation Rate",
			smooth = true,
			alpha = 0.5,
			lw = 2, linestyle = :dash, linecolor = 4, size = (600, 600))

p2 = @df data_msa scatter(
			log.(:unemployment_rate/100),
			log.(:job_destruction_rate),
			label = "",
			c = 2,
			title = "(a) Job Destruction Rate",
			xlabel = "(log) Unemployment Rate 2000-2019",
			ylabel = "(log) Job Destruction Rate",
			smooth = true,
			alpha = 0.5,
			lw = 2, linestyle = :dash, linecolor = 4, size = (600, 600))


# Read total factor productivity data
data_tfp = CSV.read("/project/high_tech_ind/high_tech_job_flows/data/disp/total_factor_productivity.csv", DataFrame)



# Job Creation and Job Destruction
# Get national data

data_nat = CSV.read("/project/high_tech_ind/high_tech_job_flows/data/bds_all/bds2020.csv", DataFrame)
subset!(data_nat, :year=> ByRow(>=(2000)))
p1 = @df data_nat  plot(:year, :job_creation_rate, label="Job Creation Rate", lw = 2, alpha = 0.5 )
@df data_nat plot!(:year, :job_destruction_rate, label="Job Destruction Rate", lw = 2, alpha = 0.5 )
@df data_nat plot!(:year, HP(:job_creation_rate, 1600), label="", c = 1, lw = 2, linestyle = :dash)
@df data_nat plot!(:year, HP(:job_destruction_rate, 1600), label="", c = 2, lw = 2, linestyle = :dash)
# Anotate the plot

# Get MSA level data
data_msa = CSV.read("/project/high_tech_ind/high_tech_job_flows/data/bds_all/bds2020_msa.csv", DataFrame)
subset!(data_msa, :year=> ByRow(>=(2000)))
# Get average employment for each MSA over the sample
data_msa_avg = combine(groupby(data_msa, [:msa]), :emp => mean => :emp)
# Get a list of all msa in the top 25% of employment
msa_top25 = data_msa_avg[sortperm(data_msa_avg.emp, rev=true), :msa][1:round(Int, 0.25*length(data_msa_avg.emp))]
# Filter data_msa to only include the top 25% of employment
subset!(data_msa, :msa=> ByRow(in(msa_top25)))
# Make a violin plot of the job creation and destruction rates for the top 25% of employment
p2 = @df data_msa boxplot(string.(:year), :job_creation_rate,  fillalpha=0.75, linewidth=2)
p3 = @df data_msa boxplot(string.(:year), :job_destruction_rate,  fillalpha=0.75, linewidth=2)

# Display the plots
plot(p1, p2, p3, layout = (1,3), size = (1000, 300))

p4 = @df data_nat  plot(:year, :job_creation_rate./100, label="Job Creation Rate", lw = 2, alpha = 1 )
@df data_msa dotplot!(:year, :job_creation_rate, c = 1, alpha = 0.1,  marker=(stroke(0)), label = "Job Creation Rate (MSA)")

p5 = @df data_nat  plot(:year, :job_destruction_rate./100, label="Job Destruction Rate", lw = 2, alpha = 1 , c=2)
@df data_msa dotplot!(:year, :job_destruction_rate, c = 2, alpha = 0.1,  marker=(stroke(0)), label = "Job Destruction Rate (MSA)")

plot(p4, p5, layout = (1,2), size = (1000, 300))

# Read HT data at the national level
data_ht_nat = CSV.read("/project/high_tech_ind/high_tech_job_flows/data/bds_high_tech/interim/nationwide.csv", DataFrame)
subset!(data_ht_nat, :year=> ByRow(>=(2000)))

p6 = @df data_nat  plot(:year, :job_creation_rate, label="Job Creation Rate", lw = 2, alpha = 0.5 )
@df data_nat plot!(:year, HP(:job_creation_rate, 400), label="", c = 1, lw = 2, linestyle = :dash)
@df data_ht_nat plot!(:year, :job_creation_rate_ht, label="Job Creation Rate", c = 2, lw = 2, alpha = 0.5 )
@df data_ht_nat plot!(:year, HP(:job_creation_rate_ht, 400), label="", c = 2, lw = 2, linestyle = :dash)
@df data_ht_nat plot!(:year, :job_creation_rate_nht, label="Job Creation Rate", lw = 2, alpha = 0.5 , c=3)
@df data_ht_nat plot!(:year, HP(:job_creation_rate_nht, 400), label="", c = 3, lw = 2, linestyle = :dash)

# Read HT data at the MSA level
data_ht_msa = CSV.read("high_tech_job_flows/data/bds_high_tech/interim/msac.csv", DataFrame)
subset!(data_ht_msa, :year=> ByRow(>=(2000)))
# Rename geography to msa
rename!(data_ht_msa, :geography => :msa)
# Filter data_msa to only include the top 25% of employment
subset!(data_ht_msa, :msa=> ByRow(in(msa_top25)))
# Get percentage of high tech jobs in each MSA
data_ht_msa[:, :share_ht] = data_ht_msa[:, :emp_ht]./(data_ht_msa[:, :emp_ht] + data_ht_msa[:, :emp_nht])
# Get average share of high tech jobs for each MSA over the sample
data_ht_msa_avg = combine(groupby(data_ht_msa, [:msa]), :share_ht => mean => :share_ht)
# Get a list of all msa in the top 25% of employment
msa_top25_ht = data_ht_msa_avg[sortperm(data_ht_msa_avg.share_ht, rev=true), :msa][1:round(Int, 0.25*length(data_ht_msa_avg.share_ht))]
# Tag each msa as HT or notHT acording to the top 25% of employment
data_ht_msa[:, :is_ht] = [(msa ∈ msa_top25_ht) ? "HT" : "notHT" for msa in data_ht_msa[:, :msa]]

# Add this percentage to the data_msa dataframe
data_msa = innerjoin(data_msa, data_ht_msa[:, [:msa, :year, :share_ht, :is_ht]], on = [:msa, :year])



# Repeat plots p4 and p5 but mark the high tech MSAs
p7 = @df data_nat  plot(:year, :job_creation_rate./100, label="Job Creation Rate", lw = 2, alpha = 1 )
@df data_msa dotplot!(:year, :job_creation_rate, group = :is_ht, alpha = 0.3,  marker=(stroke(0)), label = "Job Creation Rate (MSA)")

p8 = @df data_nat  plot(:year, :job_destruction_rate./100, label="Job Destruction Rate", lw = 2, alpha = 1 , c=2)
@df data_msa dotplot!(:year, :job_destruction_rate, group = :is_ht, alpha = 0.3,  marker=(stroke(0)), label = "Job Destruction Rate (MSA)")


# Calculate the share of high tech jobs for each MSA in each year
# Add the share of high tech jobs to the data_flows dataframe
data_flows = innerjoin(data_flows, data_ht_msa[:, [:msa, :year, :share_ht]], on = [:msa_orig => :msa, :year => :year])
rename!(data_flows, :share_ht => :share_ht_orig)
data_flows = innerjoin(data_flows, data_ht_msa[:, [:msa, :year, :share_ht]], on = [:msa_dest => :msa, :year => :year])
rename!(data_flows, :share_ht => :share_ht_dest)
# Group the data by year and calculate the quartiles of the share_ht column for each group
quartiles = combine(groupby(data_flows, :year), :share_ht_dest => (q -> quantile(q, [0.90])) => :HT_threshold)
data_flows = innerjoin(data_flows, quartiles, on = :year)
data_flows[:, :HT_orig] = data_flows.share_ht_orig .≥ data_flows.HT_threshold
data_flows[:, :HT_dest] = data_flows.share_ht_dest .≥ data_flows.HT_threshold

# Calculate yearly quartiles of the share_ht_dest column
Q1 = combine(groupby(data_flows, :year), :share_ht_dest => (q -> quantile(q, [0.25])) => :Q1)
Q2 = combine(groupby(data_flows, :year), :share_ht_dest => (q -> quantile(q, [0.50])) => :Q2)
Q3 = combine(groupby(data_flows, :year), :share_ht_dest => (q -> quantile(q, [0.75])) => :Q3)

# Add the quartiles to the data_flows dataframe
data_flows = innerjoin(data_flows, Q1, on = :year)
data_flows = innerjoin(data_flows, Q2, on = :year)
data_flows = innerjoin(data_flows, Q3, on = :year)

# Create quartile columns
data_flows[:, :HT_Q_orig] = CategoricalArray( 4 .- ( (data_flows.share_ht_orig .≤ data_flows.Q1) + (data_flows.share_ht_orig .≤ data_flows.Q2) + (data_flows.share_ht_orig .≤ data_flows.Q3) ) )
data_flows[:, :HT_Q_dest] = CategoricalArray( 4 .- ( (data_flows.share_ht_dest .≤ data_flows.Q1) + (data_flows.share_ht_dest .≤ data_flows.Q2) + (data_flows.share_ht_dest .≤ data_flows.Q3) ) )

# Normalize the share_ht_dest and share_ht_orig columns
data_flows[:, :share_ht_dest] = (data_flows[:, :share_ht_dest] .- mean(data_flows[:, :share_ht_dest])) ./ std(data_flows[:, :share_ht_dest])
data_flows[:, :share_ht_orig] = (data_flows[:, :share_ht_orig] .- mean(data_flows[:, :share_ht_orig])) ./ std(data_flows[:, :share_ht_orig])

# Add job creation and destruction rates to the data_flows dataframe
# Orig
data_flows = innerjoin(data_flows, data_ht_msa[:, [:msa, :year, :job_creation_rate_ht, :job_destruction_rate_ht, :job_creation_rate_nht, :job_destruction_rate_nht]], on = [:msa_orig => :msa, :year => :year])
# Rename the columns
rename!(data_flows, [:job_creation_rate_ht => :job_creation_rate_ht_orig, :job_destruction_rate_ht => :job_destruction_rate_ht_orig, :job_creation_rate_nht => :job_creation_rate_nht_orig, :job_destruction_rate_nht => :job_destruction_rate_nht_orig])
# Dest
data_flows = innerjoin(data_flows, data_ht_msa[:, [:msa, :year, :job_creation_rate_ht, :job_destruction_rate_ht, :job_creation_rate_nht, :job_destruction_rate_nht]], on = [:msa_dest => :msa, :year => :year])
# Rename the columns
rename!(data_flows, [:job_creation_rate_ht => :job_creation_rate_ht_dest, :job_destruction_rate_ht => :job_destruction_rate_ht_dest, :job_creation_rate_nht => :job_creation_rate_nht_dest, :job_destruction_rate_nht => :job_destruction_rate_nht_dest])


ht_msa = CSV.read("/project/high_tech_ind/high_tech_job_flows/data/bds_high_tech/interim/msac.csv", 	DataFrame)

# Calculate for each year the share of total high-tech employment in each MSA

## Calculate yearly totals of high-tech employment
ht_msa = innerjoin(ht_msa, combine(groupby(ht_msa, :year), :emp_ht => sum), on = :year)

## Calculate yearly shares of high-tech employment
ht_msa[:, :share_ht] = ht_msa[:, :emp_ht] ./ ht_msa[:, :emp_ht_sum]

# Calculate yearly percentiles of the share_ht_dest column
P5  = combine(groupby(ht_msa, :year), :share_ht => (q -> quantile(q, [0.05])) => :P5)
P10 = combine(groupby(ht_msa, :year), :share_ht => (q -> quantile(q, [0.10])) => :P10)
P25 = combine(groupby(ht_msa, :year), :share_ht => (q -> quantile(q, [0.25])) => :P25)
P50 = combine(groupby(ht_msa, :year), :share_ht => (q -> quantile(q, [0.50])) => :P50)
P75 = combine(groupby(ht_msa, :year), :share_ht => (q -> quantile(q, [0.75])) => :P75)
P90 = combine(groupby(ht_msa, :year), :share_ht => (q -> quantile(q, [0.90])) => :P90)
P95 = combine(groupby(ht_msa, :year), :share_ht => (q -> quantile(q, [0.95])) => :P95)

# Merge all the percentiles into one dataframe
percentiles = innerjoin(P5, P10, on = :year)
percentiles = innerjoin(percentiles, P25, on = :year)
percentiles = innerjoin(percentiles, P50, on = :year)
percentiles = innerjoin(percentiles, P75, on = :year)
percentiles = innerjoin(percentiles, P90, on = :year)
percentiles = innerjoin(percentiles, P95, on = :year)

# Calculate P75 - P25, P90 - P10, P95 - P5
percentiles[:, :P75_P25] = percentiles[:, :P75] - percentiles[:, :P25]
percentiles[:, :P90_P10] = percentiles[:, :P90] - percentiles[:, :P10]
percentiles[:, :P95_P5]  = percentiles[:, :P95] - percentiles[:, :P5]

# Plot the P75 - P25, P90 - P10, P95 - P5 and median
@df percentiles plot(:year,  :P50, label = "P50", legend = :topleft, title = "Percentiles of the share of high-tech employment in MSAs", xlabel = "Year", label = "Share of high-tech employment")
@df percentiles plot!(:year, :P75_P25, label = "P75 - P25")
@df percentiles plot!(:year, :P90_P10, label = "P90 - P10")
@df percentiles plot!(:year, :P95_P5, label = "P95 - P5")

df = data_bds_ht_msa_new[
	(data_bds_ht_msa_new.year .== 2000) .| (data_bds_ht_msa_new.year .== 2020) ,
	[:year, :msa,
    :job_creation_ht_rank_variate,
    :job_creation_births_ht_rank_variate,
    :job_creation_continuers_ht_rank_variate,
    :job_creation_rate_births_ht_rank_variate,
    :job_creation_rate_ht_rank_variate,
    :job_destruction_ht_rank_variate,
    :job_destruction_deaths_ht_rank_variate,
    :job_destruction_continuers_ht_rank_variate,
    :job_destruction_rate_deaths_ht_rank_variate,
    :job_destruction_rate_ht_rank_variate]]


df = outerjoin(
    unstack(df,:msa, :year, :job_creation_ht_rank_variate ),
    unstack(df,:msa, :year, :job_creation_births_ht_rank_variate ),
    unstack(df,:msa, :year, :job_creation_continuers_ht_rank_variate ),
    unstack(df,:msa, :year, :job_creation_rate_births_ht_rank_variate ),
    unstack(df,:msa, :year, :job_creation_rate_ht_rank_variate ),
    unstack(df,:msa, :year, :job_destruction_ht_rank_variate ),
    unstack(df,:msa, :year, :job_destruction_deaths_ht_rank_variate ),
    unstack(df,:msa, :year, :job_destruction_continuers_ht_rank_variate ),
    unstack(df,:msa, :year, :job_destruction_rate_deaths_ht_rank_variate ),
    unstack(df,:msa, :year, :job_destruction_rate_ht_rank_variate ),
    on = :msa, makeunique = true
	)
    
	
df[:, :job_creation_ht_diff] = df[:, "2000"] .- df[:, "2020"]
df[:, :job_creation_births_ht_diff] = df[:, "2000_1"] .- df[:, "2020_1"] 
df[:, :job_creation_continuers_ht_diff] = df[:, "2000_2"] .- df[:, "2020_2"]
df[:, :job_creation_rate_births_ht_diff] = df[:, "2000_3"] .- df[:, "2020_3"]
df[:, :job_creation_rate_ht_diff] = df[:, "2000_4"] .- df[:, "2020_4"]
df[:, :job_destruction_ht_diff] = df[:, "2000_5"] .- df[:, "2020_5"]
df[:, :job_destruction_deaths_ht_diff] = df[:, "2000_6"] .- df[:, "2020_6"]
df[:, :job_destruction_continuers_ht_diff] = df[:, "2000_7"] .- df[:, "2020_7"]
df[:, :job_destruction_rate_deaths_ht_diff] = df[:, "2000_8"] .- df[:, "2020_8"]
df[:, :job_destruction_rate_ht_diff] = df[:, "2000_9"] .- df[:, "2020_9"]

# Keep only the columns we need diff
df = df[:, [:msa, 
            :job_creation_ht_diff,
            :job_creation_births_ht_diff,
            :job_creation_continuers_ht_diff,
            :job_creation_rate_births_ht_diff,
            :job_creation_rate_ht_diff,
            :job_destruction_ht_diff,
            :job_destruction_deaths_ht_diff,
            :job_destruction_continuers_ht_diff,
            :job_destruction_rate_deaths_ht_diff,
            :job_destruction_rate_ht_diff]]

begin
    p_jc =  @df df scatter(
            :share_ht_diff, 
            :job_creation_ht_diff, 
            xlabel = "Δ share of high-tech employment",
            label = "Δ high-tech job creation")
    p_jc_b = @df df scatter(
            :share_ht_diff, 
            :job_creation_births_ht_diff, 
            xlabel = "Δ share of high-tech employment",
            label = "Δ high-tech job (births) creation")
    p_jc_c = @df df scatter(
            :share_ht_diff, 
            :job_creation_continuers_ht_diff, 
            xlabel = "Δ share of high-tech employment",
            label = "Δ high-tech job (continuers) creation")
    # p_jc_rb = @df df scatter(
    #         :share_ht_diff, 
    #         :job_creation_rate_births_ht_diff, 
    #         xlabel = "Δ share of high-tech employment",
    #         label = "Δ high-tech job creation rate (births)")
    # p_jc_r = @df df scatter(
    #         :share_ht_diff, 
    #         :job_creation_rate_ht_diff, 
    #         xlabel = "Δ share of high-tech employment",
    #         label = "Δ high-tech job creation rate")
    p_jd = @df df scatter(
            :share_ht_diff, 
            :job_destruction_ht_diff, 
            xlabel = "Δ share of high-tech employment",
            label = "Δ high-tech job destruction")
    j_jd_d = @df df scatter(
            :share_ht_diff, 
            :job_destruction_deaths_ht_diff, 
            xlabel = "Δ share of high-tech employment",
            label = "Δ high-tech job (deaths) destruction")
    p_jd_c = @df df scatter(
            :share_ht_diff, 
            :job_destruction_continuers_ht_diff, 
            xlabel = "Δ share of high-tech employment",
            label = "Δ high-tech job (continuers) destruction")
    # p_jd_rd = @df df scatter(
    #         :share_ht_diff, 
    #         :job_destruction_rate_deaths_ht_diff, 
    #         xlabel = "Δ share of high-tech employment",
    #         label = "Δ high-tech job destruction rate (deaths)")
    # p_jd_r = @df df scatter(
    #         :share_ht_diff, 
    #         :job_destruction_rate_ht_diff, 
    #         xlabel = "Δ share of high-tech employment",
    #         label = "Δ high-tech job destruction rate")
    plot(p_jc, p_jc_b, p_jc_c, p_jd, j_jd_d, p_jd_c, layout = (3,2), size = (1000, 1000))
end
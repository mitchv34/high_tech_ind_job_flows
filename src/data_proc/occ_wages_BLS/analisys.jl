using CSV
using DataFrames
using Plots
using Statistics
using StatsPlots
using FixedEffectModels
using RegressionTables

## Configure Plots
theme(:juno) 
default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

# # Questions:
# 1. Distributions of wages in high tech industries

data_agg_path = "/project/high_tech_ind/high_tech_job_flows/data/BLS/proc/ind_occ/agg_bls_data.csv"
data_agg_wages_path = "/project/high_tech_ind/high_tech_job_flows/data/BLS/proc/ind_occ/agg_bls_data_ht_stem.csv"


data_agg_wages = CSV.read(data_agg_wages_path, DataFrame);

@df subset(data_agg_wages, :HT => ByRow(==(true)))  plot(:YEAR, :H_MEAN, group = :STEM, c = [1 2], lw=2, label = ["HT + Non-STEM" "HT + STEM"])
@df subset(data_agg_wages, :HT => ByRow(!=(true))) plot!(:YEAR, :H_MEAN, group = :STEM, c = [1 2], lw=2, label = ["Non-HT + Non-STEM" "Non-HT + STEM"], linestyle = :dash)


data_agg = CSV.read(data_agg_path, DataFrame);

# Wage dispersion

data_agg.H_P9010 = data_agg.H_PCT90 - data_agg.H_PCT10
data_agg.A_P9010 = data_agg.A_PCT90 - data_agg.A_PCT10

data_agg_mean = combine(groupby(data_agg, [:YEAR, :STEM, :HT]), :H_P9010 => mean,  :A_P9010 => mean)

@df subset(data_agg_mean, :HT => ByRow(==(true)))  plot(:YEAR, :H_P9010_mean, group = :STEM, c = [1 2], lw=2, label = ["HT + Non-STEM" "HT + STEM"])
@df subset(data_agg_mean, :HT => ByRow(!=(true))) plot!(:YEAR, :H_P9010_mean, group = :STEM, c = [1 2], lw=2, label = ["Non-HT + Non-STEM" "Non-HT + STEM"], linestyle = :dash)

## Workforce composition
data_agg_wide = unstack(data_agg[:, [:YEAR, :NAICS, :OCC_CODE, :TOT_EMP]], [:YEAR, :NAICS], :OCC_CODE, :TOT_EMP,  allowduplicates=true )
data_agg_wide = ifelse.(ismissing.(data_agg_wide), 0, data_agg_wide)
data_agg_wide.TOT_EMP = sum(Matrix(data_agg_wide[:, 3:end]), dims=2)[:]
data_agg_wide.HHI = sum(Matrix(data_agg_wide[:, 3:end-1] ./ data_agg_wide.TOT_EMP).^2, dims=2)[:]

@df data_agg_wide plot(:YEAR, :HHI, group = :NAICS)
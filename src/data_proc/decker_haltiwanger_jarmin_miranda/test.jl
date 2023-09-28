using CSV
using DataFrames
using Plots
using StatsPlots
using FixedEffectModels
using Missings
using Statistics

# Download data

data_bds = CSV.read(download("https://www2.census.gov/programs-surveys/bds/tables/time-series/bds2020_vcn4_fa.csv"), DataFrame)

# Convert fage colum to categories a,b,c...l
data_bds[!, :fage] = [x[1] for x in data_bds[!, :fage]]

# Drop "l" category (f => “Left Censored”.)
data_bds = subset(data_bds, :fage => ByRow(!=('l')))

# subset to post 1992 data
data_bds = subset(data_bds, :year => ByRow(>=(1992)))

# Replace "(X)" with 0 in all columns
# Replace "(D)" and "(S)" with missing in all columns
# Replace "." with Nan in all columns
# Convert all columns to numeric
allowmissing!(data_bds)
for col in names(data_bds)[4:end]
    replace!(data_bds[!, col], "(X)" => "0")
    replace!(data_bds[!, col], "(D)" => missing)
    replace!(data_bds[!, col], "(S)" => missing)
    replace!(data_bds[!, col], "." => missing)
    data_bds[!, col] = passmissing(parse).(Float64, data_bds[!, col])
end

dropmissing!(data_bds)

ht_ind_codes = [3341, 3342, 3344, 3345, 3364, 5112, 5182, 5191, 5413, 5415, 5417]

# Tag all industries as high-tech or not
data_bds[!, :ht] = [x in ht_ind_codes for x in data_bds[!, :vcnaics4]]

# Aggregate to ht and non-ht industries, sum all columns except year, fage, and ht and anything containing "rate"
aggregated_data_bds_ht = combine(groupby(data_bds, [:year, :fage, :ht]), names(data_bds)[4:end][.!contains.(names(data_bds)[4:end], "rate")].=>sum)

# Add up without the ht column 
aggregated_data_bds = combine(groupby(data_bds, [:year, :fage]), names(data_bds)[4:end-1][.!contains.(names(data_bds)[4:end-1], "rate")].=>sum.=> names(data_bds)[4:end-1][.!contains.(names(data_bds)[4:end-1], "rate")])

# Calcuate averages for all columns except year, fage
aggregated_data_bds_fage = combine(groupby(aggregated_data_bds, [:fage]), names(aggregated_data_bds)[3:end].=>mean.=> names(aggregated_data_bds)[3:end])

plot(aggregated_data_bds_fage.job_destruction_deaths ./ sum(aggregated_data_bds_fage.job_destruction_deaths))
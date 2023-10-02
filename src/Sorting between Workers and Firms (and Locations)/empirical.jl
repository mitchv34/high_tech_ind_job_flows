using DataFrames
using CSV
using StatsPlots
using StatsBase
using Printf

# Some helper functions
# Define a custom formatting function
function my_yformatter(val)
    return @sprintf("%1.1fk", val/1000)
end

function plot_metro(metro::String, data_agg::DataFrame; 
                    initial_year::Int64=1900, educ::String = "ALL")

    #subset by metro
    data = subset(data_agg, :MSA => ByRow(==(metro))) 
    msa_name = data.MSA_NAME[1]
    # Subset or aggregate by educ
    if educ == "ALL"
        data = combine(groupby(data, :YEAR), :NET_MIGRATION => sum => :NET_MIGRATION)
    else
        data = subset(data, :COLLEGE => ByRow(==(educ)))
    end
    # Filter by initial year 
    data = subset(data, :YEAR => ByRow( >=(initial_year)))
    data[:, :NET_MIGRATION_NEGATIVE] = data[:, :NET_MIGRATION] .< 0
    p = @df data bar(:YEAR, :NET_MIGRATION, group = :NET_MIGRATION_NEGATIVE,
                    lw = 0, bar_width = 0.8, c = [2 1],
                    legend = :none, ylabel = "Net Migration", yformatter = my_yformatter, size = (800, 500))
    xticks!(p, 2000:5:2021)
    # Add some space on the y axis (top and bottom)
    hline!(p, [0], color = :white, lw = 1, label = "")
    title!(msa_name)
    return p
end

data_migration = CSV.read( "/project/high_tech_ind/high_tech_job_flows/data/acs/proc/acs_msa_counts_educ.csv"
, DataFrame)
data_migration.MSA = String.(data_migration.MSA)

# Drop non metro areas
data_migration = subset(data_migration, :MSA => ByRow( x -> ! occursin("X" , x  ) ) )

# Data Metro  population cost
path_metro_data = "/project/high_tech_ind/ht_job_flows/data/bea/BEA_MSA.csv"
data_metro = CSV.read(path_metro_data, DataFrame); 
# Rename GeoFips => MSA
rename!(data_metro, :GeoFips => :MSA)
# Convert MSA codes to string
data_metro.MSA = string.(data_metro.MSA)
# Keep only the metros in the migration data
data_metro = subset(data_metro, :MSA => ByRow( x -> x in data_migration.MSA ) )
# Keep only the data in metro in migration
data_migration = subset(data_migration, :MSA => ByRow( x -> x in data_metro.MSA ) )

# Create a dictionary of MSA - MSA_NAME
d = data_migration[
    (data_migration.YEAR .== 2010) .& 
    (data_migration.COLLEGE .== "Non College"), 
    [:MSA, :MSA_NAME]]

# Normalize the migration data by population overall population
population = data_metro.Population |> sum
data_migration[!, [:INFLOW, :OUTFLOW, :NET_MIGRATION]]  = data_migration[!, [:INFLOW, :OUTFLOW, :NET_MIGRATION]] ./ population

plot_metro("12580", data_migration, initial_year = 2012, educ = "Non College")
plot_metro("12580", data_migration, initial_year = 2012, educ = "College")
plot_metro("12580", data_migration, initial_year = 2012)

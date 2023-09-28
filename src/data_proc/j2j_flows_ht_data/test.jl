using CSV
using DataFrames
using Plots
using StatsPlots

## Configure Plots
theme(:juno) 
default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style

## Load data
bds_ht_msac = CSV.read("high_tech_job_flows/data/bds_high_tech/interim/msac.csv", DataFrame)


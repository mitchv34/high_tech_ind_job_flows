# %%
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import os

# %%
# Load data

DATA_PATH_METRO = "/project/high_tech_ind/high_tech_ind_job_flows/data/OEWS/estimated_skill_distributions/msa_skill/"
DATA_PATH_NATIONAL = "/project/high_tech_ind/high_tech_ind_job_flows/data/OEWS/estimated_skill_distributions/national_skill/"
# List files 
files_metro = os.listdir(DATA_PATH_METRO)[5:6]
files_national = os.listdir(DATA_PATH_NATIONAL)[15:16]#[end-7: end]

# %%
# Load data
df_metro = pd.read_csv(DATA_PATH_METRO+files_metro[0])
df_national = pd.read_csv(DATA_PATH_NATIONAL+files_national[0])

# Get list of msa codes
msa_codes = df_metro["AREA"].unique()

# for each msa code, get the skill distribution
# and the skill distribution for the nation

# Substampe the dataframes
N  = 1000000
DIST = stats.lognorm

sub_df = df_metro[df_metro.AREA == 31860]
sample_data = sub_df.sample(n=N, weights='EMP', replace=True).SKILL.values
fitted = stats.fit(DIST, sample_data, bounds = {"s":[0,10]})
# %%
fitted_dist = DIST(*fitted.params)
# %%


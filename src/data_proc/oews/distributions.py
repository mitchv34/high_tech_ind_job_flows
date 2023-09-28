"""
This script uses the occupation distribution from OEWS and the skill/knowledge 
for each skill from ONET to estimate the distribution of skill/knowledge in each MSA.
"""
# %%
# * Packages
import pandas as pd
import numpy as np
import os
import re

# %%
# * Read in data
path_oews = '/project/high_tech_ind/high_tech_ind_job_flows/data/OEWS/msa/proc/'
path_onet = '/project/high_tech_ind/high_tech_ind_job_flows/data/onet/knowledge/agg/knowledge_df_28_SOC.csv'

# Load O*Net data
df_onet = pd.read_csv(path_onet)
# Rename column
df_onet.rename(columns={'SOC':'OCC_CODE', "KNOWLEDGE_SOC":"SKILL"}, inplace=True)

# for every  file in the folder read it and keep only the columns we need
colums = ["AREA", "AREA_NAME", "OCC_CODE", "TOT_EMP"]

files = os.listdir(path_oews)

for file in files:
    print(f"Processing {file}...")
    year = re.search(r'\d{4}', file).group()
    df_oews = pd.read_csv(path_oews + file, usecols=colums)
    # Convert TOT_EMP to numeric 
    df_oews['TOT_EMP'] = pd.to_numeric(df_oews['TOT_EMP'], errors='coerce')
    # Merge O*Net and OEWS data on OCC_CODE
    df_oews = pd.merge(df_oews, df_onet, on='OCC_CODE', how='left').dropna()
    # Convert TOT_EMP to to percentage of total employment in each MSA
    df_oews.loc[:, "EMP"] = df_oews.groupby('AREA').TOT_EMP.transform(lambda x: x/x.sum())

    # Save to csv
    df_oews.to_csv(f"/project/high_tech_ind/high_tech_ind_job_flows/data/OEWS/estimated_skill_distributions/msa_skill/msa_skill_{year}.csv", index=False)



# %%

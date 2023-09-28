# %%
import pandas as pd
import numpy as np

data_path = "/project/high_tech_ind/high_tech_ind_job_flows/data/acs/interim/"

data_proc_path = "/project/high_tech_ind/high_tech_job_flows/data/acs/proc/"

# Load data
data = pd.read_csv(data_path + "acs_movers.csv.gz", compression='gzip', low_memory=False)

# Replace 1 and 0 with College and Non College in COLLEGE column
# data['COLLEGE'] = np.where(data['COLLEGE'] == 1, 'College', 'Non College')

# Replace " Metropolitan Statistical Area" with "" in MSANAME and MSANAMEMIG columns
data['MSANAME'] = data['MSANAME'].str.replace(" Metropolitan Statistical Area", "")
data['MSANAMEMIG'] = data['MSANAMEMIG'].str.replace(" Metropolitan Statistical Area", "")


# %%
# First I will just count the number of people who moved from each MSA to each other MSA
counts = data.groupby(['MSA', 'MSAMIG', 'YEAR']).PERWT.sum().reset_index()
# Add names using MSA_NAME column in data
counts['MSA_NAME'] = counts['MSA'].map(dict(zip(data.MSA, data.MSANAME)))
counts['MSAMIG_NAME'] = counts['MSAMIG'].map(dict(zip(data.MSAMIG, data.MSANAMEMIG)))
# Rename PERWT to count
counts.rename(columns={'PERWT': 'COUNT'}, inplace=True)
counts.sort_values("YEAR", ascending=True, inplace=True)
# Save to csv
counts.to_csv(data_proc_path + "acs_msa_flow_counts.csv", index=False)


# %%
# Count for every metro the inflow and outflow of people
iflows = counts.groupby(["MSA", "YEAR"]).COUNT.sum().reset_index().rename(columns={"COUNT": "INFLOW"})
oflows = counts.groupby(["MSAMIG", "YEAR"]).COUNT.sum().reset_index().rename(columns={"MSAMIG":"MSA", "COUNT": "OUTFLOW"})
count_msa_totals = iflows.merge(oflows, how = "inner", on=["MSA", "YEAR"])
# Create NET_MIGRATION column
count_msa_totals['NET_MIGRATION'] = count_msa_totals['INFLOW'] - count_msa_totals['OUTFLOW']
# Add names using MSA_NAME column in data
count_msa_totals['MSA_NAME'] = count_msa_totals['MSA'].map(dict(zip(data.MSA, data.MSANAME)))
# Save to csv
count_msa_totals.to_csv(data_proc_path + "acs_msa_counts.csv", index=False)


# %%
# Repeat splitting by education
counts_educ = data.groupby(['MSA', 'MSAMIG', 'YEAR', 'COLLEGE']).PERWT.sum().reset_index()
# Add names using MSA_NAME column in data
counts_educ['MSA_NAME'] = counts_educ['MSA'].map(dict(zip(data.MSA, data.MSANAME)))
counts_educ['MSAMIG_NAME'] = counts_educ['MSAMIG'].map(dict(zip(data.MSAMIG, data.MSANAMEMIG)))
# Rename PERWT to count
counts_educ.rename(columns={'PERWT': 'COUNT'}, inplace=True)
counts_educ.sort_values("YEAR", ascending=True, inplace=True)
# Save to csv
counts_educ.to_csv(data_proc_path + "acs_msa_flow_counts_educ.csv", index=False)


# %%
# Count for every metro the inflow and outflow of people
iflows_educ = counts_educ.groupby(["MSA", "YEAR", "COLLEGE"]).COUNT.sum().reset_index().rename(columns={"COUNT": "INFLOW"})
oflows_educ = counts_educ.groupby(["MSAMIG", "YEAR", "COLLEGE"]).COUNT.sum().reset_index().rename(columns={"MSAMIG":"MSA", "COUNT": "OUTFLOW"})
count_msa_totals_educ = iflows_educ.merge(oflows_educ, how = "inner", on=["MSA", "YEAR", "COLLEGE"])
# Create NET_MIGRATION column
count_msa_totals_educ['NET_MIGRATION'] = count_msa_totals_educ['INFLOW'] - count_msa_totals_educ['OUTFLOW']
# Add names using MSA_NAME column in data
count_msa_totals_educ['MSA_NAME'] = count_msa_totals_educ['MSA'].map(dict(zip(data.MSA, data.MSANAME)))
# Save to csv
count_msa_totals_educ.to_csv(data_proc_path + "acs_msa_counts_educ.csv", index=False)
# %%

# ! Testing zone
data_2019 = data[data['YEAR'] == 2019]
# %%

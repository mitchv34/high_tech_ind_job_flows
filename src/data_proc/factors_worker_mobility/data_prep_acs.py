# %% 
import pandas as pd

# %%
data = pd.read_csv('/project/high_tech_ind/high_tech_job_flows/data/acs/usa_00004.csv.gz',
                     compression='gzip', low_memory=False)

# %%
# Filter the data to only include the variables we need
data = data[[
    'YEAR', 'STATEFIP', 'COUNTYFIP', 'CITY', 'PERWT', 'SEX', 'AGE',
    'RACE', 'RACED', 'EDUC', 'EDUCD', 'OCC2010', 'IND1990', 'INDNAICS',
    'MIGRATE1', 'MIGRATE1D', 'MIGPLAC1', 'MIGCOUNTY1', 'MIGMET1']]
data = data[data.YEAR >= 1990] # Only include data from 1990 onwards
data = data[data.MIGRATE1.isin([1,2])] # Only include people who moved in the last year (within the US)
data.dropna(subset=['MIGCOUNTY1'], inplace=True) # Drop people who didn't report where they moved from
data = data[data.MIGCOUNTY1 != 0] #000 = Not in universe, or county not identifiable from public-use data

# %%
# Construct metro area from STATEFIP and COUNTYFIP
data['METRO_now'] = data.STATEFIP.astype(str).str.zfill(2) + data.COUNTYFIP.astype(str).str.zfill(3)
data['METRO_before'] = data.MIGCOUNTY1.astype(str).str.zfill(5)

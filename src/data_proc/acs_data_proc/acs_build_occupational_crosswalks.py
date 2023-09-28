# %%
import pandas as pd
import numpy as np

# %% 
#* Create Crosswalks to 2018 SOC

# URLS
path_2000_2010 = "/project/high_tech_ind/high_tech_ind_job_flows/data/aux/raw/soc_2000_to_2010_crosswalk.xls"
path_2010_2018 = "/project/high_tech_ind/high_tech_ind_job_flows/data/aux/raw/soc_2010_to_2018_crosswalk.xlsx"

# Read in the data
SOC_2000_2010 = pd.read_excel(path_2000_2010, skiprows=6)[["2000 SOC code", "2010 SOC code"]].dropna()
# Capitalize columns and remove spaces
SOC_2000_2010.rename(columns={"2000 SOC code" : "SOC_2000", "2010 SOC code" : "SOC_2010"}, inplace=True)

SOC_2010_2018 = pd.read_excel(path_2010_2018, skiprows=8)[["2010 SOC Code", "2018 SOC Code"]].dropna()
# Capitalize columns and remove spaces
SOC_2010_2018.rename(columns={"2010 SOC Code" : "SOC_2010", "2018 SOC Code" : "SOC_2018"}, inplace=True)
# Merge the two crosswalks
SOC_2000_2018 = pd.merge(SOC_2000_2010, SOC_2010_2018, left_on="SOC_2010", right_on="SOC_2010")

# Create  dictionaries to use as crosswalks
soc_2000_2010 = dict(zip(SOC_2000_2010.SOC_2000, SOC_2000_2018.SOC_2010))
soc_2000_2018 = dict(zip(SOC_2000_2018.SOC_2000, SOC_2000_2018.SOC_2018))
soc_2018_2000 = dict(zip(SOC_2000_2018.SOC_2018, SOC_2000_2018.SOC_2000))

# % 
#* Create Crosswalks to from OCC to SOC

# Crosswalks 1
path_2010 = "/project/high_tech_ind/high_tech_ind_job_flows/data/aux/raw/soc2010_occ2010.xlsx"
path_2018 = "/project/high_tech_ind/high_tech_ind_job_flows/data/aux/raw/soc2018_occ2018.xlsx"

# Crosswalks 2
url_2010 = "https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2010-occ-codes-with-crosswalk-from-2002-2011.xls"
url_2018 = "https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2018-occupation-code-list-and-crosswalk.xlsx"


# %%
# ?  2010 OCC to SOC crosswalk

# Load 2010 OCC to SOC crosswalks
occ2010_soc2010 = pd.read_excel(path_2010,  skiprows = 4, usecols = [1,3], dtype=str).dropna()
occ2010_soc2010_v2 = pd.read_excel(url_2010, sheet_name = "2010OccCodeList", usecols = [2,3], skiprows=4, dtype=str).dropna()
# Rename columns
occ2010_soc2010.rename(columns={"SOC 2010 Code" : "SOC2010", "ACS Code" : "OCC2010"}, inplace=True) 
occ2010_soc2010_v2.rename(columns={"2010 Census Code" :"OCC2010", "2010 SOC Code" : "SOC2010"}, inplace=True)

# Srip spaces from SOC2010 and OCC2010
occ2010_soc2010.SOC2010 = occ2010_soc2010.SOC2010.str.strip()
occ2010_soc2010.OCC2010 = occ2010_soc2010.OCC2010.str.strip()
occ2010_soc2010_v2.SOC2010 = occ2010_soc2010_v2.SOC2010.str.strip()
occ2010_soc2010_v2.OCC2010 = occ2010_soc2010_v2.OCC2010.str.strip()

# Re-order columns 
occ2010_soc2010_v2 = occ2010_soc2010_v2[["SOC2010", "OCC2010"]]

# Clean Second Crosswalk 
occ2010_soc2010_v2 = occ2010_soc2010_v2[~occ2010_soc2010_v2.OCC2010.str.contains("-")]
occ2010_soc2010_v2 = occ2010_soc2010_v2[~occ2010_soc2010_v2.SOC2010.str.contains("none")]

# Fill ACS Code with zeros lenght 4
occ2010_soc2010["OCC2010"] = occ2010_soc2010["OCC2010"].apply(lambda x: x.zfill(4))
occ2010_soc2010_v2["OCC2010"] = occ2010_soc2010_v2["OCC2010"].apply(lambda x: x.zfill(4))

# Merge the two crosswalks (and drop duplicates)
occ2010_soc2010 = pd.concat([occ2010_soc2010_v2, occ2010_soc2010]).drop_duplicates()

# %%

# Load OCC to SOC crosswalk (2018)

# ? 2018 OCC to SOC crosswalk

# Load 2018 OCC to SOC crosswalks
occ2018_soc2018 = pd.read_excel(path_2018,  skiprows = 4, usecols = [1,3], dtype=str).dropna()
occ2018_soc2018_v2 = pd.read_excel(url_2018, sheet_name = "2018 Census Occ Code List", usecols = [2,3], skiprows=4, dtype=str).dropna()
# Rename columns
occ2018_soc2018.rename(columns={"Matrix Occupation Code" : "SOC2018", "ACS Code" : "OCC2018"}, inplace=True)
occ2018_soc2018_v2.rename(columns={"2018 Census Code" :"OCC2018", "2018 SOC Code" : "SOC2018"}, inplace=True)

# Srip spaces from SOC2018 and OCC2018
occ2018_soc2018.SOC2018 = occ2018_soc2018.SOC2018.str.strip()
occ2018_soc2018.OCC2018 = occ2018_soc2018.OCC2018.str.strip()
occ2018_soc2018_v2.SOC2018 = occ2018_soc2018_v2.SOC2018.str.strip()
occ2018_soc2018_v2.OCC2018 = occ2018_soc2018_v2.OCC2018.str.strip()

# Re-order columns
occ2018_soc2018_v2 = occ2018_soc2018_v2[["SOC2018", "OCC2018"]]
# Clean Second Crosswalk
occ2018_soc2018_v2 = occ2018_soc2018_v2[~occ2018_soc2018_v2.OCC2018.str.contains("-")]
occ2018_soc2018_v2 = occ2018_soc2018_v2[~occ2018_soc2018_v2.SOC2018.str.contains("none")]

# Fill ACS Code with zeros lenght 4
occ2018_soc2018["OCC2018"] = occ2018_soc2018["OCC2018"].apply(lambda x: x.zfill(4))
occ2018_soc2018_v2["OCC2018"] = occ2018_soc2018_v2["OCC2018"].apply(lambda x: x.zfill(4))

# Merge the two crosswalks (and drop duplicates)
occ2018_soc2018 = pd.concat([occ2018_soc2018_v2, occ2018_soc2018]).drop_duplicates()


# %%
# Load OCC to SOC crosswalk (2002)
occ2002_soc2002 = pd.read_excel("https://www2.census.gov/programs-surveys/demo/guidance/industry-occupation/2002-census-occupation-codes.xls", skiprows = 2, usecols = [2,3], dtype=str).dropna()
occ2002_soc2002.rename(columns={"2002 Census Code":"OCC2002", "2002 SOC Code":"SOC2002"}, inplace=True)
occ2002_soc2002 = occ2002_soc2002[~occ2002_soc2002.OCC2002.str.contains("-")]
occ2002_soc2002 = occ2002_soc2002[occ2002_soc2002.SOC2002 != "none"]
occ2002_soc2002.SOC2002 = occ2002_soc2002.SOC2002.str.strip()
occ2002_soc2002.OCC2002 = occ2002_soc2002.OCC2002.str.strip()
# # occ2002_soc2002 = dict( zip( occ2002_soc2002.OCC2002, occ2002_soc2002.SOC2002 ) )
# %%
# Save all crosswalks to file

# Save SOC crosswalks
SOC_2000_2018.to_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/soc_2000_2018_crosswalk.csv", index=False)
SOC_2000_2010.to_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/soc_2000_2010_crosswalk.csv", index=False)
SOC_2010_2018.to_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/soc_2010_2018_crosswalk.csv", index=False)

# Save OCC to SOC crosswalks
occ2010_soc2010.to_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/occ2010_soc2010_crosswalk.csv", index=False)
occ2002_soc2002.to_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/occ2002_soc2002_crosswalk.csv", index=False)
occ2018_soc2018.to_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/occ2018_soc2018_crosswalk.csv", index=False)


# %%

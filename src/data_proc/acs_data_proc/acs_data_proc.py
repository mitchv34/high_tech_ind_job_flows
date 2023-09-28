# %%
import pandas as pd
import numpy as np
from ipumspy import readers
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

download_dir =  "/project/high_tech_ind/high_tech_ind_job_flows/data/acs/occupation"

with open(download_dir + "/raw/directory.json", "r") as f:
    data_paths = json.load(f)

# Load the crosswalk of PUMA to MSA
# TODO: Save all croswalks to files
path_puma2msa = '/project/high_tech_ind/high_tech_job_flows/data/acs/aux/puma2cbsa.csv'
df_crosswalk_puma_msa = pd.read_csv(path_puma2msa, encoding='latin-1')
# Drop first row
df_crosswalk_puma_msa = df_crosswalk_puma_msa.iloc[1:]
# Drop State 72 (Puerto Rico)
df_crosswalk_puma_msa = df_crosswalk_puma_msa[df_crosswalk_puma_msa['state'] != 72]
# Contruct a dictionry with traslation from
# state + puma22 top cbsa20 if cbsatype20 != Metro then assing SSXXX where SS is the state FIP code
puma2cbsa = {}
puma2cbsaname = {}
for index, row in df_crosswalk_puma_msa.iterrows():
    if row['cbsatype10'] == 'Metro':
        puma2cbsa[row['state'] + row['puma12']] = row['cbsa10']
        puma2cbsaname[row['state'] + row['puma12']] = row['cbsaname10']
    else:
        puma2cbsa[row['state'] + row['puma12']] = str(row['state']) + 'XXX'
        puma2cbsaname[row['state'] + row['puma12']] = row["stab"] + " Non-Metro"

# Load Croswalk from PUMA00 to PUMA10
url_crosswalk = 'https://usa.ipums.org/usa/resources/volii/PUMA2000_PUMA2010_crosswalk.xls'
df_crosswalk_puma_00_10 = pd.read_excel(url_crosswalk, usecols=["State00", 'PUMA00', 'PUMA10', "State10"])
# Pad with zeros PUMAs and States
df_crosswalk_puma_00_10['PUMA00'] = df_crosswalk_puma_00_10['PUMA00'].astype(str).str.zfill(5)
df_crosswalk_puma_00_10['PUMA10'] = df_crosswalk_puma_00_10['PUMA10'].astype(str).str.zfill(5)
df_crosswalk_puma_00_10['State00'] = df_crosswalk_puma_00_10['State00'].astype(str).str.zfill(2)
df_crosswalk_puma_00_10['State10'] = df_crosswalk_puma_00_10['State10'].astype(str).str.zfill(2)
# Load Croswalk from MIgPUMA00 to MIGPUMA10
url_crosswalk = "https://usa.ipums.org/usa/resources/volii/MIGPUMA2000_MIGPUMA2010_crosswalk.xls"
df_crosswalk_migpuma_00_10 = pd.read_excel(url_crosswalk, usecols=["State00", 'MigPUMA00', 'MigPUMA10', "State10"])
# Pad with zeros PUMAs and States
df_crosswalk_migpuma_00_10['MigPUMA00'] = df_crosswalk_migpuma_00_10['MigPUMA00'].astype(str).str.zfill(5)
df_crosswalk_migpuma_00_10['MigPUMA10'] = df_crosswalk_migpuma_00_10['MigPUMA10'].astype(str).str.zfill(5)
df_crosswalk_migpuma_00_10['State00'] = df_crosswalk_migpuma_00_10['State00'].astype(str).str.zfill(2)
df_crosswalk_migpuma_00_10['State10'] = df_crosswalk_migpuma_00_10['State10'].astype(str).str.zfill(2)

# Drop State 72 (Puerto Rico)
df_crosswalk_puma_00_10 = df_crosswalk_puma_00_10[df_crosswalk_puma_00_10['State00'] != '72']
df_crosswalk_migpuma_00_10 = df_crosswalk_migpuma_00_10[df_crosswalk_migpuma_00_10['State00'] != '72']

# Create a dictionary of PUMA00 to PUMA10 and MIGPUMA00 to MIGPUMA10
puma00_10 = dict(zip(df_crosswalk_puma_00_10['State00'] + df_crosswalk_puma_00_10['PUMA00'], df_crosswalk_puma_00_10['State10'] + df_crosswalk_puma_00_10['PUMA10']))
migpuma00_10 = dict(zip(df_crosswalk_migpuma_00_10['State00'] + df_crosswalk_migpuma_00_10['MigPUMA00'], df_crosswalk_migpuma_00_10['State10'] + df_crosswalk_migpuma_00_10['MigPUMA10']))

# Load the crosswalk of MigPUMA to PUMA
url_crosswalk = "https://usa.ipums.org/usa/resources/volii/puma_migpuma1_pwpuma00.xls"
df_crosswalk_migpuma_puma = pd.read_excel(url_crosswalk, skiprows=2, skipfooter=5)#, usecols=["State00", 'MigPUMA00', 'PUMA00', "State10"])
# Rename columns
df_crosswalk_migpuma_puma.columns = ["STFIP", "PUMA", "MIGPLAC1", "MIGPUMA1"]
# Pad with zeros PUMAs and States
df_crosswalk_migpuma_puma['PUMA'] = df_crosswalk_migpuma_puma['PUMA'].astype(str).str.zfill(5)
df_crosswalk_migpuma_puma['MIGPLAC1'] = df_crosswalk_migpuma_puma['MIGPLAC1'].astype(str).str.zfill(2)
df_crosswalk_migpuma_puma['MIGPUMA1'] = df_crosswalk_migpuma_puma['MIGPUMA1'].astype(str).str.zfill(5)
df_crosswalk_migpuma_puma['STFIP'] = df_crosswalk_migpuma_puma['STFIP'].astype(str).str.zfill(2)
# Create a dictionary of MigPUMA to PUMA
migpuma_puma = dict(
    zip(
        df_crosswalk_migpuma_puma.MIGPLAC1 + df_crosswalk_migpuma_puma.MIGPUMA1,
        df_crosswalk_migpuma_puma.STFIP + df_crosswalk_migpuma_puma.PUMA))


# %%
def proc_acs(year):
    DOWNLOAD_DIR = Path(data_paths[year])
    ddi_file = list(DOWNLOAD_DIR.glob("*.xml"))[0]
    ddi = readers.read_ipums_ddi(ddi_file)
    data = readers.read_microdata(ddi, DOWNLOAD_DIR / ddi.file_description.filename)
    df = data.copy()
    # Get the data
    # Convert to string and pad with zeros
    df['MIGPLAC1'] = df['MIGPLAC1'].astype(str).str.zfill(2)
    df['STATEFIP'] = df['STATEFIP'].astype(str).str.zfill(2)
    df['PUMA']     = df['PUMA'].astype(str).str.zfill(5)
    df['MIGPUMA1'] = df['MIGPUMA1'].astype(str).str.zfill(5)
    df['OCC']      = df['OCC'].astype(str).str.zfill(4)

    # Filter MIGPUMA1 to exclude the following codes 
    # 00000 = N/A (less than 1 year old or lived in same residence 1 year ago)
    # 00001 = Did not live in the United States or in Puerto Rico one year ago
    # 00002 = Lived in Puerto Rico one year ago and current residence is in the U.S.
    df = df[~df.MIGPUMA1.isin(["00000", "00001", "00002"])]
    # Create a new variable for the state and PUMA
    df['SPUMA']    = df['STATEFIP'] + df['PUMA']
    df['SPUMAMIG'] = df['MIGPLAC1'] + df['MIGPUMA1']
    # If year is before 2010, convert PUMA to 2010 PUMAs
    if int(year) < 2012:
        df["SPUMA"] = df.SPUMA.map(puma00_10)
        df["SPUMAMIG"] = df.SPUMAMIG.map(migpuma00_10)
    # Convert MIGPUMA1 to PUMA
    df.loc[:, "SPUMAMIG"] = df.SPUMAMIG.map(migpuma_puma)
    # Aggregate to MSA
    df.loc[:, "MSA"] = df.SPUMA.map(puma2cbsa)
    df.loc[:, "MSAMIG"] = df.SPUMAMIG.map(puma2cbsa)
    # Filter out non-movers (MSA level)
    df = df[df.MSA != df.MSAMIG]
    # Add MSA names
    df.loc[:, "MSANAME"] = df.SPUMA.map(puma2cbsaname)
    df.loc[:, "MSANAMEMIG"] = df.SPUMAMIG.map(puma2cbsaname)

    # # Drop anyone with missing education (EDUCD = 999)
    # df = df[df.EDUCD != 999]
    # # Drop anyone with no schooling or N/A (EDUCD = 0, 1, 2)
    # df = df[~df.EDUCD.isin([0, 1, 2])]
    # # Tag as college educated more than 4 years of college (EDUCD > 100)
    # df = df.assign(COLLEGE = lambda x: np.where(x.EDUCD > 100, 1, 0))

    # Add year
    df["YEAR"] = int(year)

    # return df[["YEAR", "PERWT", "COLLEGE", "SPUMA", "MSA",  "SPUMAMIG", "MSAMIG", "MSANAME", "MSANAMEMIG"]]
    return df[["YEAR", "PERWT", "OCC", "SPUMA", "MSA",  "SPUMAMIG", "MSAMIG", "MSANAME", "MSANAMEMIG"]]

# %%

# Load the crosswalk of OCC to SOC
occ2002_soc2002 = pd.read_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/occ2002_soc2002_crosswalk.csv", dtype=str) 
occ2010_soc2010 = pd.read_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/occ2010_soc2010_crosswalk.csv", dtype=str)
occ2018_soc2018 = pd.read_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/occ2018_soc2018_crosswalk.csv", dtype=str)

# Convert ot dictionaries
occ2002_soc2002 = dict(zip(occ2002_soc2002.OCC2002, occ2002_soc2002.SOC2002))
occ2010_soc2010 = dict(zip(occ2010_soc2010.OCC2010, occ2010_soc2010.SOC2010))
occ2018_soc2018 = dict(zip(occ2018_soc2018.OCC2018, occ2018_soc2018.SOC2018))



# Ceate empty dataframe
df = pd.DataFrame()

for year in ["2019"]:#data_paths.keys():
    print(f"Processing year {year}")
    # Process ACS data
    df_year = proc_acs(year)
    # Depenging on the year, convert OCC to SOC
    if year < "2010":
        df_year.loc[:, "SOC"] = df_year.OCC.map(occ2002_soc2002) 
    elif year < "2018":
        df_year.loc[:, "SOC"] = df_year.OCC.map(occ2010_soc2010)
    else:
        df_year.loc[:, "SOC"] = df_year.OCC.map(occ2018_soc2018)
    # Count nan in SOC
    print(f"Number of missing SOC codes: {df_year.SOC.isna().sum()}")
    # Print values OCC values with missing SOC
    print(f"OCC codes with missing SOC: {df_year.OCC[df_year.SOC.isna()].unique()}")
    # Drop missing SOC
    df_year = df_year.dropna(subset=["SOC"])
    break
    # Append to main dataframe
    df  = pd.concat([df, df_year], ignore_index=True)
# %%
# Save to disk
df.to_csv(download_dir + "/interim/acs_movers_occupations.csv.gz", index=False, compression="gzip")


# %%
# Read onet data
knowledge_SOC = pd.read_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/onet/knowledge/agg/knowledge_df_28_SOC.csv", dtype=str)
knowledge_OCC = pd.read_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/onet/knowledge/agg/knowledge_df_28_OCC.csv", dtype=str)

# Convert KNOWLEDGE_SOC and KNOWLEDGE_OCC to numeric
knowledge_SOC["KNOWLEDGE_SOC"] = pd.to_numeric(knowledge_SOC["KNOWLEDGE_SOC"])
knowledge_OCC["KNOWLEDGE_OCC"] = pd.to_numeric(knowledge_OCC["KNOWLEDGE_OCC"])

# Convert to dictionaries
knowledge_SOC = dict(zip(knowledge_SOC.SOC, knowledge_SOC.KNOWLEDGE_SOC))
knowledge_OCC = dict(zip(knowledge_OCC.OCC, knowledge_OCC.KNOWLEDGE_OCC))

# %%

# Map KNOWLEDGE to SOC and OCC
df_year["KNOWLEDGE_SOC"] = df_year.SOC.map(knowledge_SOC)
df_year["KNOWLEDGE_OCC"] = df_year.OCC.map(knowledge_OCC)
# Drop NaNs
df_year = df_year.dropna(subset=["KNOWLEDGE_SOC", "KNOWLEDGE_OCC"])

# %%
# Map KOWLEDGE to Discrete types using the percentiles
df_year["X_SOC"] = pd.qcut(df_year["KNOWLEDGE_SOC"], 3, labels=False)
df_year["X_OCC"] = pd.qcut(df_year["KNOWLEDGE_OCC"], 3, labels=False)
# %%
# Aggregate at the MSA, MSAMIG, X_SOC level
df_year_agg = df_year.groupby(["MSA", "MSAMIG", "X_SOC"]).PERWT.sum().reset_index()
# Drop non MSA to MSA flows (XX in MSAMIG or MSA)
df_year_agg = df_year_agg[~df_year_agg.MSAMIG.str.contains("X")]
df_year_agg = df_year_agg[~df_year_agg.MSA.str.contains("X")]
# %%

data_madison = df_year_agg[df_year_agg.MSA == "31540"]
data_madison.sort_values("X_SOC", inplace=True)
# For every skill level print top 3 desinations 
for skill in data_madison.X_SOC.unique():
    print(f"Skill level {skill}")
    print(data_madison[data_madison.X_SOC == skill].sort_values("PERWT", ascending=False).head(3))
    print("\n")

# %%


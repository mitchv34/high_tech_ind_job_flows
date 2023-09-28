# %%
import pandas as pd
import numpy as np
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
url_onet_model = "https://www.onetcenter.org/dl_files/database/db_28_0_text/Content%20Model%20Reference.txt"

onet_model = pd.read_csv(url_onet_model, sep="\t", encoding="latin-1")

onet_model[onet_model["Element ID"].apply(lambda x: len(x) == 1)]

# Subset to Worker Requirements (Element ID column starts with "2")
onet_model = onet_model[onet_model["Element ID"].apply(lambda x: x.startswith("2"))]

display(onet_model[onet_model["Element ID"].apply(lambda x: len(x) == 3)])

# Subset to basic skills (Element ID column starts with "2.A")
onet_model = onet_model[onet_model["Element ID"].apply(lambda x: x.startswith("2.C"))]


display(onet_model[onet_model["Element ID"].apply(lambda x: len(x) == 5)])

# Display all skills
display(onet_model[onet_model["Element ID"].apply(lambda x: len(x) == 7)])

# Filter to keep the basic level (Element ID 7 characters long)
onet_model = onet_model[onet_model["Element ID"].apply(lambda x: len(x) == 7)]

# Create a list of Knowledge
list_skills = onet_model[onet_model["Element ID"].apply(lambda x: x[4] in ["3", "4"])]["Element ID"].tolist()
print(list_skills)

# %%
# url_data = "https://www.onetcenter.org/dl_files/database/db_28_0_text/Skills.txt"
url_data = "https://www.onetcenter.org/dl_files/database/db_28_0_text/Knowledge.txt"


data_onet = pd.read_csv(url_data, sep="\t", encoding="latin-1")

# Filter on "Element ID" in list_skills
data_onet = data_onet[data_onet["Element ID"].isin(list_skills)]

# Set "Data Value" to 0 if "Recommend Suppress" == "Y"
# data_onet = data_onet[data_onet["Recommend Suppress"] != "Y"]

# Get corresponding OCC_CODE 
data_onet["OCC_CODE"] = data_onet["O*NET-SOC Code"].apply(lambda x : x.split(".")[0])

# Keep only relevant columns
data_onet = data_onet[["OCC_CODE", "Element ID", "Scale ID", "Data Value"]]

# TODO: Use pivot table instead of this mess
data_onet_importance = data_onet.loc[data_onet['Scale ID'] == "IM", ["Element ID", "OCC_CODE", "Data Value"]].reset_index(drop=True)
data_onet_importance.rename(columns={"Data Value": "IM"}, inplace=True)
data_onet_level = data_onet.loc[data_onet['Scale ID'] == "LV", ["Element ID", "OCC_CODE", "Data Value"]].reset_index(drop=True)
data_onet_level.rename(columns={"Data Value": "LV"}, inplace=True)
data_onet = pd.merge(data_onet_importance, data_onet_level, on=["OCC_CODE", "Element ID"])

# Aggregate over "Element ID" and "Scale ID" (avg)
data_onet = data_onet.groupby(["OCC_CODE", "Element ID"]).mean().reset_index()

# Normalize "IM" and "LV" to [0, 1]
data_onet["IM"] = (data_onet["IM"] - data_onet["IM"].min()) / (data_onet["IM"].max() - data_onet["IM"].min())
data_onet["LV"] = (data_onet["LV"] - data_onet["LV"].min()) / (data_onet["LV"].max() - data_onet["LV"].min())

# Create SKILL index
data_onet = data_onet.groupby("OCC_CODE").apply(lambda x: (x["IM"] * x["LV"]).sum() / len(x))

# Convert to dataframe 
data_onet = pd.DataFrame(data_onet).reset_index()
# Rename columns 
data_onet.rename(columns={0: "SKILL"}, inplace=True)

# Normalize SKILL index to [0, 1]
data_onet["SKILL"] = (data_onet["SKILL"] - data_onet["SKILL"].min()) / (data_onet["SKILL"].max() - data_onet["SKILL"].min())

# %%
# Read in BLS data

data_path = "/project/high_tech_ind/ht_job_flows/data/OEWS/national/proc/"

list_of_files = os.listdir(data_path)

data_bls = pd.read_csv(data_path + list_of_files[5])

# %%
# Merge BLS and ONET data
data = pd.merge(data_bls, data_onet, on="OCC_CODE")
# Convert to numeric all wage columns (starting with A_ or H_)
for col in data.columns:
    if col.startswith("A_") or col.startswith("H_"):
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Compute percentage of employment in each occupation
data["EMP_PCT"] = data["TOT_EMP"] / data["TOT_EMP"].sum()

# Save data to file
data.to_csv("/project/high_tech_ind/ht_job_flows/data/OEWS/national/proc/oes_nat_onet_skill.csv", index=False)

# %%
data_path = "/project/high_tech_ind/ht_job_flows/data/OEWS/msa/proc/"

list_of_files = os.listdir(data_path)

data_bls_msa = pd.read_csv(data_path + list_of_files[-2])

data_msa = pd.merge(data_bls_msa, data_onet, on="OCC_CODE", how="left")

# Drop NA values from SKILL index
data_msa.dropna(subset=["SKILL"], inplace=True)

# Convert TOT_EMP to numeric 
data_msa["TOT_EMP"] = pd.to_numeric(data_msa["TOT_EMP"], errors="coerce").fillna(0)

# Compute the percentage of employment in each occupation in each MSA
data_msa["EMP_PCT"] = data_msa.groupby("AREA_NAME")["TOT_EMP"].transform(lambda x: x / x.sum())

data_msa.to_csv("/project/high_tech_ind/ht_job_flows/data/OEWS/msa/proc/oes_nat_onet_skill.csv", index=False)

# %%
# Plot scatter plot of SKILL index vs. A_MEAN
sns.scatterplot(data=data, x="SKILL", y="H_MEAN")
# %%

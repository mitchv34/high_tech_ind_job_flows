# %% 
import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
from rich import print

# %%
# * Functions to compute SKILL index from ONET data

def process_onet_data(data_onet, list_elements, ocupation_column, name_new_col):
    """
    Process ONET data to create a SKILL index for each ocupation.
    
    Args:
    url_data (str): URL to ONET data.
    list_skills (list): List of Element IDs to filter on.
    ocupation_column (str): Name of the column containing the codes.
    name_new_col (str): Name of the new column to create.
    
    Returns:
    pandas.DataFrame: SKILL index for each ocupation.
    """

    # Filter on "Element ID" in list_skills
    data_onet = data_onet[data_onet["Element ID"].isin(list_elements)]

    # Average 
    data_onet_agg = data_onet.groupby([ocupation_column, "Element ID", "Scale ID"]).mean().reset_index()

    # Set "Data Value" to 0 if "Recommend Suppress" == "Y"
    # data_onet = data_onet[data_onet["Recommend Suppress"] != "Y"]

    # TODO: Use pivot table instead of this mess
    data_onet_importance = data_onet_agg.loc[data_onet_agg['Scale ID'] == "IM", ["Element ID", ocupation_column, "Data Value"]].reset_index(drop=True)
    data_onet_importance.rename(columns={"Data Value": "IM"}, inplace=True)
    data_onet_level = data_onet_agg.loc[data_onet_agg['Scale ID'] == "LV", ["Element ID", ocupation_column, "Data Value"]].reset_index(drop=True)
    data_onet_level.rename(columns={"Data Value": "LV"}, inplace=True)
    data_onet_agg = pd.merge(data_onet_importance, data_onet_level, on=[ocupation_column, "Element ID"])

    # Normalize "IM" and "LV" to [0, 1]
    data_onet_agg["IM"] = (data_onet_agg["IM"] - data_onet_agg["IM"].min()) / (data_onet_agg["IM"].max() - data_onet_agg["IM"].min())
    data_onet_agg["LV"] = (data_onet_agg["LV"] - data_onet_agg["LV"].min()) / (data_onet_agg["LV"].max() - data_onet_agg["LV"].min())

    # Create  index
    data_onet_agg = data_onet_agg.groupby(ocupation_column).apply(lambda x: (x["IM"] * x["LV"]).sum() / len(x))

    # Convert to dataframe 
    data_onet_agg = pd.DataFrame(data_onet_agg).reset_index()
    # Rename columns 
    data_onet_agg.rename(columns={0: name_new_col}, inplace=True)

    # Normalize  index to [0, 1]
    data_onet_agg[name_new_col] = (data_onet_agg[name_new_col] - data_onet_agg[name_new_col].min()) / (data_onet_agg[name_new_col].max() - data_onet_agg[name_new_col].min())

    return data_onet_agg


# %%
# * Select wich SKILLS and KNOWLEDGE to keep
url_onet_model = "https://www.onetcenter.org/dl_files/database/db_28_0_text/Content%20Model%20Reference.txt"

onet_model = pd.read_csv(url_onet_model, sep="\t", encoding="latin-1")
# Subset to Worker Requirements (Element ID column starts with "2")
# onet_model = onet_model[onet_model["Element ID"].apply(lambda x: x.startswith("2"))]
#display(onet_model[onet_model["Element ID"].apply(lambda x: len(x) == 3)])
# Subset to basic skills (Element ID column starts with "2.A")
# onet_model = onet_model[onet_model["Element ID"].apply(lambda x: x.startswith("2.C"))]
#display(onet_model[onet_model["Element ID"].apply(lambda x: len(x) == 5)])
# Display all skills
#display(onet_model[onet_model["Element ID"].apply(lambda x: len(x) == 7)])
# Filter to keep the basic level (Element ID 7 characters long)
onet_model = onet_model[onet_model["Element ID"].apply(lambda x: len(x) == 7)]
onet_model_knoledge = onet_model[onet_model["Element ID"].apply(lambda x: x[4] in ["3", "4"] and "2.C" in x )]
# Print in rich format that im selectiong thesse knowledge use colors
print("[bold red]Knoledge[/bold red]")
display(onet_model_knoledge[["Element ID", "Element Name"]])
# Create a list of Knowledge
list_knoledege = onet_model_knoledge["Element ID"].tolist()


# %% 
# * Load Crosswalks
occ2010_soc2010 = pd.read_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/occ2010_soc2010_crosswalk.csv", dtype=str)
# occ2002_soc2002 = pd.read_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/occ2002_soc2002_crosswalk.csv", dtype=str)
occ2018_soc2018 = pd.read_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/occ2018_soc2018_crosswalk.csv", dtype=str)

# Convert ot dictionaries
occ2010_soc2010 = dict(zip(occ2010_soc2010.SOC2010, occ2010_soc2010.OCC2010))
# occ2002_soc2002 = dict(zip(occ2002_soc2002.SOC2002, occ2002_soc2002.OCC2002))
occ2018_soc2018 = dict(zip(occ2018_soc2018.SOC2018, occ2018_soc2018.OCC2018))

# %%

related_df = pd.read_csv("https://www.onetcenter.org/dl_files/database/db_28_0_text/Related%20Occupations.txt", sep = "\t")

# Filter by Relatedness Tier == Primary-Short and drop the column
# related_df = related_df[related_df["Relatedness Tier"] == "Primary-Short"]
# related_df.drop(columns = ["Relatedness Tier"], inplace = True)
# Rename columns 
related_df.rename(columns={"O*NET-SOC Code": "ONET", "Related O*NET-SOC Code": "RELATED_ONET"}, inplace=True)
# Create dictionary mapping ONET to RELATED_ONET (should deliver a 1 to list mapping and respect the order in column Index)

related_dict = related_df.groupby("ONET").RELATED_ONET.unique().to_dict()


# %% 
def download_onet_data(name, url, c_walk, related_dict):
    """
    Downloads and extracts the Knowledge.txt and Skills.txt files from the specified version of the ONET database.
    Loads the files into pandas dataframes, keeps only the columns of interest, renames columns, and maps SOC codes to OCC codes.

    Args:
    name (str): Name of the ONET database version.
    url (str): URL to the ONET database version.
    c_walk (dict): Dictionary mapping SOC codes to OCC codes.
    related_dict (dict): Dictionary mapping ONET codes to lists of related ONET codes.

    Returns:
    tuple: A tuple of two pandas dataframes containing the Knowledge and Skills data, respectively.
    """

    # Download the zip file
    urllib.request.urlretrieve(url, f"{name}.zip")

    # Extract the Knowledge.txt and Skills.txt files from the zip file
    with zipfile.ZipFile(f"{name}.zip", "r") as zip_ref:
        # print(zip_ref.namelist())
        zip_ref.extract(f"{name}/Knowledge.txt")
        zip_ref.extract(f"{name}/Skills.txt")

    # Load the Knowledge.txt and Skills.txt files into pandas dataframes
    knowledge_df = pd.read_csv(f"./{name}/Knowledge.txt", delimiter="\t")
    skills_df = pd.read_csv(f"./{name}/Skills.txt", delimiter="\t")

    # Keep only the columns of interest
    knowledge_df = knowledge_df[["O*NET-SOC Code","Element ID", "Scale ID", "Data Value"]]
    # Rename "O*NET-SOC Code" to "ONET"
    knowledge_df.rename(columns={"O*NET-SOC Code": "ONET"}, inplace=True)

    # Delete the zip file and the db_{version} folder (not empty)
    os.remove(f"{name}.zip")
    os.remove(f"./{name}/Knowledge.txt")
    os.remove(f"./{name}/Skills.txt")
    os.rmdir(f"{name}")

    # Replace missing occupations wiht the average values of related occupations
    list_occupations = list(related_dict.keys())

    # ? For Knowledge
    list_missing = [onet for onet in list_occupations if onet not in knowledge_df.ONET.unique()]

    # For each missing ocuppation subset knowledge_df_28_SOC to the related occupations and compute the mean
    for missing_occ in list_missing:
        synthetic_values = knowledge_df[knowledge_df.ONET.isin(related_dict[missing_occ])].groupby(["Element ID", "Scale ID"]).mean().reset_index()
        synthetic_values["ONET"] = missing_occ
        # Add  compputed values to knowledge_df_28_SOC
        knowledge_df = pd.concat([knowledge_df, synthetic_values], ignore_index=True)

    # ? For Skills
    # TODO: Add missing skills

    # Obtain SOC codes for each occupation (remove .XX from the end of the code)
    knowledge_df.loc[:, "SOC"] = knowledge_df["ONET"].apply(lambda x: x.split(".")[0])
    skills_df.loc[:, "SOC"] = skills_df["O*NET-SOC Code"].apply(lambda x: x.split(".")[0])

    # Use crosswalk to map SOC codes to OCC (census) codes
    knowledge_df.loc[:, "OCC"] = knowledge_df.SOC.map(c_walk)
    skills_df.loc[:, "OCC"] = skills_df.SOC.map(c_walk)

    return knowledge_df, skills_df

## %
def create_scores_from_related_occupations():
    """
    Create a score for each occupation based on the scores of related occupations.
    """
    pass



# %%
#* ONET data for 2009 (The last year before the 2010 SOC revision)
# name = "db_14_0"
# url = f"https://www.onetcenter.org/dl_files/{name}.zip"
# knowledge_df_14, skills_df_14 = download_onet_data(name, url, occ2002_soc2002)
# %%
# * ONET Data from 2020 (The last year before the 2018 SOC revision)
name = "db_25_0_text"
url = f"https://www.onetcenter.org/dl_files/database/{name}.zip"
knowledge_df_25, skills_df_25 = download_onet_data(name, url, occ2010_soc2010)


# %%
# * ONET Latest Data
name = "db_28_0_text"
url = f"https://www.onetcenter.org/dl_files/database/{name}.zip"
knowledge_df_28, skills_df_28 = download_onet_data(name, url, occ2018_soc2018, related_dict=related_dict)

# Aggregate Knolege to a single valu per occupation (using all occupation clasifications)
knowledge_df_28_ONET = process_onet_data(knowledge_df_28, list_knoledege, "ONET", "KNOWLEDGE")
knowledge_df_28_SOC = process_onet_data(knowledge_df_28, list_knoledege, "SOC", "KNOWLEDGE_SOC")
knowledge_df_28_OCC = process_onet_data(knowledge_df_28, list_knoledege, "OCC", "KNOWLEDGE_OCC")

#%%
# Save the dataframes to csv files
path = "/project/high_tech_ind/high_tech_ind_job_flows/data/onet/knowledge/agg/"
# knowledge_df_14.to_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/knowledge_df_14.csv", index=False)
# knowledge_df_25.to_csv("/project/high_tech_ind/high_tech_ind_job_flows/data/aux/proc/knowledge_df_25.csv", index=False)
knowledge_df_28_ONET.to_csv(path + "knowledge_df_28_ONET.csv", index = False)
knowledge_df_28_SOC.to_csv(path + "knowledge_df_28_SOC.csv", index = False)
knowledge_df_28_OCC.to_csv(path + "knowledge_df_28_OCC.csv", index = False)

# %%

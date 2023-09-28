# %%
import pandas as pd
import numpy as np
import os
import re


data_path = "/project/high_tech_ind/ht_job_flows/data/OEWS/national/"

list_of_files = os.listdir(data_path+"raw/")
# %%

# %%
cols = ['OCC_CODE', 'OCC_TITLE', 'TOT_EMP',
        'H_MEAN', 'A_MEAN', 'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75',
        'H_PCT90','A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']

count = 0
for file in list_of_files:
    count += 1
    print(f"Processing file {count} of {len(list_of_files)}")
    # Get year from file name using regex
    year = re.findall(r"\d{4}", file)
    if len(year) == 0:
        year = "2019"
    else:
        year = year[0]
    # Read in data (xls)
    df = pd.read_excel(data_path+"raw/"+file)
    # Rename columns to upper case
    df.columns = df.columns.str.upper()
    # Rename some columns
    df = df.rename(columns={"OCC_TITL": "OCC_TITLE"})
    # Remove the letter "W" from column names
    df.columns = df.columns.str.replace("W", "")
    df = df[cols]
    # Save as csv
    df.to_csv(data_path+"proc/occ_national_"+year+".csv", index=False)

# %%
data_path = "/project/high_tech_ind/ht_job_flows/data/OEWS/msa/"

list_of_files = os.listdir(data_path+"raw/")
# Filter files startgin with "."
list_of_files = [file for file in list_of_files if file[0] != "."]

cols = ["AREA", "AREA_NAME", 'OCC_CODE', 'OCC_TITLE', 'TOT_EMP',
        'H_MEAN', 'A_MEAN', 'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75',
        'H_PCT90', 'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']

count = 0
for file in list_of_files:
    count += 1
    print(f"Processing file {count} of {len(list_of_files)}")
    # Get year from file name using regex
    year = re.findall(r"\d{4}", file)
    if len(year) == 0:
        year = "2019"
    else:
        year = year[0]
    # Read in data (xls)
    df = pd.read_excel(data_path+"raw/"+file)
    # Rename columns to upper case
    df.columns = df.columns.str.upper()
    # Rename some columns
    df = df.rename(columns={"OCC_TITL": "OCC_TITLE", "AREA_TITLE": "AREA_NAME"})
    # Remove the letter "W" from column names
    df.columns = df.columns.str.replace("W", "")
    df = df[cols]
    # Save as csv
    df.to_csv(data_path+"proc/occ_msa_"+year+".csv", index=False)
# %%
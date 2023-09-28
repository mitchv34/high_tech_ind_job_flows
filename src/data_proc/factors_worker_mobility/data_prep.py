# %% 
# Imports
import pandas as pd
from geopy.distance import geodesic
import geopandas as gpd
from tqdm import tqdm
import numpy as np

# %%
# Load MSA employemnt data
bds2020_msa_sec = pd.read_csv("/project/high_tech_ind/high_tech_job_flows/data/bds_all/bds2020_msa_sec.csv")
# Filter to keep only dates after 1990
# And only keep the columns we need ["year", "msa", "sector", "emp"]
bds2020_msa_sec = bds2020_msa_sec.loc[bds2020_msa_sec["year"] >= 2000, ["year", "msa", "sector", "emp"]]
# Determine for wich msa and sector we have no data
no_data = bds2020_msa_sec[bds2020_msa_sec.emp.apply(lambda x : not x.isdigit())]
list_msa_no_data = no_data["msa"].unique()
# Drop MSA with no data
bds2020_msa_sec = bds2020_msa_sec.loc[~bds2020_msa_sec["msa"].isin(list_msa_no_data)]
# Reset the index
bds2020_msa_sec = bds2020_msa_sec.reset_index(drop=True)


#%%
# Read populaiton data
msa_pop = pd.read_csv("/project/high_tech_ind/high_tech_job_flows/data/aux/msa_populations.csv")
msa_pop = msa_pop[msa_pop.msa_name.str.contains("Metro")]
# Replace " Metro Area" with ""
msa_pop["msa_name"] = msa_pop["msa_name"].str.replace(" Metro Area", "")
# Split by comma and save as name and state
msa_pop["state"] = msa_pop["msa_name"].str.split(",").str[1]
msa_pop["name"] = msa_pop["msa_name"].str.split(",").str[0]
# Drop the msa_name column
msa_pop = msa_pop.drop(columns=["msa_name"])
# Drop bottom bottom quatile of population
msa_pop = msa_pop.loc[msa_pop["population"] > msa_pop["population"].quantile(0.5)]
# Drop if State is AK, HI, or PR
msa_pop = msa_pop.loc[~msa_pop["state"].isin([" AK", " HI", " PR"])]
# %%
# Filter bds2020_msa_sec to keep only msa in msa_pop
bds2020_msa_sec = bds2020_msa_sec.loc[bds2020_msa_sec["msa"].isin(msa_pop["msa_code"].unique())].reset_index(drop=True)

# %%
# Load MSA geo data
msa_geo = gpd.read_file("/project/high_tech_ind/high_tech_job_flows/data/shape_files/tl_2019_us_cbsa/tl_2019_us_cbsa.shp", dtype={"CBSAFP": int})
# Aggregate county shapes to MSA shapes
msa_geo = msa_geo.dissolve(by="CBSAFP").reset_index()
# Compute the centroid coordinates for each MSA
msa_geo["centroid"] = msa_geo.centroid

# Extract the latitude and longitude coordinates from the centroid column
msa_geo["latitude"] = msa_geo["centroid"].y
msa_geo["longitude"] = msa_geo["centroid"].x

# Keep only the columns we need ["CBSAFP", "NAME", "latitude", "longitude"]
msa_geo = msa_geo[["CBSAFP", "NAME", "centroid", "latitude", "longitude"]]
# Rename the columns CBSAFP to msa
msa_geo = msa_geo.rename(columns={"CBSAFP": "msa", "NAME": "name"})

# Remove the MSA with no data
msa_geo.msa = msa_geo.msa.astype(int)
msa_geo = msa_geo.loc[~msa_geo["msa"].isin(list_msa_no_data)]
# Reset the index
msa_geo = msa_geo.reset_index(drop=True)
# Filter msa_geo to keep only msa in bds2020_msa_sec
msa_geo = msa_geo.loc[msa_geo["msa"].isin(bds2020_msa_sec["msa"].unique())].reset_index(drop=True)

# %%
# Compute the distance between each MSA
msa_dist = {
    "msa_orig": [],
    "msa_dest": [],
    "distance": []
}

num_iterations = len(msa_geo) * len(msa_geo)
progress_bar = tqdm(total=num_iterations)
for i in range(len(msa_geo)):
    msa_orig = msa_geo["msa"][i]
    for j in range(len(msa_geo)):
        msa_dest = msa_geo["msa"][j]
        if msa_orig == msa_dest:
            continue
        msa_dist["msa_orig"].append(msa_orig)
        msa_dist["msa_dest"].append(msa_dest)
        msa_dist["distance"].append(geodesic((msa_geo["latitude"][i], msa_geo["longitude"][i]), (msa_geo["latitude"][j], msa_geo["longitude"][j])).miles)
        progress_bar.update(1)


progress_bar.close()
msa_dist = pd.DataFrame(msa_dist)




# %% 
bds2020_msa_sec.emp = bds2020_msa_sec.emp.astype(int)
# Compute the share of employment in each sector for each MSA
# Calculate the total employment for each year-msa group
bds2020_msa_sec["total_emp"] = bds2020_msa_sec.groupby(['year', 'msa'])['emp'].transform('sum')
# Calculate the share of each sector's employment within each year-msa group
bds2020_msa_sec['share_emp'] = bds2020_msa_sec['emp'] / bds2020_msa_sec["total_emp"]
bds2020_msa_sec_shares = pd.pivot_table(bds2020_msa_sec, values='share_emp', index=['year', 'msa'], columns='sector').reset_index()

# iterate over years 
years = bds2020_msa_sec_shares.year.unique()
msas = bds2020_msa_sec_shares.msa.unique()
msa_sector_dsim = {
    "year": [],
    "msa_orig": [],
    "msa_dest": [],
    "dissimilarity": []
}



for year in years:
    # Iterate over MSA
    print(year)
    sub_bds2020_msa_sec = bds2020_msa_sec_shares.loc[bds2020_msa_sec_shares["year"] == year].reset_index(drop=True)
    num_iterations = len(sub_bds2020_msa_sec) * len(sub_bds2020_msa_sec)
    progress_bar = tqdm(total=num_iterations)
    for i in range(len(sub_bds2020_msa_sec)):
        msa_orig_shares = bds2020_msa_sec_shares.iloc[i]
        # Get the MSA code
        msa_orig = msa_orig_shares["msa"]
        # Remvoe the year and msa columns
        msa_orig_shares = msa_orig_shares.drop(["year", "msa"]).copy()
        # convert to numpy array
        msa_orig_shares = msa_orig_shares.to_numpy()
        for j in range(len(sub_bds2020_msa_sec)):
            msa_dest_shares = bds2020_msa_sec_shares.iloc[j].copy()
            # Get the MSA code
            
            msa_dest = msa_dest_shares["msa"]
            if msa_orig == msa_dest:
                continue
            # Compute the dissimilarity between the two MSA
            msa_dest_shares = msa_dest_shares.drop(["year", "msa"])
            # convert to numpy array
            msa_dest_shares = msa_dest_shares.to_numpy()
            dis = np.abs(msa_dest_shares - msa_orig_shares).sum()
            # Add all the information to the dictionary
            msa_sector_dsim["year"].append(year)
            msa_sector_dsim["msa_orig"].append(msa_orig)
            msa_sector_dsim["msa_dest"].append(msa_dest)
            msa_sector_dsim["dissimilarity"].append(dis)
            progress_bar.update(1)

    progress_bar.close()

msa_sector_dsim = pd.DataFrame(msa_sector_dsim)
msa_sector_dsim.msa_orig = msa_sector_dsim.msa_orig.astype(int)
msa_sector_dsim.msa_dest = msa_sector_dsim.msa_dest.astype(int)
# %% 
# Merge the MSA distance and the MSA sector dissimilarity
msa_dist_sector_dsim = msa_sector_dsim.merge(msa_dist, on=["msa_orig", "msa_dest"], how="left")
# Normalize the distance
msa_dist_sector_dsim["distance_norm"] = msa_dist_sector_dsim["distance"] / msa_dist_sector_dsim["distance"].max()
# Normalize the dissimilarity each year
msa_dist_sector_dsim["dissimilarity_norm"] = msa_dist_sector_dsim.groupby("year")["dissimilarity"].transform(lambda x: x / x.max())
# %% 
# Load worker mobility data
j2jod = pd.read_csv("/project/high_tech_ind/high_tech_job_flows/data/j2j_od/sex1_age_lower_07/proc_anual_no_own.csv")
# Rename geography_dest to msa_dest and geography_orig to msa_orig
j2jod = j2jod.rename(columns={"geography_dest": "msa_dest", "geography_orig": "msa_orig"})
# %%
# Calculate the total outflow for each year-msa group
j2jod[["EE_total_outflow", "AQHire_total_outflow"]] = j2jod.groupby(['year', 'msa_orig'])[["EE", "AQHire"]].transform('sum')
# Calculate the share of migration to each msa for each year-msa group
j2jod['share_EE'] = j2jod['EE'] / j2jod["EE_total_outflow"]
j2jod['share_AQHire'] = j2jod['AQHire'] / j2jod["AQHire_total_outflow"]
# Calculate earnings difference between the two MSAs
j2jod["EESearnings_diff"] = j2jod["EESEarn_Dest"] - j2jod["EESEarn_Orig"]
j2jod["AQHireearnings_diff"] = j2jod["AQHireSEarn_Dest"] - j2jod["AQHireSEarn_Orig"]

# %%
# Merge msa_dist_sector_dsim and j2jod
msa_dist_sector_dsim_j2jod = msa_dist_sector_dsim.merge(
    j2jod[[ "year", "msa_orig", "msa_dest", "EE", "AQHire", "share_EE", "share_AQHire", "EESearnings_diff", "AQHireearnings_diff"]],
                                                        on=["year", "msa_orig", "msa_dest"], how="inner")
# %%
# Add names to the MSAs 
msa_pop.rename(columns={"msa_code": "msa"}, inplace=True)
msa_dist_sector_dsim_j2jod = msa_dist_sector_dsim_j2jod.merge(msa_pop[["msa", "name"]], left_on="msa_orig", right_on="msa", how="left")
# Drop the msa column
msa_dist_sector_dsim_j2jod.drop(columns=["msa"], inplace=True)
msa_dist_sector_dsim_j2jod = msa_dist_sector_dsim_j2jod.rename(columns={"name": "msa_orig_name"})
msa_dist_sector_dsim_j2jod = msa_dist_sector_dsim_j2jod.merge(msa_pop[["msa", "name"]], left_on="msa_dest", right_on="msa", how="left")
# Drop the msa column
msa_dist_sector_dsim_j2jod.drop(columns=["msa"], inplace=True)
msa_dist_sector_dsim_j2jod = msa_dist_sector_dsim_j2jod.rename(columns={"name": "msa_dest_name"})
# %%
# Drop NaN from share_EE	share_AQHire
msa_dist_sector_dsim_j2jod.dropna(subset=["share_EE", "share_AQHire"], inplace=True)
# %%
# Save the data
msa_dist_sector_dsim_j2jod.to_csv("/project/high_tech_ind/high_tech_job_flows/data/j2j_od/proc/msa_dist_sector_dsim_j2jod_sex1_age_lower07.csv", index=False)
# %%

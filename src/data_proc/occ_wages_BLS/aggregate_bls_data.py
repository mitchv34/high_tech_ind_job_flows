import pandas as pd
from os import listdir
from rich import print
from rich.progress import track

data_path = "high_tech_job_flows/data/BLS/interim/ind_occ/"
save_path = "high_tech_job_flows/data/BLS/proc/ind_occ/"
files = [f for f in listdir(data_path)]

final_file = pd.DataFrame()
final_file_agg_wages = pd.DataFrame()

for file in track(files):
    # yellow print file name
    print(f"[bold yellow]Processing {file}[/bold yellow]")
    data = pd.read_csv(data_path + file, low_memory=False) # Read data
    # Get total wages
    data.loc[:, "TOTAL_A"] = data.A_MEAN * data.TOT_EMP
    data.loc[:, "TOTAL_H"] = data.H_MEAN * data.TOT_EMP
    data_agg_wages = data.groupby(["STEM", "HT"]).agg({"TOT_EMP": "sum", "TOTAL_A": "sum", "TOTAL_H": "sum"}).reset_index()
    # Get average wages
    data_agg_wages.loc[:, "A_MEAN"] = data_agg_wages.TOTAL_A / data_agg_wages.TOT_EMP
    data_agg_wages.loc[:, "H_MEAN"] = data_agg_wages.TOTAL_H / data_agg_wages.TOT_EMP
    # add year
    data_agg_wages.loc[:, "YEAR"] = file[:4]
    data.loc[:, "YEAR"] = file[:4]
    # concatenate
    final_file = pd.concat([final_file, data])
    final_file_agg_wages = pd.concat([final_file_agg_wages, data_agg_wages])

# Save data
final_file.to_csv(save_path + "/agg_bls_data.csv", index=False)
final_file_agg_wages.to_csv(save_path + "/agg_bls_data_ht_stem.csv", index=False)
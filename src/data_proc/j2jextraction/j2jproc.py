from os import listdir, makedirs
import pandas as pd
from rich import print
from os.path import exists
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

cols = [
    'geography',
    # 'industry',
    'education', 
    'year',
    'quarter',
    'agg_level',
    'geography_orig',
    # 'industry_orig',
    'EE',
    'AQHire',
    'EES',
    'AQHireS',
    'EESEarn_Orig',
    'EESEarn_Dest', 
    'AQHireSEarn_Orig',
    'AQHireSEarn_Dest']

folder_raw = "/project/high_tech_ind/high_tech_job_flows/data/j2j_od/raw/"
# if processed_folder does not exist, create it
folder_proc = "/project/high_tech_ind/high_tech_job_flows/data/j2j_od/interim/educ/"
if exists(folder_proc):
    proceced_files = listdir(folder_proc)
else:
    makedirs(folder_proc)
    proceced_files = []

file_list = listdir(folder_raw)

file_list = [file for file in file_list if file not in proceced_files]

print(f"[bold yellow]Processing {len(file_list)} files...[/bold yellow]")

def process_file(args):
    file, agg_level = args[0], args[1]
    print(f"[bold green]Processing {file}...[/bold green]")
    df = pd.read_csv(folder_raw + file, chunksize=1000000,
        compression="gzip", low_memory=False, usecols=cols)
    data_educ_ind = pd.DataFrame()
    for chunk in df:
        sub_df = chunk[chunk.agg_level == agg_level]
        # data_educ_ind = pd.concat([data_educ_ind,  sub_df[ (sub_df.industry != "00") & (sub_df.education != "E0") & (sub_df.education != "E5") ]])
        data_educ_ind = pd.concat([data_educ_ind, sub_df])
    # data_educ_ind.to_csv(f"./data/j2j/interim/educ_ind/{file[:-3]}", index=False)
    data_educ_ind.to_csv(f"{folder_proc}{file[:-3]}", index=False)
    print(f"[bold blue]{file} processed.[/bold blue]")


def process_parallel(args):
    cpus = cpu_count()
    ThreadPool(cpus-1).map(process_file, args)


if __name__ == "__main__":
    agg_level = 592913 # education
    # agg_level = 642321 # industry and education
    process_parallel(zip(file_list, [agg_level] * len(file_list)))



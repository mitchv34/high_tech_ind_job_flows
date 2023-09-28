# %%
# TODO: Write better comments for this file 
from pathlib import Path
import json


from ipumspy import IpumsApiClient, UsaExtract, readers, ddi

IPUMS_API_KEY = "59cba10d8a5da536fc06b59d5c537572377c4af09a213a069de3e774"

download_dir =  "/project/high_tech_ind/high_tech_ind_job_flows/data/acs/occupation/raw"

ipums = IpumsApiClient(IPUMS_API_KEY)
# %%
# Submit an API extract request
variables = ["AGE", "EDUC", "MIGPLAC1", "STATEFIP", "PUMA", "MIGPUMA1", "OCC"]
age_filter = [f"{i}".zfill(3) for i in range(21, 65)]
data_paths = {}

years = range(2005, 2022)
for year in years:
    print (f"Processing year {year} of {max(years)}")
    
    extract = UsaExtract(
        [f"us{year}a"],
        variables,
        description= f"ACS 1-year sample for {year}, variables : {' -- '.join(variables)}"
    )
    # Select cases 
    extract.select_cases("AGE", age_filter)

    ipums.submit_extract(extract)
    print(f"Extract submitted with id {extract.extract_id}")

    # wait for the extract to finish
    ipums.wait_for_extract(extract)

    DOWNLOAD_DIR = Path(download_dir + f"/{year}a")
    # Check if the download directory exists if not create it else empty it
    if not DOWNLOAD_DIR.exists():
        DOWNLOAD_DIR.mkdir(parents=True)
    else:
        for file in DOWNLOAD_DIR.glob("*"):
            file.unlink()

    # Download the extract
    ipums.download_extract(extract, download_dir=DOWNLOAD_DIR)
    data_paths[year] = DOWNLOAD_DIR    
# %%
# Get the DDI
# DOWLOAD_DIR = data_paths[2005]
# ddi_file = list(DOWNLOAD_DIR.glob("*.xml"))[0]
# ddi = readers.read_ipums_ddi(ddi_file)

# # Get the data
# ipums_df = readers.read_microdata(ddi, DOWNLOAD_DIR / ddi.file_description.filename)

# %%
# Convert Path objects to strings
data_paths = {k: str(v) for k, v in data_paths.items()}
with open(download_dir + "/directory.json", "w") as f:
    json.dump(data_paths, f)

# %%

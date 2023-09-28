import pandas as pd
from os import listdir
from rich import print
from rich.progress import track

data_path = "high_tech_job_flows/data/BLS/raw/ind_occ/"
save_path = "high_tech_job_flows/data/BLS/interim/ind_occ/"


columns = [ 'NAICS', 'NAICS_TITLE', 'OCC_CODE', 'OCC_TITLE', "TOT_EMP",
            'H_MEAN', 'A_MEAN', 'H_PCT10', 'H_PCT25', 'H_MEDIAN', 'H_PCT75', 
            'H_PCT90', 'A_PCT10', 'A_PCT25', 'A_MEDIAN', 'A_PCT75', 'A_PCT90']

files = [f for f in listdir(data_path)]

# High tech industries:
NAICS_ht = ["3341", "3342", "3344", "3345", "3364", "5112", "5182", "5191", "5413", "5415", "5417"]

# Read the data from the file
# print blue downloading STEM occupations message
print("[bold blue]Downloading STEM occupations...[/bold blue]")
data = pd.read_excel("https://www.bls.gov/soc/Attachment_C_STEM_2018.xlsx", header=14)
data.rename(columns={"Sub-domain and Type of Occupation":"sdato"},  inplace=True)
# Filter data with empty keys
# For now I'm just going keep the data with some key
data_stem = data.loc[data["sdato"] == data["sdato"], ["sdato", "2018 SOC code"]]
data_stem.rename(columns={"2018 SOC code": "2018_SOC"}, inplace=True)
# data_stem = data_stem.loc[ data_stem.sdato.apply(lambda x : any(i[0] == "1" for i in x.split(" and "))) ].head(10)

print(f"STEM occuppations are {100*len(data_stem)/len(data):.2f}% of the total")

# print green finished downloading STEM occupations message
print("[bold green]Finished downloading STEM occupations[/bold green]")

# Create list of STEM occupations
stem = data_stem["2018_SOC"].tolist()


# Print blue start processing message
print("[bold blue]Processing BLS data...[/bold blue]")
for file in track(files):
    # yellow print file name
    print(f"[bold yellow]Processing {file}[/bold yellow]")
    data = pd.read_csv(data_path + file, low_memory=False) # Read data
    data.rename(columns={col: col.upper() for col in data.columns}, inplace=True) # Rename columns to uppercase
    data.NAICS = data.NAICS.apply(lambda x: str(x)[:-2]) # Remove last two digits of NAICS
    data = data[~(data.OCC_CODE.str.endswith("0"))] # Keep only detailed occupations
    data = data[columns] # Keep only the columns we need
    data.loc[:, "HT"] = data.NAICS.isin(NAICS_ht) # Add column for high tech industries
    data.loc[:, "STEM"] = data.OCC_CODE.isin(stem)# Create a stem column
    # Drop unavailable data:
    #- *  = indicates that a wage estimate is not available		 
    #- **  = indicates that an employment estimate is not available		 
    #-  = indicates a wage equal to or greater than $100.00 per hour or $208,000 per year 		 
    #- ~ =indicates that the percent of establishments reporting the occupation is less than 0.5%
    for col in data.columns:
        data = data.loc[data[col] != "*"]
        data = data.loc[data[col] != "**"]
        data = data.loc[data[col] != "~"]
    # Replace #  = indicates a wage equal to or greater than $100.00 per hour or $208,000 per year 	
    # H prefix:
    data.H_MEAN = data.H_MEAN.apply(lambda x: 100 if x == "#" else x).astype(float)
    data.H_PCT10 = data.H_PCT10.apply(lambda x: 100 if x == "#" else x).astype(float)
    data.H_PCT25 = data.H_PCT25.apply(lambda x: 100 if x == "#" else x).astype(float)
    data.H_MEDIAN = data.H_MEDIAN.apply(lambda x: 100 if x == "#" else x).astype(float)
    data.H_PCT75 = data.H_PCT75.apply(lambda x: 100 if x == "#" else x).astype(float)
    data.H_PCT90 = data.H_PCT90.apply(lambda x: 100 if x == "#" else x).astype(float)
    # A prefix:
    data.A_MEAN = data.A_MEAN.apply(lambda x: 208000 if x == "#" else x).astype(float)
    data.A_PCT10 = data.A_PCT10.apply(lambda x: 208000 if x == "#" else x).astype(float)
    data.A_PCT25 = data.A_PCT25.apply(lambda x: 208000 if x == "#" else x).astype(float)
    data.A_MEDIAN = data.A_MEDIAN.apply(lambda x: 208000 if x == "#" else x).astype(float)
    data.A_PCT75 = data.A_PCT75.apply(lambda x: 208000 if x == "#" else x).astype(float)
    data.A_PCT90 = data.A_PCT90.apply(lambda x: 208000 if x == "#" else x).astype(float)
    # Save data
    data.to_csv(save_path + file, index=False)

# Print green finished processing message
print("[bold green]Finished processing BLS data[/bold green]")

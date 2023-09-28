# %%
# Import packages
from os import listdir
import pandas as pd 
import requests
import json
from rich import print
from rich.progress import Progress


class CensusAPI:
    """_summary_
    """
    def __init__(self, api_key, params):
        """_summary_
        """
        self.api_key = api_key
        self.time_init = params["time_init"]
        self.time_final = params["time_final"]
        self.base_url = "https://api.census.gov/data/timeseries/qwi/"
        self.endpoint = params["endpoint"]
        self.variables = params["variables"]
        # self.sex = params["sex"]
        # self.education = params["education"]
        self.states = params["states"]
        self.geo = ",".join(params["geo"]) if params["geo"] != "all" else "*"
        self.geo_type = params["geo_type"]
        # self.industry = params["industry"]

    def contruct_url(self):
        """_summary_
        """
        # education = "&education=E" + "&education=E".join(map(str, self.education))
        variables = ",".join(self.variables)
        states =  ",".join(map(str, self.states))
        # geography = f"{self.geo_type}={self.geo}" if self.geo != "" else ""
        # Construct the url
        self.request_url = self.base_url # Start with the base url
        self.request_url += self.endpoint + "?" # Add the endpoint
        self.request_url += f"get=education,{variables}" # Add the variables
        self.request_url += f"&for={self.geo_type}:{self.geo}" # Add the geography
        self.request_url += f"&in=state:{states}" # Add the states
        self.request_url += f"&time=from{self.time_init}to{self.time_final}" # Add the time range
        self.request_url += f"&key=80572bbd288e26df52d7113091fc646c7458946b" # Add the api key
                # + f"{self.endpoint}?get={variables}&for={geography}&in=state:{states}&time=from{self.time_init}to{self.time_final}{education}&industry={self.industry}&key="+self.api_key

    def get_data(self):
        if self.request_url == "":
            self.contruct_url()
        response = requests.get(self.request_url)
        self.data = response.json()

    def get_dataframe(self):
        self.get_data()
        return pd.DataFrame(self.data[1:], columns=self.data[0])

    

# %%
# Testing
variables = [
                "Emp",          #	Beginning-of-Quarter Employment	Estimate of the total number of jobs on the first day of the reference quarter
                "EmpEnd",       #	End-of-Quarter Employment	Estimate of the number of jobs on the last day of the quarter
                "EmpS",         #	Full-Quarter Employment (Stable)	Estimate of stable jobs - the number of jobs that are held on both the first and last day of the quarter with the same employer
                "EmpTotal",     #	Employment - Reference Quarter	Estimated count of people employed in a firm at any time during the quarter
                "EmpSpv",       #	Full-Quarter Employment in the Previous Quarter	Estimate of stable jobs in the quarter before the reference quarter
                "HirA",         #   Estimated number of workers who started a new job in the specified quarter"
                "HirN",         #   Estimated number of workers who started a new job excluding recall hires"
                "HirR",         #   Estimated number of workers who returned to the same employer where they had worked within the previous year"
                "Sep",          #   Estimated number of workers whose job with a given employer ended in the specified quarter"
                "HirAEnd",      #   Estimated number of workers who started a new job in the specified quarter, which continued into next quarter"
                "SepBeg",       #   Estimated number of workers whose job in the previous quarter continued and ended in the given quarter"
                "HirAEndRepl",  #   Hires into continuous quarter employment in excess of job creation"
                "HirAEndR",     #   Hires as a percent of average employment"
                "SepBegR",      #   Separations as a percent of average employment"
                "SepS",         #   Estimated number of workers who had a job for at least a full quarter and then the job ended"
                "SepSnx",       #   Estimated number of workers in the next quarter who had a job for at least a full quarter and then the job ended"
                "TurnOvrS",     #   The rate at which stable jobs begin and end"
                # "HirAEndReplR", # "Replacement hires as a percent of the average of beginning- and end-of-quarter employment"
                # "HirAS", # "Estimated number of workers that started a job that lasted at least one full quarter with a given employer"
                # "HirNS", # "Estimated number of workers who started a job that they had not held within the past year and the job turned into a job that lasted at least a full quarter with a given employer"
            ]
params = {
    "endpoint" : "se",
    "time_init" : "2000",
    "time_final" : "2030",
    "variables" : variables[1:2],
    # "sex" : [1,2],
    "education" : [1,2,3,4],
    "states" : ["02"],
    # "geo_type" : "county",
    "geo_type" : "metropolitan+statistical+area/micropolitan+statistical+area",
    "geo" : "all"#["013", "290"]
}

# Read the api key from a file
# api_keys_path = "./high_tech_job_flows/src/census_api/api_key"
# api_keys = [f for f in listdir(api_keys_path) if "key" in f]

# with open(api_keys_path + '/' + api_keys[0], 'r') as f:
#     api_key = f.read()

api_key = "b2af5736a9608a3f9865b1ba0567d8fe7435220b"

a = CensusAPI(api_key, params)
a.contruct_url()

# print(a.get_dataframe())


# %%
state_fips_codes = {
    'AL': '01',
    'AK': '02',
    'AZ': '04',
    'AR': '05',
    'CA': '06',
    'CO': '08',
    'CT': '09',
    'DE': '10',
    'DC': '11',
    'FL': '12',
    'GA': '13',
    'HI': '15',
    'ID': '16',
    'IL': '17',
    'IN': '18',
    'IA': '19',
    'KS': '20',
    'KY': '21',
    'LA': '22',
    'ME': '23',
    'MD': '24',
    'MA': '25',
    'MI': '26',
    'MN': '27',
    'MS': '28',
    'MO': '29',
    'MT': '30',
    'NE': '31',
    'NV': '32',
    'NH': '33',
    'NJ': '34',
    'NM': '35',
    'NY': '36',
    'NC': '37',
    'ND': '38',
    'OH': '39',
    'OK': '40',
    'OR': '41',
    'PA': '42',
    'RI': '44',
    'SC': '45',
    'SD': '46',
    'TN': '47',
    'TX': '48',
    'UT': '49',
    'VT': '50',
    'VA': '51',
    'WA': '53',
    'WV': '54',
    'WI': '55',
    'WY': '56'
}



data = {var : pd.DataFrame() for var in variables}

# Define the number of iterations for each loop
n1 = len(variables)
n2 = len(state_fips_codes.keys())

# Create a new Progress object with two Task objects
progress = Progress()
task1 = progress.add_task("[blue]Loop 1", total=n1)
task2 = progress.add_task("[green]Loop 2", total=n2)

j = 0
for data_var in variables:
    j += 1
    print(f"Processing variable: {data_var} ({j}/{n1})")
    progress.update(task1, advance=1)  # Update progress bar for loop 1
    params["variables"] = [data_var]
    for state in state_fips_codes.keys():
        progress.update(task2, advance=1)  # Update progress bar for loop 2
        params["states"] = [state_fips_codes[state]]
        a = CensusAPI(api_key, params)
        a.contruct_url()
        df = a.get_dataframe()
        # Replace state fips code with state name in the dataframe
        df["state"] = state
        data[data_var] = pd.concat([data[data_var], df])
        # print(df)

# Mark both tasks as complete
progress.update(task1, completed=True)
progress.update(task2, completed=True)
        

    
# %%
data_final = pd.DataFrame()
for k in data.keys():
    if len(data[k]) > 0:
        print(k)
        if len(data_final) == 0:
            data_final = data[k]
        else:
            data_final = pd.merge(data_final, data[k], on=["time", "state", "education", params['geo_type'].replace("+", " ")])
# Rename metropolitan statistical area/micropolitan statistical area to msa
data_final = data_final.rename(columns={params['geo_type'].replace("+", " ") : "msa"})
# Reorder columns
data_final = data_final[["time", "state", "msa"] + [c for c in data_final.columns if c not in ["time", "state", "msa"]]]
data_final
# %%
data_final.to_csv("/project/high_tech_ind/high_tech_job_flows/data/qwi/qwi_msa_educ.csv", index=False)

# %%
# Read unemployment data from BLS load geography as string
unemployment = pd.read_csv("/project/high_tech_ind/high_tech_job_flows/data/BLS/proc/bls_msa_unemployment_quaterly.csv", dtype={"geography" : "object"})
# Rename geography to msa
unemployment = unemployment.rename(columns={"geography" : "msa"})
# %%
data_final_unemp = pd.merge(unemployment, data_final, on=["time", "msa"], how = "right")
data_final_unemp = data_final_unemp[(data_final_unemp.unemployment == data_final_unemp.unemployment)]
data_final_unemp = data_final_unemp[~data_final_unemp.Emp.isnull()]
# %%
# Get correlaiton between unemployment and employment
data_final_unemp[["unemployment", "Emp"]].corr()
# 
# %%
data_final_unemp[["HirN", "unemployment"]]
# %%

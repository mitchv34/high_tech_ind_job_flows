# %%
import requests
import pandas as pd
import json


api_key = '80572bbd288e26df52d7113091fc646c7458946b'
# %%

source = 'acs/acs5'
variables = 'B19013_001E'
geography = 'metropolitan%20statistical%20area/micropolitan%20statistical%20area:*'

df = pd.DataFrame()

for year in range(2009, 2022):
    url = f'https://api.census.gov/data/{year}/{source}?get={variables}&for={geography}&key={api_key}'
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        headers = data.pop(0)
        df_year = pd.DataFrame(data, columns=headers)
        df_year['year'] = year
        df = pd.concat([df, df_year])
    else:
        print(f'Error: {response.status_code} for year {year}')

# %%

# Set the year and state code
year = 2013
state_code = '36'  # New York State

# Define the API endpoint and parameters for the County-to-MSA crosswalk data
api_url = f'https://api.census.gov/data/{year}/acs/acs5'
variables = 'GEO_ID,NAME,NAMELSAD,METDIVFP,CSAFP,NECTAFP,CBSAFP'
params = {'get': variables, 'for': 'county:*', 'in': f'state:{state_code}', 'key': api_key}

# Send the API request and get the response data
response = requests.get(api_url, params=params)
data = response.json()

# Convert the response data to a pandas DataFrame and rename the columns
df = pd.DataFrame(columns=data[0], data=data[1:])
df = df.rename(columns={'GEO_ID': 'geo_id', 'NAME': 'county_name', 'NAMELSAD': 'county_description',
                        'METDIVFP': 'md_code', 'CSAFP': 'csa_code', 'NECTAFP': 'necta_code', 'CBSAFP': 'msa_code',
                        'state': 'state_code', 'county': 'county_code'})

# Filter the DataFrame to only include counties that are part of an MSA
df = df[df['msa_code'].notnull()]

# Group the DataFrame by MSA and aggregate the counties into a list
msa_df = df.groupby(['msa_code', 'county_description'])[['county_name', 'county_code']].agg(list).reset_index()


# %%

url = 'https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2020/delineation-files/list1_2020.xls'
msa_table = pd.read_excel(url, header=2, dtype=str, usecols=lambda x: x in ['FIPS State Code', 'FIPS County Code','CBSA Code'], skipfooter=4)

# Combine 'FIPS State Code' and 'FIPS County Code' into a new column named 'FIPS'
msa_table['FIPS'] = msa_table['FIPS State Code'].str.zfill(2) + msa_table['FIPS County Code'].str.zfill(3)

# Rename 'CBSA Code' to 'CBSA'
msa_table = msa_table.rename(columns={'CBSA Code': 'CBSA'})

# Create a new dataframe with just the desired columns
new_df = msa_table[['CBSA', 'FIPS']]

# Print the first few rows of the new dataframe
print(new_df.tail())

# Save
new_df.to_csv('../../data/aux/county_to_msa.csv', index=False)

# %%
# Set the year and variables
year = 2010
variables = 'NAME,B01003_001E'

# Define the API endpoint and parameters for the MSA population data
api_url = f'https://api.census.gov/data/{year}/acs/acs5'
params = {'get': variables, 'for': 'metropolitan statistical area/micropolitan statistical area:*', 'key': api_key}

# Send the API request and get the response data
response = requests.get(api_url, params=params)
data = response.json()

# Convert the response data to a pandas DataFrame and rename the columns
df = pd.DataFrame(columns=data[0], data=data[1:])
df = df.rename(columns={'NAME': 'msa_name', 'B01003_001E': 'population', 'metropolitan statistical area/micropolitan statistical area': 'msa_code'})

# Convert the population column to numeric
df['population'] = pd.to_numeric(df['population'])

# Filter the DataFrame to only include MSAs (exclude Micropolitan Statistical Areas)
df = df[~df['msa_code'].str.startswith('µ')]

# Print the top 10 MSAs by population
print(df.nlargest(10, 'population'))

# Save
df.to_csv('../../data/aux/msa_populations.csv', index=False)


# %%

# Define the API endpoint and parameters
endpoint = 'https://api.census.gov/data/timeseries/bds/firms'
params = {
    'get': 'fage4,fsize,geo_stusab,naics,metro',
    'for': 'metropolitan statistical area/micropolitan statistical area:*',
    'time': '2000:2019',
    'key': api_key
}

# Make the API request
response = requests.get(endpoint, params=params)

# Parse the JSON data into a Pandas dataframe
data = pd.DataFrame(response.json()[1:], columns=response.json()[0])

# Convert the job creation data to numeric values
data['fage4'] = pd.to_numeric(data['fage4'])

# Group the data by MSA and sum the job creation data
jobs_created = data.groupby(['metro']).agg({'fage4': 'sum'})

# Print the top 10 MSAs by jobs created
print(jobs_created.sort_values(by='fage4', ascending=False).head(10))# %%


# %%
import requests
import pandas as pd

# Define BLS API endpoint and API key
endpoint = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'
api_key = '763465ae393348b886a5ee94cf774d29'

# Define BLS series ID for job finding rate
series_id = 'JTS00000000'

# Define start and end years for data request
start_year = '1990'
end_year = '2019'

# Define API request parameters
headers = {'Content-type': 'application/json'}

# Create empty list to store dataframes for each MSA
msa_dfs = []

# Make API request for each MSA code and append resulting dataframe to msa_dfs list
for msa_code in range(10180, 79600):  # loop over all MSA codes (10180 to 79600)
    # Create BLS series ID for current MSA and make API request
    current_series_id = series_id + str(msa_code) + 'JFR'
    data = {'seriesid': [current_series_id], 'startyear': start_year, 'endyear': end_year, 'registrationkey': api_key}
    response = requests.post(endpoint, headers=headers, json=data)
    # Extract job finding rates data from response and create dataframe
    try:
        jfr_data = response.json()['Results']['series'][0]['data']
        jfr_df = pd.DataFrame(jfr_data, columns=['year', 'jfr']).set_index('year').astype('float64')
        jfr_df.columns = [f"MSA{msa_code}"]
        msa_dfs.append(jfr_df)
    except:
        print(f"No data available for MSA code {msa_code}")
        continue

# Merge dataframes for all MSAs into a single dataframe
jfr_df_all = pd.concat(msa_dfs, axis=1, sort=False)

# Print first 5 rows of resulting dataframe
print(jfr_df_all.head())


# %%
import requests
import pandas as pd

# Define the API endpoint for the JOLTS data
api_url = 'https://api.bls.gov/publicAPI/v2/timeseries/data/'

# Define the parameters for the API request
series_id = 'JTS000000000000000TSL' # Series ID for total separations
start_year = '2018'
end_year = '2019'
area_codes = ['35620', "18880"] # MSA code for San Francisco-Oakland-Hayward, CA
annual_average = "true"

# Iterate through the list of area codes
df_list = []
for code in area_codes:
    # Make the API request
    data = {
        'seriesid': series_id,
        'startyear': start_year,
        'endyear': end_year,
        'area': code,
        'annualaverage': annual_average,
        'registrationkey': '2adfe607d18e4da0bdf3c511fb80e10f'
    }
    response = requests.post(api_url, data=data)

    # Extract the separation rate data for the MSA
    df = pd.DataFrame(response.json()['Results']['series'][0]['data'])
    df = df[['periodName', 'value']]
    df_list.append(df)

# Concatenate the DataFrames for each area code into a single DataFrame
result_df = pd.concat(df_list, axis=1)

# Print the separation rate data for all MSAs
print(result_df)
# Print the separation rate data for the MSA
print(df)

# %%
# Download 
import pandas as pd

url = "https://www2.census.gov/programs-surveys/bds/tables/time-series/bds2020_msa.csv"
df = pd.read_csv(url, low_memory=False)

# convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
df = df.dropna()

# convert rate columns to decimal
rate_cols = [col for col in df.columns if 'rate' in col.lower()]

# divide rate columns by 1000
df[rate_cols] = df[rate_cols].div(100)


# save the cleaned data to a CSV file
df.to_csv("../../data/bds_all/bds2020_msa.csv", index=False)

# %%
import requests
import pandas as pd


# Set variables for MSA code and quarter
msa_code = '35620' # example MSA code for New York-Newark-Jersey City, NY-NJ-PA
quarter = '2020Q4' # example quarter in YYYYQ format

# Set API key and base URL for QWI data
api_key = 'YOUR_API_KEY_HERE'
qwi_url = f'https://api.census.gov/data/timeseries/qwi/sa/'

# Set variables for QWI tables and variables of interest
table_name = 'jfs'
variable_list = ['EmpEnd', 'EmpS', 'HirA', 'Sep', 'FrmJbGn', 'FrmJbLs', 'Cov', 'Area']

# Build API request URL
qwi_request_url = f'{qwi_url}{table_name}'

# Set API request parameters
params = {
    'get': ','.join(variable_list),
    'for': f'msa:{msa_code}',
    'time': quarter,
    'key': api_key
}

# Send API request and get response
response = requests.get(qwi_request_url, params=params)

# Convert response to Pandas DataFrame
df_qwi = pd.DataFrame(response.json()[1:], columns=response.json()[0])

# Convert variable types to numeric
df_qwi[variable_list[:-1]] = df_qwi[variable_list[:-1]].apply(pd.to_numeric)

# Calculate job finding rate
jf_rate = df_qwi['FrmJbGn'].sum() / df_qwi['EmpS'].sum()

# Print result
print(f"Job finding rate for MSA {msa_code} in {quarter}: {jf_rate:.2%}")


# %%
import pandas as pd
data_census = pd.read_html("https://api.census.gov/data.html")[0]



# %%
import pandas as pd
import requests

# Set the base URL for the API
base_url = "https://api.census.gov/data/"

# Set the parameters for the API request
year_list = list(range(2011, 2021))  # years from 2000 to 2020
variables = ["B23025_005E"]  # variable code for unemployed people in arts, design, entertainment, sports, and media occupations

# Initialize an empty dataframe to store the data
msa_data = pd.DataFrame()

# Loop over the years and variables and retrieve the data
for year in year_list:
    print(year)
    for variable in variables:
        url = f"{base_url}{year}/acs/acs5?get={variable}&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:*"

        # Make the API request and convert the response to a dataframe
        response = requests.get(url)
        data = pd.DataFrame(response.json()[1:], columns=response.json()[0])

        # Add columns for the year and variable
        data["year"] = year
        data["variable"] = variable

        # Append the data to the main dataframe
        msa_data = pd.concat([msa_data, data], axis=0)

# Write the data to a CSV file
msa_data.to_csv("B23025_005E_msa_data.csv", index=False)

# %%
import pandas as pd
import requests

# Set the base URL for the API
base_url = "https://api.census.gov/data/"

# Set the parameters for the API request
year_list = list(range(2000, 2011))  # years from 2000 to 2010
variables = ["B23025_005E"]  # variable code for unemployed people in arts, design, entertainment, sports, and media occupations

# Initialize an empty dataframe to store the data
msa_data = pd.DataFrame()

# Loop over the years and variables and retrieve the data
for year in year_list:
    print(year)
    for variable in variables:
        url = f"{base_url}{year}/acs5?get={variable}&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:*"

        # Make the API request and convert the response to a dataframe
        response = requests.get(url)
        data = pd.DataFrame(response.json()[1:], columns=response.json()[0])

        # Add columns for the year and variable
        data["year"] = year
        data["variable"] = variable

        # Append the data to the main dataframe
        msa_data = msa_data.append(data)

# Write the data to a CSV file
msa_data.to_csv("B23025_005E_msa_data.csv", index=False)

# %%

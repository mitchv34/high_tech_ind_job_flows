# %%
from io import StringIO
import requests
import json
import pandas as pd
import prettytable
from rich import print

# Read a list of Metropolitan Statistical Areas (MSAs) from the BLS website
data_msa = pd.read_csv("/project/high_tech_ind/high_tech_job_flows/data/aux/la.area.txt", sep = "\t")
# Filter the list to only include MSAs
data_msa = data_msa[data_msa.area_code.str.startswith("MT")]
# Get a list of MSAs
area = data_msa.area_code.tolist()
total = len(area)
# Get the start and end years
start_year = 1978
end_year = 2019
# Iterate over years in chunks of 20 years
# Series contruction
prefix = "LA" # LA = Local Area Unemployment Statistics
# prefix = "SM" # SM = State and Metro Area Employment, Hours, and Earnings
adj = "U" # U = unadjusted, S = seasonally adjusted
measure = "03" # 03 = unemployment rate, 04 = unemployment, 05 = employment, 06 = labor force
value_name = {
    "03": "unemployment_rate",
    "04": "unemployment",
    "05": "employment",
    "06": "labor_force"
}
# measure = "03" # Average Hourly Earnings of All Employees, In Dollars

# Create an empty dataframe to store the data
data_final = pd.DataFrame()

# Segment the list of MSAs into chunks of 50 MSAs
# This is because the BLS API only allows 50 series to be queried at a time
area = [area[i:i + 50] for i in range(0, len(area), 50)]
# %%
api_call_count = 0
for year in range(start_year, end_year, 20):
    s_year = str(year)
    e_year = str(min(year + 20, end_year))
    print(f"Downloading data for {s_year} to {e_year}")
    # Iterate over the chunks of 50 MSAs
    total_downloaded = 0
    for sub_area in area:
        api_call_count += 1
        total_downloaded += len(sub_area)
        print(f"Downloading data for {total_downloaded} MSAs of {total} ({api_call_count} API calls)")
        series = [f"{prefix}{adj}{a}{measure}" for a in sub_area]
        headers = {'Content-type': 'application/json'}
        data = json.dumps( {
                    "seriesid": series,
                    "startyear": s_year,
                    "endyear": e_year,
                    "registrationkey": "763465ae393348b886a5ee94cf774d29"
                })

        p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
        json_data = json.loads(p.text)

        # print(json_data)
        for series in json_data['Results']['series']:
            x=prettytable.PrettyTable(["series id","year","period","value","footnotes"])
            seriesId = series['seriesID']
            for item in series['data']:
                year = item['year']
                period = item['period']
                value = item['value']
                footnotes=""
                for footnote in item['footnotes']:
                    if footnote:
                        footnotes = footnotes + footnote['text'] + ','
                if 'M01' <= period <= 'M12':
                    x.add_row([seriesId,year,period,value,footnotes[0:-1]])
            data = pd.read_csv(StringIO(x.get_csv_string()))
            # # # Average the monthly data to get annual data
            # data = data.groupby(['series id','year']).mean(numeric_only=True).reset_index()
            # Convert from series id to area code
            data['series id'] = data['series id'].str[7:-8]
            # Rename columns "series id" => "geography", "value" => value_name[measure]
            data = data.rename(columns={"series id": "geography", "value": value_name[measure]})
            # Keep only the columns "geography", "year" and "unemployment"
            # data = data[['year', 'geography', 'unemployment_rate']]
            # data = data[['year', 'period', 'geography', 'earnings']]
            # print(data)
            data_final = pd.concat([data_final, data])
# Remove footnotes column
data_final = data_final.drop(columns=['footnotes'])
#%%
# Convert month column to a datetime format
data_final['period'] = pd.to_datetime('2022-' + data_final['period'].str.replace('M', '') + '-01')
# %%
# Save monthly data
data_final.to_csv(f"/project/high_tech_ind/high_tech_job_flows/data/BLS/proc/bls_msa_{value_name[measure]}_monthly.csv", index=False)
# %% 
# Average the monthly data to get annual data
data_final_yearly = data_final.groupby(["geography",'year']).mean(numeric_only=True).reset_index()
# save annual data
data_final_yearly.to_csv(f"/project/high_tech_ind/high_tech_job_flows/data/BLS/proc/bls_msa_{value_name[measure]}_yearly.csv", index=False)

# %%
data_final.loc[:, "Q"] = "Q" + data_final.period.dt.quarter.astype(str)
# Average the monthly data to get quarterly data

data_final_quaterly = data_final.groupby(["geography",'year','Q']).mean(numeric_only=True).reset_index()
data_final_quaterly.loc[:, "time"] = data_final_quaterly.year.astype(str) + "-" + data_final_quaterly.Q
# Drop year and Q columns
data_final_quaterly = data_final_quaterly.drop(columns=['year', 'Q'])
# Save quarterly data
data_final_quaterly.to_csv(f"/project/high_tech_ind/high_tech_job_flows/data/BLS/proc/bls_msa_{value_name[measure]}_quaterly.csv", index=False)

# %%

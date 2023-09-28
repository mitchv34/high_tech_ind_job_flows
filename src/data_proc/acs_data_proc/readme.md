# ACS Data Process

Python scripts in this folder download an process the ACS data 

Files should be run in this order:

- `ipums_py.py` Uses the IPUMS API to get the data from the ACS. 
- `acs_data_proc.py` Processes the data from the ACS. Saves interim data readu to be used in the analysis.
- `acs_data_analize.py` Analyzes the data from the ACS. Saves interim data readu to be used in the analysis.

## Output

Analysis output the following files:

- `acs_msa_flow_counts.csv` Contains the number of workers that moved from one MSA to another in a given year.
- `acs_msa_counts.csv` Contains outflows, inflows, and net flows for each MSA in each year.
- `acs_msa_flow_counts_educ.csv` Contains the number of workers that moved from one MSA to another in a given year, by education level (College vs NonCollege).
- `acs_msa_counts_educ.csv` Contains outflows, inflows, and net flows for each MSA in each year, by education level (College vs NonCollege).
- `acs_build_occupational_crosswalks` Creates the crosswalks from the ACS occupation to SOC and ONET codes. Saves the crosswalks in the `data/aux/occ_crosswalks` folder.

<!-- TODO: Include OCCUPATION instead of  just EDUCATION-->

As occupation data im using the variable:

OCC2010 is a harmonized occupation coding scheme based on the Census Bureau's 2010 ACS occupation classification scheme. Similar variables are offered for the 1950 (OCC1950) and 1990 (OCC1990) classifications. OCC2010 offers researchers a consistent, long-term classification of occupations.

I will use the crosswalk from BLS to convert from OCC to SOC i'm assuming that the crosswalk is form 2010 to 2010 waiting for an  anwer on this

I will probably need to use a crosswalk to homogenize the data. Since ONET Data is based on 2018 SOC, I will probably need to use the 2018 SOC crosswalk.   
    - 2000 to 2010 croswalk [here](https://www.bls.gov/soc/soc_2000_to_2010_crosswalk.xls)
    - 2010 to 2018 crosswalk [here](https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx)
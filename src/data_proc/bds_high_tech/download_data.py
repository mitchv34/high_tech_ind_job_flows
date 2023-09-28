import pandas as pd
from rich import print

def download_data(table):
    """Download data from the Census Bureau's experimental data site

    Args:
        table (str): Table name to download
    """

    url = f"https://www2.census.gov/ces/bds/experimental/bds2020_ht_{table}.csv"
    df = pd.read_csv(url, low_memory=False)

    # Convert columns to numeric
    for col in df.columns[3:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # Convert ht column to boolean
    df["ht"] = df["ht"] == "High Tech"

    # Rename columns
    df.rename(columns={"msacoarse": "geography"}, inplace=True)

    # Drop micropolitan areas
    df = df[df["geography"] != "Micro"]
    # Drop rows with missing values
    df = df.dropna()

    # Convert to wide format (one row per geography-year)
    df_wide = df[df.ht == True].merge(df[df.ht == False], on = ["year", 'geography'], suffixes=('_ht', '_nht'))
    # Drop "ht_ht" and "ht_nht" columns
    df_wide = df_wide.drop(columns=["ht_ht", "ht_nht"])

    return df_wide

def main():
    """_summary_
    """
    save_path = "high_tech_job_flows/data/bds_high_tech/interim/"
    df = download_data("msac")
    print(df.head())
    # Save file
    df.to_csv(save_path + "/msac.csv", index=False)

if __name__ == "__main__":
    main()
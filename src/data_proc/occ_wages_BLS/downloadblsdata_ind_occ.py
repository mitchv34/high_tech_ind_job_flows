import requests, zipfile, io
import pandas as pd
from rich import print
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

download_to = "./data/BLS/raw/ind_occ/"

def download_year(year):
    zip_file_url = f"https://www.bls.gov/oes/special.requests/oesm{str(year)[-2:]}in4.zip"
    print(f"[bold yellow]Processing {year}...[/bold yellow]")
    print(f"[bold yellow]Downloading {zip_file_url}[/bold yellow]")
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    files = z.filelist
    files = [f for f in files if "4d" in f.filename.split("/")[-1].lower()]
    df = pd.DataFrame()
    for file in files:
        df_temp = pd.read_excel(z.open(file))
        df = pd.concat([df, df_temp])
    df.to_csv(download_to+f"{year}.csv", index=False)

# TODO: Fix parallel download
def download_parallel(args):
    print("[bold yellow]Downloading BLS data[/bold yellow]")
    print(args)
    cpus = cpu_count()
    print(f"[bold yellow]Using {cpus} CPUs[/bold yellow]")
    ThreadPool(cpus - 1).imap_unordered(download_year, args)


if __name__ == "__main__":
    # download_parallel( list(range(2003, 2022)))
    for year in range(2003, 2022):
        download_year(year)
        
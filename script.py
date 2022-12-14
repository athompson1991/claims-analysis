import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import geopandas as gpd
import os
import censusdata as census
import urllib.request
import argparse

from library import Runner


parser = argparse.ArgumentParser(description="Analysis driver")
parser.add_argument('--file', type=str, required=False, default="full.csv")
parser.add_argument('--sample', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--download', action=argparse.BooleanOptionalAction, default=False)


def announcement(text):
    breaker = "|>------------------------------------------------------------------------------<|"
    print(breaker)
    print(text)
    print(breaker + "\n")

def setup(download=False):
    files =  os.listdir()
    if "plots" in files:
        for f in os.listdir("plots"):
            os.remove(f"plots/{f}")
        os.rmdir("plots")
    os.mkdir("plots")
    if "data" not in files:
        os.mkdir("data")
    if download:
        print("Downloading the data.... many rows, will take a while")
        urllib.request.urlretrieve("https://data.ny.gov/api/views/jshw-gkgu/rows.csv?accessType=DOWNLOAD", "data/full.csv")

def do_sample(n=100000):
    df = pd.read_csv("data/full.csv")
    df_sub = df.sample(100000)
    df_sub.to_csv("data/sub.csv")

if __name__ == "__main__":
    matplotlib.use("Agg")
    args = parser.parse_args()

    start = time.time()
    
    announcement("Setup")
    setup(args.download)
    if args.sample:
        do_sample()

    announcement("Reading data")

    df = pd.read_csv(f"data/{args.file}")
    runner = Runner(df)

    zip_map = gpd.read_file("https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ny_new_york_zip_codes_geo.min.json")
    counties = gpd.read_file("https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/new-york-counties.geojson")
    census_data = census.download('acs5', 2015, census.censusgeo([('state', '36'), ('zip code tabulation area', '*')]),  ['B01003_001E'])
    

    counties["name"] = counties["name"].str[:-7].str.upper()

    announcement("Data has been read, basic manipulations to clean it up now")

    print(f"Rows before cleaning: {runner.df.shape[0]}")
    runner.clean_df()
    print(f"Rows after cleaning: {runner.df.shape[0]}")
    
    announcement("Creating map plot")

    runner.map_analysis(zip_map)
    runner.census_analysis(census_data)
    runner.map_plot(zip_map, "Number of Claims by Zip Code")
    runner.map_plot(zip_map, "Per Capita Claims by Zip Code", target_col="claims_per_capita")
    runner.map_plot(counties, "Number of Claims by County", is_zip_map=False)

    announcement("Exploring injury types")

    runner._calc_ranks()
    runner.bar_plot()

    announcement("Creating time series plots")

    runner.calculate_ts()
    runner.time_series_plot(outname="overall")
    runner.time_series_plot(xlim=("2016-01-01", "2022-05-01"), outname="recent")
    runner.time_series_plot(xlim=("2021-01-01", "2022-05-01"), outname="past-year", just_daily=True)

    announcement("Running attorney analysis")

    runner.attorney_analysis()

    announcement("Running dates analysis")

    runner.plot_density()
    runner.final_plot()

    announcement("Done with analysis, wrapping up")

    runner.pickle_plots()
    runner.pickle_data()

    print("End of script")
    end = time.time() - start
    print(f"Total execution time: {end}")
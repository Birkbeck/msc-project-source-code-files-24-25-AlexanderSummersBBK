from pathlib import Path
import pandas as pd


path = Path.cwd()
paths = {"path_hackmageddon": path / "Hackmageddon.csv", 
         "path_dates": path / "Dates.csv", 
         "path_ISO": path / "ISO.csv", 
         "path_wcci": path / "WCCI.csv"}

paths_ds = {}
for each in paths:
    paths_ds[each] = pd.DataFrame(pd.read_csv(paths[each]))

countries = []
for each in paths_ds["path_wcci"]["Country"]:
    country = paths_ds["path_ISO"][paths_ds["path_ISO"]["Country"].str.contains(each)]["Alpha 2"].tolist()
    countries.append(country[0])

paths_ds["path_wcci"]["Alpha 2"] = countries

print(paths_ds["path_wcci"])
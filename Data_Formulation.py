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

hack_dates = pd.merge(paths_ds["path_hackmageddon"], paths_ds["path_dates"], on = "Date", how = "left")


main_table = pd.merge(hack_dates, paths_ds["path_wcci"], left_on = "Country", right_on = "Alpha 2", how = "left")
main_table = main_table.drop(columns = "Alpha 2")
main_table = main_table.rename(columns={'Country_x': 'Alpha 2', 'Country_y': 'Country'})
main_table = main_table.dropna(subset = ["WCI Score"])
print(main_table)

'''a=paths_ds["path_hackmageddon"][paths_ds["path_hackmageddon"]["ID"]==364]["Country"].tolist()[0]
print(a.replace("\n", " "))
'''
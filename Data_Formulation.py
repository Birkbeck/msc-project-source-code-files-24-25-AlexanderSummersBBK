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



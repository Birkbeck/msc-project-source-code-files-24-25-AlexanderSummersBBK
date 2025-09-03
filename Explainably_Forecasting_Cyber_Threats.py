import pandas as pd
from pathlib import Path


path = Path.cwd()
data = pd.DataFrame(path / "main_table.csv")


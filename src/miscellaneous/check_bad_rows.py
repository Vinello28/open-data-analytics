import pandas as pd
import glob
f = '../data/raw/reclassified_multiclass_aiuti_2014.csv'
df = pd.read_csv(f, dtype=str)
missing = df[df['CLASSIFICAZIONE'].isna()]
if len(missing) > 0:
    print(missing.head())

import pandas as pd
import glob
files = glob.glob('../data/raw/reclassified_multiclass_aiuti_*.csv')
missing = 0
non_ai = 0
ai = 0
for f in files:
    df = pd.read_csv(f, dtype=str, usecols=['CLASSIFICAZIONE'])
    counts = df['CLASSIFICAZIONE'].fillna('MISSING').value_counts()
    missing += counts.get('MISSING', 0)
    non_ai += counts.get('NON_AI', 0)
    ai += counts.get('AI', 0)
print(f"MISSING: {missing}, NON_AI: {non_ai}, AI: {ai}")

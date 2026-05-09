import pandas as pd
from pathlib import Path

# Fix path to data dir
data_dir = Path("/Users/gabrielevianello/Documents/Università/UnivPM/Paper/Organizzazione dell'Impresa/open-data-analytics/data/distilled")
for path in sorted(data_dir.glob('pwc_distilled_*.csv')):
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"{path.name}: {len(df)} rows")
        if 'Label' in df.columns:
            vc = df['Label'].value_counts()
            if 'Virtual assistants' in vc:
                print(f"  Virtual assistants: {vc['Virtual assistants']}")
    except Exception as e:
        print(f"Error reading {path.name}: {e}")

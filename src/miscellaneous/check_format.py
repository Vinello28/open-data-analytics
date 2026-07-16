import pandas as pd
import glob
files = sorted(glob.glob('../data/raw/reclassified_multiclass_aiuti_*.csv'))
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        expected = len(lines[0].split(','))
        for i, l in enumerate(lines[1:], 2):
            if len(l.split(',')) != expected:
                print(f"Malformated line in {f} at index {i}: {l.strip()}")
                break
    break

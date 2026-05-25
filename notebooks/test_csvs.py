import os
import glob

data_dir = 'data/raw'
pattern = os.path.join(data_dir, 'reclassified_multiclass_aiuti_*.csv')
files = sorted(glob.glob(pattern))

all_good = True
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if not lines: continue
        expected_cols = len(lines[0].split(','))
        for idx, line in enumerate(lines[1:], 2):
            if '"' in line:
                print(f"File {f} line {idx} has quotes")
                all_good = False
                break
            cols = len(line.split(','))
            if cols != expected_cols:
                print(f"File {f} line {idx} has {cols} cols, expected {expected_cols}")
                all_good = False
                break

if all_good:
    print("ALL FILES ARE CLEAN AND ALIGNED FOR STATA!")

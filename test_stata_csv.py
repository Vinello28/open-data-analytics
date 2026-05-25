import pandas as pd
import csv

df = pd.DataFrame({
    'col1': ['"hello", world', 'test\nnewline', 'normal'],
    'col2': ['a,b,c', 'd"e"f', 'g,h,i']
})

for col in df.columns:
    df[col] = df[col].astype(str).str.replace('"', '', regex=False).str.replace(',', ' ', regex=False).str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False)

df.to_csv('test.csv', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
print("Done")

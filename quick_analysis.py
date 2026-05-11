import pandas as pd
import numpy as np

# Carica i due dataset
print("Loading datasets...")
df_bert = pd.read_csv("data/training/bert_bin_unbiased_v2.csv", encoding='latin-1')
df_regex = pd.read_csv("data/processed/regex_classification_tp_tn.csv", encoding='latin-1')

print(f"BERT: {len(df_bert)} rows, {df_bert.columns.tolist()}")
print(f"REGEX: {len(df_regex)} rows, {df_regex.columns.tolist()}")

# === ANALISI 1: BERT ===
print("\n" + "="*80)
print("ANALISI 1: BERT DATASET - DESCRIZIONI TRONCATE")
print("="*80)

df_bert['text_length'] = df_bert['text'].astype(str).str.len()
df_bert['is_truncated'] = df_bert['text'].astype(str).str.endswith('...')
truncated = df_bert['is_truncated'].sum()
complete = len(df_bert) - truncated

print(f"Descrizioni troncate (...): {truncated}")
print(f"Descrizioni complete: {complete}")
print(f"Percentuale troncate: {truncated/len(df_bert)*100:.2f}%")

print(f"\nStatistiche lunghezza BERT:")
print(f"  Min: {df_bert['text_length'].min()}")
print(f"  Max: {df_bert['text_length'].max()}")
print(f"  Media: {df_bert['text_length'].mean():.0f}")
print(f"  Mediana: {df_bert['text_length'].median():.0f}")

print(f"\nDescrizioni troncate per label:")
for label in sorted(df_bert['label'].unique()):
    subset = df_bert[df_bert['label'] == label]
    trunc = subset['is_truncated'].sum()
    pct = trunc / len(subset) * 100
    print(f"  {label}: {trunc}/{len(subset)} ({pct:.1f}%)")

# === ANALISI 2: REGEX ===
print("\n" + "="*80)
print("ANALISI 2: REGEX DATASET - DISTRIBUZIONI")
print("="*80)

print(f"Distribuzione Label_predicted:")
print(df_regex['Label_predicted'].value_counts().sort_index())
print(f"\nPercentuali:")
for label, count in df_regex['Label_predicted'].value_counts().sort_index().items():
    pct = count / len(df_regex) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

print(f"\nValori nulli REGEX:")
print(df_regex.isnull().sum())

df_regex['text_length'] = df_regex['Descrizione'].astype(str).str.len()
print(f"\nStatistiche lunghezza REGEX:")
print(f"  Min: {df_regex['text_length'].min()}")
print(f"  Max: {df_regex['text_length'].max()}")
print(f"  Media: {df_regex['text_length'].mean():.0f}")
print(f"  Mediana: {df_regex['text_length'].median():.0f}")

# === ESEMPI ===
print("\n" + "="*80)
print("ESEMPI BERT (3 righe)")
print("="*80)
for idx, row in df_bert.head(3).iterrows():
    print(f"\n[{idx}] Label: {row['label']}")
    text = str(row['text'])[:100]
    print(f"Text: {text}...")

print("\n" + "="*80)
print("ESEMPI REGEX (3 righe)")
print("="*80)
for idx, row in df_regex.head(3).iterrows():
    print(f"\n[{idx}] Label_predicted: {row['Label_predicted']}")
    text = str(row['Descrizione'])[:100]
    print(f"Desc: {text}...")

print("\n✓ Analisi completata")

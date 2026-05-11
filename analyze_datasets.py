import pandas as pd
import numpy as np

# Carica i due dataset
bert_path = "data/training/bert_bin_unbiased_v2.csv"
regex_path = "data/processed/regex_classification_tp_tn.csv"

print("=" * 80)
print("CARICAMENTO DATASET")
print("=" * 80)

# Leggi bert dataset
try:
    df_bert = pd.read_csv(bert_path, encoding='utf-8')
except:
    df_bert = pd.read_csv(bert_path, encoding='latin-1')

print(f"✓ BERT dataset caricato: {len(df_bert)} righe, {len(df_bert.columns)} colonne")

# Leggi regex dataset
try:
    df_regex = pd.read_csv(regex_path, encoding='utf-8')
except:
    df_regex = pd.read_csv(regex_path, encoding='latin-1')

print(f"✓ REGEX dataset caricato: {len(df_regex)} righe, {len(df_regex.columns)} colonne")

print("\n" + "=" * 80)
print("STRUTTURA DATASET BERT")
print("=" * 80)
print(f"Colonne: {df_bert.columns.tolist()}")
print(f"\nDtypes:")
print(df_bert.dtypes)
print(f"\nPrime 2 righe:")
print(df_bert.head(2).to_string())

print("\n" + "=" * 80)
print("STRUTTURA DATASET REGEX")
print("=" * 80)
print(f"Colonne: {df_regex.columns.tolist()}")
print(f"\nDtypes:")
print(df_regex.dtypes)
print(f"\nPrime 2 righe:")
print(df_regex.head(2).to_string())

# ANALISI 1: BERT DATASET - Descrizioni troncate
print("\n" + "=" * 80)
print("ANALISI 1: BERT DATASET - DESCRIZIONI TRONCATE")
print("=" * 80)

# Identifica colonna descrizione
desc_col = None
for col in df_bert.columns:
    if 'desc' in col.lower() or 'text' in col.lower() or 'content' in col.lower():
        desc_col = col
        break

if desc_col is None:
    desc_col = df_bert.columns[-1]  # ultima colonna è solitamente la descrizione

print(f"Colonna descrizione identificata: '{desc_col}'")

# Conta descrizioni troncate (terminano con ...)
df_bert['is_truncated'] = df_bert[desc_col].astype(str).str.endswith('...')
truncated_count = df_bert['is_truncated'].sum()
complete_count = len(df_bert) - truncated_count

print(f"\nDescrizioni troncate (terminano con ...): {truncated_count}")
print(f"Descrizioni complete: {complete_count}")
print(f"Percentuale troncate: {truncated_count/len(df_bert)*100:.2f}%")

# Statistiche di lunghezza
df_bert['text_length'] = df_bert[desc_col].astype(str).str.len()
print(f"\nStatistiche lunghezza testo (caratteri):")
print(f"  Min: {df_bert['text_length'].min()}")
print(f"  Max: {df_bert['text_length'].max()}")
print(f"  Media: {df_bert['text_length'].mean():.1f}")
print(f"  Mediana: {df_bert['text_length'].median():.1f}")
print(f"  Std Dev: {df_bert['text_length'].std():.1f}")

# Distribuzione per label
if 'label' in df_bert.columns:
    print(f"\nDescrizioni troncate per label:")
    for label in df_bert['label'].unique():
        subset = df_bert[df_bert['label'] == label]
        trunc = subset['is_truncated'].sum()
        pct = trunc / len(subset) * 100
        print(f"  {label}: {trunc}/{len(subset)} ({pct:.1f}%)")

# ANALISI 2: REGEX DATASET - TP vs TN
print("\n" + "=" * 80)
print("ANALISI 2: REGEX DATASET - DISTRIBUZIONI TP vs TN")
print("=" * 80)

if 'Label_predicted' in df_regex.columns:
    print(f"Distribuzione Label_predicted:")
    dist = df_regex['Label_predicted'].value_counts().sort_index()
    print(dist)
    print(f"\nPercentuali:")
    pct = df_regex['Label_predicted'].value_counts(normalize=True).sort_index() * 100
    for label, p in pct.items():
        print(f"  {label}: {p:.2f}%")

# Valori nulli
print(f"\nValori nulli nel dataset REGEX:")
print(df_regex.isnull().sum())

# Statistiche lunghezza descrizioni
if 'Description' in df_regex.columns:
    desc_col_regex = 'Description'
elif 'description' in df_regex.columns:
    desc_col_regex = 'description'
else:
    # Trova colonna descrizione
    for col in df_regex.columns:
        if 'desc' in col.lower() or 'text' in col.lower():
            desc_col_regex = col
            break
    if 'desc_col_regex' not in locals():
        desc_col_regex = df_regex.columns[-1]

print(f"\nColonna descrizione REGEX: '{desc_col_regex}'")
df_regex['text_length'] = df_regex[desc_col_regex].astype(str).str.len()
print(f"Statistiche lunghezza descrizioni REGEX:")
print(f"  Min: {df_regex['text_length'].min()}")
print(f"  Max: {df_regex['text_length'].max()}")
print(f"  Media: {df_regex['text_length'].mean():.1f}")
print(f"  Mediana: {df_regex['text_length'].median():.1f}")

# ESEMPI DA CIASCUN DATASET
print("\n" + "=" * 80)
print("ESEMPI DATASET BERT (prime 3 righe complete)")
print("=" * 80)
for idx, row in df_bert.head(3).iterrows():
    print(f"\nRiga {idx}:")
    for col in df_bert.columns:
        val = str(row[col])[:150]
        print(f"  {col}: {val}...")

print("\n" + "=" * 80)
print("ESEMPI DATASET REGEX (prime 3 righe complete)")
print("=" * 80)
for idx, row in df_regex.head(3).iterrows():
    print(f"\nRiga {idx}:")
    for col in df_regex.columns:
        val = str(row[col])[:150]
        print(f"  {col}: {val}...")

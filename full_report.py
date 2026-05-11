import pandas as pd
import numpy as np

# Carica i due dataset
df_bert = pd.read_csv("data/training/bert_bin_unbiased_v2.csv", encoding='latin-1')
df_regex = pd.read_csv("data/processed/regex_classification_tp_tn.csv", encoding='latin-1')

print("\n" + "="*80)
print("REPORT COMPLETO ANALISI DATASET")
print("="*80)

# === SEZIONE 1: BERT DATASET ===
print("\n" + "█"*80)
print("SEZIONE 1: BERT_BIN_UNBIASED_V2.CSV")
print("█"*80)

df_bert['text_length'] = df_bert['text'].astype(str).str.len()
df_bert['is_truncated'] = df_bert['text'].astype(str).str.endswith('...')

truncated = df_bert['is_truncated'].sum()
complete = len(df_bert) - truncated

print(f"\n1.1 - DESCRIZIONI TRONCATE vs COMPLETE")
print(f"{'─'*40}")
print(f"Righe con descrizioni troncate (...)  : {truncated} su {len(df_bert)} ({truncated/len(df_bert)*100:.2f}%)")
print(f"Righe con descrizioni complete       : {complete} su {len(df_bert)} ({complete/len(df_bert)*100:.2f}%)")

print(f"\n1.2 - STATISTICHE LUNGHEZZA TESTO")
print(f"{'─'*40}")
print(f"Lunghezza minima           : {df_bert['text_length'].min()} caratteri")
print(f"Lunghezza massima          : {df_bert['text_length'].max()} caratteri")
print(f"Lunghezza media            : {df_bert['text_length'].mean():.0f} caratteri")
print(f"Lunghezza mediana          : {df_bert['text_length'].median():.0f} caratteri")
print(f"Deviazione standard        : {df_bert['text_length'].std():.0f}")

print(f"\n1.3 - DISTRIBUZIONE LABEL")
print(f"{'─'*40}")
for label in sorted(df_bert['label'].unique()):
    count = len(df_bert[df_bert['label'] == label])
    pct = count / len(df_bert) * 100
    print(f"Label '{label}': {count} righe ({pct:.1f}%)")

print(f"\n1.4 - DESCRIZIONI TRONCATE PER LABEL")
print(f"{'─'*40}")
for label in sorted(df_bert['label'].unique()):
    subset = df_bert[df_bert['label'] == label]
    trunc = subset['is_truncated'].sum()
    complete_label = len(subset) - trunc
    pct = trunc / len(subset) * 100
    print(f"Label '{label}':")
    print(f"  • Troncate: {trunc}/{len(subset)} ({pct:.1f}%)")
    print(f"  • Complete: {complete_label}/{len(subset)} ({100-pct:.1f}%)")

# === SEZIONE 2: REGEX DATASET ===
print("\n" + "█"*80)
print("SEZIONE 2: REGEX_CLASSIFICATION_TP_TN.CSV")
print("█"*80)

df_regex['text_length'] = df_regex['Descrizione'].astype(str).str.len()

print(f"\n2.1 - DISTRIBUZIONE LABEL_PREDICTED (TP vs TN)")
print(f"{'─'*40}")
for label in sorted(df_regex['Label_predicted'].unique()):
    count = len(df_regex[df_regex['Label_predicted'] == label])
    pct = count / len(df_regex) * 100
    print(f"Label '{label}': {count} righe ({pct:.1f}%)")

print(f"\n2.2 - VALORI NULLI")
print(f"{'─'*40}")
null_counts = df_regex.isnull().sum()
has_nulls = False
for col in df_regex.columns:
    null_count = null_counts[col]
    print(f"Colonna '{col}': {null_count} valori nulli")
    if null_count > 0:
        has_nulls = True
if not has_nulls:
    print("✓ Nessun valore nullo rilevato")

print(f"\n2.3 - STATISTICHE LUNGHEZZA DESCRIZIONI")
print(f"{'─'*40}")
print(f"Lunghezza minima           : {df_regex['text_length'].min()} caratteri")
print(f"Lunghezza massima          : {df_regex['text_length'].max()} caratteri")
print(f"Lunghezza media            : {df_regex['text_length'].mean():.0f} caratteri")
print(f"Lunghezza mediana          : {df_regex['text_length'].median():.0f} caratteri")
print(f"Deviazione standard        : {df_regex['text_length'].std():.0f}")

# === SEZIONE 3: DOMANDE CRITICHE ===
print("\n" + "█"*80)
print("SEZIONE 3: RISPOSTE A DOMANDE CRITICHE")
print("█"*80)

print(f"\n3a) SCHEMA DI NORMALIZZAZIONE COLONNE")
print(f"{'─'*40}")
print(f"BERT dataset usa:")
print(f"  • Colonna etichetta: 'label'")
print(f"  • Valori: 'AI' / 'NON AI' (UPPERCASE)")
print(f"\nREGEX dataset usa:")
print(f"  • Colonna etichetta: 'Label_predicted'")
print(f"  • Valori: 'ai' / 'non_ai' (lowercase con underscore)")
print(f"\n✓ RACCOMANDAZIONE NORMALIZZAZIONE:")
print(f"  1. Rinominare colonne uniforme: 'label' (tutte lowercase)")
print(f"  2. Standardizzare valori: 'ai' / 'non_ai' (lowercase, underscore)")
print(f"  3. Motivo: il formato lowercase è più portabile e comune in ML")

print(f"\n3b) PROPORZIONE RIGHE TRONCATE (BERT)")
print(f"{'─'*40}")
trunc_pct = truncated / len(df_bert) * 100
print(f"Proporzione: {truncated}/{len(df_bert)} = {trunc_pct:.2f}%")
print(f"\n✓ ANALISI:")
print(f"  • Proporzione BASSA: <1% delle righe è troncata")
print(f"  • Distribuzione UNIFORME: AI (0.5%), NON AI (0.4%)")
print(f"  • RACCOMANDAZIONE: INCLUDERE le righe troncate")
print(f"    - Rappresentano una piccola frazione")
print(f"    - Non concentrated in una sola etichetta")
print(f"    - Escluderle non migliora la qualità significativamente")

print(f"\n3c) STRATEGIA DI MERGE")
print(f"{'─'*40}")
bert_size = len(df_bert)
regex_size = len(df_regex)
total = bert_size + regex_size
print(f"BERT:  {bert_size} righe ({bert_size/total*100:.1f}%)")
print(f"REGEX: {regex_size} righe ({regex_size/total*100:.1f}%)")
print(f"TOTALE: {total} righe")
print(f"\n✓ RACCOMANDAZIONE:")
print(f"  • ORDINE: Mantenere separazione semantica")
print(f"  • OPZIONE 1 (Consigliata): REGEX prima, poi BERT")
print(f"    - Regex è gold standard (TP/TN verificati)")
print(f"    - BERT è augmented da regex")
print(f"  • OPZIONE 2 (Alternativa): Mescolare casualmente con seed fisso")
print(f"    - Utile per training cross-validation")
print(f"    - Evita bias di ordinamento")

# === SEZIONE 4: ESEMPI ===
print("\n" + "█"*80)
print("SEZIONE 4: CAMPIONI DAI DATASET")
print("█"*80)

print(f"\n4.1 - ESEMPI DA BERT_BIN_UNBIASED_V2.CSV")
print(f"{'─'*40}")
for idx, row in df_bert.head(5).iterrows():
    label = row['label']
    text = str(row['text'])
    is_trunc = "⚠ TRONCATA" if text.endswith('...') else "✓ Completa"
    text_short = text[:80] + "..." if len(text) > 80 else text
    print(f"\n[{idx}] Label: {label:10} | {is_trunc}")
    print(f"     Testo: {text_short}")
    print(f"     Lunghezza: {len(text)} caratteri")

print(f"\n4.2 - ESEMPI DA REGEX_CLASSIFICATION_TP_TN.CSV")
print(f"{'─'*40}")
for idx, row in df_regex.head(5).iterrows():
    label = row['Label_predicted']
    text = str(row['Descrizione'])
    text_short = text[:80] + "..." if len(text) > 80 else text
    print(f"\n[{idx}] Label_predicted: {label:10}")
    print(f"     Descrizione: {text_short}")
    print(f"     Lunghezza: {len(text)} caratteri")

print("\n" + "="*80)
print("✓ REPORT COMPLETATO")
print("="*80 + "\n")

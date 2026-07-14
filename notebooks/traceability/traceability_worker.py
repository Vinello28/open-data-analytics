"""Estrazione dei record di tracciabilita' dai file annuali degli aiuti di stato.

I pattern NON sono definiti qui: vivono in `traceability_patterns.py` (oracolo unico).

Rispetto alla v1:
- la regex viene applicata anche a TITOLO_MISURA (oltre a TITOLO_PROGETTO e DESCRIZIONE_PROGETTO);
- ogni positivo espone MATCH_SOURCE (quale campo ha fatto match) e MATCH_RULE (quale regola),
  cosi' il contributo della misura e' isolabile e ogni etichetta e' spiegabile;
- l'output viene riscritto da zero (la v1 usava mode='a': rilanciare l'estrazione su file
  gia' esistenti duplicava le righe).

NB TITOLO_MISURA e' il nome dello *schema di aiuto* ("Fondo di garanzia per le PMI"), non del
progetto: un match li' significa "il progetto appartiene a un programma orientato alla
tracciabilita'", evidenza piu' debole di un match sul progetto. Una singola misura puo' coprire
oltre 100k record: MATCH_SOURCE serve proprio a misurare (ed eventualmente escludere) questo caso.
"""
import os
import re

import pandas as pd

from traceability_patterns import RULE_NAMES, get_mask, rule_labels, rule_masks

CHUNK_SIZE = 100_000

# nome sorgente -> colonna del dataset
FIELDS = {
    'titolo_misura': 'TITOLO_MISURA',
    'titolo_progetto': 'TITOLO_PROGETTO',
    'descrizione': 'DESCRIZIONE_PROGETTO',
}


def _any(masks, empty: pd.Series) -> pd.Series:
    out = None
    for m in masks:
        out = m if out is None else (out | m)
    return empty if out is None else out


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Filtra le righe di tracciabilita' e le annota con MATCH_SOURCE / MATCH_RULE."""
    empty = pd.Series(False, index=chunk.index)

    per_field_rules = {}   # campo -> {regola: maschera}
    for src, col in FIELDS.items():
        if col in chunk.columns:
            per_field_rules[src] = rule_masks(chunk[col])
        else:
            per_field_rules[src] = {r: empty for r in RULE_NAMES}

    # un campo e' "match" se scatta almeno una regola su quel campo
    source_masks = {
        src: _any(rules.values(), empty) for src, rules in per_field_rules.items()
    }
    # una regola e' "match" se scatta su almeno un campo
    rule_of_any_field = {
        rule: _any((per_field_rules[src][rule] for src in FIELDS), empty)
        for rule in RULE_NAMES
    }

    final = _any(source_masks.values(), empty)
    if not final.any():
        return chunk.iloc[0:0]

    out = chunk[final].copy()
    out['MATCH_SOURCE'] = rule_labels(
        {src: m[final] for src, m in source_masks.items()}, index=out.index
    )
    out['MATCH_RULE'] = rule_labels(
        {r: m[final] for r, m in rule_of_any_field.items()}, index=out.index
    )
    return out


def process_file(args):
    """Elabora un file annuale a chunk e scrive i soli record di tracciabilita'."""
    file_path, output_dir = args
    filename = os.path.basename(file_path)
    match = re.search(r'aiuti_(\d{4})\.csv', filename)
    if not match:
        return filename, 0
    year = match.group(1)
    output_path = os.path.join(output_dir, f'traceability_aiuti_{year}.csv')

    print(f"[{filename}] Inizio elaborazione...")
    try:
        chunk_iterator = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)
    except Exception as e:
        print(f"[{filename}] Errore apertura: {e}")
        return filename, 0

    matched = []
    for chunk in chunk_iterator:
        filtered = process_chunk(chunk)
        if not filtered.empty:
            matched.append(filtered)

    # scrittura unica (niente mode='a': l'output e' sempre riscritto da zero)
    if matched:
        result = pd.concat(matched, ignore_index=True)
        result.to_csv(output_path, index=False)
        total = len(result)
    else:
        if os.path.exists(output_path):
            os.remove(output_path)
        total = 0

    print(f"[{filename}] Fine elaborazione. Match trovati: {total}")
    return filename, total

"""Applica un metodo di estrazione all'intero sottoinsieme AI e aggiunge la
colonna TECNOLOGIE_AI (tecnologie separate da '|', con regola di fallback).
"""
import os
import sys
import time

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

SUBSET = os.path.join(REPO_ROOT, "data", "distilled", "ai_subset.csv")
COL = "TECNOLOGIE_AI"


def get_method(name):
    if name == "regex":
        from ai_tech.methods import regex_gazetteer as m
    elif name == "spacy":
        from ai_tech.methods import spacy_ruler as m
    elif name == "transformer":
        from ai_tech.methods import transformer_zeroshot as m
    elif name == "llm":
        from ai_tech.methods import llm_openai as m
    else:
        raise ValueError(f"metodo sconosciuto: {name}")
    return m


def run(method="regex", subset=SUBSET, output=None, **kwargs):
    output = output or subset
    df = pd.read_csv(subset, dtype=str).fillna("")
    m = get_method(method)
    t0 = time.time()
    preds = m.extract_batch(df["DESCRIZIONE_PROGETTO"].tolist(), **kwargs)
    df[COL] = ["|".join(p) for p in preds]
    df.to_csv(output, index=False)
    dt = time.time() - t0
    cov = (df[COL].str.len() > 0).mean()
    print(f"Metodo={method}  righe={len(df)}  coverage={cov:.1%}  tempo={dt:.1f}s -> {output}")
    return df


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "regex"
    run(method)

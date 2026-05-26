"""Costruisce un campione stratificato di descrizioni AI per il goldset di
valutazione. Pre-popola le etichette candidate (regex) come punto di partenza
per la revisione manuale. La colonna 'gold' va compilata/corretta a mano.
"""
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from ai_tech.methods import regex_gazetteer as rg  # noqa: E402
from ai_tech import taxonomy as T  # noqa: E402

SUBSET = os.path.join(REPO_ROOT, "data", "distilled", "ai_subset.csv")
OUT = os.path.join(REPO_ROOT, "data", "distilled", "goldset_ai_tecnologie.csv")
DUMP = os.path.join(REPO_ROOT, "data", "distilled", "goldset_to_annotate.txt")


def build(n_per_tech=6, n_generic=12, n_empty=6, seed=42, min_len=40):
    df = pd.read_csv(SUBSET, dtype=str).fillna("")
    df = df.drop_duplicates(subset=["DESCRIZIONE_PROGETTO"]).reset_index(drop=True)
    df = df[df["DESCRIZIONE_PROGETTO"].str.len() >= min_len].reset_index(drop=True)

    df["regex_pred"] = [rg.extract(t) for t in df["DESCRIZIONE_PROGETTO"]]
    df["_specifics"] = df["regex_pred"].apply(lambda labs: [l for l in labs if l != T.GENERIC])

    picked = set()
    rows = []

    def take(sub, k):
        sub = sub[~sub.index.isin(picked)]
        if sub.empty:
            return
        s = sub.sample(n=min(k, len(sub)), random_state=seed)
        for idx in s.index:
            picked.add(idx)
            rows.append(idx)

    # Stratifica per ciascuna tecnologia specifica
    for tech in T.specific_labels():
        mask = df["_specifics"].apply(lambda labs: tech in labs)
        take(df[mask], n_per_tech)

    # Solo generica
    mask_generic = df["regex_pred"].apply(lambda labs: labs == [T.GENERIC])
    take(df[mask_generic], n_generic)

    # Nessun match (per testare recall di modello/LLM)
    mask_empty = df["regex_pred"].apply(lambda labs: len(labs) == 0)
    take(df[mask_empty], n_empty)

    sample = df.loc[rows].copy().reset_index(drop=True)
    sample.insert(0, "id", range(1, len(sample) + 1))
    sample["regex_pred"] = sample["regex_pred"].apply(lambda labs: "|".join(labs))
    sample["gold"] = sample["regex_pred"]  # punto di partenza, da correggere

    cols = ["id", "ANNO", "DESCRIZIONE_PROGETTO", "regex_pred", "gold"]
    sample[cols].to_csv(OUT, index=False)

    with open(DUMP, "w", encoding="utf-8") as f:
        for _, r in sample.iterrows():
            f.write(f"[{r['id']}] ({r['ANNO']}) regex={r['regex_pred']}\n")
            f.write(r["DESCRIZIONE_PROGETTO"].strip() + "\n\n")

    print(f"Campione: {len(sample)} righe -> {OUT}")
    print(f"Dump leggibile -> {DUMP}")
    return sample


if __name__ == "__main__":
    build()

"""Costruisce un GOLD SET stratificato per la validazione umana del task "tracciabilita".

Campiona in modo bilanciato attraverso i quadranti dell'accordo regex-vs-modello e le
fasce di confidenza/forza-regex, poi pre-etichetta ogni record con il giudice LLM v2
(prompt migliorato + few-shot) per accelerare l'annotazione (review-and-correct).

L'annotatore compila la colonna GOLD_LABEL ('tracciabilita' o 'altro') giudicando in modo
indipendente; LLM_LABEL_v2 e' solo un suggerimento. Il file risultante permette di misurare
la precisione REALE sia del giudice LLM sia del modello.

Uso:
    python build_gold_set.py                # usa LM Studio su 127.0.0.1:1234 per il prefill
    python build_gold_set.py --no-llm       # salta il prefill LLM (GOLD da fare a mano)

Output: data/traceability/validation/gold_set_to_annotate.csv
"""
import os
import argparse

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
VAL_DIR = os.path.join(HERE, "..", "..", "data", "traceability", "validation")
PARQUET = os.path.join(VAL_DIR, "comparison_unique.parquet")
OUT_CSV = os.path.join(VAL_DIR, "gold_set_to_annotate.csv")

POS, NEG = "tracciabilita", "altro"
SEED = 42

# Termine forte inequivocabile (stessa radice del pattern "core" della regex di training).
STRONG_RE = r"\b(?:rin)?tracciabilit[aà]\b|provenienza del prodotto|origine certificata|passaporto digitale"

# Numerosita' per strato (bilanciato per far emergere i confini decisionali critici).
STRATA_SIZES = {
    "A_concordi_strong": 35,   # regex+modello=pos, con termine forte -> positivi "facili"
    "A_concordi_weak": 35,     # regex+modello=pos, solo pattern debole -> positivi "rumorosi"
    "C_modello_highconf": 40,  # solo modello=pos, confidenza alta -> over-trigger sicuro?
    "C_modello_lowconf": 35,   # solo modello=pos, confidenza bassa -> zona grigia
    "D_regex_only": 63,        # solo regex=pos (tutti: sono pochi) -> falsi positivi regex?
    "B_controllo_neg": 30,     # concordi negativi -> controllo (dovrebbero essere altro)
}


def assign_quadrant(df):
    return np.where((df.REGEX_LABEL == POS) & (df.AI_LABEL == POS), "A",
           np.where((df.REGEX_LABEL != POS) & (df.AI_LABEL != POS), "B",
           np.where((df.REGEX_LABEL != POS) & (df.AI_LABEL == POS), "C", "D")))


def build_strata(df):
    both = (df["TITOLO_PROGETTO"].fillna("") + " " +
            df["DESCRIZIONE_PROGETTO"].fillna("")).str.lower()
    strong = both.str.contains(STRONG_RE, regex=True)
    df = df.assign(quadrant=assign_quadrant(df), regex_strength=np.where(strong, "strong", "weak"))

    pools = {
        "A_concordi_strong": df[(df.quadrant == "A") & (df.regex_strength == "strong")],
        "A_concordi_weak":   df[(df.quadrant == "A") & (df.regex_strength == "weak")],
        "C_modello_highconf": df[(df.quadrant == "C") & (df.AI_CONFIDENCE >= 0.9)],
        "C_modello_lowconf":  df[(df.quadrant == "C") & (df.AI_CONFIDENCE < 0.9)],
        "D_regex_only": df[df.quadrant == "D"],
        "B_controllo_neg": df[df.quadrant == "B"],
    }
    picks = []
    for name, size in STRATA_SIZES.items():
        pool = pools[name]
        n = min(size, len(pool))
        if n == 0:
            print(f"[warn] strato '{name}' vuoto")
            continue
        s = pool.sample(n=n, random_state=SEED).copy()
        s["stratum"] = name
        picks.append(s)
        print(f"  {name:22s}: {n:3d} / {len(pool)} disponibili")
    return pd.concat(picks, ignore_index=True)


def prefill_llm(gold):
    import importlib
    import llm_judge_worker as judge
    importlib.reload(judge)
    avail = judge.ping("http://127.0.0.1:1234/v1")
    if not avail:
        print("[warn] LM Studio non raggiungibile: LLM_LABEL_v2 lasciato vuoto.")
        gold["LLM_LABEL_v2"] = ""
        return gold
    print(f"LM Studio ok ({avail[0]}). Pre-etichetto {len(gold)} record col prompt {judge.PROMPT_VERSION}...")
    recs = [{"titolo": t, "descrizione": d}
            for t, d in zip(gold["TITOLO_PROGETTO"].fillna(""), gold["DESCRIZIONE_PROGETTO"].fillna(""))]
    gold["LLM_LABEL_v2"] = judge.judge_batch(recs, model="openai/gpt-oss-20b",
                                             base_url="http://127.0.0.1:1234/v1", concurrency=4)
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="salta il prefill col giudice LLM")
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    print(f"Comparison table: {len(df):,} descrizioni uniche.")
    print("Campionamento stratificato:")
    gold = build_strata(df)
    gold = gold.sample(frac=1, random_state=SEED).reset_index(drop=True)  # mescola
    gold.insert(0, "gold_id", range(1, len(gold) + 1))

    if not args.no_llm:
        gold = prefill_llm(gold)
    else:
        gold["LLM_LABEL_v2"] = ""

    gold["GOLD_LABEL"] = ""   # da compilare a mano: 'tracciabilita' oppure 'altro'
    gold["GOLD_NOTE"] = ""    # eventuali note dell'annotatore

    # Ordine colonne: prima quelle per annotare, poi i "suggerimenti", infine i testi.
    cols = ["gold_id", "GOLD_LABEL", "GOLD_NOTE",
            "stratum", "quadrant", "regex_strength",
            "LLM_LABEL_v2", "AI_LABEL", "AI_CONFIDENCE", "AI_POSITIVE_PROB", "REGEX_LABEL",
            "ANNO", "TITOLO_PROGETTO", "DESCRIZIONE_PROGETTO"]
    gold = gold[[c for c in cols if c in gold.columns]]

    os.makedirs(VAL_DIR, exist_ok=True)
    gold.to_csv(OUT_CSV, index=False)
    print(f"\nGold set salvato: {os.path.abspath(OUT_CSV)}  ({len(gold)} record)")

    if "LLM_LABEL_v2" in gold and gold["LLM_LABEL_v2"].astype(str).str.len().gt(0).any():
        print("\nDistribuzione suggerimenti LLM_v2 per strato:")
        print(pd.crosstab(gold["stratum"], gold["LLM_LABEL_v2"]))


if __name__ == "__main__":
    main()

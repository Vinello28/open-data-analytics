"""Sweep della soglia di decisione del modello su P(tracciabilita) contro il GOLD umano.

Prerequisito: aver compilato la colonna GOLD_LABEL di
data/traceability/validation/gold_set_to_annotate.csv (valori: 'tracciabilita' | 'altro').

Per ogni soglia tau, il modello predice 'tracciabilita' sse AI_POSITIVE_PROB >= tau; lo script
stampa precision/recall/F1 vs GOLD e propone la tau che massimizza F1. Copiare la tau scelta in
config/model_config.yml -> inference.positive_threshold, poi `docker compose build classifier`.

NOTA metodologica: il gold e' STRATIFICATO (non un campione casuale della popolazione), quindi
precision/recall qui sono per-gold, utili a *ordinare* le soglie e a leggere il trade-off, non
una stima non distorta della precision di popolazione. Le colonne per-stratum aiutano a capire
dove il modello sbaglia.

Uso:
    python sweep_threshold.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
GOLD = os.path.join(HERE, "..", "..", "data", "traceability", "validation", "gold_set_to_annotate.csv")
POS, NEG = "tracciabilita", "altro"


def _norm(s):
    return s.astype(str).str.strip().str.lower()


def main():
    df = pd.read_csv(GOLD)
    if "GOLD_LABEL" not in df.columns:
        sys.exit("Manca la colonna GOLD_LABEL.")
    if "AI_POSITIVE_PROB" not in df.columns:
        sys.exit("Manca AI_POSITIVE_PROB: rigenera il gold set con build_gold_set.py aggiornato.")

    df["GOLD_LABEL"] = _norm(df["GOLD_LABEL"])
    df["AI_POSITIVE_PROB"] = pd.to_numeric(df["AI_POSITIVE_PROB"], errors="coerce")
    ann = df[df["GOLD_LABEL"].isin([POS, NEG]) & df["AI_POSITIVE_PROB"].notna()].copy()
    if len(ann) < 20:
        sys.exit(f"Pochi record annotati ({len(ann)}): compila GOLD_LABEL prima di calibrare.")

    y = (ann["GOLD_LABEL"] == POS).to_numpy()
    p = ann["AI_POSITIVE_PROB"].to_numpy()
    print(f"Record annotati: {len(ann)}  (positivi gold: {y.sum()}, negativi: {(~y).sum()})")
    print(f"{'tau':>5} {'prec':>6} {'rec':>6} {'F1':>6} {'pred+':>6}")

    best = (None, -1.0)
    for tau in np.round(np.arange(0.50, 0.99, 0.02), 2):
        pred = p >= tau
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and not np.isnan(prec) and not np.isnan(rec)) else 0.0
        print(f"{tau:>5} {prec:>6.3f} {rec:>6.3f} {f1:>6.3f} {int(pred.sum()):>6}")
        if f1 > best[1]:
            best = (tau, f1)

    print(f"\nSoglia suggerita (max F1 sul gold): tau = {best[0]}  (F1 = {best[1]:.3f})")
    print("-> aggiorna config/model_config.yml: inference.positive_threshold, poi rebuild del gateway.")


if __name__ == "__main__":
    main()

"""Valuta regex, modello e giudice LLM contro il GOLD SET annotato a mano.

Prerequisito: aver compilato la colonna GOLD_LABEL di
`data/traceability/validation/gold_set_to_annotate.csv` con 'tracciabilita' o 'altro'.

Per ciascuno dei tre etichettatori (REGEX_LABEL, AI_LABEL del modello, LLM_LABEL_v2)
calcola accuracy, precision/recall/F1 sulla classe positiva e la confusion matrix,
usando l'annotazione umana come verita' di riferimento.

Uso:
    python score_gold_set.py [percorso_csv]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                             accuracy_score)

POS, NEG = "tracciabilita", "altro"
DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                           "traceability", "validation", "gold_set_to_annotate.csv")


def _norm(s):
    return s.astype(str).str.strip().str.lower()


def evaluate(gold, pred, name):
    p, r, f1, _ = precision_recall_fscore_support(gold, pred, labels=[POS], average=None,
                                                  zero_division=0)
    acc = accuracy_score(gold, pred)
    cm = confusion_matrix(gold, pred, labels=[POS, NEG])
    print(f"\n=== {name} (vs GOLD umano) ===")
    print(f"Accuracy {acc:.3f} | Precision(pos) {p[0]:.3f} | Recall(pos) {r[0]:.3f} | F1(pos) {f1[0]:.3f}")
    print("Confusion matrix [righe=GOLD, colonne=pred] labels=[tracciabilita, altro]:")
    print(pd.DataFrame(cm, index=[f"gold={POS}", f"gold={NEG}"],
                       columns=[f"pred={POS}", f"pred={NEG}"]).to_string())


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    df = pd.read_csv(path)
    if "GOLD_LABEL" not in df.columns:
        sys.exit("Manca la colonna GOLD_LABEL.")

    df["GOLD_LABEL"] = _norm(df["GOLD_LABEL"])
    annotated = df[df["GOLD_LABEL"].isin([POS, NEG])].copy()
    print(f"Record totali: {len(df)} | annotati (GOLD valido): {len(annotated)}")
    if len(annotated) < 20:
        print("Pochi record annotati: compila GOLD_LABEL prima di valutare.")
        if len(annotated) == 0:
            return
    print("Bilanciamento GOLD:", annotated["GOLD_LABEL"].value_counts().to_dict())

    gold = annotated["GOLD_LABEL"]
    for col, name in [("REGEX_LABEL", "REGEX"),
                      ("AI_LABEL", "MODELLO (UmBERTo)"),
                      ("LLM_LABEL_v2", "GIUDICE LLM v2")]:
        if col not in annotated.columns:
            continue
        pred = _norm(annotated[col])
        mask = pred.isin([POS, NEG])
        if mask.sum() == 0:
            print(f"\n[{name}] nessuna predizione valida, salto.")
            continue
        evaluate(gold[mask], pred[mask], name)

    # Precisione per strato del modello: dove sbaglia di piu'?
    if "stratum" in annotated.columns:
        print("\n=== Precisione MODELLO per strato ===")
        for st, sub in annotated.groupby("stratum"):
            pos_pred = sub[_norm(sub["AI_LABEL"]) == POS]
            if len(pos_pred):
                prec = (_norm(pos_pred["GOLD_LABEL"]) == POS).mean()
                print(f"  {st:22s}: precision={prec:.2f} su {len(pos_pred)} positivi-modello")


if __name__ == "__main__":
    main()

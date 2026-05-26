"""Confronta gli approcci di estrazione sul goldset annotato.

Calcola le metriche multilabel per regex, spaCy, transformer (con calibrazione
della soglia sul goldset) e LLM (se disponibile la chiave OpenAI). Ritorna una
tabella riassuntiva e il dettaglio per metodo.
"""
import os
import sys
import time

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from ai_tech import evaluate as E  # noqa: E402
from ai_tech import taxonomy as T  # noqa: E402

GOLDSET = os.path.join(REPO_ROOT, "data", "distilled", "goldset_ai_tecnologie.csv")


def _parse(cell):
    cell = (cell or "").strip()
    return [x for x in cell.split("|") if x]


def load_gold():
    df = pd.read_csv(GOLDSET, dtype=str).fillna("")
    texts = df["DESCRIZIONE_PROGETTO"].tolist()
    gold = [_parse(c) for c in df["gold"]]
    return df, texts, gold


def tune_transformer(texts, gold, grid=None):
    """Trova la soglia che massimizza il micro-F1 sul goldset."""
    from ai_tech.methods import transformer_zeroshot as tz
    if grid is None:
        grid = [round(x / 100, 2) for x in range(50, 100, 5)] + [0.97, 0.99]
    scores = tz.scores_batch(texts)
    best = None
    for thr in grid:
        preds = [tz.labels_from_scores(d, thr) for d in scores]
        m = E.evaluate(gold, preds, labels=T.canonical_labels())
        if best is None or m["micro_f1"] > best[1]["micro_f1"]:
            best = (thr, m, preds)
    return best  # (threshold, metrics, preds)


def run(include_llm=None):
    df, texts, gold = load_gold()
    labels = T.canonical_labels()
    rows = []
    details = {}

    from ai_tech.methods import regex_gazetteer as rg
    t0 = time.time()
    p = rg.extract_batch(texts)
    rt = time.time() - t0
    details["regex"] = p
    rows.append(E.summary_row("regex", E.evaluate(gold, p, labels), rt))

    from ai_tech.methods import spacy_ruler as sr
    t0 = time.time()
    p = sr.extract_batch(texts)
    rt = time.time() - t0
    details["spacy"] = p
    rows.append(E.summary_row("spacy", E.evaluate(gold, p, labels), rt))

    t0 = time.time()
    thr, m, p = tune_transformer(texts, gold)
    rt = time.time() - t0
    details["transformer"] = p
    sr_row = E.summary_row(f"transformer@{thr}", m, rt)
    rows.append(sr_row)

    from ai_tech.methods import llm_openai as llm
    if include_llm is None:
        include_llm = llm.available()
    if include_llm:
        t0 = time.time()
        p = llm.extract_batch(texts)
        rt = time.time() - t0
        details["llm"] = p
        rows.append(E.summary_row("llm", E.evaluate(gold, p, labels), rt))
    else:
        print("[info] LLM saltato: OPENAI_API_KEY non disponibile.")

    table = pd.DataFrame(rows)
    return table, details, df, gold


if __name__ == "__main__":
    table, details, df, gold = run()
    pd.set_option("display.width", 160)
    print(table.to_string(index=False))

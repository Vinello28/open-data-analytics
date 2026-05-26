"""Metriche di confronto fra approcci di estrazione (multilabel).

gold e pred sono liste allineate per riga, ognuna una lista/insieme di
tecnologie canoniche. Calcola precision/recall/F1 micro e macro, Jaccard medio
per riga, exact-match e coverage.
"""
import collections


def _to_sets(list_of_lists):
    return [set(x) for x in list_of_lists]


def evaluate(gold, pred, labels=None):
    gold = _to_sets(gold)
    pred = _to_sets(pred)
    assert len(gold) == len(pred), "gold e pred devono avere la stessa lunghezza"
    n = len(gold)

    if labels is None:
        labels = sorted({l for s in gold for l in s} | {l for s in pred for l in s})

    # Micro
    tp = sum(len(g & p) for g, p in zip(gold, pred))
    fp = sum(len(p - g) for g, p in zip(gold, pred))
    fn = sum(len(g - p) for g, p in zip(gold, pred))
    micro_p = tp / (tp + fp) if (tp + fp) else 0.0
    micro_r = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    # Macro (per-label)
    per_label = {}
    for lab in labels:
        ltp = sum(1 for g, p in zip(gold, pred) if lab in g and lab in p)
        lfp = sum(1 for g, p in zip(gold, pred) if lab not in g and lab in p)
        lfn = sum(1 for g, p in zip(gold, pred) if lab in g and lab not in p)
        lp = ltp / (ltp + lfp) if (ltp + lfp) else 0.0
        lr = ltp / (ltp + lfn) if (ltp + lfn) else 0.0
        lf = 2 * lp * lr / (lp + lr) if (lp + lr) else 0.0
        per_label[lab] = {"precision": lp, "recall": lr, "f1": lf,
                          "support": ltp + lfn}
    macro_f1 = sum(v["f1"] for v in per_label.values()) / len(labels) if labels else 0.0

    # Jaccard medio ed exact match
    jacc = []
    exact = 0
    for g, p in zip(gold, pred):
        if not g and not p:
            jacc.append(1.0)
            exact += 1
            continue
        inter = len(g & p)
        union = len(g | p)
        jacc.append(inter / union if union else 1.0)
        if g == p:
            exact += 1
    mean_jacc = sum(jacc) / n if n else 0.0

    coverage = sum(1 for p in pred if p) / n if n else 0.0

    return {
        "n": n,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "mean_jaccard": mean_jacc,
        "exact_match": exact / n if n else 0.0,
        "coverage": coverage,
        "per_label": per_label,
    }


def summary_row(name, metrics, runtime=None):
    """Riga compatta per tabella comparativa."""
    return {
        "metodo": name,
        "micro_P": round(metrics["micro_precision"], 3),
        "micro_R": round(metrics["micro_recall"], 3),
        "micro_F1": round(metrics["micro_f1"], 3),
        "macro_F1": round(metrics["macro_f1"], 3),
        "Jaccard": round(metrics["mean_jaccard"], 3),
        "exact": round(metrics["exact_match"], 3),
        "coverage": round(metrics["coverage"], 3),
        "runtime_s": round(runtime, 1) if runtime is not None else None,
    }

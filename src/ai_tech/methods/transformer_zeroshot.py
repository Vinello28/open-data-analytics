"""Approccio 3 — Zero-shot classification con transformer su GPU.

Usa un modello NLI multilingue (mDeBERTa-v3 xnli) in modalita zero-shot
multi-label: per ogni descrizione stima la probabilita di ciascuna tecnologia
e tiene quelle sopra soglia. Gira sulla RTX 5070Ti (device=0) con fallback CPU.
"""
from .. import taxonomy as T

NAME = "transformer"
MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
HYPOTHESIS = "Questo progetto utilizza {}."

# Etichetta candidata (frase naturale) -> tecnologia canonica.
ZS_LABELS = {
    "machine learning": "machine learning",
    "deep learning": "deep learning",
    "reti neurali artificiali": "reti neurali",
    "computer vision o visione artificiale": "computer vision",
    "elaborazione del linguaggio naturale": "elaborazione del linguaggio naturale",
    "intelligenza artificiale generativa o large language model": "ai generativa",
    "apprendimento per rinforzo": "reinforcement learning",
    "sistemi di raccomandazione": "sistemi di raccomandazione",
    "analisi predittiva o manutenzione predittiva": "analisi predittiva",
    "chatbot o assistente virtuale": "chatbot e assistenti virtuali",
    "robotica": "robotica",
    "guida autonoma": "guida autonoma",
    "riconoscimento vocale o del parlato": "riconoscimento vocale e del parlato",
    "riconoscimento ottico dei caratteri": "riconoscimento ottico caratteri",
    "sistemi esperti basati su regole": "sistemi esperti",
    "data mining o big data analytics": "data mining e big data",
    "intelligenza artificiale generica": T.GENERIC,
}

_clf = None


def load():
    global _clf
    if _clf is not None:
        return _clf
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    _clf = pipeline("zero-shot-classification", model=MODEL, device=device)
    return _clf


def scores_batch(texts, batch_size=16):
    """Ritorna, per ogni testo, {tecnologia_canonica: score} (max sui sinonimi)."""
    clf = load()
    texts = [str(t) if t is not None else "" for t in texts]
    cand = list(ZS_LABELS.keys())
    results = clf(texts, candidate_labels=cand, multi_label=True,
                  hypothesis_template=HYPOTHESIS, batch_size=batch_size)
    if isinstance(results, dict):
        results = [results]
    out = []
    for r in results:
        d = {}
        for lab, sc in zip(r["labels"], r["scores"]):
            canon = ZS_LABELS[lab]
            d[canon] = max(d.get(canon, 0.0), sc)
        out.append(d)
    return out


def labels_from_scores(score_dict, threshold):
    chosen = [c for c, s in score_dict.items() if s >= threshold]
    return T.apply_fallback(chosen)


def extract_batch(texts, threshold=0.5, batch_size=16):
    return [labels_from_scores(d, threshold) for d in scores_batch(texts, batch_size)]


def extract(text, threshold=0.5):
    return extract_batch([text], threshold=threshold)[0]

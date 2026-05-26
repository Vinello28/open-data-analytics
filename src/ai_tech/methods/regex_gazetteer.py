"""Approccio 1 — Regex / dizionario (gazetteer).

Deterministico e velocissimo: applica i pattern della tassonomia al testo.
Espone la firma comune extract(text) / extract_batch(texts).
"""
from multiprocessing import Pool, cpu_count

from .. import taxonomy as T

NAME = "regex"


def extract(text):
    """Tecnologie canoniche presenti nel testo (con regola di fallback)."""
    return T.apply_fallback(T.match_text(text))


def extract_batch(texts, n_workers=None):
    """Versione parallela su lista di testi. Per il gazetteer il costo per riga
    e' minimo, ma su volumi grandi il multiprocessing aiuta."""
    texts = list(texts)
    if not texts:
        return []
    if n_workers is None:
        n_workers = min(cpu_count(), 8)
    if n_workers <= 1 or len(texts) < 2000:
        return [extract(t) for t in texts]
    with Pool(n_workers) as pool:
        return pool.map(extract, texts, chunksize=500)

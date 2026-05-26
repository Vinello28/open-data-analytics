"""Approccio 4 — Estrazione via LLM (OpenAI) con output strutturato.

Per ogni descrizione chiede al modello di elencare, scegliendo SOLO dal
vocabolario canonico della tassonomia, le tecnologie AI menzionate o
chiaramente implicite. Output JSON validato contro il vocabolario.

Richiede la variabile d'ambiente OPENAI_API_KEY. Concorrenza via thread
(chiamate I/O-bound) e cache su disco per evitare costi ripetuti.
"""
import json
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor

from .. import taxonomy as T

NAME = "llm"
MODEL = os.environ.get("AI_TECH_LLM_MODEL", "gpt-4o-mini")
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                           "data", "distilled", ".cache_llm_tech.json")

_client = None
_cache = None
_ALLOWED = set(T.canonical_labels())


def available():
    return bool(os.environ.get("OPENAI_API_KEY"))


def load():
    global _client
    if _client is None:
        if not available():
            raise RuntimeError("OPENAI_API_KEY non impostata: l'approccio LLM non e' utilizzabile.")
        from openai import OpenAI
        _client = OpenAI()
    return _client


def _load_cache():
    global _cache
    if _cache is None:
        if os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        else:
            _cache = {}
    return _cache


def _save_cache():
    if _cache is not None:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False)


def _key(text):
    return hashlib.sha1((MODEL + "||" + str(text)).encode("utf-8")).hexdigest()


_SYSTEM = (
    "Sei un estrattore di tecnologie di intelligenza artificiale da descrizioni di "
    "progetti in italiano. Devi indicare quali tecnologie AI sono usate o chiaramente "
    "implicite nel testo, scegliendo ESCLUSIVAMENTE dalla lista consentita. "
    "Non inventare etichette fuori lista. Se il testo cita l'AI solo in modo generico "
    "senza una tecnica specifica, usa '" + T.GENERIC + "'. "
    "Rispondi solo con JSON nel formato {\"tecnologie\": [\"...\"]}."
)


def _one(text):
    cache = _load_cache()
    k = _key(text)
    if k in cache:
        return cache[k]
    client = load()
    allowed = ", ".join(f'"{c}"' for c in T.canonical_labels())
    user = f"Lista consentita: [{allowed}]\n\nDescrizione:\n{str(text)[:2000]}"
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user}],
    )
    raw = resp.choices[0].message.content
    try:
        labels = json.loads(raw).get("tecnologie", [])
    except Exception:
        labels = []
    labels = [l for l in labels if l in _ALLOWED]
    result = T.apply_fallback(labels)
    cache[k] = result
    return result


def extract(text):
    out = _one(text)
    _save_cache()
    return out


def extract_batch(texts, max_workers=8):
    texts = list(texts)
    _load_cache()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        out = list(ex.map(_one, texts))
    _save_cache()
    return out

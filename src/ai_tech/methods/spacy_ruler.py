"""Approccio 2 — NER rule-based con spaCy EntityRuler.

Carica il modello italiano (it_core_news_lg) per la tokenizzazione e aggiunge
un EntityRuler con pattern derivati dalla tassonomia. I pattern token sono
costruiti usando il tokenizer di spaCy, cosi le frasi multi-token (es.
"machine learning", "self-driving") si allineano correttamente. Gli acronimi
restano case-sensitive (TEXT), le frasi case-insensitive (LOWER).
"""
from .. import taxonomy as T

NAME = "spacy"
MODEL = "it_core_news_lg"

_nlp = None


def _build_patterns(nlp):
    patterns = []
    for canon, groups in T.TAXONOMY.items():
        for phrase in groups.get("ci", []):
            toks = [{"LOWER": tok.lower_} for tok in nlp.make_doc(phrase)]
            if toks:
                patterns.append({"label": "AI_TECH", "pattern": toks, "id": canon})
        for acro in groups.get("cs", []):
            toks = [{"TEXT": tok.text} for tok in nlp.make_doc(acro)]
            if toks:
                patterns.append({"label": "AI_TECH", "pattern": toks, "id": canon})
    return patterns


def load():
    global _nlp
    if _nlp is not None:
        return _nlp
    import spacy
    # Disabilitiamo le componenti statistiche non necessarie: ci serve solo
    # tokenizzazione + EntityRuler. Lo speed-up e' notevole.
    nlp = spacy.load(MODEL, disable=["tagger", "morphologizer", "parser",
                                     "lemmatizer", "attribute_ruler", "ner"])
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    ruler.add_patterns(_build_patterns(nlp))
    _nlp = nlp
    return _nlp


def extract(text):
    nlp = load()
    doc = nlp(str(text) if text else "")
    found = [ent.ent_id_ for ent in doc.ents if ent.label_ == "AI_TECH" and ent.ent_id_]
    return T.apply_fallback(found)


def extract_batch(texts, batch_size=256):
    nlp = load()
    texts = [str(t) if t is not None else "" for t in texts]
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        found = [ent.ent_id_ for ent in doc.ents if ent.label_ == "AI_TECH" and ent.ent_id_]
        out.append(T.apply_fallback(found))
    return out

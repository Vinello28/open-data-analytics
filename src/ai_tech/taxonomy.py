"""Tassonomia delle tecnologie AI da riconoscere nelle descrizioni progetto.

Per ogni tecnologia canonica si elencano le varianti/sinonimi:
- "ci": frasi cercate case-INSENSITIVE (parole intere, es. "machine learning").
- "cs": acronimi cercati case-SENSITIVE in maiuscolo (es. "ML", "CNN") per evitare
        falsi positivi (in italiano "ai"/"ml" minuscoli ricorrono spesso).

Seed iniziale dalla AI_REGEX_PATTERN di src/reclassify_annual.py, ampliato con
conoscenza di dominio. La lista va rivista/raffinata sulle frequenze reali del
corpus (vedi notebook). Usata da tutti i metodi: regex, EntityRuler spaCy
(pattern), zero-shot (set di etichette) e LLM (vocabolario consentito).
"""
import re

GENERIC = "intelligenza artificiale (generica)"

TAXONOMY = {
    "machine learning": {
        "ci": ["machine learning", "apprendimento automatico", "apprendimento macchina",
               "apprendimento statistico", "supervised learning", "unsupervised learning",
               "apprendimento supervisionato", "apprendimento non supervisionato"],
        "cs": ["ML"],
    },
    "deep learning": {
        "ci": ["deep learning", "apprendimento profondo"],
        "cs": ["DL"],
    },
    "reti neurali": {
        "ci": ["reti neurali", "rete neurale", "neural network", "neural networks",
               "rete convoluzionale", "reti convoluzionali", "convolutional",
               "rete ricorrente", "reti ricorrenti", "percettrone", "perceptron",
               "reti generative avversarie", "generative adversarial"],
        "cs": ["CNN", "RNN", "LSTM", "GAN", "GNN", "MLP"],
    },
    "computer vision": {
        "ci": ["computer vision", "visione artificiale", "visione computerizzata",
               "elaborazione delle immagini", "elaborazione di immagini",
               "riconoscimento delle immagini", "riconoscimento di immagini",
               "image recognition", "object detection", "riconoscimento di oggetti",
               "riconoscimento oggetti", "segmentazione delle immagini",
               "riconoscimento facciale", "face recognition", "analisi delle immagini"],
        "cs": [],
    },
    "elaborazione del linguaggio naturale": {
        "ci": ["natural language processing", "elaborazione del linguaggio naturale",
               "elaborazione linguaggio naturale", "comprensione del linguaggio",
               "text mining", "analisi testuale", "analisi del testo",
               "sentiment analysis", "analisi del sentiment", "analisi semantica",
               "classificazione semantica", "named entity", "estrazione di entita",
               "language understanding"],
        "cs": ["NLP", "NLU", "NER"],
    },
    "ai generativa": {
        "ci": ["ai generativa", "intelligenza artificiale generativa", "generative ai",
               "modelli generativi", "large language model", "modelli linguistici",
               "modello linguistico", "stable diffusion", "diffusion model",
               "generazione di testo", "generazione di immagini", "chatgpt"],
        "cs": ["LLM", "LLMs", "GPT", "GenAI", "VLM"],
    },
    "reinforcement learning": {
        "ci": ["reinforcement learning", "apprendimento per rinforzo",
               "apprendimento con rinforzo"],
        "cs": ["RL"],
    },
    "sistemi di raccomandazione": {
        "ci": ["sistema di raccomandazione", "sistemi di raccomandazione",
               "motore di raccomandazione", "recommendation system",
               "recommender system", "recommender"],
        "cs": [],
    },
    "analisi predittiva": {
        "ci": ["analisi predittiva", "algoritmi predittivi", "algoritmo predittivo",
               "modelli predittivi", "modello predittivo", "predictive analytics",
               "manutenzione predittiva", "manutenzione preventiva",
               "predictive maintenance", "previsione della domanda", "forecasting"],
        "cs": [],
    },
    "chatbot e assistenti virtuali": {
        "ci": ["chatbot", "chat bot", "assistente virtuale", "assistenti virtuali",
               "virtual assistant", "virtual assistants", "conversational ai",
               "agente conversazionale", "intelligenza artificiale conversazionale",
               "voicebot"],
        "cs": [],
    },
    "robotica": {
        "ci": ["robotica collaborativa", "robotica", "robotico", "robotici",
               "cobot", "robot autonomo", "robot autonomi", "bracci robotici"],
        "cs": ["AGV", "AMR"],
    },
    "guida autonoma": {
        "ci": ["guida autonoma", "veicoli autonomi", "veicolo autonomo",
               "autonomous driving", "self-driving", "automazione della guida"],
        "cs": [],
    },
    "riconoscimento vocale e del parlato": {
        "ci": ["riconoscimento vocale", "riconoscimento del parlato", "speech recognition",
               "speech-to-text", "text-to-speech", "sintesi vocale", "sintesi della voce",
               "elaborazione del parlato", "trascrizione automatica", "voice recognition"],
        "cs": ["ASR", "TTS", "STT"],
    },
    "riconoscimento ottico caratteri": {
        "ci": ["riconoscimento ottico dei caratteri", "riconoscimento ottico",
               "optical character recognition", "lettura automatica dei documenti",
               "document understanding"],
        "cs": ["OCR"],
    },
    "sistemi esperti": {
        "ci": ["sistema esperto", "sistemi esperti", "expert system",
               "sistema basato su regole", "sistemi basati su regole",
               "ragionamento automatico", "base di conoscenza", "knowledge base"],
        "cs": [],
    },
    "data mining e big data": {
        "ci": ["data mining", "big data analytics", "analisi dei big data",
               "estrazione della conoscenza", "knowledge discovery", "clustering",
               "data analytics"],
        "cs": [],
    },
    "intelligenza artificiale (generica)": {
        "ci": ["intelligenza artificiale", "artificial intelligence",
               "tecnologie ai", "tecnologie ia", "modelli di ai", "algoritmi di ai",
               "algoritmi di ia", "sistemi di ai", "sistema di ai"],
        "cs": ["IA", "AI"],
    },
}


def _compile(variants):
    if not variants:
        return None
    alternation = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True))
    return alternation


def build_patterns():
    """Ritorna {canonica: (regex_ci|None, regex_cs|None)} con pattern compilati."""
    compiled = {}
    for canon, groups in TAXONOMY.items():
        ci = _compile(groups.get("ci", []))
        cs = _compile(groups.get("cs", []))
        regex_ci = re.compile(rf"\b(?:{ci})\b", re.IGNORECASE) if ci else None
        regex_cs = re.compile(rf"\b(?:{cs})\b") if cs else None
        compiled[canon] = (regex_ci, regex_cs)
    return compiled


_PATTERNS = None


def match_text(text):
    """Ritorna la lista (ordinata) delle tecnologie canoniche presenti nel testo."""
    global _PATTERNS
    if _PATTERNS is None:
        _PATTERNS = build_patterns()
    if not text:
        return []
    s = str(text)
    found = []
    for canon, (rci, rcs) in _PATTERNS.items():
        if (rci and rci.search(s)) or (rcs and rcs.search(s)):
            found.append(canon)
    return found


def canonical_labels():
    """Etichette canoniche (utile per zero-shot e LLM)."""
    return list(TAXONOMY.keys())


def specific_labels():
    """Etichette canoniche escludendo quella generica."""
    return [c for c in TAXONOMY if c != GENERIC]


def apply_fallback(labels):
    """Regola concordata: se ci sono tech specifiche uso quelle (scartando la
    generica); altrimenti tengo la generica solo se presente."""
    labels = list(dict.fromkeys(labels))  # dedup preservando l'ordine
    specifics = [l for l in labels if l != GENERIC]
    if specifics:
        return specifics
    return [GENERIC] if GENERIC in labels else []


def entity_ruler_patterns():
    """Pattern in formato spaCy EntityRuler: lista di dict {label, pattern}.

    Le frasi case-insensitive diventano pattern token con LOWER; gli acronimi
    diventano pattern con TEXT (case-sensitive). label = tecnologia canonica.
    """
    patterns = []
    for canon, groups in TAXONOMY.items():
        for phrase in groups.get("ci", []):
            toks = [{"LOWER": w.lower()} for w in phrase.split()]
            patterns.append({"label": "AI_TECH", "pattern": toks, "id": canon})
        for acro in groups.get("cs", []):
            patterns.append({"label": "AI_TECH", "pattern": [{"TEXT": acro}], "id": canon})
    return patterns

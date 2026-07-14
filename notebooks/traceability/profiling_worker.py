"""Profilazione dei progetti di tracciabilita': tecnologia, tipo di intervento, settore, lessico.

Usato da `traceability_profiling.ipynb`. Tutte le funzioni pesanti stanno qui (non nel notebook)
per poter girare in multiprocessing su Ryzen 9 9900X: N_PROC = 10 (12 core fisici - 2 liberi).

Tre tassonomie multi-label + una pipeline lessicale spaCy:

- TECNOLOGIA      : cosa usano   (blockchain, IoT, RFID/NFC, QR, AI/ML, cloud, ERP/MES, GPS)
- TIPO_INTERVENTO : cosa fanno   (implementazione, formazione, consulenza, R&S, certificazione)
                    estende i pattern gia' presenti in src/regex_multiprocessing.py
- SETTORE         : dove         (agroalimentare, vitivinicolo, tessile, farmaceutico, ...)

Multi-label: un progetto puo' usare blockchain *e* IoT, o fare formazione *e* implementazione.
Le categorie non sono mutuamente esclusive, quindi ogni asse produce una colonna ';'-separated
piu' colonne booleane one-hot per l'analisi incrociata.
"""
import re
from collections import Counter

import pandas as pd

N_PROC = 10  # 12 core fisici - 2 lasciati liberi

# --------------------------------------------------------------------------------------
# Tassonomie (regex compilate una volta sola; il testo in ingresso e' gia' minuscolo)
# --------------------------------------------------------------------------------------

TECNOLOGIA = {
    'blockchain':      r'\b(?:blockchain|dlt|registri distribuiti|distributed ledger|smart contract|token|nft)\b',
    'iot_sensori':     r'\b(?:iot|internet of things|sensor[ei]|sensoristica|telemetria|domotica)\b',
    'rfid_nfc':        r'\b(?:rfid|nfc|tag (?:rfid|nfc)|etichett[ae] intelligent[ei])\b',
    'qr_barcode':      r'\b(?:qr[- ]?code|codice qr|barcode|codice a barre|datamatrix|codici? a barre)\b',
    # NB niente `\bai\b`: in italiano "ai" e' la preposizione ("ai prodotti", "ai clienti").
    # Misurato: matchava 1.156 record contro gli ~83 che citano davvero l'intelligenza artificiale.
    'ai_ml':           r'\b(?:intelligenza artificiale|machine learning|deep learning|reti neurali|computer vision|visione artificiale|apprendimento automatico)\b',
    'cloud':           r'\b(?:cloud|saas|piattaforma cloud|cloud computing)\b',
    'erp_mes':         r'\b(?:erp|mes|gestionale|sistema informativo|wms|plm)\b',
    'gps_geo':         r'\b(?:gps|geolocalizzazione|georeferenziazione|satellitare)\b',
}

TIPO_INTERVENTO = {
    # estende REGEX_IMPLEMENTAZIONE di src/regex_multiprocessing.py
    'implementazione': r'\b(?:implementazion\w*|sviluppo|realizzazion\w*|creazion\w*|installazion\w*|adozion\w*|introduzion\w*|digitalizzazion\w*)\b',
    # estende REGEX_FORMAZIONE di src/regex_multiprocessing.py
    'formazione':      r'\b(?:formazion\w*|corso|corsi|training|tirocinio|aggiornamento professionale|docenza|addestramento|percorso formativo)\b',
    'consulenza':      r'\b(?:consulenz\w*|consulente|affiancamento|advisory|supporto specialistico)\b',
    'ricerca_sviluppo': r'\b(?:ricerca (?:e |ed )?sviluppo|r&s|r&d|ricerca industriale|sperimentazion\w*|prototip\w*|brevett\w*)\b',
    'certificazione':  r'\b(?:certificazion\w*|accreditament\w*|audit|iso \d+|disciplinare|dop|igp|docg|\bdoc\b)\b',
}

SETTORE = {
    'agroalimentare':  r'\b(?:agroaliment\w*|agricol\w*|aliment\w*|food|zootecni\w*|ortofrutt\w*|caseari\w*|lattiero\w*|carne|olio|oliv\w*|cereal\w*)\b',
    'vitivinicolo':    r'\b(?:vitivinicol\w*|vino|vini|cantina|enolog\w*|uva|vigneto)\b',
    'tessile_moda':    r'\b(?:tessil\w*|abbigliamento|moda|calzatur\w*|concia|pell[ei]|filat[oi])\b',
    'farmaceutico':    r'\b(?:farmaceutic\w*|farmac\w*|biomedical\w*|sanitari\w*|dispositiv[oi] medic\w*)\b',
    'automotive':      r'\b(?:automotive|autoveicol\w*|componentistica auto|ricambi)\b',
    'legno_arredo':    r'\b(?:legno|arredo|arredamento|mobil[ei]|falegnameria)\b',
    # NB `magazzino` e `distribuzione` sono FUNZIONI aziendali, non settori: quasi ogni impresa ne ha
    # una. Tenerle dentro gonfiava il settore "logistica" (1.759 progetti, di cui 1.046 solo per la
    # parola "magazzino"). Il settore e' il business dell'impresa, non un suo reparto.
    'logistica':       r'\b(?:logistic\w*|trasport\w*|spedizion\w*|corriere|autotrasport\w*)\b',
    'edilizia':        r'\b(?:edilizi\w*|costruzion\w*|calcestruzzo|cantiere)\b',
}

TAXONOMIES = {
    'TECNOLOGIA': TECNOLOGIA,
    'TIPO_INTERVENTO': TIPO_INTERVENTO,
    'SETTORE': SETTORE,
}

_COMPILED = {
    axis: {cat: re.compile(pat) for cat, pat in cats.items()}
    for axis, cats in TAXONOMIES.items()
}


def _text(df: pd.DataFrame) -> pd.Series:
    """Testo su cui profilare: titolo + descrizione del progetto, minuscolo."""
    titolo = df.get('TITOLO_PROGETTO', pd.Series('', index=df.index)).fillna('').astype(str)
    descr = df.get('DESCRIZIONE_PROGETTO', pd.Series('', index=df.index)).fillna('').astype(str)
    return (titolo + ' . ' + descr).str.lower()


def tag_taxonomies(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge, per ogni asse, la colonna dei VALORI (';'-separated) e il conteggio `ASSE__n`.

    Multi-label: nessuna categoria esclude le altre, quindi il valore e' una lista
    (`TECNOLOGIA = "blockchain;iot_sensori"`), non una categoria singola.

    NB: NON vengono materializzate le colonne one-hot `ASSE__categoria`. Sarebbero 21 colonne
    di True/False che ripetono un'informazione gia' contenuta nella colonna dei valori, rendendo
    il dataset illeggibile. Per l'analisi si ricavano al volo con `onehot()`, che e' una sola
    riga di pandas: il dato canonico resta uno solo.
    """
    out = df.copy()
    text = _text(df)

    for axis, cats in _COMPILED.items():
        flags = pd.DataFrame(
            {cat: text.str.contains(rx, regex=True) for cat, rx in cats.items()},
            index=df.index,
        )
        out[axis] = flags.apply(lambda r: ';'.join(c for c in flags.columns if r[c]), axis=1)
        out[f'{axis}__n'] = flags.sum(axis=1)
    return out


def onehot(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Ricava le colonne booleane di un asse dalla colonna dei valori. Per grafici e incroci.

    Le categorie assenti dai dati esistono comunque (a 0), cosi' i grafici hanno sempre
    le stesse barre e restano confrontabili tra sottoinsiemi.
    """
    dummies = df[axis].fillna('').str.get_dummies(sep=';').astype(bool)
    return dummies.reindex(columns=list(TAXONOMIES[axis]), fill_value=False)


# --------------------------------------------------------------------------------------
# Pipeline lessicale (spaCy it_core_news_lg, gia' installato)
# --------------------------------------------------------------------------------------

# Parole onnipresenti nel bureaucratese dei bandi: non discriminano nulla.
DOMAIN_STOPWORDS = {
    'progetto', 'progetti', 'azienda', 'aziendale', 'aziendali', 'impresa', 'imprese',
    'obiettivo', 'obiettivi', 'attivita', 'attività', 'realizzazione', 'realizzare',
    'prevedere', 'prevede', 'previsto', 'consentire', 'permettere', 'consistere',
    'srl', 'spa', 'snc', 'sas', 'società', 'societa', 'euro', 'investimento',
    'intervento', 'ambito', 'settore', 'nuovo', 'nuova', 'nuovi', 'nuove',
    'sistema', 'sistemi', 'soluzione', 'soluzioni', 'gestione', 'processo', 'processi',
    'gia', 'già', 'inoltre', 'quindi', 'grazie', 'seguente', 'presente', 'proprio',
    'gestire', 'utilizzare', 'gia_', 'gruppo', 'gia’',
}


def lemmatize_corpus(texts, n_process: int = N_PROC, batch_size: int = 200):
    """Lemmatizza il corpus con spaCy in multiprocessing.

    Ritorna una lista di liste di lemmi (solo NOUN/PROPN/ADJ/VERB, no stopword, len>2).
    Disabilita parser/ner: servono solo tagger+lemmatizer -> molto piu' veloce.
    """
    import spacy

    nlp = spacy.load('it_core_news_lg', disable=['parser', 'ner'])
    stops = nlp.Defaults.stop_words | DOMAIN_STOPWORDS
    keep_pos = {'NOUN', 'PROPN', 'ADJ', 'VERB'}

    docs = []
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
        lemmas = [
            t.lemma_.lower()
            for t in doc
            if t.pos_ in keep_pos
            and not t.is_stop
            and not t.is_punct
            and not t.like_num
            and len(t.lemma_) > 2
            and t.lemma_.lower() not in stops
            and t.is_alpha
        ]
        docs.append(lemmas)
    return docs


def count_terms(docs, ngram: int = 1) -> Counter:
    """Conta unigrammi (ngram=1) o bigrammi (ngram=2) sui lemmi."""
    c = Counter()
    for lemmas in docs:
        if ngram == 1:
            c.update(lemmas)
        else:
            c.update(
                ' '.join(lemmas[i:i + ngram]) for i in range(len(lemmas) - ngram + 1)
            )
    return c

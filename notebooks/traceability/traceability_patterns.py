"""Oracolo unico dei pattern di tracciabilita' (v2).

ROOT CAUSE della v1
-------------------
Le regole composte della v1 verificavano che due termini comparissero *entrambi nello
stesso testo*, a qualsiasi distanza. Misurato sui match della sola regola azione+oggetto:
i due termini distavano **mediana 148 caratteri, 42% oltre 200**, su descrizioni lunghe in
media ~1.000 caratteri. Non e' una relazione semantica: e' una collisione lessicale.
In piu' i verbi generici (`monitoraggio` 1.831 hit, `identificazione` 595) accoppiati a un
qualsiasi `prodotti` sparso nel testo etichettavano come "tracciabilita'" consulenze di
cybersecurity, librerie online e gestionali di magazzino.
Risultato: solo il 18,2% dei 6.581 positivi conteneva un termine forte di tracciabilita'.

FIX v2
------
1. `near(A, B, window)` vincola i due termini alla **prossimita'** (~una proposizione),
   in entrambi gli ordini.
2. I verbi generici escono da ACTION: rientrano solo accoppiati a un target *forte*
   (filiera / lotto / materie prime), mai al generico "prodotti".
3. Ogni positivo espone la regola che lo ha generato (`rule_labels`) -> classificazione
   spiegabile e auditabile.

Questo modulo e' l'UNICA sorgente di verita' dei pattern (cfr. tasks/lessons.md:
"Un solo oracolo di label tra training e validazione"). `traceability_worker.py`,
`prepare_traceability_data.py` e `validation_worker.py` importano da qui.
"""
import re

import pandas as pd

# Finestra di prossimita' in caratteri (~ una proposizione).
WINDOW = 60
# Finestra piu' stretta per le regole che usano verbi generici.
WINDOW_TIGHT = 40


def near(a: str, b: str, window: int = WINDOW) -> str:
    """Pattern che matcha A e B entro `window` caratteri, in entrambi gli ordini."""
    return rf'(?:{a}.{{0,{window}}}?{b}|{b}.{{0,{window}}}?{a})'


# --- Termini forti: bastano da soli, nessun contesto richiesto -----------------
STRONG = (
    r'\b(?:'
    r'(?:rin)?tracciabil\w*'                      # tracciabilita', tracciabile, rintracciabilita'
    r'|provenienza (?:del|dei|della|delle) (?:prodott\w+|materi\w+|aliment\w+)'
    r'|origine certificat\w+'
    r'|passaporto digitale'
    r'|digital product passport'
    r'|caten[ae] di custodia'
    r'|chain of custody'
    r'|anti[- ]?contraffazione'
    r'|dalla stalla alla tavola'
    r'|dal campo alla tavola'
    r'|from farm to fork'
    r'|farm to fork'
    r')\b'
)

# --- Verbi/azioni SPECIFICI della tracciabilita' -------------------------------
# Rimossi rispetto a v1: monitoraggio, identificazione, autenticazione (generici).
ACTION = (
    r'\b(?:'
    r'tracciar\w*|tracciament\w*|tracciatur\w*|rintracci\w*|serializzazion\w*'
    r')\b'
)

# Azioni AMBIGUE: etichettatura/marcatura sono tracciabilita' solo in un contesto di filiera.
# `(?!\s+ce\b)` esclude "marcatura CE", che e' conformita' di prodotto, non tracciabilita'
# (falso amico trovato nell'audit: 20 occorrenze).
ACTION_LABELING = r'\b(?:etichettatur\w*|marcatur\w*)\b(?!\s+ce\b)'

# --- La filiera, in tutti i modi in cui la si nomina ---------------------------
# `catena/sistema di approvvigionamento` e' un SINONIMO di supply chain (lo chiedeva la
# specifica). Non serve una regola nuova: entrando nel vocabolario eredita i vincoli di
# prossimita' gia' validati. Da solo NON basta: l'audit mostra che nel 90% dei casi e'
# procurement (riorganizzare gli acquisti, gestionali di magazzino), non tracciabilita'.
SUPPLY_CHAIN = (
    r'filier[ae]|supply chain|caten[ae] di fornitura'
    r'|(?:caten[ae]|sistem[ai]) di approvvigionamento'
)

# --- Oggetti della tracciabilita' ---------------------------------------------
TARGET = (
    r'\b(?:'
    + SUPPLY_CHAIN +
    r'|lott[oi]|materi[ae] prim[ae]|semilavorat\w*'
    r'|prodott[oi]|merc[ei]|aliment[oi]'
    r')\b'
)

# --- Contesto di filiera: l'unico che rende "etichettatura/marcatura" tracciabilita' ---
TARGET_TRACE = (
    r'\b(?:' + SUPPLY_CHAIN + r'|lott[oi]'
    r'|materi[ae] prim[ae]|origine|provenienza)\b'
)

# --- Tecnologie abilitanti + contesto -----------------------------------------
TECH = (
    r'\b(?:'
    r'blockchain|dlt|registri distribuiti|distributed ledger|smart contract'
    r'|qr[- ]?code|codice qr|barcode|codice a barre'
    r'|rfid|nfc|internet of things|iot'
    r')\b'
)

CTX = (
    r'\b(?:'
    + SUPPLY_CHAIN +
    r'|provenienza|origine|logistic\w*'          # `logistic\w*` copre gia' "logistica intelligente"
    r'|anti[- ]?contraffazione|lott[oi]|magazzin\w*'
    r')\b'
)

# --- Filiera + attributi di trasparenza/controllo -----------------------------
# `caten[ae] del valore` sta qui e SOLO qui: da sola e' business-speak (l'audit trova negozi di
# abbigliamento e "presenza commerciale all'estero", 356 record). Vale solo se qualificata da
# CHAIN_MOD (tracciata, trasparente, certificata...).
CHAIN = r'\b(?:' + SUPPLY_CHAIN + r'|caten[ae] del valore)\b'
CHAIN_MOD = (
    r'\b(?:trasparen\w*|sicurezza alimentare|certificat\w*|garantit\w*|tracciat\w*)\b'
)


# --- DEVIAZIONI DALLA SPECIFICA (misurate, non opinioni) ------------------------------
# La lista di termini della specifica e' coperta, con tre eccezioni deliberate. Ogni numero
# viene da `audit_spec_terms.py` / `sample_new_terms.py` sui 23,96M record.
#
# 1. Termini cercati ma CONDIZIONATI a un contesto di filiera, non trigger autonomi:
#    blockchain, QR/barcode, IoT, DLT, tracciamento, filiera, supply chain.
#    Da soli pescano: DeFi e air mobility (blockchain), QR turistici di chiese e hotel,
#    "IoT per lo Smart Living", "Tracciamento Sanitario". Il termine c'e', la tracciabilita' no.
#
# 2. Termini ESCLUSI perche' falsi amici (il termine esiste, significa altro):
#    - `autenticazione`        -> login, pagamenti, cyber security, biometria (284 orfani)
#    - `identificazione + prodotto` -> "Formazione per l'addetto ai trattamenti fitosanitari" (213)
#    - `sicurezza del prodotto` -> le "schede di sicurezza" (SDS) dei prodotti chimici (118)
#    - `marcatura CE`          -> conformita' di prodotto, non tracciabilita' (20)
#    - `caten[ae] del valore` da sola -> business-speak generico (356) -> vale solo con CHAIN_MOD
#
# 3. La specifica chiede i termini larghi "solo nel titolo del progetto". NON implementato:
#    `filiera` nel solo titolo = 36.147 record, 7,5x l'intero set di positivi, e sono
#    "Filiera Bollicine in Rete", ristoranti a km 0, "filiera edile". Il titolo e' corto ma
#    non e' un vincolo semantico. Decisione presa con l'utente: si resta sulla prossimita'.

# --- Regole: nome -> pattern ---------------------------------------------------
# NB una regola `generic_strong` (monitoraggio/controllo/identificazione + target forte) era
# stata provata e RIMOSSA: l'audit su dati reali mostrava che produceva quasi solo falsi
# positivi (controllo di gestione delle commesse, standard SA8000, gioielleria). I verbi
# generici restano fuori dall'oracolo.
RULES = {
    'strong':        STRONG,
    'action_target': near(ACTION, TARGET),
    # etichettatura/marcatura contano solo in contesto di filiera, finestra stretta
    'labeling':      near(ACTION_LABELING, TARGET_TRACE, WINDOW_TIGHT),
    'tech_ctx':      near(TECH, CTX),
    'chain_mod':     near(CHAIN, CHAIN_MOD),
}

RULE_NAMES = tuple(RULES)


def _normalize(series: pd.Series) -> pd.Series:
    """Contratto d'ingresso: qualunque Series -> testo minuscolo, senza NaN, dtype object.

    IL dtype object NON E' UN DETTAGLIO: e' un fix.

    pandas >= 3.0 usa di default stringhe Arrow-backed, che per le regex passano da **RE2**.
    In RE2 `\\b` e `\\w` sono **ASCII-only**: dopo una lettera accentata la 'à' non conta come
    word char, quindi `\\btracciabilit[aà]\\b` NON matcha "tracciabilità:" — mentre il modulo `re`
    di Python (Unicode-aware) lo matcha correttamente.

        pd.Series(["tracciabilità:"]).str.contains(r'\\btracciabilit[aà]\\b')  -> False  (Arrow/RE2)
        re.search(r'\\btracciabilit[aà]\\b', "tracciabilità:")                 -> True   (re)

    Effetto misurato: la regex v1 (scritta quando pandas usava `re`) ha smesso silenziosamente di
    riconoscere ~2.000 record che contengono esplicitamente la parola "tracciabilità". Non era un
    errore di scrittura: e' una **regressione introdotta dall'upgrade di pandas**.

    Forzare object dtype riporta il motore a `re`, con semantica Unicode piena (e supporto ai
    lookahead, usati per escludere "marcatura CE").
    """
    return series.fillna('').astype(str).str.lower().astype(object)


def rule_masks(series: pd.Series) -> dict:
    """Maschera booleana per ciascuna regola. Chiave = nome regola."""
    s = _normalize(series)
    return {name: s.str.contains(pat, regex=True) for name, pat in RULES.items()}


def get_mask(series: pd.Series) -> pd.Series:
    """Maschera positiva = OR di tutte le regole.

    Firma identica alla v1 (Series di testo -> Series booleana), cosi' gli import
    esistenti (prepare_traceability_data.py, validation_worker.py) non si rompono.
    """
    masks = rule_masks(series)
    out = None
    for m in masks.values():
        out = m if out is None else (out | m)
    return out


def rule_labels(masks: dict, index=None) -> pd.Series:
    """Da un dict {nome: maschera} a una Series di stringhe ';'-separated."""
    df = pd.DataFrame(masks, index=index)
    return df.apply(lambda row: ';'.join(c for c in df.columns if row[c]), axis=1)

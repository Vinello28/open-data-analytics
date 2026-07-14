"""Fa etichettare il gold set da un annotatore INDIPENDENTE dalla regex (LLM locale),
poi prepara il minimo indispensabile di lavoro umano.

Perche' un LLM e non il BERT
----------------------------
Il BERT era addestrato sulle label di `get_mask`: chiedergli di validare la regex e' come
chiedere a uno studente di correggere il compito del proprio insegnante copiando da lui.
L'LLM zero-shot non ha mai visto `get_mask`: il suo giudizio e' un'evidenza *indipendente*,
quindi rompe la circolarita'. Non e' ground truth — e' un secondo parere.

Il protocollo (e perche' non basta "prendere per buoni gli accordi")
--------------------------------------------------------------------
Tentazione: dove regex e LLM concordano assumo che abbiano ragione, e faccio annotare
all'umano solo i disaccordi. E' comodo ma **gonfia le metriche**: i due possono sbagliare
INSIEME (es. accendersi entrambi sulla parola "tracciabilita'" in un contesto che non c'entra),
e quell'errore resterebbe invisibile.

Quindi il file da annotare contiene:
  - TUTTI i disaccordi regex-vs-LLM          -> e' li' che si decide chi ha ragione
  - + N_SPOT accordi pescati a caso          -> misura quanto sbagliano quando vanno d'accordo
  - il tutto MESCOLATO e SENZA le due label  -> annotazione in cieco, niente anchoring

Uso:
    # 1. avvia LM Studio (server locale, porta 1234) con gpt-oss-20b
    python judge_gold_set.py
    # 2. compila GOLD_LABEL in validation/gold_to_adjudicate.csv
    python merge_adjudication.py
    python score_gold_set.py ../../data/traceability/validation/regex_gold_set_scored.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_judge_worker as judge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.join(HERE, '..', '..', 'data', 'traceability', 'validation')
GOLD = os.path.join(VAL_DIR, 'regex_gold_set.csv')
JUDGED = os.path.join(VAL_DIR, 'regex_gold_set_judged.csv')
TO_ADJ = os.path.join(VAL_DIR, 'gold_to_adjudicate.csv')
# La chiave sta in un file SEPARATO: se `stratum`/`_motivo` restassero nel file da annotare,
# l'annotazione non sarebbe cieca (stratum='A_positivi_v2' rivela la risposta della regex,
# e '_motivo' rivela quali record sono gli spot-check). Il merge la ricongiunge via COR.
KEY_FILE = os.path.join(VAL_DIR, '.gold_adjudication_key.csv')

# Solo cio' che serve a decidere. Niente colonne amministrative da scrollare.
ANNOT_COLS = ['COR', 'TITOLO_PROGETTO', 'DESCRIZIONE_PROGETTO', 'GOLD_LABEL']

POS, NEG = 'tracciabilita', 'altro'
N_SPOT = 30        # accordi da far verificare in cieco
SEED = 42
CONCURRENCY = 4


def main():
    if judge.ping() is None:
        sys.exit(
            "LM Studio non raggiungibile su http://localhost:1234/v1\n"
            "Avvia LM Studio -> Developer -> Start Server, con il modello gpt-oss-20b caricato."
        )

    df = pd.read_csv(GOLD)
    print(f"gold set: {len(df)} record | strati: {df['stratum'].value_counts().to_dict()}")

    records = [
        {'titolo': str(r.get('TITOLO_PROGETTO', '') or ''),
         'descrizione': str(r.get('DESCRIZIONE_PROGETTO', '') or '')}
        for _, r in df.iterrows()
    ]

    print(f"giudizio LLM ({judge.DEFAULT_MODEL}) su {len(records)} record...", flush=True)
    df['LLM_LABEL'] = judge.judge_batch(records, concurrency=CONCURRENCY)

    errori = df['LLM_LABEL'].astype(str).str.startswith('error')
    if errori.any():
        print(f"ATTENZIONE: {int(errori.sum())} record senza risposta valida dall'LLM.")
        print(df.loc[errori, 'LLM_LABEL'].value_counts().head().to_string())

    valid = df[~errori].copy()
    accordo = valid['LLM_LABEL'] == valid['REGEX_V2_LABEL']

    print("\n" + "=" * 70)
    print("ACCORDO regex v2.1  vs  giudice LLM (indipendente)")
    print("=" * 70)
    print(f"  concordi   : {int(accordo.sum()):>3} / {len(valid)}  ({accordo.mean():.1%})")
    print(f"  discordi   : {int((~accordo).sum()):>3}")
    print("\n  incroci [righe=regex v2.1, colonne=LLM]:")
    print(pd.crosstab(valid['REGEX_V2_LABEL'], valid['LLM_LABEL']).to_string())
    print("\n  disaccordi per strato:")
    print(valid[~accordo]['stratum'].value_counts().to_string())

    df.to_csv(JUDGED, index=False)

    # --- pacchetto per l'umano: disaccordi + spot-check in cieco ---------------------
    disc = valid[~accordo].copy()
    disc['_motivo'] = 'disaccordo'

    conc = valid[accordo]
    spot = conc.sample(min(N_SPOT, len(conc)), random_state=SEED).copy()
    spot['_motivo'] = 'spot_check_accordo'

    todo = pd.concat([disc, spot], ignore_index=True)
    todo = todo.sample(frac=1, random_state=SEED).reset_index(drop=True)
    todo['GOLD_LABEL'] = ''

    # La chiave (stratum, _motivo) va a parte: nel file da annotare rivelerebbe le risposte.
    todo[['COR', 'stratum', '_motivo']].to_csv(KEY_FILE, index=False)
    todo[ANNOT_COLS].to_csv(TO_ADJ, index=False)

    print("\n" + "=" * 70)
    print(f"DA ANNOTARE A MANO: {len(todo)} record  (invece di {len(df)})")
    print("=" * 70)
    print(f"  {len(disc)} disaccordi  +  {len(spot)} accordi da verificare in cieco (non distinguibili)")
    print(f"  file: {TO_ADJ}")
    print(f"  colonne: {', '.join(ANNOT_COLS)}")
    print("\n  compila GOLD_LABEL con 'tracciabilita' oppure 'altro'.")
    print("  Criterio: conta COSA viene tracciato. Prodotti/lotti/materie lungo la filiera")
    print("  -> 'tracciabilita'. Processi, documenti, dati aziendali, persone -> 'altro'.")
    print("\n  poi:  python merge_adjudication.py")


if __name__ == '__main__':
    main()

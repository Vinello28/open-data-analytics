"""Ricompone il GOLD_LABEL dei 300 record dopo l'adjudicazione umana dei disaccordi.

Regola di ricomposizione:
  - record adjudicati dall'umano          -> vale la label umana (sempre, e' l'arbitro)
  - accordi regex+LLM non adjudicati      -> vale la label concordata

Il secondo caso e' un'ASSUNZIONE, non un'osservazione. Per questo `judge_gold_set.py` mette
nel pacchetto anche N accordi da verificare in cieco: qui li usiamo per stimare quanto spesso
l'assunzione e' sbagliata (`errore stimato sugli accordi`). Se quel numero non e' ~0, le
metriche finali vanno lette come ottimistiche e il gold set va allargato.

Uso:  python merge_adjudication.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.join(HERE, '..', '..', 'data', 'traceability', 'validation')
JUDGED = os.path.join(VAL_DIR, 'regex_gold_set_judged.csv')
TO_ADJ = os.path.join(VAL_DIR, 'gold_to_adjudicate.csv')
KEY_FILE = os.path.join(VAL_DIR, '.gold_adjudication_key.csv')
OUT = os.path.join(VAL_DIR, 'regex_gold_set_scored.csv')

POS, NEG = 'tracciabilita', 'altro'
KEY = 'COR'          # identificativo del record: il join non passa piu' dal testo


def main():
    for p in (JUDGED, TO_ADJ, KEY_FILE):
        if not os.path.exists(p):
            sys.exit(f"manca {p} — lancia prima judge_gold_set.py")

    judged = pd.read_csv(JUDGED)
    adj = pd.read_csv(TO_ADJ)
    # `stratum` e `_motivo` stanno qui e non nel file annotato: l'annotazione era in cieco.
    keys = pd.read_csv(KEY_FILE)

    adj['GOLD_LABEL'] = adj['GOLD_LABEL'].astype(str).str.strip().str.lower()
    done = adj[adj['GOLD_LABEL'].isin([POS, NEG])]
    if len(done) < len(adj):
        print(f"ATTENZIONE: {len(adj) - len(done)}/{len(adj)} record ancora senza GOLD_LABEL.")
        if done.empty:
            sys.exit("nessun record annotato: compila GOLD_LABEL in " + TO_ADJ)

    human = dict(zip(done[KEY].astype(str), done['GOLD_LABEL']))
    motivo = dict(zip(keys[KEY].astype(str), keys['_motivo']))

    valid = judged[~judged['LLM_LABEL'].astype(str).str.startswith('error')].copy()
    key = valid[KEY].astype(str)

    valid['GOLD_LABEL'] = [
        human.get(k, llm if llm == rgx else None)
        for k, llm, rgx in zip(key, valid['LLM_LABEL'], valid['REGEX_V2_LABEL'])
    ]
    valid['GOLD_SOURCE'] = [
        'umano' if k in human else 'accordo_regex_llm' for k in key
    ]

    # --- quanto sbagliano regex e LLM QUANDO VANNO D'ACCORDO? ------------------------
    spot = valid[[motivo.get(k) == 'spot_check_accordo' for k in key]]
    if len(spot):
        sbagliati = (spot['GOLD_LABEL'] != spot['LLM_LABEL']).sum()
        tasso = sbagliati / len(spot)
        print("=" * 72)
        print("SANITY CHECK — errore degli ACCORDI (regex e LLM concordi, ma sbagliati)")
        print("=" * 72)
        print(f"  spot-check in cieco: {len(spot)} record, {sbagliati} sbagliati -> {tasso:.1%}")
        if tasso > 0.10:
            print("  ⚠ oltre il 10%: le metriche qui sotto sono OTTIMISTICHE.")
            print("    L'assunzione 'se concordano hanno ragione' non regge: allarga il gold set.")
        else:
            print("  ok: l'assunzione sugli accordi regge, le metriche sono attendibili.")

    out = valid.dropna(subset=['GOLD_LABEL'])
    out.to_csv(OUT, index=False)
    print(f"\nScritto {OUT}  ({len(out)} record con GOLD_LABEL)")
    print("  provenienza:", out['GOLD_SOURCE'].value_counts().to_dict())
    print("  bilanciamento:", out['GOLD_LABEL'].value_counts().to_dict())
    print(f"\nOra: python score_gold_set.py {OUT}")


if __name__ == '__main__':
    main()

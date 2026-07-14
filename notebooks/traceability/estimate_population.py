"""Riporta le stime del gold set STRATIFICATO alla popolazione reale.

Perche' serve
-------------
Il gold set sovracampiona apposta gli strati rari (A=120, B=80, C=100), quindi le metriche
aggregate calcolate su di esso NON sono metriche di popolazione: dire "recall 93%" leggendo
la confusion matrix del gold set e' un errore. Le quote di veri positivi PER STRATO sono
invece stime non distorte, perche' ogni strato e' campionato a caso al proprio interno.
Qui le ripesiamo per la dimensione reale di ciascun bacino.

Bacini (in spazio di DESCRIZIONI UNICHE: e' cosi' che sono stati campionati gli strati)
  S1 = positivi v2.1                        -> quota veri = precision
  S2 = near-miss, non presi dalla v2        -> qui vivono i falsi negativi
  S3 = positivi v1 scartati dalla v2, FUORI dai near-miss (per non contarli due volte)
Tutto cio' che sta fuori da S1|S2|S3 e' assunto negativo: non contiene alcun termine di
dominio. E' l'assunzione dichiarata del gold set (il recall e' "sul pool raggiungibile").
"""
import glob
import multiprocessing as mp
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_regex_versions import mask_v1  # noqa: E402
from traceability_patterns import get_mask  # noqa: E402
from build_regex_gold_set import NEAR_MISS, _text  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, '..', '..', 'data', 'technology_mapping')
N_PROC, CHUNK = 10, 100_000


def scan(fp):
    """Ritorna le descrizioni uniche di ciascun bacino, per questo file."""
    s1, s2, s3 = set(), set(), set()
    for chunk in pd.read_csv(fp, chunksize=CHUNK, low_memory=False):
        text = _text(chunk)
        v2 = get_mask(text)
        t = chunk.get('TITOLO_PROGETTO', pd.Series('', index=chunk.index))
        d = chunk.get('DESCRIZIONE_PROGETTO', pd.Series('', index=chunk.index))
        v1 = mask_v1(t) | mask_v1(d)
        near = text.str.contains(NEAR_MISS, regex=True)

        descr = d.fillna('').astype(str)
        s1 |= set(descr[v2])
        s2 |= set(descr[~v2 & near])
        s3 |= set(descr[~v2 & ~near & v1])     # v1-only e FUORI dai near-miss: nessun doppio conteggio
    return s1, s2, s3


def main():
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, 'reclassified_multiclass_aiuti_*.csv')))
    with mp.get_context('fork').Pool(N_PROC) as pool:
        parts = pool.map(scan, files)

    S1 = set().union(*[p[0] for p in parts]) - {''}
    S2 = set().union(*[p[1] for p in parts]) - {''}
    S3 = set().union(*[p[2] for p in parts]) - {''}
    S2 -= S1
    S3 -= S1 | S2

    # quote di veri positivi misurate sul gold set (campione casuale dentro ogni strato)
    R1, R2, R3 = 0.508, 0.020, 0.025
    N1, N2, N3 = len(S1), len(S2), len(S3)

    TP = R1 * N1
    FP = (1 - R1) * N1
    FN = R2 * N2 + R3 * N3
    POS = TP + FN                       # tutti i veri positivi del pool raggiungibile

    print("=" * 78)
    print("BACINI (descrizioni uniche)")
    print("=" * 78)
    print(f"  S1  positivi v2.1                    : {N1:>7,}   veri {R1:.1%}")
    print(f"  S2  near-miss non presi dalla v2     : {N2:>7,}   veri {R2:.1%}")
    print(f"  S3  positivi v1 scartati, fuori S2   : {N3:>7,}   veri {R3:.1%}")
    print(f"\n  veri positivi totali stimati (S1+S2+S3): {POS:,.0f}")

    def report(nome, tp, fp, fn, tn_pool):
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        acc = (tp + tn_pool) / (tp + fp + fn + tn_pool)
        print(f"\n--- {nome} ---")
        print(f"  precision {prec:6.1%} | recall {rec:6.1%} | F1 {f1:6.3f} | accuracy {acc:6.1%}")
        print(f"  TP {tp:7,.0f} | FP {fp:7,.0f} | FN {fn:7,.0f} | TN {tn_pool:7,.0f}")
        return f1

    TOT = N1 + N2 + N3                  # pool raggiungibile
    TN_v2 = TOT - N1 - FN               # negativi corretti: fuori da S1 e davvero negativi
    report("REGEX v2.1  (tutti i positivi)", TP, FP, FN, TN_v2)

    # v1: predice positivo su (v1 dentro S1) + S3 + (v1 dentro S2). Approssimazione: la v1
    # copre il 60,4% dei positivi v2 (misurato da compare_regex_versions) con la stessa qualita'.
    v1_in_S1 = 0.604 * N1
    tp1 = R1 * v1_in_S1
    fp1 = (1 - R1) * v1_in_S1 + (1 - R3) * N3
    fn1 = POS - tp1
    report("REGEX v1  (storica, approssimata)", tp1, fp1, fn1, TOT - v1_in_S1 - N3 - fn1)

    # cascata: l'LLM tiene il 43% dei positivi regex, con precision 96,2% (misurato su strato A)
    keep = 0.43 * N1
    tp_c = 0.962 * keep
    fp_c = (1 - 0.962) * keep
    fn_c = POS - tp_c
    report("REGEX v2.1 + GIUDICE LLM (cascata)", tp_c, fp_c, fn_c, TOT - keep - fn_c)

    print("\n" + "=" * 78)
    print("Nota: il recall e' relativo al POOL RAGGIUNGIBILE (S1|S2|S3), non all'intero corpus.")
    print("Fuori da questi bacini nessun record contiene un termine di dominio: e' l'assunzione")
    print("dichiarata del gold set, non una misura.")


if __name__ == '__main__':
    main()

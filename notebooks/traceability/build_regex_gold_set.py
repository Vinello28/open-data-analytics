"""Costruisce un gold set STRATIFICATO per validare la regex v2 contro annotazione umana.

Perche' serve
-------------
Il BERT sembrava buono (F1 0.928) solo perche' era valutato contro le label prodotte dalla regex
che lo aveva addestrato: valutazione circolare, priva di valore. L'unico modo per sapere davvero
quanto vale un classificatore e' confrontarlo con etichette umane. Questo script produce il
campione da annotare.

Tre strati, ciascuno risponde a una domanda precisa:

  A  positivi v2                      -> PRECISION di v2: quanti dei suoi positivi sono veri?
  B  positivi v1 ma NON v2            -> gli scartati dalla v2 erano davvero falsi positivi?
  C  NON positivi v2, ma "near-miss"  -> RECALL: quanti veri positivi la v2 si perde?
     (contengono un termine di dominio ma nessuna regola scatta)

Lo strato C non e' un campione casuale della popolazione: pescare a caso su ~930k record con base
rate ~0.2% non troverebbe quasi nulla. Si campiona dove i falsi negativi possono realisticamente
annidarsi. Il recall stimato va quindi letto come "recall sul pool dei near-miss", non come recall
di popolazione: e' un limite noto e va dichiarato nel report.

Output: data/traceability/validation/regex_gold_set.csv con colonna GOLD_LABEL VUOTA da compilare
        a mano con 'tracciabilita' oppure 'altro'.

Uso:
    python build_regex_gold_set.py
"""
import glob
import multiprocessing as mp
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_regex_versions import mask_v1  # noqa: E402  (v1 congelata: unica baseline storica)
from traceability_patterns import get_mask  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, '..', '..', 'data', 'technology_mapping')
V2_DIR = os.path.join(HERE, '..', '..', 'data', 'traceability')
OUT_DIR = os.path.join(V2_DIR, 'validation')
OUT_CSV = os.path.join(OUT_DIR, 'regex_gold_set.csv')

N_PROC = 10          # 12 core fisici - 2 liberi
CHUNK_SIZE = 100_000
SEED = 42

# Quanti record per strato (dimensionati per essere annotabili a mano in tempi umani)
N_A, N_B, N_C = 120, 80, 100

COLS = ['COR', 'ANNO', 'TITOLO_MISURA', 'TITOLO_PROGETTO', 'DESCRIZIONE_PROGETTO']

# Un record e' "near-miss" se parla di temi limitrofi alla tracciabilita' ma la v2 non scatta.
NEAR_MISS = (
    r'\b(?:tracciab\w*|rintracci\w*|filier[ae]|supply chain|blockchain|rfid|nfc'
    r'|qr[- ]?code|iot|internet of things|provenienza|origine|lott[oi]'
    r'|etichettatur\w*|anti[- ]?contraffazione|certificazion\w*)\b'
)


def _text(df):
    t = df.get('TITOLO_PROGETTO', pd.Series('', index=df.index)).fillna('').astype(str)
    d = df.get('DESCRIZIONE_PROGETTO', pd.Series('', index=df.index)).fillna('').astype(str)
    m = df.get('TITOLO_MISURA', pd.Series('', index=df.index)).fillna('').astype(str)
    return (m + ' . ' + t + ' . ' + d).str.lower()


def scan_file(file_path):
    """Una sola passata sul file annuale -> (near-miss non catturati, positivi v1 scartati da v2).

    Gli strati B e C si ricavano ENTRAMBI dal corpus, ricalcolando le maschere. Nessuna
    dipendenza da CSV su disco: una versione precedente di questo script leggeva la v1 da
    `_v1_backup/`, e quando quella cartella e' sparita ha prodotto in SILENZIO uno strato B
    vuoto (220 record invece di 300). Il corpus e' l'unica fonte che non puo' diventare stale.
    """
    near_out, dropped_out = [], []
    try:
        it = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)
    except Exception as e:
        print(f"[{os.path.basename(file_path)}] errore: {e}")
        return pd.DataFrame(columns=COLS), pd.DataFrame(columns=COLS)

    for chunk in it:
        cols = [c for c in COLS if c in chunk.columns]
        text = _text(chunk)
        v2 = get_mask(text)

        # C: parla di temi limitrofi ma nessuna regola v2 scatta
        near = text.str.contains(NEAR_MISS, regex=True) & ~v2
        if near.any():
            sub = chunk.loc[near, cols]
            # sotto-campiono subito: il pool near-miss e' grande, non serve tenerlo tutto
            if len(sub) > 400:
                sub = sub.sample(400, random_state=SEED)
            near_out.append(sub)

        # B: la v1 li dava positivi, la v2 no -> erano davvero falsi positivi?
        t = chunk.get('TITOLO_PROGETTO', pd.Series('', index=chunk.index))
        d = chunk.get('DESCRIZIONE_PROGETTO', pd.Series('', index=chunk.index))
        dropped = (mask_v1(t) | mask_v1(d)) & ~v2
        if dropped.any():
            dropped_out.append(chunk.loc[dropped, cols])

    n = pd.concat(near_out, ignore_index=True) if near_out else pd.DataFrame(columns=COLS)
    dr = pd.concat(dropped_out, ignore_index=True) if dropped_out else pd.DataFrame(columns=COLS)
    print(f"[{os.path.basename(file_path)}] near-miss={len(n)} scartati_v1={len(dr)}", flush=True)
    return n, dr


def _load(pattern):
    frames = []
    for f in sorted(glob.glob(pattern)):
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as e:
            print(f"errore su {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _dedup(df):
    return df.dropna(subset=['DESCRIZIONE_PROGETTO']).drop_duplicates(subset=['DESCRIZIONE_PROGETTO'])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    v2 = _load(os.path.join(V2_DIR, 'traceability_aiuti_*.csv'))
    print(f"positivi v2 (da disco): {len(v2)}")

    # --- Strati B e C: una sola passata sul corpus ------------------------------------
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, 'reclassified_multiclass_aiuti_*.csv')))
    ctx = mp.get_context('fork')   # Python 3.14 usa forkserver di default
    with ctx.Pool(processes=N_PROC) as pool:
        parts = pool.map(scan_file, files)

    near = _dedup(pd.concat([p for p, _ in parts if len(p)], ignore_index=True))
    dropped = _dedup(pd.concat([d for _, d in parts if len(d)], ignore_index=True))
    print(f"pool near-miss (dedup): {len(near)} | scartati da v2 (dedup): {len(dropped)}")

    # --- Strato A: positivi v2 (precision) --------------------------------------------
    a = v2.drop_duplicates(subset=['DESCRIZIONE_PROGETTO'])
    a = a.sample(min(N_A, len(a)), random_state=SEED).copy()
    a['stratum'] = 'A_positivi_v2'

    # --- Strato B: positivi v1 scartati da v2 -----------------------------------------
    b = dropped.sample(min(N_B, len(dropped)), random_state=SEED).copy()
    b['stratum'] = 'B_scartati_da_v2'

    # --- Strato C: near-miss non catturati (recall) -----------------------------------
    c = near.sample(min(N_C, len(near)), random_state=SEED).copy()
    c['stratum'] = 'C_near_miss_non_catturati'

    # Uno strato vuoto rende il gold set inservibile per la domanda che quello strato pone:
    # meglio fallire rumorosamente che consegnare all'utente un campione monco.
    for nome, strato, atteso in (('A', a, N_A), ('B', b, N_B), ('C', c, N_C)):
        if len(strato) < atteso:
            raise SystemExit(
                f"strato {nome}: {len(strato)} record invece di {atteso}. "
                "Il gold set sarebbe sbilanciato — controlla il corpus prima di annotare."
            )

    keep = COLS + ['stratum', 'MATCH_SOURCE', 'MATCH_RULE']
    gold = pd.concat([a, b, c], ignore_index=True)
    gold = gold[[k for k in keep if k in gold.columns]]

    # etichetta della regex, per il confronto successivo
    gold['REGEX_V2_LABEL'] = ['tracciabilita' if s == 'A_positivi_v2' else 'altro'
                              for s in gold['stratum']]
    gold['REGEX_V1_LABEL'] = ['tracciabilita' if s in ('A_positivi_v2', 'B_scartati_da_v2')
                              else 'altro' for s in gold['stratum']]
    gold['GOLD_LABEL'] = ''   # <-- da compilare a mano: 'tracciabilita' | 'altro'

    gold = gold.sample(frac=1, random_state=SEED).reset_index(drop=True)  # mescola: evita bias
    gold.to_csv(OUT_CSV, index=False)

    print(f"\nScritto {OUT_CSV}  ({len(gold)} record da annotare)")
    print(gold['stratum'].value_counts().to_string())
    print("\nCompila la colonna GOLD_LABEL con 'tracciabilita' oppure 'altro',")
    print("poi lancia: python score_gold_set.py ../../data/traceability/validation/regex_gold_set.csv")


if __name__ == '__main__':
    main()

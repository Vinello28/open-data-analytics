"""Confronto ONESTO v1 vs v2 della regex di tracciabilita', sullo STESSO corpus.

Perche' esiste questo script
---------------------------
I CSV in data/traceability/ erano stati generati tempo fa, con un corpus e una regex non
piu' allineati al codice attuale: confrontarli con l'output della v2 sarebbe apples-to-oranges.
L'unico confronto valido e' ricalcolare **entrambe** le maschere sugli stessi record, nella
stessa passata. La v1 e' congelata qui sotto come baseline storica.

Uso:
    python compare_regex_versions.py
"""
import glob
import multiprocessing as mp
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traceability_patterns import get_mask as mask_v2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(HERE, '..', '..', 'data', 'technology_mapping')
OUT_DIR = os.path.join(HERE, '..', '..', 'data', 'traceability', 'validation')

N_PROC = 10
CHUNK_SIZE = 100_000


# --- v1 CONGELATA (copia storica, non importarla altrove) ---------------------------
def mask_v1(series):
    # object dtype -> motore `re` di Python (Unicode-aware), com'era all'epoca in cui la v1 fu
    # scritta. Senza questo, pandas 3.0 userebbe Arrow/RE2 con \b ASCII-only e la v1 perderebbe
    # ~2.000 record contenenti "tracciabilità": staremmo confrontando un bug di engine, non i due
    # design. Vedi traceability_patterns._normalize.
    s = series.fillna('').astype(str).str.lower().astype(object)
    p_core = (r'\b(?:(?:rin)?tracciabilit[aà]|provenienza del prodotto|origine certificata'
              r'|passaporto digitale)\b')
    m_core = s.str.contains(p_core, regex=True)

    p_action = r'\b(?:tracciamento|tracciatura|identificazione|monitoraggio|autenticazione)\b'
    p_target = (r'\b(?:prodotto|prodotti|materia|materie|filiera|supply chain|merce|merci'
                r'|lotto|lotti|end-to-end)\b')
    m_at = s.str.contains(p_action, regex=True) & s.str.contains(p_target, regex=True)

    p_tech = (r'\b(?:blockchain|registri distribuiti|distributed ledger|qr code|codice qr'
              r'|rfid|nfc|internet of things|iot)\b')
    p_tech_t = r'\b(?:filiera|supply chain|provenienza|logistica|anticontraffazione)\b'
    m_tt = s.str.contains(p_tech, regex=True) & s.str.contains(p_tech_t, regex=True)

    p_chain = r'\b(?:filiera|caten[ae] del valore|supply chain)\b'
    p_chain_mod = r'\b(?:trasparente|trasparenza|sicurezza alimentare|certificata|garantita)\b'
    m_ch = s.str.contains(p_chain, regex=True) & s.str.contains(p_chain_mod, regex=True)

    m5 = (s.str.contains(r'\bindustria 4\.0\b', regex=True)
          & s.str.contains(r'\b(?:rin)?tracciabilit[aà]\b', regex=True))
    m6 = (s.str.contains(r'\blogistica\b', regex=True)
          & s.str.contains(r'\bintelligente\b', regex=True)
          & s.str.contains(r'\b(?:merci|prodotti|tracciamento)\b', regex=True))
    return m_core | m_at | m_tt | m_ch | m5 | m6


def compare_file(file_path):
    """Ritorna (anno, n_record, n_v1, n_v2, n_solo_v1, n_solo_v2, n_entrambi) + i campioni."""
    year = os.path.basename(file_path).split('_')[-1].replace('.csv', '')
    stats = dict(anno=year, record=0, v1=0, v2=0, solo_v1=0, solo_v2=0, entrambi=0)
    samples_only_v1, samples_only_v2 = [], []

    try:
        it = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)
    except Exception as e:
        print(f"[{year}] errore: {e}")
        return stats, pd.DataFrame(), pd.DataFrame()

    for chunk in it:
        t = chunk.get('TITOLO_PROGETTO', pd.Series('', index=chunk.index))
        d = chunk.get('DESCRIZIONE_PROGETTO', pd.Series('', index=chunk.index))
        m = chunk.get('TITOLO_MISURA', pd.Series('', index=chunk.index))

        # v1: solo titolo progetto + descrizione (come da codice storico)
        v1 = mask_v1(t) | mask_v1(d)
        # v2: + TITOLO_MISURA
        v2 = mask_v2(t) | mask_v2(d) | mask_v2(m)

        stats['record'] += len(chunk)
        stats['v1'] += int(v1.sum())
        stats['v2'] += int(v2.sum())
        stats['solo_v1'] += int((v1 & ~v2).sum())
        stats['solo_v2'] += int((~v1 & v2).sum())
        stats['entrambi'] += int((v1 & v2).sum())

        cols = [c for c in ['ANNO', 'TITOLO_PROGETTO', 'DESCRIZIONE_PROGETTO'] if c in chunk.columns]
        if (v1 & ~v2).any() and len(samples_only_v1) < 3:
            samples_only_v1.append(chunk.loc[v1 & ~v2, cols].head(10))
        if (~v1 & v2).any() and len(samples_only_v2) < 3:
            samples_only_v2.append(chunk.loc[~v1 & v2, cols].head(10))

    print(f"[{year}] record={stats['record']:>8,} v1={stats['v1']:>5} v2={stats['v2']:>5} "
          f"soloV1={stats['solo_v1']:>5} soloV2={stats['solo_v2']:>4}", flush=True)
    s1 = pd.concat(samples_only_v1, ignore_index=True) if samples_only_v1 else pd.DataFrame()
    s2 = pd.concat(samples_only_v2, ignore_index=True) if samples_only_v2 else pd.DataFrame()
    return stats, s1, s2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, 'reclassified_multiclass_aiuti_*.csv')))
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=N_PROC) as pool:
        results = pool.map(compare_file, files)

    stats = pd.DataFrame([r[0] for r in results]).sort_values('anno')
    tot = stats[['record', 'v1', 'v2', 'solo_v1', 'solo_v2', 'entrambi']].sum()

    print("\n" + "=" * 78)
    print("DELTA v1 -> v2  (stesso corpus, stessa passata)")
    print("=" * 78)
    print(stats.to_string(index=False))
    print("-" * 78)
    print(f"TOTALE record analizzati : {tot['record']:,}")
    print(f"positivi v1              : {tot['v1']:,}  ({tot['v1']/tot['record']:.3%})")
    print(f"positivi v2              : {tot['v2']:,}  ({tot['v2']/tot['record']:.3%})")
    print(f"  in comune              : {tot['entrambi']:,}")
    print(f"  persi dalla v2 (solo v1): {tot['solo_v1']:,}   <- attesi: falsi positivi della v1")
    print(f"  nuovi in v2 (solo v2)  : {tot['solo_v2']:,}   <- attesi: veri positivi persi dalla v1")
    if tot['v1']:
        print(f"recall v2 vs v1          : {tot['entrambi']/tot['v1']:.1%} "
              f"(quanta parte della v1 e' confermata)")

    stats.to_csv(os.path.join(OUT_DIR, 'regex_v1_v2_delta.csv'), index=False)

    s1 = pd.concat([r[1] for r in results if len(r[1])], ignore_index=True)
    s2 = pd.concat([r[2] for r in results if len(r[2])], ignore_index=True)
    s1.to_csv(os.path.join(OUT_DIR, 'sample_only_v1.csv'), index=False)
    s2.to_csv(os.path.join(OUT_DIR, 'sample_only_v2.csv'), index=False)

    print("\n--- CAMPIONE: scartati dalla v2 (erano positivi v1) ---")
    for i, r in s1.drop_duplicates('DESCRIZIONE_PROGETTO').head(6).iterrows():
        print(f"  - {str(r['DESCRIZIONE_PROGETTO'])[:110]}")
    print("\n--- CAMPIONE: nuovi trovati dalla v2 (la v1 li perdeva) ---")
    for i, r in s2.drop_duplicates('DESCRIZIONE_PROGETTO').head(6).iterrows():
        print(f"  - {str(r['DESCRIZIONE_PROGETTO'])[:110]}")
    print(f"\nSalvato: {OUT_DIR}/regex_v1_v2_delta.csv")


if __name__ == '__main__':
    main()

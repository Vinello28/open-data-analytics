"""Audit di COPERTURA: la regex v2 cerca davvero i termini della specifica?

Per ogni termine della specifica conta, sul corpus completo:
  - hits_titolo  : record in cui il termine compare nel TITOLO_PROGETTO
  - hits_descr   : record in cui compare nella DESCRIZIONE_PROGETTO
  - gia_v2       : di quelli, quanti sono GIA' positivi per la v2 (quindi il termine non aggiunge nulla)
  - orfani       : hits NON catturati dalla v2 -> e' il recall che la specifica chiede e noi perdiamo

La specifica distingue due blocchi:
  BLOCCO 1 (ovunque): termini specifici della tracciabilita'
  BLOCCO 2 (SOLO NEL TITOLO del progetto): termini larghi (filiera, supply chain, logistica...)
    -> il titolo e' corto e deliberato: e' esso stesso la finestra di prossimita'.
       Cercare "filiera" in una descrizione da 1.000 char e' rumore; nel titolo e' un segnale.

Uso: python audit_spec_terms.py
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

# --- BLOCCO 1: termini della specifica, cercati OVUNQUE -------------------------------
SPEC_ANY = {
    'tracciabilita':        r'\btracciabilit[aà]\w*\b',
    'rintracciabilita':     r'\brintracciabilit[aà]\w*\b',
    'tracciamento':         r'\btracciament\w*\b',
    'autenticazione':       r'\bautenticazion\w*\b',
    'blockchain':           r'\bblockchain\b',
    'registri_distribuiti': r'\b(?:registri distribuiti|distributed ledger(?: technology)?|dlt)\b',
    'monitoraggio_e2e':     r'\bmonitoraggio\b.{0,60}?\bend[- ]?to[- ]?end\b|\bend[- ]?to[- ]?end\b.{0,60}?\bmonitoraggio\b',
    'provenienza_prodotto': r'\bprovenienza (?:del|dei|della|delle) (?:prodott\w+|materi\w+|aliment\w+)\b',
    'origine_certificata':  r'\borigine certificat\w+\b',
    'qr_barcode':           r'\b(?:qr[- ]?code|codice qr|codic[ei] a barre|barcode)\b',
    'localizz_prodotto':    r'\blocalizzazion\w*\b.{0,60}?\b(?:prodott\w+|materi\w+)\b|\b(?:prodott\w+|materi\w+)\b.{0,60}?\blocalizzazion\w*\b',
    'identif_prodotto':     r'\bidentificazion\w*\b.{0,60}?\b(?:prodott\w+|materi\w+)\b|\b(?:prodott\w+|materi\w+)\b.{0,60}?\bidentificazion\w*\b',
}

# --- BLOCCO 2: termini larghi, la specifica li vuole SOLO NEL TITOLO ------------------
SPEC_TITLE = {
    'filiera':              r'\bfilier[ae]\b',
    'catena_valore':        r'\bcaten[ae] del valore\b',
    'approvvigionamento':   r'\b(?:caten[ae]|sistem[ai]) di approvvigionamento\b',
    'supply_chain':         r'\bsupply chain\b',
    'industria40_tracc':    r'\bindustria 4\.0\b.{0,80}?\btracciabilit[aà]|\btracciabilit[aà]\w*\b.{0,80}?\bindustria 4\.0\b',
    'internet_of_things':   r'\b(?:internet of things|iot)\b',
    'sicurezza_prodotto':   r'\bsicurezza (?:del |dei )?prodott\w*\b',
    'logistica_intelligente': r'\blogistica intelligente\b',
    'smart_logistics':      r'\bsmart logistics\b',
}


def _norm(s):
    # object dtype -> motore `re` Unicode-aware (vedi traceability_patterns._normalize)
    return s.fillna('').astype(str).str.lower().astype(object)


def audit_file(file_path):
    year = os.path.basename(file_path).split('_')[-1].replace('.csv', '')
    rows = {k: dict(hits=0, orfani=0) for k in list(SPEC_ANY) + [f'TIT:{k}' for k in SPEC_TITLE]}
    n = 0
    orphan_samples = []

    try:
        it = pd.read_csv(file_path, chunksize=CHUNK_SIZE, low_memory=False)
    except Exception as e:
        print(f"[{year}] errore: {e}", flush=True)
        return year, rows, 0, pd.DataFrame()

    for chunk in it:
        t = _norm(chunk.get('TITOLO_PROGETTO', pd.Series('', index=chunk.index)))
        d = _norm(chunk.get('DESCRIZIONE_PROGETTO', pd.Series('', index=chunk.index)))
        m = chunk.get('TITOLO_MISURA', pd.Series('', index=chunk.index))
        n += len(chunk)

        v2 = mask_v2(t) | mask_v2(d) | mask_v2(m)

        for name, pat in SPEC_ANY.items():
            hit = t.str.contains(pat, regex=True) | d.str.contains(pat, regex=True)
            rows[name]['hits'] += int(hit.sum())
            rows[name]['orfani'] += int((hit & ~v2).sum())

        for name, pat in SPEC_TITLE.items():
            hit = t.str.contains(pat, regex=True)      # SOLO titolo, come da specifica
            key = f'TIT:{name}'
            rows[key]['hits'] += int(hit.sum())
            orph = hit & ~v2
            rows[key]['orfani'] += int(orph.sum())
            if orph.any() and len(orphan_samples) < 40:
                s = chunk.loc[orph, ['TITOLO_PROGETTO']].head(3).copy()
                s['termine'] = name
                orphan_samples.append(s)

    print(f"[{year}] {n:,} record", flush=True)
    smp = pd.concat(orphan_samples, ignore_index=True) if orphan_samples else pd.DataFrame()
    return year, rows, n, smp


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, 'reclassified_multiclass_aiuti_*.csv')))
    ctx = mp.get_context('fork')
    with ctx.Pool(processes=N_PROC) as pool:
        results = pool.map(audit_file, files)

    total = {}
    n_tot = 0
    for _, rows, n, _ in results:
        n_tot += n
        for k, v in rows.items():
            acc = total.setdefault(k, dict(hits=0, orfani=0))
            acc['hits'] += v['hits']
            acc['orfani'] += v['orfani']

    df = pd.DataFrame([
        dict(termine=k, hits=v['hits'], gia_coperti=v['hits'] - v['orfani'], orfani=v['orfani'])
        for k, v in total.items()
    ])
    df['copertura'] = (df['gia_coperti'] / df['hits'].replace(0, pd.NA) * 100).round(1)
    df = df.sort_values('orfani', ascending=False)

    print("\n" + "=" * 82)
    print(f"COPERTURA DELLA SPECIFICA da parte della regex v2 — {n_tot:,} record")
    print("  'orfani' = record che contengono il termine ma che la v2 NON classifica tracciabilita'")
    print("  TIT: = termine cercato SOLO nel titolo del progetto (come da specifica)")
    print("=" * 82)
    print(df.to_string(index=False))

    df.to_csv(os.path.join(OUT_DIR, 'spec_coverage_audit.csv'), index=False)

    smp = pd.concat([r[3] for r in results if len(r[3])], ignore_index=True)
    if len(smp):
        smp.to_csv(os.path.join(OUT_DIR, 'spec_orphans_title.csv'), index=False)
        print("\n--- CAMPIONE di titoli ORFANI (specifica li vuole, v2 li perde) ---")
        for _, r in smp.drop_duplicates('TITOLO_PROGETTO').head(15).iterrows():
            print(f"  [{r['termine']:<22}] {str(r['TITOLO_PROGETTO'])[:90]}")
    print(f"\nSalvato: {OUT_DIR}/spec_coverage_audit.csv")


if __name__ == '__main__':
    main()

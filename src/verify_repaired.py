"""Dimostra che `technology_mapping_repaired` ha gli STESSI dati di `technology_mapping`,
e che l'unica differenza e' il codec del testo.

Perche'
-------
`technology_mapping_repaired` e' stato prodotto riparando il mojibake di `technology_mapping`.
Prima di usarlo come sorgente di un'analisi bisogna provare due cose OPPOSTE, e servono
entrambe:

  1. le colonne di DECISIONE non sono cambiate  -> zero differenze attese
  2. le colonne di TESTO   sono cambiate        -> se fossero zero, il repair non e' avvenuto
                                                   e `repaired` e' solo una copia

Un solo controllo non basta: "0 differenze ovunque" verrebbe letto come successo, mentre
significherebbe che il file non e' stato riparato affatto.

Come
----
Entrambi i file sono QUOTE_NONE (misurato: 0 virgolette) e hanno 27 campi per riga, senza
newline dentro i campi: una riga = un record e `split(',')` e' esatto. Quindi si confronta
in **lockstep riga contro riga**, senza pandas: qualche KB di RAM per worker invece di
centinaia di MB, e nessun rischio che il parser reinterpreti il quoting.

L'allineamento e' **posizionale**: `COR` NON e' una chiave (duplicato 2.275 volte nel 2024),
serve solo come assertion che le due sequenze non siano sfasate.

    ./.venv/bin/python src/verify_repaired.py            # tutti gli anni
    ./.venv/bin/python src/verify_repaired.py --anni 2015 2016
"""
import argparse
import multiprocessing as mp
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PUB_DIR = os.path.join(REPO_ROOT, 'data', 'technology_mapping')
REP_DIR = os.path.join(REPO_ROOT, 'data', 'technology_mapping_repaired')

N_PROC = 10        # regola utente: 12 core - 2. Qui e' I/O bound, la RAM non e' il vincolo.
ANNI = list(range(2014, 2026))

# Le colonne che NON devono cambiare: le decisioni dei modelli + il technology mapping.
DECISIONE = ['CLASSIFICAZIONE', 'CLASSIFICAZIONE_CONFIDENZA',
             'CLASSIFICAZIONE_MULTICLASS', 'CLASSIFICAZIONE_MULTICLASS_CONFIDENZA',
             'TIPO_AI', 'TECNOLOGIE_AI']

# Le colonne che DEVONO cambiare (almeno un po'): e' li' che vive il mojibake.
TESTO = ['TITOLO_MISURA', 'TITOLO_PROGETTO', 'DESCRIZIONE_PROGETTO',
         'DENOMINAZIONE_BENEFICIARIO', 'SETTORI_ATTIVITA', 'OBIETTIVO']

MAX_ESEMPI = 3


def verifica_anno(anno):
    """Confronta i due file di un anno riga per riga. Ritorna un dict di esito."""
    nome = f'reclassified_multiclass_aiuti_{anno}.csv'
    p_pub, p_rep = os.path.join(PUB_DIR, nome), os.path.join(REP_DIR, nome)

    out = {'anno': anno, 'errori': [], 'diff_decisione': {}, 'diff_testo': {},
           'esempi': [], 'righe': 0}

    for p in (p_pub, p_rep):
        if not os.path.exists(p):
            out['errori'].append(f'manca {p}')
            return out

    # utf-8 con errors='replace': il pubblicato contiene mojibake, non deve far esplodere il read
    with open(p_pub, encoding='utf-8', errors='replace', newline='') as fa, \
         open(p_rep, encoding='utf-8', errors='replace', newline='') as fb:

        ha = fa.readline().rstrip('\r\n').split(',')
        hb = fb.readline().rstrip('\r\n').split(',')
        if ha != hb:
            out['errori'].append(f'header diversi: {set(ha) ^ set(hb)}')
            return out

        idx = {c: ha.index(c) for c in DECISIONE + TESTO if c in ha}
        mancanti = [c for c in DECISIONE + TESTO if c not in idx]
        if mancanti:
            out['errori'].append(f'colonne assenti: {mancanti}')
            return out
        i_cor = ha.index('COR') if 'COR' in ha else None

        diff_dec = {c: 0 for c in DECISIONE}
        diff_txt = {c: 0 for c in TESTO}
        n = 0
        cor_sfasati = 0

        for n, (la, lb) in enumerate(zip(fa, fb), 1):
            if la == lb:
                continue                        # identiche: nessuna colonna e' cambiata
            a = la.rstrip('\r\n').split(',')
            b = lb.rstrip('\r\n').split(',')

            if len(a) != len(ha) or len(b) != len(hb):
                if len(out['esempi']) < MAX_ESEMPI:
                    out['esempi'].append(f'riga {n}: campi {len(a)} vs {len(b)} (atteso {len(ha)})')
                out['errori'].append(f'riga {n}: numero di campi inatteso')
                continue

            if i_cor is not None and a[i_cor] != b[i_cor]:
                cor_sfasati += 1

            for c in DECISIONE:
                if a[idx[c]] != b[idx[c]]:
                    diff_dec[c] += 1
                    if len(out['esempi']) < MAX_ESEMPI:
                        out['esempi'].append(
                            f'riga {n} {c}: pub={a[idx[c]]!r} rep={b[idx[c]]!r}')
            for c in TESTO:
                if a[idx[c]] != b[idx[c]]:
                    diff_txt[c] += 1

        # le due sequenze devono finire insieme: se una ha piu' righe, zip() l'avrebbe troncata
        resto_a, resto_b = fa.readline(), fb.readline()
        if resto_a or resto_b:
            out['errori'].append(
                f'righe diverse: il {"pubblicato" if resto_a else "repaired"} ne ha di piu\' dopo la {n}')

    out['righe'] = n
    out['diff_decisione'] = diff_dec
    out['diff_testo'] = diff_txt
    out['cor_sfasati'] = cor_sfasati
    if cor_sfasati:
        out['errori'].append(f'{cor_sfasati} COR sfasati: allineamento posizionale rotto')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anni', nargs='+', type=int, default=ANNI)
    args = ap.parse_args()

    with mp.get_context('fork').Pool(N_PROC) as pool:
        esiti = pool.map(verifica_anno, args.anni)
    esiti.sort(key=lambda e: e['anno'])

    print(f'\n{"anno":<6}{"righe":>12}{"diff decisione":>16}{"celle testo diverse":>22}')
    print('-' * 56)
    tot_righe = tot_dec = tot_txt = 0
    for e in esiti:
        d = sum(e['diff_decisione'].values())
        t = sum(e['diff_testo'].values())
        tot_righe += e['righe']; tot_dec += d; tot_txt += t
        print(f'{e["anno"]:<6}{e["righe"]:>12,}{d:>16,}{t:>22,}')
    print('-' * 56)
    print(f'{"TOT":<6}{tot_righe:>12,}{tot_dec:>16,}{tot_txt:>22,}')

    print('\nCelle di testo riparate per colonna:')
    for c in TESTO:
        print(f'  {c:<28}{sum(e["diff_testo"].get(c, 0) for e in esiti):>12,}')

    if tot_dec:
        print('\nDifferenze nelle colonne di DECISIONE per colonna:')
        for c in DECISIONE:
            n = sum(e['diff_decisione'].get(c, 0) for e in esiti)
            if n:
                print(f'  {c:<40}{n:>12,}')

    errori = [(e['anno'], msg) for e in esiti for msg in e['errori']]
    esempi = [(e['anno'], x) for e in esiti for x in e['esempi']]
    if esempi:
        print('\nEsempi:')
        for anno, x in esempi[:15]:
            print(f'  [{anno}] {x}')

    # I due cancelli, in direzioni opposte.
    ok = True
    if errori:
        print('\nERRORI STRUTTURALI:')
        for anno, msg in errori[:20]:
            print(f'  [{anno}] {msg}')
        ok = False
    if tot_dec:
        print(f'\nFALLITO: {tot_dec:,} differenze nelle colonne di decisione (attese: 0).')
        ok = False
    if tot_txt == 0:
        print('\nFALLITO: 0 celle di testo diverse -> `repaired` e\' una copia, '
              'nessun mojibake e\' stato riparato.')
        ok = False

    if ok:
        print(f'\nOK: {tot_righe:,} righe, decisioni identiche, {tot_txt:,} celle di testo riparate.')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

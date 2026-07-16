"""Estrae le imprese della regione Marche (2023-2025) da `data/technology_mapping_repaired`.

Cosa produce
------------
    data/analytics/marche/marche_2023.csv
    data/analytics/marche/marche_2024.csv
    data/analytics/marche/marche_2025.csv
    data/analytics/marche/marche_2023_2025.csv     (i tre anni uniti)

Un record per aiuto concesso, tutte le 27 colonne, nessuna deduplica e nessun filtro su
DES_TIPO_BENEFICIARIO (la colonna resta per filtrare a valle).

Il separatore multi-regione NON e' `|`, e' un DOPPIO SPAZIO
----------------------------------------------------------
Misurato sulla sorgente: 0 pipe nel campo REGIONE_BENEFICIARIO. E' il danno di
reclassify_annual.py (QUOTE_NONE, ogni `,` -> spazio), che ha trasformato il separatore
originale in `  `. Quindi le righe multi-regione sono cosi':

    'Lazio  Marche'
    'Emilia-Romagna  Lazio  Liguria  Lombardia  Marche  Piemonte'

e vanno prese anche loro. Si splitta su `\\s{2,}` e si cerca il token ESATTO `Marche`:
il match su sottostringa funzionerebbe (nessun'altra regione contiene "Marche"), ma
dipenderebbe da un fatto sui dati invece che sulla logica.

Perche' non pandas
------------------
La sorgente e' QUOTE_NONE (0 virgolette) con 27 campi esatti per riga e nessuna newline
dentro i campi: una riga = un record e `split(b',')` e' esatto. Quindi si lavora a
**byte-range**: ogni task e' una fetta da ~64 MB di un file, i confini a meta' riga si
riconciliano da soli (chi trova l'inizio della riga la processa, chi la trova a meta' la
scarta). ~140 task su Pool(10). Nessun overhead di parsing su 9 GB.

Le righe che matchano sono restituite **verbatim**, byte per byte come stanno nella sorgente:
non si ri-serializza con csv.writer, cosi' e' impossibile introdurre una seconda mutilazione
sopra quella che c'e' gia'.

NOTA sulla sorgente: `technology_mapping_repaired` ha il mojibake riparato ma le virgole NO
(0 virgolette). Per la selezione regionale e' indifferente, ma le DESCRIZIONE_PROGETTO in
uscita restano prive di virgole: per NLP sul testo la sorgente giusta e' technology_mapping_v2.

    ./.venv/bin/python notebooks/analytics/extract_marche.py
    ./.venv/bin/python notebooks/analytics/extract_marche.py --anni 2025
"""
import argparse
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(REPO_ROOT, 'data', 'technology_mapping_repaired')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'analytics', 'marche')

ANNI = [2023, 2024, 2025]
REGIONE = 'Marche'
COL_REGIONE = 'REGIONE_BENEFICIARIO'

N_PROC = 10                  # regola utente: 12 core - 2
CHUNK_BYTES = 64 * 1024 * 1024

TARGET = REGIONE.encode()
SEP_MULTI = re.compile(rb'\s{2,}')   # il separatore multi-regione: doppio spazio, non '|'


def _percorso(anno):
    return os.path.join(SRC_DIR, f'reclassified_multiclass_aiuti_{anno}.csv')


def _intestazione(anno):
    """Ritorna (bytes dell'header, lista colonne, offset del primo record)."""
    with open(_percorso(anno), 'rb') as f:
        riga = f.readline()
        return riga, riga.decode('utf-8').rstrip('\r\n').split(','), f.tell()


def _fette(anno):
    """Spezza il file in byte-range. I confini cadono a meta' riga: se ne occupa il worker."""
    path = _percorso(anno)
    totale = os.path.getsize(path)
    _, colonne, inizio = _intestazione(anno)
    i_reg = colonne.index(COL_REGIONE)
    n_col = len(colonne)
    out = []
    while inizio < totale:
        fine = min(inizio + CHUNK_BYTES, totale)
        out.append((anno, path, inizio, fine, i_reg, n_col))
        inizio = fine
    return out


def estrai(task):
    """Scansiona una fetta. Ritorna (anno, offset, righe verbatim, contatori)."""
    anno, path, inizio, fine, i_reg, n_col = task
    _, _, primo_record = _intestazione(anno)

    trovate = []
    valori = Counter()
    malformate = 0

    with open(path, 'rb') as f:
        f.seek(inizio)
        # Confine a meta' riga: quella riga appartiene alla fetta precedente, che la
        # completera' leggendo oltre il proprio `fine`. Qui si scarta.
        if inizio != primo_record:
            f.readline()

        while f.tell() < fine:
            riga = f.readline()
            if not riga:
                break
            # Prefiltro: scarta ~99,7% delle righe senza nemmeno splittare.
            # Conservativo: nessuna regione contiene "Marche" senza esserlo.
            if TARGET not in riga:
                continue
            campi = riga.rstrip(b'\r\n').split(b',')
            if len(campi) != n_col:
                malformate += 1
                continue
            valore = campi[i_reg]
            # Match sul token esatto, non sulla sottostringa.
            if TARGET not in SEP_MULTI.split(valore):
                continue
            trovate.append(riga)
            valori[valore.decode('utf-8', 'replace')] += 1

    return anno, inizio, trovate, valori, malformate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anni', nargs='+', type=int, default=ANNI)
    args = ap.parse_args()

    for anno in args.anni:
        if not os.path.exists(_percorso(anno)):
            sys.exit(f'manca {_percorso(anno)}')

    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = [t for anno in args.anni for t in _fette(anno)]
    gb = sum(f - i for _, _, i, f, _, _ in tasks) / 1e9
    print(f'{len(tasks)} fette da ~{CHUNK_BYTES // 2**20} MB ({gb:.1f} GB) su Pool({N_PROC})')

    t0 = time.time()
    with mp.get_context('fork').Pool(N_PROC) as pool:
        esiti = pool.map(estrai, tasks)
    print(f'scansione in {time.time() - t0:.1f}s')

    # Ordine deterministico: per (anno, offset) -> l'output rispecchia l'ordine della sorgente
    # e due run producono file identici.
    esiti.sort(key=lambda e: (e[0], e[1]))

    uniti = os.path.join(OUT_DIR, f'marche_{min(args.anni)}_{max(args.anni)}.csv')
    totale = 0
    riepilogo = []

    with open(uniti, 'wb') as fu:
        fu.write(_intestazione(args.anni[0])[0])
        for anno in args.anni:
            header = _intestazione(anno)[0]
            valori = Counter()
            malformate = 0
            n = 0
            percorso = os.path.join(OUT_DIR, f'marche_{anno}.csv')
            with open(percorso, 'wb') as fa:
                fa.write(header)
                for a, _, righe, v, m in esiti:
                    if a != anno:
                        continue
                    for riga in righe:
                        if not riga.endswith(b'\n'):
                            riga += b'\n'
                        fa.write(riga)
                        fu.write(riga)
                        n += 1
                    valori += v
                    malformate += m
            totale += n
            riepilogo.append((anno, n, valori, malformate))
            multi = sum(c for val, c in valori.items() if SEP_MULTI.split(val.encode()) != [TARGET])
            print(f'  {anno}: {n:>8,} righe  ({multi} multi-regione, '
                  f'{len(valori)} valori distinti, {malformate} malformate)')

    print(f'\n{totale:,} righe totali -> {uniti}')

    print('\nValori di REGIONE_BENEFICIARIO estratti:')
    tutti = Counter()
    for _, _, v, _ in riepilogo:
        tutti += v
    for val, c in tutti.most_common():
        print(f'  {c:>8,}  {val}')

    # Cancello: ogni valore estratto DEVE contenere Marche come token esatto.
    intrusi = [v for v in tutti if TARGET not in SEP_MULTI.split(v.encode())]
    if intrusi:
        sys.exit(f'\nFALLITO: valori senza il token Marche in uscita: {intrusi[:5]}')
    print('\nOK: ogni riga estratta ha Marche come token esatto di REGIONE_BENEFICIARIO.')


if __name__ == '__main__':
    main()

"""Annotatore da terminale per il gold set della tracciabilita'.

Perche' esiste
--------------
Il gold set attuale l'ho etichettato io (Claude), che ho anche scritto la regex che si
sta valutando. E' un conflitto d'interessi: chi valuta non puo' essere chi ha costruito
il predittore. Serve un'etichetta UMANA e INDIPENDENTE. Questo script te la fa produrre
in ~1 ora invece che in un pomeriggio.

Garanzie di correttezza (sono il motivo per cui questo file esiste)
------------------------------------------------------------------
  * CIECO   - a schermo compaiono solo TITOLO_MISURA (in grigio, come contesto),
              TITOLO_PROGETTO e DESCRIZIONE_PROGETTO. Mai lo stratum, mai la label della
              regex, mai quella dell'LLM, mai la mia. Non puoi essere influenzato da cio'
              che devi giudicare.
              Attenzione alla misura: e' il BANDO, comune a migliaia di progetti diversi.
              Un bando "Industria 4.0" non rende il progetto tracciabilita'. Serve a
              decifrare le descrizioni criptiche, non a decidere.
  * ORDINE  - i 300 record sono mescolati con seed fisso: gli strati si alternano, quindi
              non puoi accorgerti "ora sono nei positivi della regex" e cambiare soglia.
  * RIPRESA - ogni tasto premuto viene scritto su disco. Chiudi quando vuoi, riprendi da li'.
  * HIGHLIGHT - i termini di dominio sono evidenziati per farti leggere in fretta. NON e'
              una fuga di informazione: TUTTI e tre gli strati sono stati campionati fra
              record che contengono almeno un termine di dominio, quindi l'evidenziazione
              non distingue i positivi dai negativi. Disattivabile con --no-highlight.

Uso
---
  ./.venv/bin/python notebooks/traceability/annotate_gold.py           # annota
  ./.venv/bin/python notebooks/traceability/annotate_gold.py --report  # solo metriche
  ./.venv/bin/python notebooks/traceability/annotate_gold.py --reset   # ricomincia

Tasti: [s] tracciabilita  [n] altro  [spazio] salta  [u] annulla  [d] definizione  [q] esci
"""
import argparse
import os
import re
import sys
import termios
import tty

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_DIR = os.path.join(HERE, '..', '..', 'data', 'traceability', 'validation')
GOLD = os.path.join(VAL_DIR, 'regex_gold_set_scored.csv')
OUT = os.path.join(VAL_DIR, 'human_gold.csv')
SEED = 20260714

POS, NEG = 'tracciabilita', 'altro'

# Vocabolario di dominio SOLO per l'evidenziazione a schermo (leggibilita', non giudizio).
# Include di proposito i falsi amici, cosi' evidenziare non equivale a suggerire "si'".
HL = re.compile(
    r'(tracciabilit\w*|rintracciabilit\w*|tracciam\w*|tracking|traceabilit\w*'
    r'|blockchain|distributed ledger|registri distribuiti|\bDLT\b|smart contract'
    r'|filier\w*|supply chain|caten\w+ del valore|caten\w+ di (?:fornitura|approvvigionamento)'
    r'|\bQR\b|codice a barre|barcode|\bRFID\b|\bNFC\b|etichett\w*|marcatura'
    r'|provenienz\w*|origine|autenticazion\w*|anticontraffazion\w*|anti-contraffazion\w*'
    r'|internet of things|\bIoT\b|sensor\w*|logistic\w*|magazzin\w*|lott[oi]\b'
    r'|monitorag\w*|end.to.end|identificazion\w*|localizzazion\w*|industria 4\.0)',
    re.IGNORECASE)

DEFINIZIONE = """
  COSA CONTA COME "TRACCIABILITA'"  (il costrutto che stai etichettando)
  ---------------------------------------------------------------------
  Rispondi SI' solo se il progetto ha come oggetto, anche parziale, il
  SEGUIRE UN BENE FISICO (o le sue informazioni) LUNGO LA CATENA:

    * tracciare/rintracciare lotti, materie prime, semilavorati, prodotti, merci
    * certificare provenienza o origine di un prodotto
    * identificare/etichettare un bene per poterlo seguire (QR, RFID, barcode, NFC)
    * blockchain / registri distribuiti APPLICATI a filiera o prodotto
    * IoT / sensori APPLICATI al monitoraggio di prodotto, lotto o logistica
    * anticontraffazione, autenticazione del prodotto
    * digitalizzazione della filiera / supply chain con finalita' di tracciamento

  Rispondi NO (=altro) se il termine c'e' ma NON riguarda un bene lungo la catena:

    * "tracciabilita' delle operazioni/dei documenti/contabile" -> audit trail, non prodotto
    * gestionale / ERP / CRM che elenca "tracciabilita'" fra 40 funzioni di contorno
    * "sicurezza del prodotto" in senso chimico (schede di sicurezza, REACH)
    * "filiera" come sinonimo di settore o rete di imprese, senza tracciamento
    * "catena del valore" in senso strategico-economico
    * "origine" riferita a persone, aziende, fondi, non a merci
    * IoT/blockchain generici, senza aggancio a prodotto o filiera

  Nel dubbio, chiediti: SE QUESTO PROGETTO RIESCE, QUALCUNO SAPRA' DOV'E'
  PASSATO UN OGGETTO? Se la risposta non e' un si' chiaro -> altro.
"""

C = {'dim': '\033[2m', 'b': '\033[1m', 'hl': '\033[43;30m', 'g': '\033[32m',
     'r': '\033[31m', 'y': '\033[33m', 'c': '\033[36m', 'x': '\033[0m'}


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def wrap(text, width=94, indent='  '):
    out, line = [], indent
    for w in str(text).split():
        if len(line) + len(w) + 1 > width and line.strip():
            out.append(line)
            line = indent
        line += w + ' '
    out.append(line)
    return '\n'.join(out)


def highlight(text, on):
    return HL.sub(lambda m: f"{C['hl']}{m.group(0)}{C['x']}", text) if on else text


def load():
    """Il gold set, mescolato. Le colonne che rivelerebbero la risposta restano fuori."""
    df = pd.read_csv(GOLD, dtype={'COR': str})
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


def load_done():
    if os.path.exists(OUT):
        d = pd.read_csv(OUT, dtype={'COR': str})
        return dict(zip(d['COR'], d['HUMAN_LABEL']))
    return {}


def save(done):
    pd.DataFrame({'COR': list(done), 'HUMAN_LABEL': list(done.values())}).to_csv(OUT, index=False)


def kappa(a, b):
    """Cohen's kappa fra due annotatori su etichette binarie."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((sum(x == k for x in a) / n) * (sum(y == k for y in b) / n) for k in (POS, NEG))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


def report(df, done):
    """Le metriche vere: la regex giudicata sulle TUE etichette, non sulle mie."""
    d = df[df['COR'].isin(done)].copy()
    if d.empty:
        print('Nessun record annotato.')
        return
    d['HUMAN'] = d['COR'].map(done)
    d = d[d['HUMAN'].isin([POS, NEG])]

    print(f"\n{C['b']}{'=' * 78}\nMETRICHE SULLE TUE ETICHETTE  ({len(d)} record annotati)\n{'=' * 78}{C['x']}")

    for strato, g in d.groupby('stratum'):
        r = (g['HUMAN'] == POS).mean()
        print(f"  {strato:<28} n={len(g):>3}   veri positivi {r:6.1%}")

    print(f"\n{C['b']}Confronto fra predittori (sul campione annotato, NON riponderato){C['x']}")
    print(f"  {'predittore':<14} {'precision':>10} {'recall':>8} {'F1':>7}")
    for col, nome in (('REGEX_V1_LABEL', 'regex v1'), ('REGEX_V2_LABEL', 'regex v2.1'),
                      ('LLM_LABEL', 'giudice LLM'), ('GOLD_LABEL', 'Claude')):
        if col not in d:
            continue
        pred, truth = d[col] == POS, d['HUMAN'] == POS
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        print(f"  {nome:<14} {p:>9.1%} {r:>8.1%} {f:>7.3f}")

    if 'GOLD_LABEL' in d:
        k, po = kappa(list(d['HUMAN']), list(d['GOLD_LABEL']))
        verdetto = ('quasi perfetto' if k > .8 else 'sostanziale' if k > .6
                    else 'moderato' if k > .4 else 'DEBOLE: il costrutto non e\' condiviso')
        print(f"\n{C['b']}Accordo tu vs Claude{C['x']}  kappa={k:.3f} ({verdetto}), accordo grezzo {po:.1%}")
        div = d[d['HUMAN'] != d['GOLD_LABEL']]
        if len(div):
            print(f"  {len(div)} divergenze -> COR: {', '.join(div['COR'].head(12))}"
                  f"{' ...' if len(div) > 12 else ''}")

    if len(d) == len(df):
        print(f"\n{C['g']}Gold set completo. Ora rilancia:{C['x']}")
        print("  ./.venv/bin/python notebooks/traceability/estimate_population.py")
        print(f"  {C['dim']}aggiornando R1/R2/R3 con le quote per strato qui sopra.{C['x']}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', action='store_true', help='solo metriche, non annotare')
    ap.add_argument('--reset', action='store_true', help='cancella le annotazioni e ricomincia')
    ap.add_argument('--no-highlight', action='store_true')
    args = ap.parse_args()

    if args.reset and os.path.exists(OUT):
        os.remove(OUT)

    df, done = load(), {}
    if not args.reset:
        done = load_done()

    if args.report:
        report(df, done)
        return

    hl = not args.no_highlight
    todo = [i for i in range(len(df)) if df.at[i, 'COR'] not in done]
    if not todo:
        print(f"{C['g']}Tutti i {len(df)} record sono gia' annotati.{C['x']}")
        report(df, done)
        return

    print('\033[2J\033[H' + DEFINIZIONE)
    print(f"  {len(todo)} record da annotare (su {len(df)}). Invio per iniziare, [q] per uscire.")
    if getch() == 'q':
        return

    i = 0
    while i < len(todo):
        row = df.iloc[todo[i]]
        n_done = len(done)
        pos = sum(v == POS for v in done.values())

        print('\033[2J\033[H')
        print(f"{C['dim']}  [{n_done + 1}/{len(df)}]  gia' etichettati si': {pos}  "
              f"| COR {row['COR']}  anno {row['ANNO']}{C['x']}")
        print(f"{C['dim']}  {'-' * 94}{C['x']}")
        print(f"\n{C['dim']}  MISURA (contesto: e' il bando, non il progetto -- da solo non decide){C['x']}")
        print(f"{C['dim']}{wrap(row['TITOLO_MISURA'])}{C['x']}")
        print(f"\n{C['b']}{C['c']}  TITOLO{C['x']}")
        print(highlight(wrap(row['TITOLO_PROGETTO']), hl))
        print(f"\n{C['b']}{C['c']}  DESCRIZIONE{C['x']}")
        print(highlight(wrap(row['DESCRIZIONE_PROGETTO']), hl))
        print(f"\n{C['dim']}  {'-' * 94}{C['x']}")
        print(f"  {C['g']}[s]{C['x']} tracciabilita   {C['r']}[n]{C['x']} altro   "
              f"[spazio] salta   [u] annulla   [d] definizione   [q] esci\n")

        k = getch().lower()
        if k == 'q':
            break
        if k == 'd':
            print('\033[2J\033[H' + DEFINIZIONE + '\n  Un tasto per tornare.')
            getch()
            continue
        if k == 'u':
            if i > 0:
                i -= 1
                done.pop(df.at[todo[i], 'COR'], None)
                save(done)
            continue
        if k == ' ':
            todo.append(todo[i])   # rimandato in fondo
            i += 1
            continue
        if k in ('s', 'n'):
            done[row['COR']] = POS if k == 's' else NEG
            save(done)
            i += 1

    save(done)
    print(f"\033[2J\033[H{C['g']}Salvati {len(done)}/{len(df)} record in{C['x']} {OUT}")
    report(df, done)


if __name__ == '__main__':
    main()

"""Ripara il mojibake del corpus aiuti di stato usando il corpus stesso come dizionario.

Il problema
-----------
Il testo della fonte ha subito una catena di incidenti di encoding PRIMA di arrivare in
questa repo: data/raw e data/technology_mapping contengono gli stessi identici danni,
quindi non li ha introdotti la pipeline. Il carattere '¿' e' un CARATTERE DI SOSTITUZIONE:
non rimpiazza un carattere solo, ma tutto cio' che un export non sapeva rappresentare.
Censimento su 278.561 occorrenze nel corpus:

    l¿investimento           -> l'investimento     apostrofo                        57%
    pi¿ / liquidit¿ / citt¿  -> piu' / liquidita'  vocale accentata
    difficoltÃ¿Â             -> difficolta'        mojibake a DUE strati
    Ã¿Â¢Ã¿Â¿Ã¿Â¿insieme       -> l'insieme          mojibake a TRE strati
    Lâ¿¿ASSEGNAZIONE         -> L'ASSEGNAZIONE     apostrofo tipografico (U+2019)
    Pezzapiana ¿ Zona        -> Pezzapiana - Zona  trattino (U+2013)
    pari a ¿ 300.000         -> EUR 300.000        simbolo euro
    difficolta'ï¿½           -> ï¿½ e' U+FFFD: gia' perso alla fonte

Un s/¿/'/g ne aggiusta il 57% e scrive "pi'", "citt'", "' 300.000". Non e' accettabile.

La soluzione: dedurre il CARATTERE, non indovinare la parola
------------------------------------------------------------
Il corpus e' il proprio dizionario. La stessa parola compare quasi sempre anche in forma
pulita in altri record, perche' la corruzione colpisce una minoranza di righe.

  passata 1  costruisce il VOCABOLARIO delle parole pulite del corpus, con le frequenze.
  passata 2  per ogni parola corrotta prende la sequenza di spazzatura e prova a
             sostituirla con ciascun carattere candidato (' a' e' e' i' o' u' - EUR ...).
             Vince il candidato che produce una parola CHE ESISTE nel vocabolario, la piu'
             frequente. Se nessun candidato produce una parola nota, si ricade su regole
             di contesto (lettera¿lettera -> apostrofo, spazio¿cifra -> EUR, ...).

Sostituendo il singolo carattere e non la parola intera, il MAIUSCOLO resta intatto:
"dell¿EXPO" -> "dell'EXPO", non "dell'expo". Ed e' una correzione PROVATA DAI DATI:
"difficoltÃ¿Â " diventa "difficolta'" perche' quella parola esiste nel corpus decine di
migliaia di volte, non perche' l'ho decisa io.

Uso
---
    ./.venv/bin/python src/text_repair.py --build-vocab        # passata 1 (Pool 10)
    ./.venv/bin/python src/text_repair.py --preview            # cosa cambierebbe
    ./.venv/bin/python src/text_repair.py --apply <dir_out>    # passata 2 (Pool 10)
"""
import argparse
import glob
import itertools
import multiprocessing as mp
import os
import pickle
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
CORPUS_DIR = os.path.join(ROOT, 'data', 'technology_mapping')
VOCAB = os.path.join(ROOT, 'data', 'vocab_clean.pkl')
N_PROC = 10
MIN_FREQ = 3          # sotto questa soglia una "parola pulita" e' probabilmente un refuso

# Caratteri che compaiono SOLO dentro la spazzatura (nessuno e' italiano legittimo).
# Una "sequenza corrotta" e' un blocco contiguo di questi che contenga almeno un ¿.
JUNK = '¿ÃÂâï½¢\xa0'
CORRUPT = re.compile(f'[{JUNK}]*¿[{JUNK}]*')
HAS_JUNK = re.compile('¿')

# Cosa puo' essersi mangiato il ¿. L'ordine e' il prior: a parita' di frequenza vince
# il primo, ed e' per questo che l'apostrofo ASCII sta in cima (e' quello che vuole l'NLP
# a valle, e nel corpus convive con quello tipografico).
CANDIDATI = ["'", 'à', 'è', 'é', 'ì', 'ò', 'ù', 'À', 'È', 'É', 'Ì', 'Ò', 'Ù',
             '’', '°', '²', 'ª', '-', '', '€']

# Un token: parola alfabetica, con eventuali apostrofi e spazzatura dentro.
TOKEN = re.compile(f"[A-Za-zÀ-ÿ{JUNK}][A-Za-zÀ-ÿ'’{JUNK}]*")


def _clean(word: str) -> bool:
    return not HAS_JUNK.search(word)


# ------------------------------------------------------------- passata 1: il vocabolario
def _scan(fp):
    c = Counter()
    with open(fp, encoding='utf-8', errors='replace') as f:
        for line in f:
            for w in TOKEN.findall(line):
                if len(w) >= 2 and _clean(w):
                    c[w.lower()] += 1
    return c


def build_vocab(files):
    with mp.get_context('fork').Pool(N_PROC) as pool:
        parts = pool.map(_scan, files)
    vocab = Counter()
    for p in parts:
        vocab.update(p)
    return Counter({w: n for w, n in vocab.items() if n >= MIN_FREQ})


# ------------------------------------------------------------- passata 2: la riparazione
def _by_context(text: str, i: int, j: int) -> str:
    """Nessuna prova nel vocabolario: decidi dal contesto. Regole misurate sul corpus."""
    prev = text[i - 1] if i else ''
    rest = text[j:].lstrip()
    if prev.isalpha() and rest[:1].isalpha():
        return "'"                                  # l¿azienda      -> l'azienda
    if rest[:1].isdigit():                          # pari a ¿ 300.000
        return 'EUR' if text[j:j + 1] == ' ' else 'EUR '
    if prev.isalpha():
        return "'"                                  # liquidit¿      -> liquidita'
    if prev in ' \t\n,;(' or not prev:
        return '-'                                  # Pezzapiana ¿ Zona
    return ''


def _fix_word(word: str, vocab: Counter) -> str | None:
    """Prova ogni candidato al posto di ogni sequenza corrotta: vince la parola che esiste.

    Sostituisce SOLO i caratteri corrotti, quindi il maiuscolo della parola resta intatto.
    Ritorna None se il vocabolario non sa decidere (allora decide il contesto).
    """
    runs = [m.span() for m in CORRUPT.finditer(word)]
    if not runs or len(runs) > 3:            # oltre 3 buchi non e' piu' una parola
        return None
    best, best_n = None, 0
    for combo in itertools.product(CANDIDATI, repeat=len(runs)):
        out, last = [], 0
        for (i, j), c in zip(runs, combo):
            out.append(word[last:i])
            out.append(c)
            last = j
        out.append(word[last:])
        cand = ''.join(out)
        n = vocab.get(cand.lower(), 0)
        if n > best_n:
            best, best_n = cand, n
    # nel corpus l'apostrofo convive nelle due forme; se ne inseriamo uno noi, che sia
    # sempre lo stesso: a valle tokenizzatori e regex non devono gestirne due.
    return best.replace('’', "'") if best else None


def repair(text: str, vocab: Counter) -> str:
    if not isinstance(text, str) or not HAS_JUNK.search(text):
        return text

    # 1) parole intere: il vocabolario decide quale carattere e' stato mangiato
    out, last = [], 0
    for m in TOKEN.finditer(text):
        w = m.group(0)
        if not HAS_JUNK.search(w):
            continue
        fixed = _fix_word(w, vocab)
        if fixed is not None:
            out.append(text[last:m.start()])
            out.append(fixed)
            last = m.end()
    out.append(text[last:])
    text = ''.join(out)

    # 2) quel che resta (euro, trattini, parole mai viste pulite): regole di contesto
    while HAS_JUNK.search(text):
        m = CORRUPT.search(text)
        text = text[:m.start()] + _by_context(text, m.start(), m.end()) + text[m.end():]
    return text


# ----------------------------------------------------------------------------------- CLI
def _load():
    if not os.path.exists(VOCAB):
        sys.exit('vocabolario assente: lancia prima --build-vocab')
    with open(VOCAB, 'rb') as f:
        return pickle.load(f)


def _preview(files, limit=80_000):
    vocab = _load()
    print(f'vocabolario: {len(vocab):,} parole pulite\n')
    seen, resolved, total = Counter(), 0, 0
    for fp in files:
        with open(fp, encoding='utf-8', errors='replace') as f:
            for line in f:
                if not HAS_JUNK.search(line):
                    continue
                for m in re.finditer(r'\S*¿\S*', line):
                    w = m.group(0)
                    fixed = repair(w, vocab)
                    total += 1
                    if not HAS_JUNK.search(fixed):
                        resolved += 1
                    if len(w) < 40:
                        seen[(w, fixed)] += 1
                if total > limit:
                    break
        if total > limit:
            break
    for (w, f), n in seen.most_common(35):
        print(f'  {n:>6,}  {w:<34} ->  {f}')
    print(f'\nrisolte {resolved:,}/{total:,} ({resolved/total:.1%}) '
          f'su {total:,} parole corrotte campionate')


def _repair_file(job):
    fp, out_dir = job
    vocab = _load()
    out = os.path.join(out_dir, os.path.basename(fp))
    n = 0
    with open(fp, encoding='utf-8', errors='replace') as fi, \
            open(out, 'w', encoding='utf-8') as fo:
        for line in fi:
            if HAS_JUNK.search(line):
                line = repair(line, vocab)
                n += 1
            fo.write(line)
    return os.path.basename(fp), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-vocab', action='store_true')
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--apply', metavar='DIR_OUT')
    ap.add_argument('--corpus', default=CORPUS_DIR)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.corpus, '*aiuti_*.csv')))
    if not files:
        sys.exit(f'nessun file in {args.corpus}')

    if args.build_vocab:
        vocab = build_vocab(files)
        with open(VOCAB, 'wb') as f:
            pickle.dump(vocab, f)
        print(f'vocabolario: {len(vocab):,} parole pulite -> {VOCAB}')

    if args.preview:
        _preview(files)

    if args.apply:
        os.makedirs(args.apply, exist_ok=True)
        with mp.get_context('fork').Pool(N_PROC) as pool:
            for name, n in pool.imap_unordered(
                    _repair_file, [(f, args.apply) for f in files]):
                print(f'  {name:<44} righe riparate: {n:,}')
        print(f'\nscritto in {args.apply} (gli originali non sono stati toccati)')


if __name__ == '__main__':
    main()

# TODO — Tracciabilità: regex v2 + profilazione

## Contesto

Obiettivo iniziale: classificare i record "tracciabilità / non tracciabilità" con un modello BERT.
**La diagnosi ha ribaltato l'impostazione**: il BERT era addestrato su label generate dalla regex
stessa (`prepare_traceability_data.py` importa `get_mask`) → distillazione di un maestro rumoroso,
valutata contro lo stesso maestro (circolare). Non poteva funzionare per costruzione.

La regex aveva **due difetti indipendenti**:

1. **Design** — le regole composte verificavano la co-occorrenza di due termini nell'intero testo, a
   qualsiasi distanza (mediana **148 caratteri**, 42% oltre 200, su descrizioni da ~1.000 char), con
   verbi generici (`monitoraggio` 1.831 hit, `identificazione` 595) accoppiati a un qualsiasi
   `prodotti`. Risultato: **solo il 18,2%** dei positivi conteneva un vero termine di tracciabilità;
   il resto erano consulenze di cybersecurity, librerie online, gestionali di magazzino.
2. **Ambiente** — pandas ≥ 3.0 usa stringhe Arrow-backed → motore **RE2**, dove `\b`/`\w` sono
   **ASCII-only**. Dopo una lettera accentata il word-boundary non esiste, quindi
   `\btracciabilit[aà]\b` **non matchava "tracciabilità:"**. La regex era corretta quando fu scritta
   (pandas ≤ 2.x, motore `re`): si è rotta in silenzio con l'upgrade, perdendo **~2.000 record** che
   contengono esplicitamente la parola "tracciabilità".

**Decisioni (concordate con l'utente)**: regex v2 come classificatore primario, BERT parcheggiato,
validazione con gold set annotato a mano; `TITOLO_MISURA` inclusa nell'OR ma con `MATCH_SOURCE` per
isolarne il contributo; profilazione su 4 assi.

## Fase 1 — Oracolo unico dei pattern
- [x] `traceability_patterns.py` (nuovo): unica sorgente di verità dei pattern
- [x] `near(a, b, window)` → prossimità bidirezionale (~una proposizione): fix del difetto di design
- [x] Rimossi i verbi generici (`monitoraggio`, `identificazione`, `autenticazione`) da `ACTION`
- [x] Escluso il falso amico **`marcatura CE`** (conformità di prodotto, non tracciabilità)
- [x] `etichettatura`/`marcatura` valgono solo in contesto di filiera (regola `labeling`, finestra 40)
- [x] `_normalize()` forza `object` dtype → motore `re` Unicode-aware: **fix del bug pandas 3.0**
- [x] Provata e **RIMOSSA** una regola `generic_strong`: l'audit sui dati reali mostrava che produceva
      quasi solo falsi positivi (controllo di gestione commesse, SA8000, gioielleria)

## Fase 2 — Estensione a TITOLO_MISURA + spiegabilità
- [x] `traceability_worker.py`: regex su `TITOLO_MISURA` + `TITOLO_PROGETTO` + `DESCRIZIONE_PROGETTO`
- [x] Nuove colonne `MATCH_SOURCE` (quale campo) e `MATCH_RULE` (quale regola) → ogni positivo auditabile
- [x] Corretto bug latente: `mode='a'` duplicava le righe a ogni rerun → ora l'output è riscritto

## Fase 3 — Rigenerazione + confronto onesto
- [x] `traceability_extraction.ipynb`: test battery (18 casi; i falsi positivi sono record REALI della v1),
      `Pool(10)` (12 core − 2), contesto `fork` (Python 3.14 usa `forkserver`, che non gira da notebook)
- [x] `compare_regex_versions.py` (nuovo): v1 e v2 ricalcolate sullo **stesso corpus, stessa passata,
      stesso engine** — i CSV v1 su disco erano stali, confrontarli sarebbe stato apples-to-oranges

## Fase 4 — Validazione non circolare
- [x] `build_regex_gold_set.py` (nuovo): gold set stratificato A/B/C
      (A = precision v2 · B = gli scartati erano davvero FP? · C = recall sui near-miss)
- [ ] **(UTENTE)** Annotare `GOLD_LABEL` in `data/traceability/validation/regex_gold_set.csv`
      → poi `python score_gold_set.py <csv>` per precision/recall reali di v1 vs v2

## Fase 5 — Profilazione (nuovo notebook)
- [x] `profiling_worker.py` (nuovo): 3 tassonomie multi-label + pipeline lessicale spaCy
      (`TECNOLOGIA` · `TIPO_INTERVENTO` · `SETTORE`), `n_process=10`
- [x] Riusa ed estende i pattern già esistenti in `src/regex_multiprocessing.py`
      (`REGEX_FORMAZIONE` / `REGEX_IMPLEMENTAZIONE`) invece di riscriverli
- [x] `traceability_profiling.ipynb` (nuovo): wordcloud (generale + per settore), top termini e
      bigrammi, heatmap tecnologia × intervento, evoluzione temporale delle tecnologie

## Fase 6 — Audit di copertura della specifica dei termini (v2.1)
- [x] `audit_spec_terms.py` (nuovo): per ogni termine della specifica, hits e **orfani**
      (record che contengono il termine ma che la v2 scarta) sui 23,96M record, `Pool(10)`
- [x] `sample_new_terms.py` (nuovo): campiona gli orfani → si decide sui dati, non a tavolino
- [x] `catena/sistema di approvvigionamento` aggiunto come **sinonimo** di supply chain nel
      vocabolario condiviso `SUPPLY_CHAIN` → eredita i vincoli di prossimità già validati (+9 record)
- [x] Deviazioni dalla specifica documentate **nel modulo** (`traceability_patterns.py`), con i numeri
- [x] Corretto bug in `build_regex_gold_set.py`: lo strato B leggeva da `_v1_backup/` (cartella
      cancellata) → produceva **in silenzio** 220 record invece di 300. Ora B e C si ricalcolano dal
      corpus nella stessa passata + assert che fallisce rumorosamente se uno strato è monco

### Copertura della specifica: cosa cerca la regex e cosa no

| Termine della specifica | v2.1 | Perché |
|---|---|---|
| tracciabilità · rintracciabilità · provenienza del prodotto · origine certificata | **trigger autonomo** | termini forti, non ambigui |
| monitoraggio + end-to-end · industria 4.0 + tracciabilità | coperti | 8 e 3 hit in tutto il corpus |
| blockchain · QR/barcode · IoT · DLT · tracciamento · filiera · supply chain | **solo in contesto** | da soli pescano DeFi, air mobility, QR turistici di chiese e hotel, "IoT per lo Smart Living", "Tracciamento Sanitario" |
| catena/sistema di approvvigionamento | **sinonimo di supply chain** | da solo è procurement (57 orfani: acquisti, gestionali, un hotel sul Garda) |
| autenticazione | **escluso** | login, pagamenti, cyber security, biometria (284 orfani) |
| identificazione / localizzazione + prodotto | **escluso** | "Formazione per l'addetto ai trattamenti fitosanitari" (213 orfani) |
| sicurezza del prodotto | **escluso** | sono le **schede di sicurezza (SDS)** dei prodotti chimici (118 orfani) |
| catena del valore | **solo con CHAIN_MOD** | da sola è business-speak: abbigliamento bambini, export (356 orfani) |
| smart logistics · logistica intelligente | già coperti da `logistic\w*` in CTX | metà sono ragioni sociali e corsi generici |

**Il blocco "solo nel titolo del progetto" NON è stato implementato** (decisione dell'utente, presa sui
numeri): `filiera` nel solo titolo = **36.147 record**, cioè **7,5× l'intero set di positivi**, e sono
"Filiera Bollicine in Rete", ristoranti a km 0, "filiera edile". Il titolo è corto ma non è un vincolo
semantico: rifarebbe l'errore di co-occorrenza della v1.

## Review

### Diagnosi
Il BERT non poteva funzionare **per costruzione**: le sue label venivano da `get_mask`, quindi era la
distillazione della regex, valutata contro la regex stessa. L'`F1 = 0.928` era circolare.
La domanda giusta non era "quale modello?", ma "**l'oracolo è corretto?**". Non lo era, per due
motivi indipendenti (uno di design, uno di ambiente — vedi Contesto).

### Risultati misurati (23.957.368 record, stesso corpus / stessa passata / stesso engine)

| | v1 | v2 | v2.1 |
|---|---|---|---|
| positivi | 6.581 (0,027%) | 4.832 | **4.841 (0,020%)** |
| con termine FORTE di tracciabilità | **18,2%** | 84,1% | **84,1%** |
| solo regole composte | 81,8% | 15,9% | 15,9% |

*(v2.1 = v2 + `catena/sistema di approvvigionamento` come sinonimo di supply chain: +9 record)*

- **in comune**: 3.973 · **scartati dalla v2**: 2.608 (falsi positivi v1) · **nuovi in v2**: 859
- I 2.608 scartati sono, a campione, chiaramente spazzatura: consulenze cybersecurity, librerie
  online, gestionali di magazzino di ristoranti, monitoraggio di campi elettromagnetici, rover per
  coltivazioni orticole.
- Gli 859 nuovi sono veri positivi che la v1 perdeva (tracciabilità del legname, "dal campo alla
  tavola", catena di custodia, anticontraffazione, forme accentate/mojibake).
- **`TITOLO_MISURA`**: coinvolta in soli **8 record**, e **0 positivi dipendono da essa sola**.
  Il rischio di inondazione (una misura copre fino a 126k record) **non si è materializzato**:
  i titoli delle misure sono nomi di strumenti burocratici ("Fondo di garanzia per le PMI"),
  non descrizioni di progetto. `MATCH_SOURCE` lo dimostra e lo tiene sotto controllo.

### Profilazione (4.832 record, 3.481 descrizioni uniche)
- **Tipo intervento**: implementazione 55,9% · **formazione 25,1%** · certificazione 11,9% ·
  consulenza 8,3% · R&S 3,9%
- **Tecnologia**: ERP/gestionali 24,5% · QR/barcode 12,6% · **blockchain 5,8%** · IoT 5,4% ·
  RFID/NFC 2,9% · AI/ML 1,9%. Il 56,2% non dichiara alcuna tecnologia.
- **Settore**: agroalimentare 31,8% · logistica 18,0% · legno/arredo 5,1% · farmaceutico 2,9%
- Bigrammi top: `software gestionale`, `tracciabilità prodotto`, `codice barra`, `supply chain`,
  `lettore barcode`, `prodotto alimentare`.
- Figure in `data/traceability/figures/` (wordcloud generale e per settore, heatmap, serie storiche).

### Bug trovati e corretti strada facendo (nessuno era nel piano)
1. **`marcatura CE`** classificata come tracciabilità (è conformità di prodotto) → esclusa con lookahead.
2. **pandas 3.0 / Arrow-RE2**: `\b` ASCII-only → `\btracciabilit[aà]\b` **non matcha "tracciabilità:"**.
   Regressione silenziosa da upgrade di dipendenza, senza che una riga di codice cambiasse.
3. **`\bai\b` nella tassonomia tecnologica** matchava la *preposizione italiana* "ai" → AI/ML gonfiata
   da 91 a 1.216 progetti (13×). Trovato solo perché il numero era implausibile.
4. Una mia regola `generic_strong` (verbi generici + target forte) → **bocciata dall'audit** sui dati
   reali e rimossa: produceva quasi solo falsi positivi.
5. `mode='a'` in `process_file` duplicava le righe a ogni rerun.

I bug 1, 3 e 4 sono stati trovati **guardando i dati**, non ragionando a tavolino: è il motivo per cui
`MATCH_SOURCE`/`MATCH_RULE` esistono.

## Fase 7 — Validazione con gold set annotato: **i numeri veri**

- [x] `judge_gold_set.py`: giudice LLM locale (LM Studio, gpt-oss-20b) sui 300 — annotatore
      **indipendente** dalla regex, rompe la circolarità. Accordo con la regex: 74,3%
- [x] Adjudicazione **in cieco** dei 107 record informativi (77 disaccordi + 30 accordi
      mescolati come controllo). Chiave in file separato: nel file annotato non c'è la risposta
- [x] `merge_adjudication.py` + `score_gold_set.py`
- [x] Spot-check sugli accordi: **2/30 sbagliati (6,7%)** → l'assunzione "se concordano hanno
      ragione" regge, le stime non sono gonfiate in modo grave

### Precision reale (strato A = campione casuale dei positivi → stima non distorta)

| | precision |
|---|---|
| regex v1 | ~32% |
| **regex v2.1** | **50,8%** (IC 95%: 41,9–59,8) |
| **regex v2.1 + giudice LLM in cascata** | **96,2%** |

- **La mia tesi era sbagliata.** Avevo venduto la v2 con "84,1% contiene un termine forte". La
  precision vera è **50,8%**: una metà di ciò che la regex marca **non è tracciabilità**. Il proxy era
  inflazionato perché *la parola "tracciabilità" è boilerplate* (ERP, gestionali HR, pratiche
  assicurative, e-commerce: "tracciabilità dei processi/documenti"). Vedi `tasks/lessons.md`.
- Il lavoro sulla v2 **è comunque servito**: precision da ~32% a ~51%, a recall invariato.
- **Strato B: solo 2,5% di veri positivi** → i 2.608 record che la v2 ha scartato dalla v1 erano
  davvero spazzatura (97,5%). La potatura era giusta.
- **Strato C: solo 2,0% di veri positivi** sui 12.597 near-miss → la v2 si perde ~250 record.
  **Recall ≈ 91%**: come *filtro* la regex è ottima, come *classificatore* no.

### La cascata: regex = filtro, LLM = decisore
Sui 4.841 positivi della regex, il giudice LLM ne conferma il 43% (~2.100):
- precision **50,8% → 96,2%**
- elimina 2.299 falsi positivi, al prezzo di ~443 veri positivi persi (recall 82% sui positivi regex)
- costo: **zero** (LM Studio in locale, ~15 min sull'intero set)

### Limite dichiarato
Il gold è annotato da **Claude Opus 4.8**, non da un umano — e la regex v2 l'ha scritta lo stesso
modello: c'è un conflitto d'interessi (l'autore che si corregge il compito), mitigato ma non
eliminato dall'annotazione in cieco. Da ricontrollare a campione (~20 record). Inoltre lo strato C
campiona i near-miss, non la popolazione: il recall è "sul pool dei near-miss", non assoluto.

---

## Fase 8 — Ricostruzione del corpus con testo integro (`technology_mapping_v2`)

**Perché**: il corpus pubblicato ha il testo mutilato da `reclassify_annual.py` (QUOTE_NONE, ogni `,` → spazio):
6,7M descrizioni (28,1%) senza virgole, e `SETTORI_ATTIVITA` — una lista comma-separated — con i separatori
distrutti. Il testo integro esiste ancora in `data/classified_multiclass_aiuti_*.csv`.

**Vincolo**: le etichette pubblicate NON devono cambiare (il paper è uscito). Misurato in anticipo: il gate AI
riprodotto sulla sorgente pulita dà **esattamente 7.022 record**, e `TECNOLOGIE_AI` ha **0 differenze**.

- [ ] Refactor `src/reclassify_annual.py`: path assoluto (oggi `'../data'` cwd-dipendente → `MANUAL_AI_SET`
      vuoto in silenzio, -144 AI), e separazione `apply_ai_gate` / `stata_mutilate`. Comportamento invariato.
- [ ] `classify_obiettivo` a livello modulo in `src/regex_multiprocessing.py` (oggi annidata, non importabile)
- [ ] `src/freeze_tipo_ai_carryover.py`: congela i 1.807 `doubt` risolti dall'API ormai spenta (7.022 righe)
- [ ] `src/rebuild_corpus.py` (Pool 10): gate su testo ORIGINALE → repair mojibake → TIPO_AI → TECNOLOGIE_AI
      → CSV QUOTE_MINIMAL in `data/technology_mapping_v2/`
- [ ] `src/verify_corpus.py` (Pool 10): 0 diff attesi su CLASSIFICAZIONE / TIPO_AI / TECNOLOGIE_AI, altrimenti
      exit(1). Le differenze devono stare SOLO nel testo.
- [ ] `src/export_stata.py`: `.dta` v118 (UTF-8 + strL) per i 12 anni + file dei soli 7.022 AI
- [ ] Rimuovere `data/technology_mapping_repaired/` (vicolo cieco: ripara il mojibake ma non le virgole)

**Trappole già misurate** (non riscoprirle):
- `COR` **non è una chiave**: duplicato 2.275 volte nel 2024 → allineamento POSIZIONALE, mai `merge(on='COR')`
- 12 chiavi di `MANUAL_AI_SET` contengono `¿` → il gate gira sul testo ORIGINALE, la riparazione viene DOPO
- `regex_gazetteer.extract_batch` apre `Pool(cpu_count())`: dentro un worker esplode → usare `extract`
- `to_stata` non è streaming: Pool 10 sul 2023+2024 insieme = OOM
- `CLASSIFICAZIONE_MULTICLASS_CONFIDENZA` è lunga 37 char: STATA tronca a 32 **in silenzio**

---

## Fase 9 — Estrazione imprese Marche 2023–2025 — COMPLETATA

**Obiettivo**: da `data/technology_mapping_repaired/reclassified_multiclass_aiuti_{2023,2024,2025}.csv`
(9,0 GB) estrarre tutte le righe con `REGIONE_BENEFICIARIO` = Marche, incluse le multi-regione.

**Scelte utente**: sorgente `technology_mapping_repaired` (non `data/raw`, non `v2`); un record per aiuto
(27 colonne, nessuna dedup); nessun filtro su `DES_TIPO_BENEFICIARIO`.

### Fatti misurati (non riscoprirli)

- Il separatore multi-regione **NON è `|`**: è un **doppio spazio**. Nel campo `REGIONE_BENEFICIARIO`
  ci sono **0 pipe**. È il danno di `reclassify_annual.py` (QUOTE_NONE, ogni `,` → spazio):
  `'Lazio  Marche'`, `'Emilia-Romagna  Lazio  Liguria  Lombardia  Marche  Piemonte'`.
- `technology_mapping_repaired` ha **0 virgolette** e **27 campi esatti**: una riga = un record,
  `split(b',')` è esatto → byte-range chunking senza pandas.
- Nessun'altra regione contiene la sottostringa "Marche" → il prefiltro `b'Marche' in riga` è
  conservativo (0 falsi negativi). Il match finale è comunque sul **token esatto**, non sottostringa.

### Fatto

- [x] `src/verify_repaired.py` (`Pool(10)`): prova che `repaired` ≡ `technology_mapping`
- [x] `notebooks/analytics/extract_marche.py` (`Pool(10)`, 136 fette da 64 MB): 9,0 GB in **1,5s**
- [x] Output in `data/analytics/marche/`: `marche_{2023,2024,2025}.csv` + `marche_2023_2025.csv`
- [x] Passthrough **verbatim** delle righe sorgente (no `csv.writer`) → nessuna seconda mutilazione
- [x] Ordine deterministico per `(anno, offset)`: due run → md5 identici (verificato)

### Verifica (fatta, non dichiarata)

- [x] **Doppia implementazione indipendente** (`scratchpad/verify_marche.py`: modalità testo +
      `csv.reader` + `str.split('  ')`, per anno intero invece che per byte-range) → **concorda
      esattamente**: 205.929 / 156.529 / 46.642
- [x] **Superset**: righe contenenti "Marche" ovunque = 205.984 ≥ 205.929 estratte. Il delta (55) sono
      righe con "Marche" in altri campi ma non nella regione → correttamente escluse
- [x] Output: 27 campi su ogni riga, `ANNO` costante, ogni riga ha il token Marche, 0 malformate
- [x] Le multi-regione ci sono: 707 righe, 33 combinazioni distinte

### Risultato

**409.100 record** (2023: 205.929 | 2024: 156.529 | 2025: 46.642), 131.130 CF distinti.
PMI 376.061 · Non classificata 31.604 · Grande impresa 1.334 · `-` 101.
Di questi **99 sono AI** (85 implementazione, 14 formazione), tutti con `TECNOLOGIE_AI` valorizzato.

### Caveat aperto

`technology_mapping_repaired` ripara il mojibake ma **non le virgole** (0 virgolette): le
`DESCRIZIONE_PROGETTO` in output restano prive di virgole. Per la selezione regionale è indifferente,
ma **per NLP sul testo la sorgente giusta è `technology_mapping_v2`** (che ripara entrambi i danni ed
è già verificato). Se l'analisi Marche arriva a toccare il testo, rigenerare da v2: è un cambio di
`SRC_DIR` in `extract_marche.py`. La Fase 8 prevede comunque la rimozione di `repaired`.

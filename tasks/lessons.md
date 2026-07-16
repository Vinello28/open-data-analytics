# Lessons

## Un proxy non è una metrica: "contiene un termine forte" NON è precision
- **Contesto**: avevo venduto la regex v2 con "l'**84,1%** dei positivi contiene un termine forte di
  tracciabilità (nella v1 era il 18,2%)", presentandolo come prova che funzionasse. Il gold set
  annotato dice che la **precision reale è 50,8%** (IC 95%: 42–60%). Il proxy era inflazionato dal
  fatto che *la parola "tracciabilità" è boilerplate*: compare in ERP, gestionali HR, pratiche
  assicurative, e-commerce ("tracciabilità dei processi/documenti"). Contenere il termine forte non
  dimostra che il progetto **tracci un prodotto**.
- **Regola**: quando non puoi misurare la metrica vera, un proxy va bene per **orientarti**, mai per
  **concludere**. E va dichiarato come proxy ogni volta che lo citi, non solo la prima. Se ti accorgi
  di usarlo come se fosse la metrica ("l'84% è buono"), ti stai auto-ingannando.
- **Come applicarla**: il proxy misura la *presenza di un segnale*; la metrica misura la *correttezza
  di una decisione*. Sono la stessa cosa solo se il segnale è deciso — e le parole non lo sono mai.

## Se il gold viene dall'accordo tra due predittori, non puoi valutare quei due predittori su di esso
- **Contesto**: nel gold set 193/300 label venivano dall'accordo regex+LLM (solo i 107 discordanti
  erano annotati a mano). Sui 193 **regex e LLM coincidono col gold per costruzione**: valutarli lì
  dentro dava accuracy gonfiate (LLM 0,94!). Il confronto onesto esiste solo sui record con gold
  **indipendente** da entrambi.
- **Regola**: la ground truth deve essere indipendente da **ogni** sistema che ci misuri contro.
  È la stessa trappola del BERT (addestrato su `get_mask`, valutato contro `get_mask`), rientrata
  dalla finestra sotto forma di "assumiamo che gli accordi siano giusti".
- **Come applicarla**: separa sempre (a) le metriche **non distorte** — la precision si stima sullo
  strato A, che è un campione *casuale* dei positivi; (b) i confronti testa a testa, solo sui record
  ad annotazione indipendente. Non citare mai l'aggregato su un campione stratificato: gli strati
  sono sovracampionati e non rispettano le proporzioni di popolazione.

## Se prometti un'annotazione "in cieco", controlla che il file non contenga la risposta
- **Contesto**: avevo progettato l'adjudicazione in cieco (l'umano non deve sapere cosa hanno detto
  regex e LLM, altrimenti insegue la loro risposta). Avevo tolto le colonne `REGEX_V2_LABEL` e
  `LLM_LABEL`… ma avevo lasciato **`stratum`** (dove `A_positivi_v2` significa letteralmente "la regex
  ha detto tracciabilità") e **`_motivo`** (che rivelava quali record erano gli spot-check).
  Il cieco non era cieco. Trovato solo perché l'utente ha chiesto "ci sono solo le colonne di interesse?".
- **Regola**: il blinding si verifica sull'**artefatto finale**, non sull'intenzione. Apri il file che
  consegni e chiediti: "da qui posso dedurre la risposta?". Le colonne di servizio (id di strato,
  motivo del campionamento, ordine delle righe) sono canali di fuga quanto la label esplicita.
- **Come applicarla**: le chiavi vanno in un file **separato** (`.gold_adjudication_key.csv`),
  ricongiunto in fase di merge via id. Nel file da annotare solo: id, testo, colonna da compilare.

## Il dataset esportato è un prodotto, non un dump della memoria di lavoro
- **Contesto**: `tag_taxonomies` materializzava 21 colonne one-hot `TECNOLOGIA__blockchain=True/False`
  *dentro il CSV esportato* (56 colonne totali). Servivano solo ai grafici, ma ripetevano
  un'informazione già presente nella colonna dei valori (`TECNOLOGIA = "blockchain;iot_sensori"`).
  L'utente le ha giustamente definite inutilizzabili.
- **Regola**: separa la **rappresentazione di lavoro** (one-hot, comoda per plottare e incrociare)
  dal **dato canonico** (i valori). Nel file che consegni va il dato canonico, una volta sola.
  Se una colonna si ricava da un'altra con una riga di pandas, non è un dato: è una vista.
- **Come applicarla**: `onehot(df, axis)` in `profiling_worker.py` ricava le booleane al volo con
  `str.get_dummies(sep=';')`. Export: 56 → **35 colonne**, zero ridondanza.

## Multiprocessing NON è solo per la pipeline "ufficiale": vale per OGNI passata sul corpus
- **Contesto**: l'utente aveva già dato la regola (Pool(10), 12 core − 2). L'ho applicata a
  `traceability_worker`, `compare_regex_versions`, `profiling_worker`… ma ho scritto gli script
  **usa e getta** di audit (`sample_spec_orphans.py`) con un banale `for file in files:` sequenziale.
  Su 24M record significa far aspettare l'utente minuti per una cosa che ne richiede uno.
  Richiamato: *"ci stai mettendo troppo, usa i miei core"*.
- **Regola**: se lo script tocca il corpus completo, va in `Pool(10)` con `mp.get_context('fork')`.
  **Non esiste lo script "di scarto"**: uno script esplorativo lento fa perdere tempo esattamente
  come uno di produzione. La regola dell'utente vale sul *carico*, non sull'importanza del file.
- **Come applicarla**: prima di lanciare qualunque cosa che apra i file di `data/technology_mapping/`,
  fermarsi e chiedersi "sto ciclando in sequenza?". Se sì, riscrivere con `pool.map(fn, files)`.
  Template pronto: `audit_spec_terms.py`.

## Un termine può essere nella specifica ed essere comunque un falso amico: controlla cosa pesca
- **Contesto**: la specifica dei termini chiedeva `identificazione + prodotto`, `autenticazione`,
  `sicurezza del prodotto`, `catena del valore`, `filiera nel titolo`. Campionando gli orfani:
  `sicurezza del prodotto` = le **schede di sicurezza (SDS) dei prodotti chimici**;
  `identificazione + prodotto` = corsi di **trattamenti fitosanitari**; `autenticazione` = login e
  pagamenti; `catena del valore` = negozi di abbigliamento; `filiera` nel solo titolo = **36.147
  record** (7,5× l'intero set positivo): "Filiera Bollicine in Rete", ristoranti a km 0.
- **Regola**: una lista di keyword è un'**ipotesi**, non una verità. Prima di implementarla,
  misura *cosa aggiunge davvero* — non quanti record, ma **quali**. Un termine giusto in astratto
  può essere dominato, nei dati reali, da un omonimo di un altro dominio.
- **Come applicarla**: `audit_spec_terms.py` (copertura per termine) + `sample_new_terms.py`
  (campiona gli orfani). Riporta i campioni all'utente **prima** di cambiare l'oracolo: qui hanno
  ribaltato una decisione già presa. I sinonimi veri (`catena di approvvigionamento` = supply chain)
  entrano nel **vocabolario esistente**, così ereditano i vincoli di prossimità già validati, invece
  di diventare una regola nuova senza rete.

## Verifica la premessa di un bug report prima di "fixare" ciò che l'utente indica
- **Contesto**: task tracciabilità — l'utente sospettava che il modello fosse addestrato solo sulla
  descrizione (non sul titolo), a differenza della regex. In realtà `dataset.py:231` costruiva già
  l'input come `f"{title}: {desc}"`, identico all'inference. La premessa non reggeva.
- **Regola**: quando l'utente indica una causa presunta, verificala nel codice PRIMA di implementare.
  Se cade, fermati e ripianifica (come da CLAUDE.md §1) invece di applicare un fix già presente.
- **Come applicarla**: in plan mode, dedica un'esplorazione esplicita a confermare/smentire l'ipotesi
  dell'utente; riporta l'esito e ridiscuti la direzione con AskUserQuestion.

## Un solo oracolo di label tra training e validazione
- **Contesto**: i negativi di training erano filtrati con una regex debole (`tracciabil|filiera|
  blockchain`) mentre la ground truth di validazione usava la `get_mask` completa → label
  contraddittorie, confine decisionale sporco.
- **Regola**: la funzione che definisce la label positiva deve essere UNA sola, importata (non
  duplicata) sia dal preparatore del dataset sia dal validatore.
- **Come applicarla**: `get_mask` in `notebooks/traceability/traceability_worker.py` è l'oracolo;
  `prepare_traceability_data.py` e `validation_worker.py` la importano entrambi.

## Base-rate shift: calibra la soglia, non (solo) il training set
- **Contesto**: modello addestrato ~50/50 ma applicato a popolazione con ~0.3% positivi → argmax@0.5
  over-triggera (positive-rate 15×, precision 0.06).
- **Regola**: separa "learnability" (training bilanciato) da "operating point" (soglia di decisione
  calibrata sul base-rate reale/gold). Esponi P(classe positiva) in inference per poter ritarare a
  valle senza rieseguire il modello.
- **Come applicarla**: `inference.positive_threshold` in `model_config.yml`, applicato dall'helper
  condiviso `src/inference/labeling.py::probs_to_results`.

## Un modello addestrato su label di una regex NON può superare la regex
- **Contesto**: il BERT dava "F1 = 0.928 held-out" e sembrava ottimo, ma le sue label venivano da
  `get_mask` (`prepare_traceability_data.py` la importa). Era la **distillazione di un maestro
  rumoroso**, valutata contro lo stesso maestro: valutazione circolare, numero privo di significato.
  Sulla popolazione reale over-triggerava 20,8×.
- **Regola**: se le label di training e la ground truth di validazione vengono dalla stessa funzione,
  non stai misurando la qualità del modello — stai misurando quanto bene imita quella funzione.
  Il modello non può, per costruzione, scoprire nulla che l'oracolo non sapesse già.
- **Come applicarla**: prima di investire in un modello supervisionato, chiedi "da dove vengono le
  label?". Se la risposta è "da una regola", allora (a) la regola è il classificatore, aggiustala; e
  (b) l'unica validazione sensata è un **gold set annotato a mano**. Vedi `build_regex_gold_set.py`.

## Le regole composte richiedono PROSSIMITÀ, non co-occorrenza
- **Contesto**: `get_mask` v1 marcava un record se `azione` e `oggetto` comparivano entrambi nel
  testo, **a qualsiasi distanza**. Misurato: mediana **148 caratteri** di distanza, 42% oltre 200,
  su descrizioni di ~1.000 caratteri. Risultato: consulenze di cybersecurity e gestionali di
  magazzino classificati "tracciabilità"; solo il **18,2%** dei positivi aveva un termine forte.
- **Regola**: `A.*B` su un documento lungo non è una relazione semantica, è una collisione lessicale.
  Vincola i termini a una finestra (~una proposizione) e tieni fuori i verbi generici
  (`monitoraggio`, `identificazione`, `controllo`): da soli non significano nulla.
- **Come applicarla**: helper `near(a, b, window)` in `traceability_patterns.py`. Ogni positivo
  espone `MATCH_RULE`/`MATCH_SOURCE` → l'audit dei match è ciò che ha smascherato il problema
  (e ha bocciato una mia regola `generic_strong`, poi rimossa perché produceva solo spazzatura).

## pandas ≥ 3.0: `\b` e `\w` sono ASCII-only (Arrow/RE2) → le parole accentate non matchano
- **Contesto**: `pd.Series(["tracciabilità:"]).str.contains(r'\btracciabilit[aà]\b')` → **False**,
  mentre `re.search(...)` sullo stesso testo → **True**. pandas 3.0 usa stringhe Arrow-backed che
  passano da RE2, dove `\b` è ASCII-only: dopo `à` il word-boundary non esiste. La regex v1 era
  **corretta quando fu scritta** (pandas ≤ 2.x, motore `re`) e si è rotta in silenzio con l'upgrade,
  perdendo **~2.000 record che contengono esplicitamente "tracciabilità"**.
- **Regola**: una regressione può arrivare da un upgrade di dipendenza senza che una riga di codice
  cambi. Se una regex con `\b`/`\w` gira su testo non-ASCII (italiano, accenti!), verifica il motore.
- **Come applicarla**: `_normalize()` in `traceability_patterns.py` forza `.astype(object)` →
  motore `re` di Python, Unicode-aware, con supporto ai lookahead. Test di non-regressione nella
  test battery del notebook: "tracciabilità:" (accento + punteggiatura) DEVE matchare.

## Un carattere di sostituzione non si sostituisce a tappeto: si *deduce* dai dati

- **Contesto**: il corpus conteneva 278.561 `¿`. La lettura ovvia ("gli apostrofi sono diventati `¿`")
  era vera solo al **57%**. Il `¿` è un *replacement char*: si è mangiato apostrofi, vocali accentate
  (`pi¿`→`più`, `citt¿`→`città`), trattini (`Pezzapiana ¿ Zona`), il simbolo euro (`pari a ¿ 300.000`),
  virgolette tipografiche — e in alcuni file c'erano **tre strati di mojibake sovrapposti**
  (`Ã¿Â¢Ã¿Â¿Ã¿Â¿interno` → `all'interno`).
- **Regola**: prima di sostituire, **censisci i contesti**. Un `sed 's/¿/'/g'` avrebbe scritto
  `pi'`, `citt'`, `' 300.000` su decine di migliaia di record, e nessuno se ne sarebbe accorto.
- **Come applicarla**: `src/text_repair.py` non indovina la parola, **deduce il carattere**. Costruisce
  il vocabolario delle parole PULITE del corpus (il corpus è il proprio dizionario: la corruzione
  colpisce una minoranza di righe), poi per ogni sequenza corrotta prova ogni candidato e tiene quello
  che produce una parola **che esiste davvero**. Toccando solo i caratteri rotti, il maiuscolo
  sopravvive: `dell¿EXPO` → `dell'EXPO`, non `dell'expo` (era il bug della prima versione, che
  sostituiva la parola intera).
- **Verifica**: righe identiche prima/dopo su tutti i 12 file, zero `¿` residui, delta di dimensione
  -0,0057% (solo i byte di spazzatura collassati). Batteria di test sui casi difficili prima di
  toccare 16GB.
- **E soprattutto**: il danno era **già in `data/raw/`**, identico. Non l'aveva introdotto la pipeline.
  Prima di correggere a valle, controlla sempre se il difetto viene da monte: cambia completamente
  che cosa vai a riparare.

## RECIDIVA: la regola del Pool vale anche per il controllo di tre righe fatto "al volo"
- **Contesto**: seconda violazione della stessa regola già scritta sopra ("Multiprocessing NON è solo
  per la pipeline ufficiale"), sullo stesso corpus, **pochi minuti dopo aver letto questo file**.
  `src/verify_repaired.py` era correttamente in `Pool(10)`; poi, per un cross-check "veloce" del
  conteggio AI, ho scritto `cat data/*/reclassified_*.csv | awk ...`: **15 GB in pipe sequenziale**.
  L'utente ha interrotto con "stai usando il multiprocessing?".
- **Perché è ricascato**: la prima volta la scusa era "è uno script usa e getta". Qui era peggio:
  non sembrava nemmeno uno *script*, era "un comando". La regola l'avevo applicata all'artefatto che
  *si vedeva* (il file .py in `src/`) e non al comando in Bash, come se il costo dipendesse da dove
  scrivi il codice invece che da quanti byte legge.
- **Regola**: il trigger non è "sto scrivendo uno script", è **"sto per leggere il corpus"**. Se una
  riga di Bash apre più di un file da un GB, non è un comando: è una passata, e va in `Pool(10)`.
  In pratica: `cat`/`awk`/`grep` su `data/*/reclassified_*.csv` sono **sempre** un errore.
- **Come applicarla**: il cross-check parallelo (24 task su `Pool(10)`) ha impiegato **7,8s** contro i
  minuti del pipe. Non è una questione di stile: è il tempo dell'utente, e l'avevo già sprecato una volta.

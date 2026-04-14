# Piano di Sviluppo: Augmentation Dataset PWC

- [ ] 1. Installazione librerie necessarie (`datasets` di Hugging Face).
- [ ] 2. Download del dataset `pwc-archive/papers-with-abstracts` e salvataggio in `data/raw/pwc-archive`.
- [ ] 3. Caricamento in DataFrame pandas e selezione delle colonne: `title`, `abstract`, `short_abstract`, `tasks`, `methods`.
- [ ] 4. Filtraggio records AI tramite regex avanzata e salvataggio in `data/processed/pwc_ai_filtered.csv`.
- [ ] 5. Generazione Dataset Multiclasse (description, label) con mapping deterministico sui domini applicativi, salvataggio in `data/processed/pwc_ai_multiclass.csv`.
- [ ] 6. Generazione Dataset Multilabel (description, labels) tramite normalizzazione della colonna `tasks`, salvataggio in `data/processed/pwc_ai_multilabel.csv`.

# Piano di Sviluppo: Multilabel Consolidation e Traduzione (max 20-30 label)
- [x] 1. Analisi delle label esistenti ed estrazione frequenze dal dataset `pwc_ai_multilabel.csv`.
- [x] 2. Definizione del dizionario di mapping (circa 20-30 macro-categorie AI, es. 'Reinforcement Learning', 'Computer Vision', 'Natural Language Processing') da applicare alle label attuali.
- [x] 3. Integrazione di nuovo codice nel notebook `09_eda_pwc_multilabel.ipynb` per:
    - [x] Mappare e normalizzare le liste di label per ogni riga.
    - [x] Dedurre duplicati interni (es. se un record ha 'dr' e 'rl' che mappano entrambi a 'Reinforcement Learning', va mantenuta 1 sola occorrenza).
    - [x] Visualizzare graficamente la nuova distribuzione.
- [x] 4. Salvataggio del nuovo dataset normalizzato in `data/processed/` (es. `pwc_ai_multilabel_mapped.csv`).

# Piano di Sviluppo: Refining "Research" Category Multiclasse
- [x] 1. Caricamento del dataset `pwc_ai_multiclass.csv` nel notebook `08_eda_pwc_multiclass.ipynb` ed isolamento dei record classificati come "Research".
- [x] 2. Sviluppo logica di ri-classificazione: mappatura dei termini ricorrenti in base a keyword applicative per disambiguare il linguaggio accademico dalla ricerca in altri settori (e.g. Healthcare, NLP, Computer Vision/Robotics).
- [x] 3. Esecuzione della nuova passata di classificazione e aggiornamento delle label sul DataFrame originale.
- [x] 4. Visualizzazione e confronto della nuova distribuzione dei dati.
- [x] 5. Salvataggio in sovrascrittura di `data/processed/pwc_ai_multiclass.csv`.

## Review - Refining "Research" Category Multiclasse
- **Risultato**: Sviluppate regole regex mirate ad individuare gli effettivi casi d'uso ("Healthcare", "Virtual Assistants", "Fintech", ecc.) all'interno degli abstract etichettati genericamente come "Research". 
- **Impatto misurato**: Ricollocati circa 129.313 record da "Research" alle rispettive categorie applicative (il conteggio è sceso da ~337.000 a ~207.000).
- **Prossimi step**: Eventuale training del modello potrà trarre beneficio dalla maggiore granularità applicativa.

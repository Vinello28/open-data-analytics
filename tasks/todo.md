# Piano di Sviluppo: Augmentation Dataset PWC

- [ ] 1. Installazione librerie necessarie (`datasets` di Hugging Face).
- [ ] 2. Download del dataset `pwc-archive/papers-with-abstracts` e salvataggio in `data/raw/pwc-archive`.
- [ ] 3. Caricamento in DataFrame pandas e selezione delle colonne: `title`, `abstract`, `short_abstract`, `tasks`, `methods`.
- [ ] 4. Filtraggio records AI tramite regex avanzata e salvataggio in `data/processed/pwc_ai_filtered.csv`.
- [ ] 5. Generazione Dataset Multiclasse (description, label) con mapping deterministico sui domini applicativi, salvataggio in `data/processed/pwc_ai_multiclass.csv`.
- [ ] 6. Generazione Dataset Multilabel (description, labels) tramite normalizzazione della colonna `tasks`, salvataggio in `data/processed/pwc_ai_multilabel.csv`.

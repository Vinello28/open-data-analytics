# Lessons

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

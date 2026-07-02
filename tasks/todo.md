# TODO — Risanamento pipeline tracciabilità (dati + calibrazione + gold)

Piano completo: `~/.claude/plans/contesto-stavo-lavorando-per-virtual-gadget.md`

## Contesto
La premessa iniziale (titolo escluso dal training) NON reggeva: training e inference usano già
`f"{titolo}: {desc}"` (`dataset.py:231`). Le cause reali dello scarso risultato (precision 0.06,
positive-rate 15.4× regex): base-rate 50/50 vs 0.3% reale, negativi etichettati con regex più
debole della ground truth, encoder quasi congelato, validazione circolare.

## Fase 1 — Fix dati (label oracle unico)
- [x] `prepare_traceability_data.py`: import `get_mask` canonica; helper `traceability_mask` (titolo OR descr)
- [x] Filtro augmentation ora usa `get_mask` invece del vecchio `tracciabil|filiera|blockchain`
- [x] Rigenerazione label dall'oracolo dopo dedup (log righe cambiate) — zero contraddizioni con la ground truth
- [x] Eseguito il prepare → nuovi split; 0 contraddizioni con get_mask, balance 50/50 (1769 label corrette, 30%!)

## Fase 2 — Fine-tune meno congelato
- [x] `model_config.yml`: `unfreeze_top_layers` 2 → 4 (esperimento, confrontare F1 held-out)

## Fase 3 — Calibrazione soglia (base-rate)
- [x] `model_config.yml`: `inference.positive_threshold: 0.5`
- [x] `config.py` + `config_loader.py`: campo `positive_threshold`
- [x] `labeling.py` (helper condiviso numpy): soglia su P(pos) + espone `positive_prob`
- [x] `pytorch_engine.py`, `triton_engine.py`, `tensorrt_engine.py`: usano `probs_to_results`
- [x] `server.py`: `Prediction` espone `positive_prob`

## Fase 4 — Gold set indipendente
- [x] Rigenerata comparison_unique.parquet col NUOVO modello (929.962 righe; A=2711 B=868548 C=58476 D=227)
- [x] Eseguito `build_gold_set.py` → gold_set_to_annotate.csv (238 record); aggiunta colonna AI_POSITIVE_PROB
- [x] Creato `sweep_threshold.py` per calibrare τ su P(pos) vs GOLD dopo annotazione
- [ ] **(UTENTE)** Annotare `GOLD_LABEL` (priorità strati C_modello_*; opzionale prefill LLM riavviando LM Studio + rerun build_gold_set.py)

## Fase 5 — Retrain + ri-validazione
- [x] `train.py --data-source csv` (K-Fold): CV F1 ~0.91-0.93; HELD-OUT TEST P=0.935 R=0.922 F1=0.928
- [x] Export ONNX fp16 → Triton; deploy `docker compose` (triton + classifier gateway 8080) healthy
- [x] Spot-check /classify OK (tracciabilità-solo-titolo → tracciabilita 0.977; controlli negativi < 0.02)
- [ ] Full reclassification 930k via notebook (rigenera AI_LABEL + AI_POSITIVE_PROB col nuovo modello) — 21 min
- [ ] Notebook validation: sweep soglia su P(pos), scelta τ
- [ ] `score_gold_set.py` con gold compilato → precision/recall regex vs modello vs giudice
- [ ] Fissare `positive_threshold = τ` in `model_config.yml`

**Nota deploy**: `config.py`/`config_loader.py` sono baked nell'immagine serve (montati solo `src/inference`,
`src/training`), quindi modifiche a config richiedono `docker compose build classifier`.

## Fix collaterale
- [ ] `traceability_classification.ipynb`: input `f"{titolo}: {desc}" if titolo else desc` (no "nan: desc")

## Review

**Diagnosi**: la premessa iniziale (titolo escluso dal training) era falsa — `dataset.py:231` e il
notebook di inference usano già `f"{titolo}: {desc}"`. Cause reali: (1) ~30% label contraddittorie
(negativi filtrati con regex più debole della ground truth), (2) base-rate 50/50 vs 0.3% reale →
over-trigger a soglia 0.5, (3) encoder quasi congelato.

**Interventi consegnati e verificati**
- `prepare_traceability_data.py`: oracolo di label unico = `get_mask` canonica (titolo OR descr).
  Rigenerato l'intero dataset → **1769/5808 label corrette (30%)**; split con 0 contraddizioni, 50/50.
- `model_config.yml`: `unfreeze_top_layers` 2→4; nuovo `inference.positive_threshold`.
- `labeling.py` (helper condiviso): soglia su P(pos) + `positive_prob`; usato da pytorch/triton/tensorrt engine.
- `server.py`: `Prediction` espone `positive_prob`. `config.py`/`config_loader.py`: campo threshold.
- Notebook classificazione: input allineato al training (no `"nan: desc"`) + cattura `positive_prob`.
- **Retrain K-Fold**: CV F1 ~0.91-0.93; **held-out test P=0.935 R=0.922 F1=0.928**.
- Export ONNX fp16 → Triton; stack deployato (8080) healthy; **spot-check /classify OK** (tracciabilità
  solo-titolo → tracciabilita 0.977; negativi < 0.02).
- Riclassificati 930k record col nuovo modello; comparison table + gold set (238) rigenerati.

**Numeri chiave sulla popolazione (nuovo modello, soglia 0.5)**: positive-rate modello 6.58% vs regex
0.316% (20.8×), recall vs regex 0.923. NB: la regex NON è ground truth → il gold set dirà se i ~58k
positivi-solo-modello (quadrante C) sono veri o falsi. Range soglia utile ~0.90–0.98 (il modello FP16
satura P(pos) a ~0.98, quindi τ=0.99 → 0 positivi).

**Rimanente (utente)**: annotare `GOLD_LABEL`, poi `python sweep_threshold.py` per scegliere τ e
`python score_gold_set.py` per precision/recall onesti; infine fissare `positive_threshold=τ` e
`docker compose build classifier`.

# Validazione Dati CUP per Stata/Excel

**Data validazione:** 25 marzo 2026
**Dataset:** `04_analisi_cup.ipynb` → `data/cup_clean/`
**Totale record:** 4.701.314
**Colonne:** 22

---

## ✅ Verifiche Superate

| Check | Risultato |
|---|---|
| **Nomi colonne** | 22 colonne, tutti compatibili Stata (≤32 char, no spazi, no numeri iniziali) |
| **Lunghezza stringhe** | Solo `SETTORI_ATTIVITA` sfora (max 5.517 char, 1.245 record) — richiede `strL` in Stata 13+ |
| **Timezone in DATA_CONCESSIONE** | ✅ Rimosso correttamente, 100% formato `YYYY-MM-DD` |
| **A-capo nei campi testo** | ✅ Rimossi (0 residui) |
| **Virgole nei campi multi-valore** | ✅ Sostituite con `\|` |
| **Valori numerici** | ✅ Tutti parsabili come `double`, nessun overflow |
| **Consistenza righe** | ✅ Aggregato = Somma annuali = 4.701.314 |
| **File annuali per Excel** | ✅ Tutti ≤ 1.048.576 righe |

---

## ⚠️ Punti di Attenzione

### 1. Mojibake (Encoding) — 0,22%

**10.357 record** (0,22%) contengono testo con codifica corrotta (es. `Ã¨` invece di `è`).

**Causa:** dati OpenData originali, non introdotto dal notebook.

**Impatto:**
- Stata: non bloccante, ma il testo sarà illeggibile
- Excel: dipende dall'editor, ma il testo apparirà corrotto

**Distribuzione per anno:**
- 2014–2018: <230 record
- 2019–2021: 8.538 record (picco)
- 2022–2025: <900 record

**Soluzione (opzionale):** se necessario, correggere con regex UTF-8→Latin1:
```python
# Fix doppia codifica UTF-8 in Stata 13+
# Per ora, considerare accettabile dato il volume basso
```

### 2. `SETTORI_ATTIVITA` Lunghe

**1.245 record** superano il limite Stata di 2.045 caratteri (max osservato: 5.517).

**Impatto:**
- **Stata < 13:** troncamento automatico a 2.045 char (perdita dati)
- **Stata 13+:** usare tipo `strL` per preservare interamente
- **Excel:** nessun problema (limite cella = 32.767 char)

**Raccomandazione:** usare Stata 13+ con `strL`:
```stata
import delimited "cup_clean_2024.csv", encoding(utf-8) bindquote(strict) clear
// SETTORI_ATTIVITA verrà importato come strL automaticamente in Stata 13+
```

### 3. CUP Non-Standard

**11.171 record** (0,24%) hanno CUP con formato anomalo:
- Lunghezza: min=1, max=15 (moda=15)
- Esempi: `"0000"` (4 char), valori con spaziature

**Impatto:** se usate CUP come chiave di join/linkage, validare prima.

**Raccomandazione Stata:**
```stata
// Identificare CUP anomali
gen cup_invalid = !regexm(CUP, "^[A-Z0-9]{15}$")
tab cup_invalid
// Escludere dall'analisi o trattare separatamente
```

### 4. File Aggregato Supera Limite Excel

**`cup_clean_all.csv`: 4.701.314 righe** > limite Excel (1.048.576)

**Raccomandazione:**
- Per Excel: usare **file annuali** (`cup_clean_2014.csv` ... `cup_clean_2025.csv`)
- Per Stata: importare direttamente il file aggregato oppure uno annuale per volta

---

## Istruzioni Import Stata

### Opzione 1: File singolo annuale

```stata
// Esempio: anno 2024
import delimited "data/cup_clean/cup_clean_2024.csv", encoding(utf-8) bindquote(strict) clear

// Conversione formati
destring IMPORTO_NOMINALE_TOTALE ELEMENTO_DI_AIUTO_TOTALE NUM_COMPONENTI NUM_STRUMENTI, replace double
gen data_conc = date(DATA_CONCESSIONE, "YMD")
format data_conc %td

// Verifica
describe
summarize IMPORTO_NOMINALE_TOTALE ELEMENTO_DI_AIUTO_TOTALE
```

### Opzione 2: File aggregato (più potente, richiede RAM)

```stata
// Caricamento completo (4.7M righe)
import delimited "data/cup_clean/cup_clean_all.csv", encoding(utf-8) bindquote(strict) clear

// Stesse conversioni
destring IMPORTO_NOMINALE_TOTALE ELEMENTO_DI_AIUTO_TOTALE NUM_COMPONENTI NUM_STRUMENTI, replace double
gen data_conc = date(DATA_CONCESSIONE, "YMD")
format data_conc %td
```

### Opzione 3: Loop su anni

```stata
// Carica e accumula per anni
clear
foreach year in 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 {
    append using "data/cup_clean/cup_clean_`year'.csv"
}

destring IMPORTO_NOMINALE_TOTALE ELEMENTO_DI_AIUTO_TOTALE NUM_COMPONENTI NUM_STRUMENTI, replace double
gen data_conc = date(DATA_CONCESSIONE, "YMD")
format data_conc %td
```

---

## Istruzioni Import Excel

### Per file singolo annuale

1. Apri Excel
2. **Data** → **From Text/CSV** → seleziona `cup_clean_2024.csv` (esempio)
3. **Encoding:** UTF-8
4. **Delimitatore:** Comma (virgola)
5. **Import**

⚠️ **Non usare `cup_clean_all.csv`** — supera il limite di righe.

### Per multiple anni

Opzione A: Importa ogni anno in un foglio separato
Opzione B: Usa pivot table / analisi con Power Query per aggregare

---

## Schema Colonne

| # | Colonna | Tipo | Note |
|---|---|---|---|
| 1 | `CAR` | str | Codice attività ricerca |
| 2 | `TITOLO_MISURA` | str | Titolo della misura di aiuto |
| 3 | `DES_TIPO_MISURA` | str | Descrizione tipo misura |
| 4 | `TITOLO_PROGETTO` | str | max 445 char |
| 5 | `DESCRIZIONE_PROGETTO` | str | max 1.300 char |
| 6 | `DATA_CONCESSIONE` | date | Formato `YYYY-MM-DD` |
| 7 | `CUP` | str | Codice Unico Progetto (⚠️ 11k anomali) |
| 8 | `DENOMINAZIONE_BENEFICIARIO` | str | max 390 char |
| 9 | `CODICE_FISCALE_BENEFICIARIO` | str | max 19 char |
| 10 | `DES_TIPO_BENEFICIARIO` | str | Tipo beneficiario |
| 11 | `REGIONE_BENEFICIARIO` | str | Regione o multi-valore separato da `\|` |
| 12 | `FILE_SOURCE` | str | Nome file sorgente |
| 13 | `COR` | str | ? |
| 14 | `ANNO` | int | Anno concessione (2014–2025) |
| 15 | `IMPORTO_NOMINALE_TOTALE` | double | EUR, max 2.5B |
| 16 | `ELEMENTO_DI_AIUTO_TOTALE` | double | EUR, max 2.06B |
| 17 | `NUM_COMPONENTI` | int | Numero beneficiari (1–12) |
| 18 | `NUM_STRUMENTI` | int | Numero strumenti aiuto (0–19) |
| 19 | `COD_STRUMENTI` | str | Codice strumento (11 char max) |
| 20 | `SETTORI_ATTIVITA` | strL | ATECO, max 5.517 char (⚠️ 1.2k record) |
| 21 | `OBIETTIVO` | str | Obiettivo aiuto, max 867 char |
| 22 | `SETTORE_MACRO` | str | Macro-settore estratto da ATECO |

---

## Checklist Pre-Analisi

- [ ] Dataset scelto: file annuale o aggregato?
- [ ] Stata versione: 13+ (necessario per `strL` su `SETTORI_ATTIVITA`)
- [ ] RAM disponibile: >8GB se usi file aggregato
- [ ] Encoding: UTF-8 confermato in Stata
- [ ] Record con CUP anomali: controllati/esclusi se rilevanti
- [ ] Mojibake: accettato o corretto prima analisi
- [ ] Data concessione: riconvertita in date Stata (`%td`)

---

## Conteggio Righe per Anno

```
2014:      15,434
2015:       2,106
2016:       8,132
2017:     181,968
2018:     525,825
2019:     284,698
2020:     654,762
2021:     573,143
2022:     522,647
2023:     768,756
2024:     597,320
2025:     566,523
─────────────────
TOTALE:  4,701,314
```

Tutti i file annuali sono sotto il limite Excel (1.048.576 righe).

---

**Conclusione:** Dataset pulito e pronto per Stata ed Excel. Attenzione ai 3 punti minori (mojibake, SETTORI_ATTIVITA lunghi, CUP anomali), ma nessuno è bloccante.

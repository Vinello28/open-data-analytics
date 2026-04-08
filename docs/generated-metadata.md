# Documentazione Dataset `cup_clean`

Dataset generato dal notebook `04_analisi_cup.ipynb`. Contiene i dati degli aiuti di stato italiani (2014–2025) aggregati per CUP (Codice Unico di Progetto), puliti e normalizzati per compatibilità con STATA.

## File di output

Directory: `data/cup_clean/`

| File | Righe | Dimensione |
|---|--:|--:|
| `cup_clean_2014.csv` | 15.434 | 8,5 MB |
| `cup_clean_2015.csv` | 2.106 | 1,0 MB |
| `cup_clean_2016.csv` | 8.132 | 3,4 MB |
| `cup_clean_2017.csv` | 181.968 | 81,8 MB |
| `cup_clean_2018.csv` | 525.825 | 270,4 MB |
| `cup_clean_2019.csv` | 284.698 | 115,1 MB |
| `cup_clean_2020.csv` | 654.762 | 281,6 MB |
| `cup_clean_2021.csv` | 573.143 | 271,5 MB |
| `cup_clean_2022.csv` | 522.647 | 245,7 MB |
| `cup_clean_2023.csv` | 768.756 | 320,3 MB |
| `cup_clean_2024.csv` | 597.320 | 258,5 MB |
| `cup_clean_2025.csv` | 566.523 | 263,6 MB |
| `cup_clean_all.csv` | 4.701.314 | 2.121,4 MB |

Formato: CSV, encoding UTF-8, senza indice.

## Dati di origine

I file sorgente si trovano in `data/cup_only/aggregated_cup_YYYY.csv` (12 file, uno per anno 2014–2025).

## Schema delle colonne

Il dataset pulito contiene **22 colonne** (21 originali + 1 derivata).

| # | Colonna | Tipo | Null (%) | Valori unici | Descrizione |
|--:|---|---|--:|--:|---|
| 1 | `CAR` | int64 | 0,00 | 13.245 | Codice Autorità Responsabile |
| 2 | `TITOLO_MISURA` | str | 0,00 | 10.756 | Titolo della misura di aiuto |
| 3 | `DES_TIPO_MISURA` | str | 0,00 | 2 | Tipo di misura (es. regime di aiuto, aiuto ad hoc) |
| 4 | `TITOLO_PROGETTO` | str | ~0,00 | 772.023 | Titolo del progetto finanziato |
| 5 | `DESCRIZIONE_PROGETTO` | str | ~0,00 | 758.478 | Descrizione testuale del progetto |
| 6 | `DATA_CONCESSIONE` | str | 0,00 | 3.450 | Data di concessione dell'aiuto (formato `YYYY-MM-DD`, timezone rimossa) |
| 7 | `CUP` | str | 0,00 | 1.816.649 | Codice Unico di Progetto — identificativo univoco del progetto di investimento pubblico (15 caratteri alfanumerici) |
| 8 | `DENOMINAZIONE_BENEFICIARIO` | str | 0,00 | 1.740.822 | Ragione sociale del beneficiario |
| 9 | `CODICE_FISCALE_BENEFICIARIO` | str | 0,00 | 1.768.625 | Codice fiscale o partita IVA del beneficiario |
| 10 | `DES_TIPO_BENEFICIARIO` | str | 0,00 | 3 | Tipo di beneficiario (es. PMI, grande impresa) |
| 11 | `REGIONE_BENEFICIARIO` | str | 0,00 | 503 | Regione del beneficiario (**multi-valore**, separatore `\|`) |
| 12 | `FILE_SOURCE` | str | 0,00 | 164 | File sorgente originale |
| 13 | `COR` | int64 | 0,00 | 4.698.992 | Codice identificativo del record originale |
| 14 | `ANNO` | int64 | 0,00 | 12 | Anno di riferimento (2014–2025) |
| 15 | `IMPORTO_NOMINALE_TOTALE` | float64 | 0,00 | 654.201 | Importo nominale totale dell'aiuto (euro) |
| 16 | `ELEMENTO_DI_AIUTO_TOTALE` | float64 | 0,00 | 788.639 | Elemento di aiuto equivalente (euro) |
| 17 | `NUM_COMPONENTI` | int64 | 0,00 | 12 | Numero di beneficiari coinvolti nel progetto CUP |
| 18 | `NUM_STRUMENTI` | int64 | 0,00 | 16 | Numero di strumenti di aiuto utilizzati |
| 19 | `COD_STRUMENTI` | str | 1,06 | 1.623 | Codice dello strumento di aiuto (sovvenzioni, prestiti, garanzie, ecc.) |
| 20 | `SETTORI_ATTIVITA` | str | 1,06 | 5.968 | Codici ATECO dei settori di attivita (**multi-valore**, separatore `\|`) |
| 21 | `OBIETTIVO` | str | 1,06 | 758 | Obiettivo dell'aiuto (**multi-valore**, separatore `\|` gia presente nei dati originali) |
| 22 | `SETTORE_MACRO` | str | ~1,06 | 21 | **Colonna derivata** — lettera del macro-settore ATECO, estratta dal primo codice in `SETTORI_ATTIVITA` |

## Colonne multi-valore (separatore pipe `|`)

Tre colonne contengono valori multipli separati dal carattere pipe (`|`):

### `REGIONE_BENEFICIARIO`

Quando un progetto coinvolge beneficiari in piu regioni, queste sono elencate separate da `|`.

- Esempio: `Emilia-Romagna|Provincia Autonoma di Trento`
- Record multi-valore: 22.870 (0,49% del totale)
- Valori unici (comprese combinazioni): 503

### `SETTORI_ATTIVITA`

Codici ATECO dei settori di attivita del progetto. Piu codici indicano attivita trasversali a piu settori.

- Esempio: `M.74.0|M.75.0`
- Record multi-valore: 28.143 (0,60% del totale)
- Valori unici (comprese combinazioni): 5.968
- Nota: i valori duplicati all'interno dello stesso record sono stati rimossi (64 record corretti)

### `OBIETTIVO`

Obiettivo dell'aiuto di stato. Questa colonna utilizzava gia il separatore `|` nei dati originali (non e stata modificata durante la pulizia).

- Esempio: `Ricerca, sviluppo e innovazione|Sviluppo regionale`
- Le virgole presenti in questa colonna fanno parte del testo naturale (es. "Ricerca, sviluppo e innovazione") e non sono separatori
- Valori unici (comprese combinazioni): 758

## Operazioni di pulizia applicate

Le seguenti trasformazioni sono state applicate ai dati originali (`data/cup_only/`) per produrre il dataset pulito:

1. **Rimozione timezone da `DATA_CONCESSIONE`**: il suffisso timezone (es. `+02:00`) e stato rimosso, trasformando `2014-06-30+02:00` in `2014-06-30`. Record corretti: 3.526.814.

2. **Normalizzazione separatore multi-valore in `REGIONE_BENEFICIARIO` e `SETTORI_ATTIVITA`**: la virgola-spazio (`, `) usata come separatore nei dati originali e stata sostituita con il carattere pipe (`|`). Questo evita ambiguita con le virgole presenti nel testo naturale di altri campi. I valori duplicati all'interno dello stesso record sono stati rimossi.

3. **Rimozione a-capo nei campi testuali**: i caratteri `\r\n` e `\n` sono stati sostituiti con uno spazio nei campi `TITOLO_PROGETTO` (21.221 record), `DESCRIZIONE_PROGETTO` (138.233), `DENOMINAZIONE_BENEFICIARIO` (217).

4. **Normalizzazione doppi apici**: i doppi apici (`"`) interni ai campi testuali sono stati sostituiti con apici singoli (`'`) per evitare problemi di parsing CSV. Campi interessati: `TITOLO_PROGETTO` (49.899), `DESCRIZIONE_PROGETTO` (207.760), `TITOLO_MISURA` (181.198), `DENOMINAZIONE_BENEFICIARIO` (115.900).

5. **Aggiunta colonna `SETTORE_MACRO`**: derivata estraendo la prima lettera maiuscola dal codice ATECO in `SETTORI_ATTIVITA` (es. `C.33.2` -> `C` = Manifattura).

## Note per l'import in STATA

- Usare `import delimited` con opzione `bindquote(strict)` per gestire le virgole nei campi
- La colonna `SETTORI_ATTIVITA` puo superare i 2.045 caratteri: usare il tipo `strL` (STATA 13+)
- Il formato date (`YYYY-MM-DD`) e compatibile con la funzione `date()` di STATA
- I nomi colonna sono tutti compatibili con STATA (alfanumerici con underscore, max 32 caratteri)

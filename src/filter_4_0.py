"""Filtro Industria / Transizione / Impresa 4.0 su CSV annuali.

Modulo importabile (necessario per ProcessPoolExecutor su Windows).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TEXT_COLS = ["TITOLO_MISURA", "TITOLO_PROGETTO", "DESCRIZIONE_PROGETTO"]

PATTERN_STR = (
    r"\b(?:piano\s+(?:nazionale\s+)?)?"
    r"(?:industria|transizione|impresa)\s*[-_.]?\s*4[\s.]*0\b"
)
PATTERN = re.compile(PATTERN_STR, re.IGNORECASE)
KEYWORDS = re.compile(r"industria|transizione|impresa", re.IGNORECASE)


def _match_mask(chunk: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=chunk.index)
    for col in TEXT_COLS:
        if col not in chunk.columns:
            continue
        s = chunk[col].fillna("")
        cand = s.str.contains(KEYWORDS, regex=True, na=False)
        if cand.any():
            mask |= cand & s.str.contains(PATTERN, regex=True, na=False)
    return mask


def filter_year(args: tuple[int, Path] | tuple[int, str]) -> pd.DataFrame:
    year, data_dir = args
    path = Path(data_dir) / f"classified_multiclass_aiuti_{year}.csv"
    if not path.exists():
        return pd.DataFrame()

    kept: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
        mask = _match_mask(chunk)
        if mask.any():
            kept.append(chunk.loc[mask].copy())

    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()

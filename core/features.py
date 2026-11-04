"""Copie des variables "match" et "contexte" du dépôt de classification (pour le simulateur).

À garder synchronisé avec `football_league_classification/flc/features.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAT_PAIRS = {
    "shots": ("HS", "AS"),
    "shots_on_target": ("HST", "AST"),
    "fouls": ("HF", "AF"),
    "corners": ("HC", "AC"),
    "yellow": ("HY", "AY"),
    "red": ("HR", "AR"),
}


def _safe_ratio(num, den):
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def match_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    for name, (h, a) in STAT_PAIRS.items():
        f[f"home_{name}"] = df[h]
        f[f"away_{name}"] = df[a]
        f[f"total_{name}"] = df[h] + df[a]
        f[f"diff_{name}"] = df[h] - df[a]
    f["home_goals"] = df["FTHG"]
    f["away_goals"] = df["FTAG"]
    f["total_goals"] = df["FTHG"] + df["FTAG"]
    f["diff_goals"] = df["FTHG"] - df["FTAG"]
    f["ht_total_goals"] = df["HTHG"] + df["HTAG"]
    f["second_half_goals"] = f["total_goals"] - f["ht_total_goals"]
    f["home_shot_accuracy"] = _safe_ratio(df["HST"], df["HS"])
    f["away_shot_accuracy"] = _safe_ratio(df["AST"], df["AS"])
    f["conversion_rate"] = _safe_ratio(f["total_goals"], f["total_shots_on_target"])
    f["fouls_per_card"] = f["total_fouls"] / (f["total_yellow"] + f["total_red"] + 1)
    f["corners_per_shot"] = _safe_ratio(f["total_corners"], f["total_shots"])
    return f


def context_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    f["kickoff_hour"] = df["kickoff_hour"]  # manquant avant 2019-20, pour toutes les ligues
    f["weekday"] = df["Date"].dt.weekday
    f["month"] = df["Date"].dt.month
    f["is_weekend"] = (f["weekday"] >= 5).astype(int)
    return f

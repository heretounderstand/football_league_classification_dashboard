"""Chargement des artefacts produits par le dépôt `football_league_classification`."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data"

MODEL_LABELS = {"dummy": "Baseline (fréquences)", "logreg": "Régression logistique", "lightgbm": "LightGBM"}
SET_LABELS = {
    "—": "—",
    "match": "Stats du match",
    "match_context": "+ contexte calendaire",
    "full": "+ forme récente des équipes",
}
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Accuracy équilibrée",
    "macro_f1": "F1 macro",
    "top2_accuracy": "Top-2 accuracy",
    "log_loss": "Log-loss",
    "country_accuracy": "Pays correct",
    "tier_accuracy": "Division correcte",
}


@st.cache_data
def results() -> dict:
    return json.loads((DATA / "results.json").read_text())


@st.cache_data
def legacy_audit() -> pd.DataFrame:
    return pd.DataFrame(json.loads((DATA / "legacy_audit.json").read_text()))


@st.cache_data
def coverage() -> pd.DataFrame:
    return pd.read_csv(DATA / "coverage.csv")


@st.cache_data
def predictions() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "test_predictions.parquet")
    df["correct"] = df["league"] == df["pred"]
    return df


@st.cache_data
def calibration() -> pd.DataFrame:
    return pd.read_csv(DATA / "calibration.csv")


@st.cache_data
def importance() -> pd.DataFrame:
    return pd.read_csv(DATA / "permutation_importance.csv")


@st.cache_data
def profiles() -> pd.DataFrame:
    return pd.read_csv(DATA / "league_season_profiles.csv")


@st.cache_data
def match_sample() -> pd.DataFrame:
    return pd.read_parquet(DATA / "match_sample.parquet")


@st.cache_data
def feature_ranges() -> pd.DataFrame:
    return pd.read_csv(DATA / "simulator_feature_ranges.csv", index_col=0)


@st.cache_resource
def simulator_model() -> dict:
    return joblib.load(DATA / "simulator_model.joblib")


def results_table() -> pd.DataFrame:
    """Une ligne par modèle évalué avec métriques de validation, de test et IC à 95 %."""
    rows = []
    for r in results()["results"]:
        row = {"key": r["key"], "model": r["model"], "feature_set": r["feature_set"],
               "Modèle": MODEL_LABELS[r["model"]], "Variables": SET_LABELS[r["feature_set"]]}
        for k, v in r["test"].items():
            row[k] = v
            row[f"{k}_lo"], row[f"{k}_hi"] = r["test_ci"][k]
            row[f"valid_{k}"] = r["valid"][k]
        rows.append(row)
    return pd.DataFrame(rows)


def model_label(key: str) -> str:
    if key == "baseline":
        return MODEL_LABELS["dummy"]
    model, fs = key.split("__")
    return f"{MODEL_LABELS[model]} · {SET_LABELS[fs]}"


def proba_matrix(pred: pd.DataFrame, classes: list[str]) -> np.ndarray:
    return pred[[f"p_{c}" for c in classes]].to_numpy()

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, style

R = data.results()
LEAGUES = R["leagues"]
CLASSES = R["classes"]
table = data.results_table()
preds = data.predictions()

st.title("Performance des modèles")
st.write(
    "Toutes les métriques portent sur les saisons de test, prédites en *walk-forward*. "
    "Les barres d'erreur sont des intervalles de confiance à 95 % (bootstrap sur les matchs)."
)

# --- Comparaison --------------------------------------------------------------
metric = st.segmented_control(
    "Métrique", list(data.METRIC_LABELS), default="accuracy",
    format_func=data.METRIC_LABELS.get) or "accuracy"
model_colors = {"dummy": style.ink("muted"), "logreg": "#eb6834" if style.mode() == "light" else "#d95926",
                "lightgbm": style.ink("accent")}
fig = go.Figure()
order = ["match", "match_context", "full"]
for model in ["dummy", "logreg", "lightgbm"]:
    t = table[table["model"] == model]
    if model == "dummy":
        ys = ["Baseline"]
    else:
        t = t.set_index("feature_set").loc[order].reset_index()
        ys = [data.SET_LABELS[f] for f in t["feature_set"]]
    fig.add_trace(go.Scatter(
        x=t[metric], y=ys, mode="markers", name=data.MODEL_LABELS[model],
        marker=dict(size=12, color=model_colors[model], line=dict(width=2, color="rgba(0,0,0,0)")),
        error_x=dict(type="data", symmetric=False, array=t[f"{metric}_hi"] - t[metric],
                     arrayminus=t[metric] - t[f"{metric}_lo"], color=model_colors[model], thickness=2, width=0),
        customdata=np.stack([t[f"{metric}_lo"], t[f"{metric}_hi"], t[f"valid_{metric}"]], axis=1),
        hovertemplate=(f"{data.MODEL_LABELS[model]} · %{{y}}<br>{data.METRIC_LABELS[metric]} : %{{x:.3f}}"
                       "<br>IC 95 % : %{customdata[0]:.3f} – %{customdata[1]:.3f}"
                       "<br>Validation : %{customdata[2]:.3f}<extra></extra>"),
    ))
style.layout(fig, height=320, left=210)
fig.update_yaxes(categoryorder="array", categoryarray=["Baseline"] + [data.SET_LABELS[f] for f in order],
                 autorange="reversed", showgrid=False)
fig.update_xaxes(title=data.METRIC_LABELS[metric] + (" (plus bas = mieux)" if metric == "log_loss" else ""),
                 tickformat=".3f" if metric == "log_loss" else ".0%")
style.show(fig)

st.divider()

# --- Détail d'un modèle -------------------------------------------------------
keys = [r["key"] for r in R["results"]]
key = st.selectbox("Modèle à détailler", keys, index=keys.index(R["best_model"]), format_func=data.model_label)
p = preds[preds["model_key"] == key]
res = next(r for r in R["results"] if r["key"] == key)

cols = st.columns(len(res["test_by_season"]) + 2)
for c, (season, m) in zip(cols, res["test_by_season"].items()):
    c.metric(f"Accuracy {season}", f"{100 * m['accuracy']:.1f} %")
errors = p[~p["correct"]]
same_country = (errors["league"].map(lambda l: LEAGUES[l]["country"])
                == errors["pred"].map(lambda l: LEAGUES[l]["country"])).mean()
cols[-2].metric("Erreurs dans le bon pays", f"{100 * same_country:.0f} %",
                help="Parmi les matchs mal classés, part où la ligue prédite est du même pays (D1 ↔ D2).")
cols[-1].metric("Confiance moyenne", f"{100 * p['confidence'].mean():.0f} %")

left, right = st.columns([1.25, 1], gap="large")
with left:
    st.subheader("Matrice de confusion")
    normalize = st.toggle("Normaliser par ligue réelle (rappel)", value=True)
    cm = pd.crosstab(p["league"], p["pred"]).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    counts = cm.to_numpy()
    zmat = counts / counts.sum(axis=1, keepdims=True) if normalize else counts
    txt = [[(f"{v:.0%}" if normalize else f"{v:d}") if v else "" for v in row] for row in zmat]
    fig = go.Figure(go.Heatmap(
        z=zmat, x=CLASSES, y=CLASSES, text=txt, texttemplate="%{text}", xgap=2, ygap=2,
        colorscale=[[i / (len(style.SEQUENTIAL) - 1), c] for i, c in enumerate(style.SEQUENTIAL)],
        customdata=counts, showscale=False,
        hovertemplate="Réel : %{y}<br>Prédit : %{x}<br>%{customdata} matchs<extra></extra>",
    ))
    style.layout(fig, height=560, left=130, bottom=120)
    fig.update_yaxes(autorange="reversed", title="Ligue réelle", showgrid=False)
    fig.update_xaxes(title="Ligue prédite", tickangle=-40, showgrid=False)
    style.show(fig)
with right:
    st.subheader("Par ligue")
    by = []
    for lg in CLASSES:
        tp = ((p["league"] == lg) & (p["pred"] == lg)).sum()
        by.append({"league": lg, "Rappel": tp / max((p["league"] == lg).sum(), 1),
                   "Précision": tp / max((p["pred"] == lg).sum(), 1)})
    by = pd.DataFrame(by)
    which = st.segmented_control("Mesure", ["Rappel", "Précision"], default="Rappel") or "Rappel"
    fig = go.Figure(go.Bar(
        x=by[which], y=by["league"], orientation="h",
        marker=dict(color=[style.league_color(l, LEAGUES) for l in by["league"]],
                    pattern=dict(shape=[style.tier_pattern(LEAGUES[l]["tier"]) for l in by["league"]],
                                 fgcolor=style.ink("primary"), size=6, solidity=0.25),
                    cornerradius=4),
        text=[f"{v:.0%}" for v in by[which]], textposition="outside", cliponaxis=False,
        textfont=dict(color=style.ink("primary")),
        hovertemplate="%{y}<br>" + which + " : %{x:.1%}<extra></extra>",
    ))
    style.layout(fig, height=470, left=130, right=50, showlegend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, 1.1], tickformat=".0%")
    style.show(fig)
    style.legend_note()

st.subheader("Les probabilités sont-elles fiables ?")
cal = data.calibration()
cal = cal[(cal["model_key"] == key) & (cal["count"] >= 20)]
c1, c2 = st.columns([1.3, 1], gap="large")
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite",
                             line=dict(color=style.ink("muted"), dash="dot", width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=cal["confidence"], y=cal["accuracy"], mode="lines+markers", name="Modèle",
        line=dict(color=style.ink("accent"), width=2), marker=dict(size=9),
        customdata=cal["count"],
        hovertemplate="Confiance moyenne %{x:.0%}<br>Taux de réussite %{y:.0%}<br>%{customdata} matchs<extra></extra>",
    ))
    style.layout(fig, height=360, left=70, bottom=60)
    fig.update_xaxes(title="Probabilité attribuée à la ligue prédite", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title="Taux de bonnes réponses", range=[0, 1], tickformat=".0%")
    style.show(fig)
with c2:
    gap = float(((cal["confidence"] - cal["accuracy"]) * cal["count"]).sum() / cal["count"].sum())
    st.write(
        "Chaque point regroupe les matchs selon la confiance du modèle. Sur la diagonale, un « 60 % » "
        "correspondrait exactement à 60 % de bonnes réponses. "
        + (f"Ici, les points sont **sous la diagonale** : le modèle est un peu trop sûr de lui "
           f"(en moyenne {100 * gap:.0f} points de confiance en trop). C'est l'effet de la dérive entre les "
           "saisons ; un recalibrage sur la saison la plus récente serait une piste d'amélioration."
           if gap > 0.03 else "Les probabilités sont bien calibrées.")
    )
    st.metric("Log-loss (test)", f"{res['test']['log_loss']:.3f}",
              f"{res['test']['log_loss'] - table.set_index('key').loc['baseline', 'log_loss']:+.3f} vs baseline",
              delta_color="inverse")

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, style

R = data.results()
LEAGUES = R["leagues"]
CLASSES = R["classes"]

STAT_FR = {"shots_on_target": "tirs cadrés", "shots": "tirs", "fouls": "fautes", "corners": "corners",
           "yellow": "cartons jaunes", "red": "cartons rouges", "goals": "buts"}
PREFIX_FR = {"home_": "domicile · ", "away_": "extérieur · ", "total_": "total · ", "diff_": "écart dom.-ext. · "}
SPECIAL = {
    "kickoff_hour": "Heure du coup d'envoi", "weekday": "Jour de la semaine", "month": "Mois",
    "is_weekend": "Week-end", "ht_total_goals": "Buts à la mi-temps", "second_half_goals": "Buts en 2e période",
    "home_shot_accuracy": "Précision des tirs · domicile", "away_shot_accuracy": "Précision des tirs · extérieur",
    "conversion_rate": "Buts / tir cadré", "fouls_per_card": "Fautes par carton", "corners_per_shot": "Corners par tir",
    "form_own_fouls": "Forme · fautes commises par l'équipe", "form_own_yellow": "Forme · cartons de l'équipe",
    "form_n_prev": "Forme · nb de matchs d'historique",
}


def feature_fr(name: str) -> str:
    if name in SPECIAL:
        return SPECIAL[name]
    if name.startswith("form_tot_"):
        return f"Forme · {STAT_FR.get(name[9:], name[9:])} par match"
    for pre, fr in PREFIX_FR.items():
        if name.startswith(pre):
            stat = name[len(pre):]
            return f"{STAT_FR.get(stat, stat).capitalize()} ({fr.strip(' ·')})"
    return name


st.title("Ce qui distingue les ligues")
st.write(
    "Importance par permutation : on mélange une variable (ou un groupe de variables) entre les matchs et on "
    "mesure de combien la log-loss se dégrade sur la saison 2022-23. Un groupe est permuté d'un bloc, ce qui "
    "évite de sous-estimer des variables corrélées (tirs domicile / extérieur / total…)."
)

imp = data.importance()
keys = [k for k in imp["model_key"].unique()]
key = st.selectbox("Modèle", keys, index=keys.index(R["best_model"]) if R["best_model"] in keys else 0,
                   format_func=data.model_label)
imp = imp[imp["model_key"] == key]

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Par famille de variables")
    g = imp[imp["kind"] == "group"].sort_values("importance", ascending=False)
    fig = go.Figure(go.Bar(
        x=g["importance"].clip(lower=0), y=g["name"], orientation="h",
        marker=dict(color=style.ink("accent"), cornerradius=4),
        error_x=dict(type="data", array=g["std"], color=style.ink("muted"), thickness=1.5, width=0),
        text=[f"{v:+.3f}" if v > 0.0005 else "≈ 0" for v in g["importance"]],
        textposition="outside", cliponaxis=False,
        textfont=dict(color=style.ink("primary")),
        customdata=g["importance"],
        hovertemplate="%{y}<br>%{customdata:+.4f} de log-loss<extra></extra>",
    ))
    style.layout(fig, height=380, left=150, right=60, bottom=60, showlegend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Hausse de la log-loss")
    style.show(fig)
    st.caption("« ≈ 0 » : la famille n'apporte rien de plus que les autres variables du modèle.")
with right:
    st.subheader("Variables individuelles (top 15)")
    f = imp[imp["kind"] == "feature"].nlargest(15, "importance")
    fig = go.Figure(go.Bar(
        x=f["importance"], y=[feature_fr(n) for n in f["name"]], orientation="h",
        marker=dict(color=style.ink("accent"), cornerradius=4),
        customdata=f[["name", "group"]].to_numpy(),
        hovertemplate="%{y}<br>%{customdata[0]} · %{customdata[1]}<br>+%{x:.4f} de log-loss<extra></extra>",
    ))
    style.layout(fig, height=380, left=260, bottom=60, showlegend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont_size=11)
    fig.update_xaxes(title="Hausse de la log-loss")
    style.show(fig)

st.info(
    "**À retenir.** L'**heure du coup d'envoi** et le **calendrier** portent beaucoup d'information "
    "(un tiers des matchs de LaLiga débutent à 20 h ou plus tard, le vendredi pèse ~20 % en Bundesliga 2 et en "
    "Ligue 2, et 42 % des matchs du mardi ou mercredi soir sont des matchs de Championship…). C'est un signal réel, mais il relève de la programmation TV plus que du jeu. Parmi les "
    "statistiques de jeu, ce sont les **fautes et cartons** (culture arbitrale) qui distinguent le mieux les ligues, "
    "loin devant les tirs et les corners. La **forme récente** agrège plusieurs matchs du même championnat, ce qui "
    "lisse le bruit d'un match isolé. Une fois qu'elle est là, les stats du match courant n'ajoutent presque plus rien.",
    icon=":material/lightbulb:",
)

st.subheader("Les confusions les plus fréquentes")
preds = data.predictions()
p = preds[(preds["model_key"] == key) & ~preds["correct"]]
pairs = (p.groupby(["league", "pred"]).size().rename("n").reset_index()
         .assign(share=lambda d: d["n"] / preds[preds["model_key"] == key].groupby("league").size()
                 .reindex(d["league"]).to_numpy()))
pairs["Même pays"] = [LEAGUES[a]["country"] == LEAGUES[b]["country"] for a, b in zip(pairs["league"], pairs["pred"])]
top = pairs.nlargest(10, "n").rename(columns={"league": "Ligue réelle", "pred": "Prédite comme",
                                             "n": "Matchs", "share": "Part des matchs de la ligue"})
st.dataframe(top, hide_index=True, use_container_width=True,
             column_config={"Part des matchs de la ligue": st.column_config.ProgressColumn(
                 format="percent", min_value=0, max_value=float(max(top["Part des matchs de la ligue"].max(), 0.01)))})

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core import data, style

R = data.results()
LEAGUES = R["leagues"]
CLASSES = R["classes"]
STATS = ["Tirs", "Tirs cadrés", "Fautes", "Corners", "Cartons jaunes", "Cartons rouges", "Buts"]

st.title("Données & styles de jeu")
st.write("Source : football-data.co.uk, 10 championnats, saisons 1993-94 à 2022-23.")

# --- Couverture ---------------------------------------------------------------
st.subheader("Quand les statistiques existent-elles ?")
cov = data.coverage()
grid = cov.pivot(index="league", columns="season", values="coverage").reindex(CLASSES[::-1])
fig = go.Figure(go.Heatmap(
    z=grid.values, x=list(grid.columns), y=grid.index,
    customdata=[[f"{s}-{str(s + 1)[-2:]}" for s in grid.columns]] * len(grid),
    colorscale=[[0, style.SEQUENTIAL[0]], [1, style.SEQUENTIAL[5]]], zmin=0, zmax=1,
    xgap=2, ygap=2, colorbar=dict(title="Part des matchs<br>avec stats", tickformat=".0%", len=0.8),
    hovertemplate="%{y} · %{customdata}<br>%{z:.0%} des matchs avec statistiques<extra></extra>",
))
fig.add_vrect(x0=2016.5, x1=max(grid.columns) + 0.5, line=dict(color=style.ink("primary"), width=1.5),
              fillcolor="rgba(0,0,0,0)")
fig.add_annotation(x=2016.5, y=1.0, yref="paper", yanchor="bottom", xanchor="left", showarrow=False,
                   text="Fenêtre de modélisation", font=dict(color=style.ink("primary")))
style.layout(fig, height=400, left=130, top=50)
fig.update_xaxes(showgrid=False, tickmode="array", tickvals=list(grid.columns)[::3],
                 ticktext=[f"{s}-{str(s + 1)[-2:]}" for s in list(grid.columns)[::3]])
fig.update_yaxes(showgrid=False)
style.show(fig)
st.caption(
    "Avant 2017-18, les divisions 2 (hors Championship), puis LaLiga, Serie A et Ligue 1 jusqu'au "
    "milieu des années 2000, n'ont pas de statistiques. L'ancien pipeline remplaçait ces trous par 0 : "
    "la présence des données trahissait la ligue. Seule la fenêtre encadrée est utilisée."
)

# --- Profils ------------------------------------------------------------------
st.subheader("La carte d'identité statistique de chaque ligue")
prof = data.profiles()
mean = prof.groupby("league")[STATS + ["kickoff_hour"]].mean().reindex(CLASSES)
mean = mean.rename(columns={"kickoff_hour": "Heure du coup d'envoi"})
z = (mean - mean.mean()) / mean.std()
text = mean.map(lambda v: f"{v:.2f}" if v < 1 else f"{v:.1f}")
fig = go.Figure(go.Heatmap(
    z=z.values, x=z.columns, y=z.index, text=text.values, texttemplate="%{text}",
    colorscale=style.diverging(), zmid=0, zmin=-2.2, zmax=2.2, xgap=2, ygap=2,
    colorbar=dict(title="Écart à la<br>moyenne (σ)", len=0.8),
    hovertemplate="%{y}<br>%{x} : %{text} par match<br>%{z:+.2f} σ vs les 10 ligues<extra></extra>",
))
style.layout(fig, height=460, left=130, bottom=60)
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(showgrid=False)
style.show(fig)
st.caption(
    "Valeurs : moyenne par match (2017-18 → 2022-23, heure : depuis 2019-20). Rouge = au-dessus de la "
    "moyenne des 10 ligues, bleu = en dessous. La Premier League siffle peu de fautes, "
    "l'Espagne et l'Italie distribuent beaucoup de cartons, la Serie B cumule fautes et cartons."
)

# --- Distribution d'une statistique ------------------------------------------
st.subheader("Un match isolé est bruité")
stat = st.segmented_control("Statistique", STATS, default="Fautes", key="dist_stat") or "Fautes"
sample = data.match_sample()
fig = go.Figure()
for lg in CLASSES:
    v = sample.loc[sample["league"] == lg, stat]
    fig.add_trace(go.Box(
        x=v, name=lg, orientation="h", marker_color=style.league_color(lg, LEAGUES),
        line_width=1.5, boxpoints=False, fillcolor="rgba(0,0,0,0)" if LEAGUES[lg]["tier"] == 2 else None,
        hovertemplate=f"{lg}<br>%{{x}}<extra></extra>",
    ))
style.layout(fig, height=440, left=130, showlegend=False)
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(title=f"{stat} par match")
style.show(fig)
st.caption(
    "Distribution sur 600 matchs par ligue (hors saisons de test). Les boîtes se chevauchent largement : "
    "d'où une accuracy modeste avec les seules stats du match. Couleur = pays, boîte vide = division 2."
)

# --- Évolution ----------------------------------------------------------------
st.subheader("Les styles (et les données) évoluent")
stat2 = st.segmented_control("Statistique ", STATS, default="Tirs", key="evo_stat") or "Tirs"
countries = style.COUNTRIES
fig = make_subplots(rows=1, cols=5, shared_yaxes=True, subplot_titles=countries, horizontal_spacing=0.02)
for i, country in enumerate(countries, start=1):
    for lg in [l for l in CLASSES if LEAGUES[l]["country"] == country]:
        d = prof[prof["league"] == lg].sort_values("season")
        tier = LEAGUES[lg]["tier"]
        fig.add_trace(go.Scatter(
            x=d["season_label"], y=d[stat2], name=lg, mode="lines+markers",
            line=dict(color=style.country_color(country), width=2, dash=style.tier_dash(tier)),
            marker=dict(size=8, symbol="circle" if tier == 1 else "circle-open"),
            hovertemplate=f"{lg} · %{{x}}<br>{stat2} : %{{y:.2f}}<extra></extra>",
        ), row=1, col=i)
style.layout(fig, height=420, left=50, bottom=150, top=30)
fig.update_layout(legend=dict(orientation="h", y=-0.32, yanchor="top", x=0))
fig.update_xaxes(tickangle=-45, showgrid=False, tickfont_size=10)
fig.for_each_annotation(lambda a: a.update(font=dict(color=style.ink("primary"))))
style.show(fig)
style.legend_note()
st.caption(
    "Certaines ruptures viennent des données elles-mêmes (ex. tirs en Serie A : 25,6 → 20,4 entre 2017-18 "
    "et 2018-19, puis retour à ~26 en 2021-22, vraisemblablement un changement de fournisseur). La saison 2020-21, "
    "à huis clos, est atypique. C'est pourquoi l'évaluation se fait dans le temps, avec ré-entraînement annuel."
)

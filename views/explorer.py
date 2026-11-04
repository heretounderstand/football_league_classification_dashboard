import plotly.graph_objects as go
import streamlit as st

from core import data, style

R = data.results()
LEAGUES = R["leagues"]
CLASSES = R["classes"]

st.title("Explorer les prédictions")
st.write("Les matchs des saisons de test, avec ce que le modèle en a pensé.")

preds = data.predictions()
keys = [r["key"] for r in R["results"] if r["key"] != "baseline"]
f1, f2, f3, f4 = st.columns([1.6, 1, 1.4, 1])
key = f1.selectbox("Modèle", keys, index=keys.index(R["best_model"]), format_func=data.model_label)
season = f2.selectbox("Saison", ["Toutes"] + sorted(preds["season_label"].unique()))
leagues = f3.multiselect("Ligue réelle", CLASSES, placeholder="Toutes")
outcome = f4.selectbox("Résultat", ["Tous", "Bien classés", "Mal classés"])

p = preds[preds["model_key"] == key]
if season != "Toutes":
    p = p[p["season_label"] == season]
if leagues:
    p = p[p["league"].isin(leagues)]
if outcome != "Tous":
    p = p[p["correct"] == (outcome == "Bien classés")]
min_conf = st.slider("Confiance minimale du modèle", 0.0, 1.0, 0.0, 0.05, format="%.2f")
p = p[p["confidence"] >= min_conf]

c1, c2, c3 = st.columns(3)
c1.metric("Matchs", f"{len(p):,}".replace(",", " "))
c2.metric("Accuracy sur la sélection", f"{100 * p['correct'].mean():.1f} %" if len(p) else "—")
c3.metric("Confiance moyenne", f"{100 * p['confidence'].mean():.0f} %" if len(p) else "—")
st.caption("Astuce : filtrez sur une confiance élevée. Plus le modèle est sûr de lui, plus il a raison "
           "(voir la calibration).")

view = p.assign(Match=p["HomeTeam"] + " – " + p["AwayTeam"],
                Score=p["FTHG"].astype(int).astype(str) + "-" + p["FTAG"].astype(int).astype(str),
                ok=p["correct"].map({True: "✓", False: "✗"}))
view = view[["Date", "Match", "Score", "league", "pred", "confidence", "ok", "match_id"]].sort_values("Date")
event = st.dataframe(
    view.drop(columns="match_id"), hide_index=True, use_container_width=True, height=360,
    on_select="rerun", selection_mode="single-row",
    column_config={
        "Date": st.column_config.DateColumn(format="DD/MM/YYYY"),
        "league": "Ligue réelle", "pred": "Ligue prédite", "ok": "",
        "confidence": st.column_config.ProgressColumn("Confiance", format="percent", min_value=0, max_value=1),
    },
)

rows = event.selection.rows if event and event.selection else []
if not len(view):
    st.stop()
row = view.iloc[rows[0]] if rows else view.iloc[0]
m = p[p["match_id"] == row["match_id"]].iloc[0]
st.subheader(f"{m['HomeTeam']} {int(m['FTHG'])}-{int(m['FTAG'])} {m['AwayTeam']}")
st.caption(f"{m['Date']:%d/%m/%Y} · {m['league']} · {m['season_label']}"
           + ("" if rows else " · sélectionnez une ligne du tableau pour changer de match"))
probs = sorted(((c, float(m[f"p_{c}"])) for c in CLASSES), key=lambda t: -t[1])
fig = go.Figure(go.Bar(
    x=[v for _, v in probs], y=[c for c, _ in probs], orientation="h",
    marker=dict(color=[style.league_color(c, LEAGUES) if c == m["league"] else style.ink("neutral")
                       for c, _ in probs], cornerradius=4),
    text=[f"{v:.0%}" + ("  ← ligue réelle" if c == m["league"] else "") for c, v in probs],
    textposition="outside", cliponaxis=False, textfont=dict(color=style.ink("primary")),
    hovertemplate="%{y} : %{x:.1%}<extra></extra>",
))
style.layout(fig, height=380, left=130, right=120, showlegend=False)
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(range=[0, min(1.0, probs[0][1] * 1.45 + 0.05)], tickformat=".0%", title="Probabilité")
style.show(fig)

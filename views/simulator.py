import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data, style
from core.features import context_features, match_features

R = data.results()
LEAGUES = R["leagues"]
CLASSES = R["classes"]
bundle = data.simulator_model()
presets = pd.read_csv(data.DATA / "simulator_presets.csv", index_col=0)

st.title("Simulateur")
st.write(
    "Saisissez les statistiques d'un match imaginaire : le modèle *LightGBM · stats du match + contexte "
    "calendaire*, ré-entraîné sur toutes les saisons, estime de quelle ligue il provient."
)

WEEKDAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
MONTHS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre",
          "Octobre", "Novembre", "Décembre"]
FIELDS = [  # (colonne brute, libellé, max)
    ("HS", "AS", "Tirs", 40), ("HST", "AST", "Tirs cadrés", 20), ("HF", "AF", "Fautes", 35),
    ("HC", "AC", "Corners", 20), ("HY", "AY", "Cartons jaunes", 10), ("HR", "AR", "Cartons rouges", 4),
]


def apply_preset():
    lg = st.session_state.get("preset")
    if not lg:
        return
    row = presets.loc[lg]
    for col in presets.columns:
        if col in ("weekday", "month", "home", "away", "date"):
            continue
        st.session_state[col] = int(row[col])
    st.session_state["weekday"] = WEEKDAYS[int(row["weekday"])]
    st.session_state["month"] = MONTHS[int(row["month"]) - 1]


if "HS" not in st.session_state:
    st.session_state["preset"] = "Premier League"
    apply_preset()

st.selectbox("Partir d'un match réel de…", CLASSES, key="preset", on_change=apply_preset,
             help="Le match de 2022-23 dont les statistiques sont les plus proches du profil médian de la ligue.")
pr = presets.loc[st.session_state["preset"]]
st.caption(f"Préréglage : {pr['home']} – {pr['away']} ({pd.Timestamp(pr['date']):%d/%m/%Y}). "
           "Modifiez ensuite les valeurs librement.")

left, right = st.columns([1.1, 1], gap="large")
with left:
    h, a = st.columns(2)
    h.markdown("**Domicile**")
    a.markdown("**Extérieur**")
    for hc, ac, label, mx in FIELDS:
        h.number_input(label, 0, mx, key=hc)
        a.number_input(label + " ", 0, mx, key=ac)
    g1, g2, g3, g4 = st.columns(4)
    g1.number_input("Buts dom.", 0, 10, key="FTHG")
    g2.number_input("Buts ext.", 0, 10, key="FTAG")
    g3.number_input("Mi-temps dom.", 0, 10, key="HTHG")
    g4.number_input("Mi-temps ext.", 0, 10, key="HTAG")
    t1, t2, t3 = st.columns(3)
    t1.slider("Heure du coup d'envoi", 11, 22, key="kickoff_hour", help="Heure britannique, comme dans la source.")
    t2.selectbox("Jour", WEEKDAYS, key="weekday")
    t3.selectbox("Mois", MONTHS, key="month")

s = st.session_state
problems = []
if s["HST"] > s["HS"] or s["AST"] > s["AS"]:
    problems.append("plus de tirs cadrés que de tirs")
if s["HTHG"] > s["FTHG"] or s["HTAG"] > s["FTAG"]:
    problems.append("plus de buts à la mi-temps qu'à la fin")

raw = pd.DataFrame([{c: float(s[c]) for c in
                     ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
                      "FTHG", "FTAG", "HTHG", "HTAG"]}])
raw["kickoff_hour"] = float(s["kickoff_hour"])
# date fictive tombant le bon jour et le bon mois
month = MONTHS.index(s["month"]) + 1
base = pd.Timestamp(year=2022 if month >= 7 else 2023, month=month, day=1)
raw["Date"] = base + pd.Timedelta(days=(WEEKDAYS.index(s["weekday"]) - base.weekday()) % 7)
X = pd.concat([match_features(raw), context_features(raw)], axis=1)[bundle["features"]].astype(float)
proba = bundle["model"].predict_proba(X)[0]

with right:
    if problems:
        st.warning("Match incohérent : " + ", ".join(problems) + ".", icon=":material/warning:")
    order = proba.argsort()[::-1]
    top = CLASSES[order[0]]
    st.metric("Ligue la plus probable", top)
    st.caption(f"{100 * proba[order[0]]:.0f} % de probabilité · ligue du préréglage : "
               f"{100 * proba[CLASSES.index(st.session_state['preset'])]:.0f} %")
    fig = go.Figure(go.Bar(
        x=proba[order], y=[CLASSES[i] for i in order], orientation="h",
        marker=dict(color=[style.league_color(CLASSES[i], LEAGUES) for i in order],
                    pattern=dict(shape=[style.tier_pattern(LEAGUES[CLASSES[i]]["tier"]) for i in order],
                                 fgcolor=style.ink("primary"), size=6, solidity=0.25),
                    cornerradius=4),
        text=[f"{v:.0%}" for v in proba[order]], textposition="outside", cliponaxis=False,
        textfont=dict(color=style.ink("primary")),
        hovertemplate="%{y} : %{x:.1%}<extra></extra>",
    ))
    style.layout(fig, height=420, left=130, right=50, showlegend=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(range=[0, min(1, proba.max() * 1.35 + 0.05)], tickformat=".0%")
    style.show(fig)
    style.legend_note()
    st.caption("Même un match « médian » est souvent mal classé : l'heure et le jour pèsent lourd "
               "(un mardi soir évoque le Championship, un vendredi la Bundesliga 2 ou la Ligue 2…). Essayez de changer "
               "l'heure ou le nombre de cartons, et regardez comment les probabilités basculent.")

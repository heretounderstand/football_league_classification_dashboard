import plotly.graph_objects as go
import streamlit as st

from core import data, style

R = data.results()
table = data.results_table().set_index("key")
best = table.loc[R["best_model"]]
base = table.loc["baseline"]
match_only = table.loc["lightgbm__match"]

st.title("Quelle ligue ?")
st.markdown(
    "#### Peut-on reconnaître la ligue d'un match (parmi les 10 premières et deuxièmes divisions "
    "d'Angleterre, d'Espagne, d'Italie, d'Allemagne et de France) à partir de ses seules statistiques ?"
)
st.write(
    f"Le modèle retenu, **{data.model_label(R['best_model'])}**, est évalué sur les saisons "
    f"{' et '.join(R['splits']['test']['seasons'])} ({R['splits']['test']['n']:,} matchs) "
    "sans les avoir vues pendant l'entraînement.".replace(",", " ")
)


def pct(x):
    return f"{100 * x:.1f} %"


c1, c2, c3, c4 = st.columns(4)
c1.metric("Ligue correcte", pct(best["accuracy"]),
          f"{100 * (best['accuracy'] - base['accuracy']):+.1f} pts vs baseline",
          help=f"IC 95 % (bootstrap) : {pct(best['accuracy_lo'])} – {pct(best['accuracy_hi'])}. "
               f"La baseline prédit toujours la ligue la plus fréquente ({pct(base['accuracy'])}).")
c2.metric("Bonne ligue dans le top 2", pct(best["top2_accuracy"]))
c3.metric("Pays correct", pct(best["country_accuracy"]), help="Sur 5 pays (hasard ≈ 20 %).")
c4.metric("Stats du match seules", pct(match_only["accuracy"]),
          help="LightGBM avec uniquement les statistiques du match (tirs, fautes, cartons, corners, buts).")

st.divider()

left, right = st.columns([1.15, 1], gap="large")
with left:
    st.subheader("Pourquoi l'ancien score de 96 % était trompeur")
    audit = data.legacy_audit()
    audit = audit[audit["id"].isin(["A", "A0", "A1", "B", "C", "D"])].iloc[::-1]
    labels = {
        "A": "Protocole d'origine",
        "A0": "Seulement la « force » des équipes",
        "A1": "Seulement saison + « stats manquantes ? »",
        "B": "Sans la force des équipes",
        "C": "… et seulement les vraies stats",
        "D": "… et découpage temporel",
    }
    fig = go.Figure(go.Bar(
        x=audit["accuracy"], y=[labels[i] for i in audit["id"]], orientation="h",
        marker=dict(color=[style.ink("accent") if i in ("A", "A0") else style.ink("neutral")
                           for i in audit["id"]], cornerradius=4),
        text=[pct(v) for v in audit["accuracy"]], textposition="outside",
        textfont=dict(color=style.ink("primary")),
        hovertemplate="%{y}<br>Accuracy %{x:.1%}<extra></extra>", cliponaxis=False,
    ))
    fig.add_vline(x=base["accuracy"], line=dict(color=style.ink("muted"), dash="dot", width=1.5),
                  annotation_text="baseline", annotation_position="top",
                  annotation_font_color=style.ink("muted"))
    style.layout(fig, height=320, left=250, right=60)
    fig.update_xaxes(range=[0, 1.08], tickformat=".0%", showgrid=True)
    fig.update_yaxes(showgrid=False)
    style.show(fig)
with right:
    st.markdown(
        """
**Deux fuites de données expliquaient l'essentiel du score :**

1. **La « force » des équipes** (points par match calculés sur *tout* le jeu de données,
   test compris) donne à chaque club une valeur unique. Le modèle reconnaissait donc le club,
   et le club donne la ligue. Cette seule variable atteint ~87 %.
2. **Les statistiques manquantes remplacées par 0.** Avant 2017, plusieurs divisions 2 n'ont
   aucune statistique : « 0 tir, 0 faute » signifiait en réalité « Ligue 2, Serie B, LaLiga 2… ».

S'y ajoutait un **découpage aléatoire**, qui mélange les matchs d'une même saison entre
l'entraînement et le test. Une fois ces biais retirés, il reste le vrai signal, plus modeste
mais réel : le style de jeu propre à chaque championnat.
"""
    )

st.divider()
st.subheader("La nouvelle méthode")
m1, m2, m3 = st.columns(3, gap="large")
with m1:
    st.markdown("**1 · Données propres**")
    st.write(
        f"{R['n_raw_matches']:,} matchs bruts (1993-2023), dont {R['n_raw_with_stats']:,} avec des statistiques. "
        f"On garde les **{R['n_modelling_matches']:,} matchs** des saisons où les 10 ligues sont "
        "renseignées (2017-18 → 2022-23). Aucune imputation, et aucun nom d'équipe ni d'arbitre "
        "parmi les variables.".replace(",", " ")
    )
with m2:
    st.markdown("**2 · Découpage dans le temps**")
    s = R["splits"]
    st.write(
        f"Entraînement : {', '.join(s['train']['seasons'])}. Réglage des hyperparamètres sur "
        f"{', '.join(s['valid']['seasons'])}. Test en *walk-forward* sur {', '.join(s['test']['seasons'])} : "
        "chaque saison est prédite par un modèle ré-entraîné sur toutes les saisons précédentes."
    )
with m3:
    st.markdown("**3 · Trois niveaux d'information**")
    st.write(
        "Stats du match seules → + contexte calendaire (heure du coup d'envoi, jour, mois) → "
        "+ forme récente : moyennes des 5 matchs *précédents* des deux équipes, sans jamais "
        "utiliser le match courant ni le futur."
    )

st.subheader("Résultats sur le test")
show = data.results_table()
pct_cols = [k for k in data.METRIC_LABELS if k != "log_loss"]
show[pct_cols] = show[pct_cols] * 100
show = show[["Modèle", "Variables"] + pct_cols + ["log_loss"]]
st.dataframe(
    show.rename(columns=data.METRIC_LABELS),
    hide_index=True, use_container_width=True,
    column_config={data.METRIC_LABELS[k]: st.column_config.NumberColumn(format="%.1f %%") for k in pct_cols}
    | {"Log-loss": st.column_config.NumberColumn(format="%.3f")},
)
st.caption("Modèle retenu : meilleure log-loss sur la saison de validation. "
           "Détails, intervalles de confiance et matrices de confusion : page *Performance des modèles*.")

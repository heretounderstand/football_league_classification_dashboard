"""Dashboard : peut-on reconnaître la ligue d'un match à partir de ses statistiques ?"""
import streamlit as st

st.set_page_config(page_title="Quelle ligue ? · Classification de matchs", page_icon="⚽", layout="wide")

pages = [
    st.Page("views/home.py", title="Le projet", icon=":material/sports_soccer:", default=True),
    st.Page("views/data.py", title="Données & styles de jeu", icon=":material/table_chart:"),
    st.Page("views/performance.py", title="Performance des modèles", icon=":material/leaderboard:"),
    st.Page("views/explain.py", title="Ce qui distingue les ligues", icon=":material/insights:"),
    st.Page("views/explorer.py", title="Explorer les prédictions", icon=":material/search:"),
    st.Page("views/simulator.py", title="Simulateur", icon=":material/tune:"),
]
st.navigation(pages).run()

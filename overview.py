import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Football League Classification Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Dashboard d'analyse et prédiction des ligues de football"
    }
)

# Custom CSS pour améliorer l'UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f7ff;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        flex: 1;
        min-width: 150px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1565C0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
        margin-top: 5px;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 10px;
        color: #666;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger les données
@st.cache_data
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Fichier non trouvé: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

# En-tête principal avec logo et titre
col_logo, col_title = st.columns([2, 5])
with col_logo:
    st.image("image/top-5.jpg", width=275)
with col_title:
    st.markdown('<h1 class="main-header">Classification des Ligues de Football</h1>', unsafe_allow_html=True)

# Barre de navigation en tabs
tabs = st.tabs(["📊 Vue d'ensemble", "🔍 Données", "📈 Statistiques"])

# Tab 1: Vue d'ensemble
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ## 🎯 Présentation du projet
    
    Ce dashboard interactif permet d'explorer et comprendre les résultats d'un modèle de classification
    supervisée qui prédit les ligues de football à partir des statistiques de matchs. Visualisez les performances
    du modèle, explorez les caractéristiques importantes et testez vos propres prédictions.
    """)
    
    # Métriques de performance du modèle
    st.markdown("### ⭐ Performance du modèle")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Accuracy", value="0.964")
    with col2:
        st.metric(label="Precision", value="0.964")
    with col3:
        st.metric(label="Recall", value="0.964")
    with col4:
        st.metric(label="F1 Score", value="0.964")
    with col5:
        st.metric(label="ROC AUC", value="0.999")

    # Guide de navigation
    st.markdown("### 🧭 Navigation")
    st.markdown("""
    - **Vue d'ensemble**: Présentation du projet et performances
    - **Données**: Exploration des datasets originaux et nettoyés
    - **Statistiques**: Analyse approfondie et corrélations
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Données
with tabs[1]:
    # Chargement des données
    data_original = load_data("data/matches_original.csv")
    data_cleaned = load_data("data/matches.csv")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Dataset original</h3>', unsafe_allow_html=True)
        if not data_original.empty:
            expander = st.expander("Voir les premières lignes", expanded=True)
            with expander:
                st.dataframe(data_original.head(5), use_container_width=True)
            
            total_rows = data_original.shape[0]
            total_cols = data_original.shape[1]
            missing_values = data_original.isna().sum().sum()
            
            st.markdown(f"""
            - **Dimensions**: {total_rows} lignes × {total_cols} colonnes
            - **Valeurs manquantes**: {missing_values}
            - **Mémoire utilisée**: {round(data_original.memory_usage(deep=True).sum() / (1024*1024), 2)} MB
            """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Dataset nettoyé</h3>', unsafe_allow_html=True)
        if not data_cleaned.empty:
            expander = st.expander("Voir les premières lignes", expanded=True)
            with expander:
                st.dataframe(data_cleaned.head(5), use_container_width=True)
            
            total_rows = data_cleaned.shape[0]
            total_cols = data_cleaned.shape[1]
            missing_values = data_cleaned.isna().sum().sum()
            
            st.markdown(f"""
            - **Dimensions**: {total_rows} lignes × {total_cols} colonnes
            - **Valeurs manquantes**: {missing_values}
            - **Mémoire utilisée**: {round(data_cleaned.memory_usage(deep=True).sum() / (1024*1024), 2)} MB
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    if not data_cleaned.empty:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Aperçu des types de données</h3>', unsafe_allow_html=True)
        
        # Comptage des types de données
        type_counts = data_cleaned.dtypes.value_counts().reset_index()
        type_counts.columns = ["Type", "Nombre"]
        
        st.dataframe(type_counts, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Statistiques
with tabs[2]:
    if not data_cleaned.empty:
        # Distribution des ligues
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Distribution des ligues</h3>', unsafe_allow_html=True)
        
        league_cols = [col for col in data_cleaned.columns if col.startswith("League_") and col != "League_Division"]
        if league_cols:
            league_dist = data_cleaned[league_cols].sum().sort_values(ascending=False)
            league_dist = league_dist.reset_index()
            league_dist.columns = ["Ligue", "Nombre de matchs"]
            
            # Utilisation de Plotly pour un graphique interactif
            fig = px.bar(league_dist, 
                         x="Nombre de matchs", 
                         y="Ligue", 
                         orientation='h',
                         color="Nombre de matchs",
                         color_continuous_scale=px.colors.sequential.Viridis,
                         title="Répartition des matchs par ligue")
            
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Corrélation entre les variables numériques
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Matrice de corrélation</h3>', unsafe_allow_html=True)
        
        # Sélection des colonnes numériques
        numeric_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
        
        # Interface pour choisir les variables
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_vars = st.slider("Nombre de variables à afficher", min_value=5, max_value=len(numeric_cols), value=10, step=1)
            correlation_type = st.radio("Type de corrélation", ["Positive", "Négative", "Absolue", "Toutes"])
        
        with col2:
            # Filtrage des corrélations selon le choix
            corr_matrix = data_cleaned[numeric_cols].corr()
            
            if correlation_type == "Positive":
                filtered_corr = corr_matrix.where(corr_matrix > 0)
            elif correlation_type == "Négative":
                filtered_corr = corr_matrix.where(corr_matrix < 0)
            elif correlation_type == "Absolue":
                filtered_corr = corr_matrix.abs()
            else:
                filtered_corr = corr_matrix
            
            # Création de la heatmap avec masque pour le triangle supérieur
            mask = np.triu(np.ones_like(filtered_corr, dtype=bool))
            
            # Sélection des colonnes avec les corrélations les plus fortes
            if correlation_type != "Toutes":
                mean_corr = filtered_corr.abs().mean().sort_values(ascending=False)
                top_cols = mean_corr.index[:n_vars]
                filtered_corr = filtered_corr.loc[top_cols, top_cols]
                mask = np.triu(np.ones_like(filtered_corr, dtype=bool))
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(filtered_corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                        ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8})
            ax.set_title(f"Matrice de corrélation ({correlation_type.lower()})")
            st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
# Barre latérale avec des informations supplémentaires
with st.sidebar:
    st.markdown("## Informations sur le projet")
    
    ligue_pays = {
        'Premier League': 'Angleterre', 'EFL Championship': 'Angleterre',
        'Ligue 1': 'France', 'Ligue 2': 'France',
        'Serie A': 'Italie', 'Serie B': 'Italie',
        'Bundesliga': 'Allemagne', 'Bundesliga 2': 'Allemagne',
        'LaLiga': 'Espagne', 'LaLiga 2': 'Espagne'
    }
    
    st.markdown("""### 🏆 Ligues """)
    st.markdown("Les données comprennent des matchs de 10 ligues de football européennes:")
    st.write(pd.Series(ligue_pays).reset_index().rename(columns={"index": "Ligue", 0: "Pays"}))
    
    st.markdown("""
    
    ### 📂 Données utilisées
    - matches_original.csv
    - matches.csv
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📊 Méthodologie
    1. Prétraitement et Ingénierie des Caractéristiques
    2. Évaluation des Modèles de Classification
    3. Comparaison des Meilleurs Modèles de Classification
    """)
    
# Footer
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Développé avec Streamlit • Dernière mise à jour: Février 2025")
url = "https://github.com/heretounderstand/football_league_classification"
st.markdown("Github Repo : [link](%s)" % url)
st.markdown("📧 kadrilud@gmail.com")
st.markdown("</div>", unsafe_allow_html=True)
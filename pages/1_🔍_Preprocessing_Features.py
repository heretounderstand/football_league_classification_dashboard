from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(
    page_title="Preprocessing & Feature Engineering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
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

# Fonction pour créer une carte métrique
def metric_card(title, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """, unsafe_allow_html=True)

# Chargement des différents datasets
feature_scores = load_data("data/feature_scores.csv")
y_train = load_data("data/y_train.csv")
X_train = load_data("data/train_original.csv")
X_test = load_data("data/test_original.csv")
X_train_preprocessed = load_data("data/X_train_preprocessed.csv")
X_test_preprocessed = load_data("data/X_test_preprocessed.csv")
X_train_lda = load_data("data/X_train_lda.csv")
X_test_lda = load_data("data/X_test_lda.csv")
X_train_selected = load_data("data/X_train_selected.csv")
X_test_selected = load_data("data/X_test_selected.csv")
pca = load_data("data/pca.csv")

# En-tête principal avec logo et titre
col_logo, col_title = st.columns([2, 5])
with col_logo:
    st.image("image/top-5.jpg", width=275)
with col_title:
    st.markdown('<h1 class="main-header">Prétraitement et Ingénierie des Caractéristiques</h1>', unsafe_allow_html=True)

# Création des onglets pour la navigation principale
tabs = st.tabs([
    "📊 Vue d'ensemble", 
    "📐 Standardisation et Encodage", 
    "🧩 PCA", 
    "🔍 LDA", 
    "🏆 Features finales"
])

# Tab 1: Vue d'ensemble
with tabs[0]:
    # Carte d'introduction
    st.markdown("""
    ## 🎯 Présentation de l'analyse
    
    Cette page présente les différentes étapes du pipeline de préparation des données pour notre modèle de classification des ligues de football.
    
    Vous découvrirez comment les données brutes sont transformées pas à pas pour devenir exploitables par les algorithmes de machine learning,
    en passant par la standardisation, l'encodage des variables catégorielles, la réduction de dimensionnalité et la sélection des caractéristiques les plus pertinentes.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Données de test</h3>', unsafe_allow_html=True)
        if not X_test.empty:
            expander = st.expander("Voir les premières lignes", expanded=True)
            with expander:
                st.dataframe(X_test.head(5), use_container_width=True)
            
            total_rows = X_test.shape[0]
            total_cols = X_test.shape[1]
            
            st.markdown(f"""
            - **Dimensions**: {total_rows} lignes × {total_cols} colonnes
            - **Mémoire utilisée**: {round(X_test.memory_usage(deep=True).sum() / (1024*1024), 2)} MB
            """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Données d\'entraînement</h3>', unsafe_allow_html=True)
        if not X_train.empty:
            expander = st.expander("Voir les premières lignes", expanded=True)
            with expander:
                st.dataframe(X_train.head(5), use_container_width=True)
            
            total_rows = X_train.shape[0]
            total_cols = X_train.shape[1]
            
            st.markdown(f"""
            - **Dimensions**: {total_rows} lignes × {total_cols} colonnes
            - **Mémoire utilisée**: {round(X_train.memory_usage(deep=True).sum() / (1024*1024), 2)} MB
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Standardisation & Encodage
with tabs[1]:
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Données de test</h3>', unsafe_allow_html=True)
        if not X_test_preprocessed.empty:
            expander = st.expander("Voir les premières lignes", expanded=True)
            with expander:
                st.dataframe(X_test_preprocessed.head(5), use_container_width=True)
            
            total_rows = X_test_preprocessed.shape[0]
            total_cols = X_test_preprocessed.shape[1]
            
            st.markdown(f"""
            - **Dimensions**: {total_rows} lignes × {total_cols} colonnes
            - **Mémoire utilisée**: {round(X_test_preprocessed.memory_usage(deep=True).sum() / (1024*1024), 2)} MB
            """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Données d\'entraînement</h3>', unsafe_allow_html=True)
        if not X_train_preprocessed.empty:
            expander = st.expander("Voir les premières lignes", expanded=True)
            with expander:
                st.dataframe(X_train_preprocessed.head(5), use_container_width=True)
            
            total_rows = X_train_preprocessed.shape[0]
            total_cols = X_train_preprocessed.shape[1]
            
            st.markdown(f"""
            - **Dimensions**: {total_rows} lignes × {total_cols} colonnes
            - **Mémoire utilisée**: {round(X_train_preprocessed.memory_usage(deep=True).sum() / (1024*1024), 2)} MB
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    if not X_train_preprocessed.empty:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Aperçu des types de données</h3>', unsafe_allow_html=True)
        
        # Comptage des types de données
        type_counts = X_train_preprocessed.dtypes.value_counts().reset_index()
        type_counts.columns = ["Type", "Nombre"]
        
        st.dataframe(type_counts, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Guide sur les techniques appliquées
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="subsection-header">Techniques de prétraitement appliquées</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>Pourquoi standardiser les données?</h4>
        <p>La standardisation est essentielle car elle:</p>
        <ul>
            <li>Facilite la convergence des algorithmes d'optimisation</li>
            <li>Évite que les variables à grande échelle dominent celles à petite échelle</li>
            <li>Améliore l'interprétabilité des coefficients des modèles</li>
            <li>Est nécessaire pour les techniques comme PCA ou k-means</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>Étapes appliquées:</h4>
        <p><span class="step-number">1</span> <strong>Imputation des valeurs manquantes</strong>: remplacement par la médiane pour les variables numériques et par le mode pour les catégorielles</p>
        <p><span class="step-number">2</span> <strong>Standardisation</strong>: transformation en variables centrées-réduites (μ=0, σ=1)</p>
        <p><span class="step-number">3</span> <strong>Encodage des variables catégorielles</strong>: conversion en variables numériques via one-hot encoding</p>
        <p><span class="step-number">4</span> <strong>Gestion des outliers</strong>: détection et traitement des valeurs aberrantes</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: PCA
with tabs[2]:
    st.markdown('</div>', unsafe_allow_html=True)
        
    # Variance expliquée par les composantes
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Utilisation du DataFrame existant pca
    # Calcul de la variance cumulée
    pca['Variance cumulée (%)'] = np.cumsum(pca['Variance expliquée (%)'])

    # Création du graphique
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pca['Composante'],
        y=pca['Variance expliquée (%)'],
        name='Variance expliquée'
    ))

    fig.add_trace(go.Scatter(
        x=pca['Composante'],
        y=pca['Variance cumulée (%)'],
        name='Variance cumulée',
        mode='lines+markers',
        line=dict(width=3)
    ))

    fig.update_layout(
        title='Variance expliquée et cumulée par composante principale',
        xaxis_title='Composante Principale',
        yaxis_title='Variance Expliquée (%)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )

    # Ajouter une ligne horizontale à 95% pour montrer le seuil typique
    fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Seuil 95%")

    # Déterminer le nombre de composantes nécessaires pour atteindre 95% de variance
    n_components_95 = np.argmax(pca['Variance cumulée (%)'] >= 95)
    composante_95 = pca['Composante'][n_components_95]

    # Ajouter une ligne verticale à la composante qui atteint 95% de variance
    fig.add_vline(
        x=n_components_95, 
        line_dash="dash", 
        line_color="green",
        annotation_text=f"{composante_95}",
        annotation_position="top"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

     
    if not feature_scores.empty:
        # Corrélation entre les variables numériques
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Top Features Importantes</h3>', unsafe_allow_html=True)
        
        # Interface pour choisir les variables
        col1, col2 = st.columns([1, 3])
        
        with col1:
            n_vars = st.slider("Nombre de variables à afficher", min_value=5, max_value=feature_scores.shape[0], value=10, step=1)
            ascend_type = st.checkbox("Descendant", value=True)
        
        with col2:
            feature_scores.sort_values(ascending = False if ascend_type else True , by = "F_Score", inplace = True)
            
            # Utilisation de Plotly pour un graphique interactif
            fig = px.bar(feature_scores.head(n_vars), 
                         x="F_Score", 
                         y="Feature", 
                         orientation='h',
                         color="F_Score",
                         color_continuous_scale=px.colors.sequential.Viridis,
                         title="Importance des Features")
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'} if ascend_type else {'categoryorder': 'total descending'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Corrélation entre les variables numériques
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Features Non Significatives</h3>', unsafe_allow_html=True)
        
        feature_less = feature_scores[feature_scores['P_Value'] > 0.05]
        
        st.dataframe(feature_less, use_container_width=True)
            
        total_rows = feature_less.shape[0]
        previous_rows = feature_scores.shape[0]
            
        st.markdown(f"""
        - **Nombre**: {total_rows}
        - **Pourcentage**: {(total_rows / previous_rows):.2f}%
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <h3 class="subsection-header">Qu'est-ce que la PCA?</h3>
        
        L'Analyse en Composantes Principales (PCA) est une technique de réduction de dimensionnalité non supervisée qui:
        
        - Transforme les variables originales en nouvelles variables non corrélées (composantes principales)
        - Capture le maximum de variance dans les données avec moins de dimensions
        - Facilite la visualisation des données à haute dimension
        - Réduit le bruit et combat le fléau de la dimensionnalité
        """, unsafe_allow_html=True)
        
# Tab 4: LDA
with tabs[3]:
    
    # Introduction à la LDA
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown("""
    <h3 class="subsection-header">Qu'est-ce que la LDA?</h3>
        
    L'Analyse Discriminante Linéaire (LDA) est une technique de réduction de dimensionnalité supervisée qui:
        
    - Cherche les axes qui maximisent la séparation entre les classes
    - Réduit la dimensionnalité tout en préservant l'information discriminante
    - Diffère de la PCA car elle utilise les étiquettes de classe
    - Est particulièrement adaptée pour les problèmes de classification
    """, unsafe_allow_html=True)    
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Résultats de la LDA
    if not X_train_lda.empty and not X_test_lda.empty:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Résultats de la réduction de dimensionnalité</h3>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            original_dim = X_train_preprocessed.shape[1] if not X_train_preprocessed.empty else X_train.shape[1]
            metric_card("Dimensions originales", original_dim)
        
        with col2:
            lda_dim = X_train_lda.shape[1]
            metric_card("Dimensions après LDA", lda_dim)
        
        with col3:
            reduction = 100 * lda_dim / original_dim
            metric_card("Pourcentage restant", f"{reduction:.2f}%")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualisation des données LDA
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Visualisation des composantes LDA</h3>', unsafe_allow_html=True)
        
        # Simuler des données pour une meilleure visualisation
        if X_train_lda.shape[1] >= 2:
            # Ajout d'une variable pour la couleur (simulation de l'appartenance à une ligue)
            league_list = y_train['League_Division'].unique()
            
            # Créer des données simulées pour la visualisation
            vis_data = X_train_lda.copy()
            vis_data['League'] = np.random.choice(league_list, size=len(vis_data))
            
            # Option de sélection des composantes
            if X_train_lda.shape[1] > 2:
                use_3d = st.checkbox("Vue 3D LDA", value=False)
                if use_3d :
                    col1, col2, col3 = st.columns(3)
                
                    with col1:
                        pc_x = st.selectbox("Composante X LDA:", X_train_lda.columns, index=0)
                    
                    with col2:
                        pc_y = st.selectbox("Composante Y LDA:", X_train_lda.columns, index=1)
                    
                    with col3:
                        pc_z = st.selectbox("Composante Z LDA:", X_train_lda.columns, index=2)
                        
                    # Scatter 3D
                    fig = px.scatter_3d(
                        vis_data,
                        x=pc_x,
                        y=pc_y,
                        z=pc_z,
                        color='League',
                        title="Projection des données sur trois composantes LDA"
                    )
                    fig.update_layout(height=700)
                
                else :
                    col1, col2 = st.columns(2)
                
                    with col1:
                        pc_x = st.selectbox("Composante X LDA:", X_train_lda.columns, index=0)
                    
                    with col2:
                        pc_y = st.selectbox("Composante Y LDA:", X_train_lda.columns, index=1)  
                    
                    # Scatter 2D
                    fig = px.scatter(
                        vis_data,
                        x=pc_x,
                        y=pc_y,
                        color='League',
                        title="Projection des données sur deux composantes LDA",
                        marginal_x="histogram",
                        marginal_y="histogram"
                    )
                    fig.update_layout(height=700)
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Comparaison PCA vs LDA
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="subsection-header">Comparaison PCA vs LDA</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <h4>Avantages de la PCA</h4>
            <ul>
                <li>Méthode non supervisée (ne nécessite pas les étiquettes)</li>
                <li>Capture la variance maximale dans les données</li>
                <li>Moins sensible au surapprentissage</li>
                <li>Utile pour la visualisation des données</li>
                <li>Adaptée à divers types de problèmes</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>Avantages de la LDA</h4>
            <ul>
                <li>Méthode supervisée (utilise les étiquettes)</li>
                <li>Maximise la séparation entre les classes</li>
                <li>Réduit plus efficacement la dimensionnalité pour la classification</li>
                <li>Peut améliorer directement les performances du modèle</li>
                <li>Nombre de composantes limité par le nombre de classes</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Tab 5: Features finales
with tabs[4]:
    st.markdown('<h2 class="section-header">Caractéristiques Sélectionnées</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Données de test</h3>', unsafe_allow_html=True)
        if not X_test_selected.empty:
            expander = st.expander("Voir les premières lignes", expanded=True)
            with expander:
                st.dataframe(X_test_selected.head(5), use_container_width=True)
            
            total_rows = X_test_selected.shape[0]
            total_cols = X_test_selected.shape[1]
            
            st.markdown(f"""
            - **Dimensions**: {total_rows} lignes × {total_cols} colonnes
            - **Mémoire utilisée**: {round(X_test_selected.memory_usage(deep=True).sum() / (1024*1024), 2)} MB
            """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="sub-header">Données d\'entraînement</h3>', unsafe_allow_html=True)
        if not X_train_selected.empty:
            expander = st.expander("Voir les premières lignes", expanded=True)
            with expander:
                st.dataframe(X_train_selected.head(5), use_container_width=True)
            
            total_rows = X_train_selected.shape[0]
            total_cols = X_train_selected.shape[1]
            
            st.markdown(f"""
            - **Dimensions**: {total_rows} lignes × {total_cols} colonnes
            - **Mémoire utilisée**: {round(X_train_selected.memory_usage(deep=True).sum() / (1024*1024), 2)} MB
            """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Corrélation entre les variables numériques
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Matrice de corrélation</h3>', unsafe_allow_html=True)
        
    # Interface pour choisir les variables
    col1, col2 = st.columns([1, 3])
        
    with col1:
        n_vars = st.slider("Nombre de variables à afficher", min_value=5, max_value=20, value=10, step=1)
        correlation_type = st.radio("Type de corrélation", ["Positive", "Négative", "Absolue", "Toutes"])
    
    with col2:
        # Filtrage des corrélations selon le choix
        corr_matrix = X_train_selected.corr()
        
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
    
    st.markdown("""
    
    ### 📊 Datasets finaux
    
    **X_train_selected.csv** :
    - Nombre de colonnes: 20
    - Nombre de lignes: 79459
    
    **X_test_selected.csv** :
    - Nombre de colonnes: 20
    - Nombre de lignes: 34055
    
    ### 📂 Données utilisées
    - train_original.csv
    - test_original.csv
    - X_train_preprocessed.csv
    - X_test_preprocessed.csv
    - X_train_lda.csv
    - X_test_lda.csv
    - X_train_selected.csv
    - X_test_selected.csv
    - pca.csv
    - feature_scores.csv
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📊 Méthodologie
    1. Division des données en ensembles d'entraînement (70%) et de test (30%)
    2. Standardisation des données numériques et encodage des variables catégorielles
    3. Analyse de l'importance des caractéristiques avec F-Score et P-Value
    4. Application des techniques de réduction dimensionnelle (PCA, LDA)
    5. Sélection des caractéristiques les plus pertinentes
    """)

# Footer
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Développé avec Streamlit • Dernière mise à jour: Février 2025")
url = "https://github.com/heretounderstand/football_league_classification"
st.markdown("Github Repo : [link](%s)" % url)
st.markdown("📧 kadrilud@gmail.com")
st.markdown("</div>", unsafe_allow_html=True)
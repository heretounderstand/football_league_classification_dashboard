import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# Configuration de la page
st.set_page_config(
    page_title="Best Football League Model Comparison",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .highlight {
        background-color: #e8f4f8;
        padding: 5px;
        border-left: 3px solid #1E88E5;
        margin-bottom: 10px;
    }
    .model-comparison {
        font-weight: bold;
        color: #0D47A1;
    }
</style>
""", unsafe_allow_html=True)

# Définition du dictionnaire pour les ligues et pays
ligue_pays = {
    'Premier League': 'Angleterre', 'EFL Championship': 'Angleterre',
    'Ligue 1': 'France', 'Ligue 2': 'France',
    'Serie A': 'Italie', 'Serie B': 'Italie',
    'Bundesliga': 'Allemagne', 'Bundesliga 2': 'Allemagne',
    'LaLiga': 'Espagne', 'LaLiga 2': 'Espagne'
}

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
    
# Chargement des prédictions
predictions_catboost = load_data("data/predictions_catboost.csv")
predictions_lightgbm = load_data("data/predictions_lightgbm.csv")
predictions_xgboost = load_data("data/predictions_xgboost.csv")
catboost_robustesse = load_data("data/catboost_robustesse.csv")
lightgbm_robustesse = load_data("data/lightgbm_robustesse.csv")
xgboost_robustesse = load_data("data/xgboost_robustesse.csv")
catboost_learning = load_data("data/catboost_learning_curve.csv")
lightgbm_learning = load_data("data/lightgbm_learning_curve.csv")
xgboost_learning = load_data("data/xgboost_learning_curve.csv")
catboost_importance = load_data("data/catboost_feature_importance.csv")
lightgbm_importance = load_data("data/lightgbm_feature_importance.csv")
xgboost_importance = load_data("data/xgboost_feature_importance.csv")

# Fonction pour créer la matrice de confusion avec focus sur les erreurs entre ligues du même pays
def plot_errors_by_country(y_true, y_pred, labels, ligue_pays):
    # Création de la matrice de confusion classique
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Préparation des données pour le graphe d'erreurs par pays
    pays_list = list(set(ligue_pays.values()))
    error_matrix = np.zeros((len(pays_list), len(pays_list)))
    
    # On regroupe les erreurs par pays
    for i, true_ligue in enumerate(labels):
        for j, pred_ligue in enumerate(labels):
            if i != j:  # c'est une erreur
                true_pays = ligue_pays.get(true_ligue)
                pred_pays = ligue_pays.get(pred_ligue)
                
                if true_pays in pays_list and pred_pays in pays_list:
                    true_idx = pays_list.index(true_pays)
                    pred_idx = pays_list.index(pred_pays)
                    error_matrix[true_idx, pred_idx] += cm[i, j]
    
    # Création de la heatmap avec Plotly
    fig = go.Figure(data=go.Heatmap(
        z=error_matrix,
        x=pays_list,
        y=pays_list,
        colorscale='Reds',
        text=error_matrix.astype(int),
        texttemplate="%{text}",
        textfont={"size":12},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title="Erreurs de Classification entre Pays",
        xaxis_title="Pays Prédit",
        yaxis_title="Pays Réel",
        xaxis=dict(side='top'),
    )
    
    return fig

# Fonction pour le calcul des erreurs par pays
def calculate_country_errors(y_true, y_pred, ligue_pays):
    errors = {'total': 0}
    same_country_errors = {'total': 0}
    
    for country in set(ligue_pays.values()):
        errors[country] = 0
        same_country_errors[country] = 0
    
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            # C'est une erreur
            errors['total'] += 1
            
            true_country = ligue_pays.get(y_true[i])
            pred_country = ligue_pays.get(y_pred[i])
            
            if true_country:
                errors[true_country] = errors.get(true_country, 0) + 1
            
            # Si les deux ligues sont du même pays
            if true_country and pred_country and true_country == pred_country:
                same_country_errors['total'] += 1
                same_country_errors[true_country] = same_country_errors.get(true_country, 0) + 1
    
    return errors, same_country_errors

# Fonction pour créer un graphique de comparaison d'erreurs entre pays
def plot_country_error_comparison(errors_dict, same_country_errors_dict):
    countries = [country for country in errors_dict.keys() if country != 'total']
    
    error_data = []
    for country in countries:
        total_errors = errors_dict.get(country, 0)
        same_country = same_country_errors_dict.get(country, 0)
        diff_country = total_errors - same_country
        
        error_data.append({
            'Pays': country,
            'Erreurs même pays': same_country,
            'Erreurs autres pays': diff_country
        })
    
    error_df = pd.DataFrame(error_data)
    
    fig = px.bar(error_df, x='Pays', 
                 y=['Erreurs même pays', 'Erreurs autres pays'],
                 title='Répartition des erreurs par pays',
                 barmode='stack',
                 color_discrete_sequence=['#FF9800', '#E57373'])
    
    fig.update_layout(height=400, legend_title_text='Type d\'erreur')
    
    return fig

# En-tête principal avec logo et titre
col_logo, col_title = st.columns([2, 5])
with col_logo:
    st.image("image/top-5.jpg", width=275)
with col_title:
    st.markdown('<h1 class="main-header">Comparaison des Meilleurs Modèles de Classification</h1>', unsafe_allow_html=True)

# Barre de navigation en tabs
tabs = st.tabs(["📊 Vue d'ensemble", "🔍 Analyse des Erreurs", "📈 Robustesse", "🏆 Importance des Features", "📉 Learning Curves"])

# Tab 1: Vue d'ensemble
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ## 🎯 Présentation de l'analyse
    
    Cette page permet de comparer les performances de trois algorithmes de classification utilisés pour prédire les ligues de football :
    - CatBoost
    - LightGBM
    - XGBoost
    
    L'objectif est d'analyser en profondeur les forces et faiblesses de chaque modèle, en particulier concernant les erreurs 
    de classification entre ligues d'un même pays.
    """)
    
    # Section pour les liens des repos
    st.markdown("### 📚 Accès aux modèles")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("[🔗 Repo LightGBM](https://github.com/heretounderstand/football_league_classification/modeles_entraines/lightgbm_optimized.pkl)", unsafe_allow_html=False)
        st.caption("Modèle LightGBM optimisé")
        
    with col2:
        st.markdown("[🔗 Repo CatBoost](https://github.com/heretounderstand/football_league_classification/modeles_entraines/catboost_optimized.pkl)", unsafe_allow_html=False)
        st.caption("Modèle CatBoost optimisé")
        
    with col3:
        st.markdown("[🔗 Repo XGBoost](https://github.com/heretounderstand/football_league_classification/modeles_entraines/xgboost_optimized.pkl)", unsafe_allow_html=False)
        st.caption("Modèle XGBoost optimisé")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Analyse des Erreurs
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Analyse des erreurs de classification</h3>', unsafe_allow_html=True)
    
    # Sélection du modèle à analyser
    model_choice = st.selectbox("Sélectionnez le modèle à analyser:", ["CatBoost", "LightGBM", "XGBoost"])
    
    # Chargement des prédictions du modèle sélectionné
    if model_choice == "CatBoost":
        predictions_df = predictions_catboost
    elif model_choice == "LightGBM":
        predictions_df = predictions_lightgbm
    else:
        predictions_df = predictions_xgboost
    
    if not predictions_df.empty:
        # Obtention des labels uniques
        unique_labels = sorted(list(set(predictions_df['y_true'].unique()) | set(predictions_df['y_pred'].unique())))
        
        # Création de la matrice de confusion avec focus sur les erreurs entre pays
        conf_matrix_fig = plot_errors_by_country(
            predictions_df['y_true'], 
            predictions_df['y_pred'], 
            unique_labels, 
            ligue_pays
        )
        
        st.plotly_chart(conf_matrix_fig, use_container_width=True)
        
        # Analyse des erreurs par pays
        errors, same_country_errors = calculate_country_errors(
            predictions_df['y_true'], 
            predictions_df['y_pred'], 
            ligue_pays
        )
        
        # Graphique des erreurs par pays
        country_error_fig = plot_country_error_comparison(errors, same_country_errors)
        st.plotly_chart(country_error_fig, use_container_width=True)
        
        # Statistiques sur les erreurs
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">Erreurs totales</h3>', unsafe_allow_html=True)
            
            # Tableau des erreurs par pays
            error_stats = []
            for country in sorted(set(ligue_pays.values())):
                if country in errors:
                    error_stats.append({
                        'Pays': country,
                        'Nombre d\'erreurs': errors[country],
                        'Pourcentage': f"{(errors[country] / errors['total'] * 100):.1f}%"
                    })
            
            error_stats_df = pd.DataFrame(error_stats)
            st.dataframe(error_stats_df, use_container_width=True)
            
            st.markdown(f"**Nombre total d'erreurs**: {errors['total']}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3 class="sub-header">Erreurs entre ligues du même pays</h3>', unsafe_allow_html=True)
            
            # Tableau des erreurs du même pays
            same_country_stats = []
            for country in sorted(set(ligue_pays.values())):
                if country in same_country_errors and same_country_errors[country] > 0:
                    same_country_stats.append({
                        'Pays': country,
                        'Erreurs même pays': same_country_errors[country],
                        'Pourcentage des erreurs': f"{(same_country_errors[country] / errors[country] * 100):.1f}%"
                    })
            
            same_country_stats_df = pd.DataFrame(same_country_stats)
            st.dataframe(same_country_stats_df, use_container_width=True)
            
            st.markdown(f"**Nombre d'erreurs entre ligues du même pays**: {same_country_errors['total']}")
            st.markdown(f"**Pourcentage d'erreurs du même pays**: {(same_country_errors['total'] / errors['total'] * 100):.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Robustesse
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Analyse de la robustesse des modèles</h3>', unsafe_allow_html=True)
    
    if not catboost_robustesse.empty and not lightgbm_robustesse.empty and not xgboost_robustesse.empty:
        # Création d'un dataframe combiné pour comparaison
        catboost_robustesse['Modèle'] = 'CatBoost'
        lightgbm_robustesse['Modèle'] = 'LightGBM'
        xgboost_robustesse['Modèle'] = 'XGBoost'
        
        combined_robustesse = pd.concat([catboost_robustesse, lightgbm_robustesse, xgboost_robustesse])
        
        # Statistiques descriptives
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="CatBoost - F1 Score moyen", 
                value=f"{catboost_robustesse['F1-score'].mean():.4f}",
                delta=f"±{catboost_robustesse['F1-score'].std():.4f}"
            )
        
        with col2:
            st.metric(
                label="LightGBM - F1 Score moyen", 
                value=f"{lightgbm_robustesse['F1-score'].mean():.4f}",
                delta=f"±{lightgbm_robustesse['F1-score'].std():.4f}"
            )
        
        with col3:
            st.metric(
                label="XGBoost - F1 Score moyen", 
                value=f"{xgboost_robustesse['F1-score'].mean():.4f}",
                delta=f"±{xgboost_robustesse['F1-score'].std():.4f}"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
        
    # Création d'un boxplot pour comparer les distributions
    fig = px.box(
        combined_robustesse, 
        x='Modèle', 
        y='F1-score',
        color='Modèle',
        points='all',
        title='Distribution des F1 Scores pour différentes initialisations',
        color_discrete_map={'CatBoost': '#1976D2', 'LightGBM': '#43A047', 'XGBoost': '#FFA000'}
    )
    
    fig.update_layout(height=500, yaxis_range=[0.9, 1.0])
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Statistiques de robustesse</h3>', unsafe_allow_html=True)
        
    robustesse_stats = pd.DataFrame({
        'Modèle': ['CatBoost', 'LightGBM', 'XGBoost'],
        'F1 Score Min': [
            catboost_robustesse['F1-score'].min(),
            lightgbm_robustesse['F1-score'].min(),
            xgboost_robustesse['F1-score'].min()
        ],
        'F1 Score Max': [
            catboost_robustesse['F1-score'].max(),
            lightgbm_robustesse['F1-score'].max(),
            xgboost_robustesse['F1-score'].max()
        ],
        'F1 Score Moyen': [
            catboost_robustesse['F1-score'].mean(),
            lightgbm_robustesse['F1-score'].mean(),
            xgboost_robustesse['F1-score'].mean()
        ],
        'Écart-type': [
            catboost_robustesse['F1-score'].std(),
            lightgbm_robustesse['F1-score'].std(),
            xgboost_robustesse['F1-score'].std()
        ],
        'Coefficient de variation (%)': [
            (catboost_robustesse['F1-score'].std() / catboost_robustesse['F1-score'].mean()) * 100,
            (lightgbm_robustesse['F1-score'].std() / lightgbm_robustesse['F1-score'].mean()) * 100,
            (xgboost_robustesse['F1-score'].std() / xgboost_robustesse['F1-score'].mean()) * 100
        ]
    })
    
    # Formatage des colonnes numériques
    for col in robustesse_stats.columns[1:]:
        if col == 'Coefficient de variation (%)':
            robustesse_stats[col] = robustesse_stats[col].map('{:.2f}'.format)
        else:
            robustesse_stats[col] = robustesse_stats[col].map('{:.4f}'.format)
    
    st.dataframe(robustesse_stats, use_container_width=True)
    
    # Identification du modèle le plus robuste
    most_robust = robustesse_stats.loc[robustesse_stats['Écart-type'].astype(float).idxmin(), 'Modèle']
    st.markdown(f"""
    <div class="highlight">
    <p class="model-comparison">Modèle le plus robuste: <span style="color: #1E88E5;">{most_robust}</span> avec le plus faible écart-type.</p>
    </div>
    """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Importance des Features
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Analyse de l\'importance des features</h3>', unsafe_allow_html=True)
    
    if not catboost_importance.empty and not lightgbm_importance.empty and not xgboost_importance.empty:
        col1, col2 = st.columns([4,3])
        
        with col1:
            # Sélection du modèle à visualiser
            model_choice = st.radio("Choisir le modèle à visualiser:", ["CatBoost", "LightGBM", "XGBoost", "Comparaison"], horizontal=True)
        with col2:
            # Préparation des données pour la visualisation
            top_n = st.slider("Nombre de features à afficher:", 5, 20, 10)
        
        if model_choice == "CatBoost":
            
            # Tri et sélection des top features
            sorted_importance = catboost_importance.sort_values('Importance', ascending=False).head(top_n)
            
            # Création du graphique
            fig = px.bar(
                sorted_importance,
                x='Importance',
                y='Feature',
                title=f'Top {top_n} features les plus importantes selon CatBoost',
                orientation='h',
                color='Importance',
                color_continuous_scale='Oranges'
            )
            
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_choice == "LightGBM":
            
            # Tri et sélection des top features
            sorted_importance = lightgbm_importance.sort_values('Importance', ascending=False).head(top_n)
            
            # Création du graphique
            fig = px.bar(
                sorted_importance,
                x='Importance',
                y='Feature',
                title=f'Top {top_n} features les plus importantes selon LightGBM',
                orientation='h',
                color='Importance',
                color_continuous_scale='Greens'
            )
            
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_choice == "XGBoost":
            
            # Tri et sélection des top features
            sorted_importance = xgboost_importance.sort_values('Importance', ascending=False).head(top_n)
            
            # Création du graphique
            fig = px.bar(
                sorted_importance,
                x='Importance',
                y='Feature',
                title=f'Top {top_n} features les plus importantes selon XGBoost',
                orientation='h',
                color='Importance',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Création d'un dataframe commun avec toutes les features
            all_features = set(catboost_importance['Feature']) | set(lightgbm_importance['Feature']) | set(xgboost_importance['Feature'])
            
            importance_comparison = pd.DataFrame({'Feature': list(all_features)})
            
            # Ajout des# Ajout des colonnes d'importance pour chaque modèle
            importance_comparison = importance_comparison.merge(
                catboost_importance[['Feature', 'Importance']].rename(columns={'Importance': 'CatBoost'}),
                on='Feature', how='left'
            )
            
            importance_comparison = importance_comparison.merge(
                lightgbm_importance[['Feature', 'Importance']].rename(columns={'Importance': 'LightGBM'}),
                on='Feature', how='left'
            )
            
            importance_comparison = importance_comparison.merge(
                xgboost_importance[['Feature', 'Importance']].rename(columns={'Importance': 'XGBoost'}),
                on='Feature', how='left'
            )
            
            # Remplacement des NaN par 0
            importance_comparison = importance_comparison.fillna(0)
            
            # Calcul des corrélations
            corr_matrix = importance_comparison[['CatBoost', 'LightGBM', 'XGBoost']].corr()
            
            # Affichage de la matrice de corrélation
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu_r',  # Équivalent de "coolwarm" dans Plotly
                zmid=0,  # Point central de la colorscale (0 pour les corrélations)
                text=corr_matrix.round(2),  # Format à 2 décimales
                texttemplate='%{text:.2f}',
                hoverongaps=False,
                colorbar=dict(
                    title='Corrélation',
                    titleside='right'
                )
            ))
            
            fig.update_layout(
                title="Corrélation entre les importances des features des modèles",
                xaxis=dict(
                    tickangle=-45,
                    side='bottom'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Identification des features consensus (importantes pour tous les modèles)
            st.markdown('<h3 class="sub-header">Features importantes pour tous les modèles</h3>', unsafe_allow_html=True)
            
            # Calcul d'un score agrégé
            importance_comparison[['CatBoost', 'LightGBM', 'XGBoost']] = importance_comparison[['CatBoost', 'LightGBM', 'XGBoost']].div(importance_comparison[['CatBoost', 'LightGBM', 'XGBoost']].sum())

            # Calcul du score agrégé
            importance_comparison['Score agrégé'] = importance_comparison[['CatBoost', 'LightGBM', 'XGBoost']].mean(axis=1)
            
            # Top features par score agrégé
            top_consensus = importance_comparison.sort_values('Score agrégé', ascending=False)
            
            fig = px.bar(
                top_consensus,
                x='Score agrégé',
                y='Feature',
                title='Top features par importance agrégée',
                orientation='h',
                color='Score agrégé',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout( yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 5: Learning Curves
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Analyse des learning curves</h3>', unsafe_allow_html=True)
    
    if not catboost_learning.empty and not lightgbm_learning.empty and not xgboost_learning.empty:
        # Choix du modèle à visualiser
        model_choice = st.radio("Choisir le modèle à visualiser:", ["CatBoost", "LightGBM", "XGBoost", "Comparaison"], horizontal=True, key="learning_curve_model")
        
        if model_choice == "CatBoost":
            # Création du graphique pour CatBoost
            fig = go.Figure()
            
            # Courbe d'entraînement
            fig.add_trace(go.Scatter(
                x=catboost_learning['Training Size'],
                y=catboost_learning['Train Mean'],
                mode='lines+markers',
                name='Score d\'entraînement',
                line=dict(color='#FF9800'),
                error_y=dict(
                    type='data',
                    array=catboost_learning['Train Std'],
                    visible=True,
                    color='#FF9800'
                )
            ))
            
            # Courbe de test
            fig.add_trace(go.Scatter(
                x=catboost_learning['Training Size'],
                y=catboost_learning['Test Mean'],
                mode='lines+markers',
                name='Score de validation',
                line=dict(color='#FF7043'),
                error_y=dict(
                    type='data',
                    array=catboost_learning['Test Std'],
                    visible=True,
                    color='#FF7043'
                )
            ))
            
            fig.update_layout(
                title='Learning Curve de CatBoost',
                xaxis_title='Taille de l\'ensemble d\'entraînement',
                yaxis_title='Score',
                height=500,
                xaxis=dict(tickmode='array', tickvals=catboost_learning['Training Size']),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_choice == "LightGBM":
            # Création du graphique pour LightGBM
            fig = go.Figure()
            
            # Courbe d'entraînement
            fig.add_trace(go.Scatter(
                x=lightgbm_learning['Training Size'],
                y=lightgbm_learning['Train Mean'],
                mode='lines+markers',
                name='Score d\'entraînement',
                line=dict(color='#4CAF50'),
                error_y=dict(
                    type='data',
                    array=lightgbm_learning['Train Std'],
                    visible=True,
                    color='#4CAF50'
                )
            ))
            
            # Courbe de test
            fig.add_trace(go.Scatter(
                x=lightgbm_learning['Training Size'],
                y=lightgbm_learning['Test Mean'],
                mode='lines+markers',
                name='Score de validation',
                line=dict(color='#FF7043'),
                error_y=dict(
                    type='data',
                    array=lightgbm_learning['Test Std'],
                    visible=True,
                    color='#FF7043'
                )
            ))
            
            fig.update_layout(
                title='Learning Curve de LightGBM',
                xaxis_title='Taille de l\'ensemble d\'entraînement',
                yaxis_title='Score',
                height=500,
                xaxis=dict(tickmode='array', tickvals=lightgbm_learning['Training Size']),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        elif model_choice == "XGBoost":
            # Création du graphique pour XGBoost
            fig = go.Figure()
            
            # Courbe d'entraînement
            fig.add_trace(go.Scatter(
                x=xgboost_learning['Training Size'],
                y=xgboost_learning['Train Mean'],
                mode='lines+markers',
                name='Score d\'entraînement',
                line=dict(color='#2196F3'),
                error_y=dict(
                    type='data',
                    array=xgboost_learning['Train Std'],
                    visible=True,
                    color='#2196F3'
                )
            ))
            
            # Courbe de test
            fig.add_trace(go.Scatter(
                x=xgboost_learning['Training Size'],
                y=xgboost_learning['Test Mean'],
                mode='lines+markers',
                name='Score de validation',
                line=dict(color='#FF7043'),
                error_y=dict(
                    type='data',
                    array=xgboost_learning['Test Std'],
                    visible=True,
                    color='#FF7043'
                )
            ))
            
            fig.update_layout(
                title='Learning Curve de XGBoost',
                xaxis_title='Taille de l\'ensemble d\'entraînement',
                yaxis_title='Score',
                height=500,
                xaxis=dict(tickmode='array', tickvals=xgboost_learning['Training Size']),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Comparaison
            # Création du graphique de comparaison des courbes de validation
            fig = go.Figure()
            
            # Courbe CatBoost
            fig.add_trace(go.Scatter(
                x=catboost_learning['Training Size'],
                y=catboost_learning['Test Mean'],
                mode='lines+markers',
                name='CatBoost',
                line=dict(color='#FF9800'),
                error_y=dict(
                    type='data',
                    array=catboost_learning['Test Std'],
                    visible=True,
                    color='#FF9800'
                )
            ))
            
            # Courbe LightGBM
            fig.add_trace(go.Scatter(
                x=lightgbm_learning['Training Size'],
                y=lightgbm_learning['Test Mean'],
                mode='lines+markers',
                name='LightGBM',
                line=dict(color='#4CAF50'),
                error_y=dict(
                    type='data',
                    array=lightgbm_learning['Test Std'],
                    visible=True,
                    color='#4CAF50'
                )
            ))
            
            # Courbe XGBoost
            fig.add_trace(go.Scatter(
                x=xgboost_learning['Training Size'],
                y=xgboost_learning['Test Mean'],
                mode='lines+markers',
                name='XGBoost',
                line=dict(color='#2196F3'),
                error_y=dict(
                    type='data',
                    array=xgboost_learning['Test Std'],
                    visible=True,
                    color='#2196F3'
                )
            ))
            
            fig.update_layout(
                title='Comparaison des Learning Curves (Scores de validation)',
                xaxis_title='Taille de l\'ensemble d\'entraînement',
                yaxis_title='Score',
                height=500,
                xaxis=dict(tickmode='array', tickvals=catboost_learning['Training Size']),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analyse de la convergence
            st.markdown('<h3 class="sub-header">Analyse de la convergence</h3>', unsafe_allow_html=True)
            
            # On calcule la différence entre le dernier et l'avant-dernier point
            catboost_last_diff = catboost_learning['Test Mean'].iloc[-1] - catboost_learning['Test Mean'].iloc[-2]
            lightgbm_last_diff = lightgbm_learning['Test Mean'].iloc[-1] - lightgbm_learning['Test Mean'].iloc[-2]
            xgboost_last_diff = xgboost_learning['Test Mean'].iloc[-1] - xgboost_learning['Test Mean'].iloc[-2]
            
            # On calcule l'écart entre les scores d'entraînement et de validation pour le dernier point
            catboost_gap = catboost_learning['Train Mean'].iloc[-1] - catboost_learning['Test Mean'].iloc[-1]
            lightgbm_gap = lightgbm_learning['Train Mean'].iloc[-1] - lightgbm_learning['Test Mean'].iloc[-1]
            xgboost_gap = xgboost_learning['Train Mean'].iloc[-1] - xgboost_learning['Test Mean'].iloc[-1]
            
            # Création d'un tableau de comparaison
            convergence_data = pd.DataFrame({
                'Modèle': ['CatBoost', 'LightGBM', 'XGBoost'],
                'Score final': [
                    catboost_learning['Test Mean'].iloc[-1],
                    lightgbm_learning['Test Mean'].iloc[-1],
                    xgboost_learning['Test Mean'].iloc[-1]
                ],
                'Écart-type final': [
                    catboost_learning['Test Std'].iloc[-1],
                    lightgbm_learning['Test Std'].iloc[-1],
                    xgboost_learning['Test Std'].iloc[-1]
                ],
                'Progression dernière étape': [
                    catboost_last_diff,
                    lightgbm_last_diff,
                    xgboost_last_diff
                ],
                'Écart train-test': [
                    catboost_gap,
                    lightgbm_gap,
                    xgboost_gap
                ]
            })
            
            # Formatage des colonnes
            for col in convergence_data.columns[1:]:
                convergence_data[col] = convergence_data[col].map('{:.4f}'.format)
            
            st.dataframe(convergence_data, use_container_width=True)
            
            # Modèle avec le meilleur score final
            best_final = ['CatBoost', 'LightGBM', 'XGBoost'][
                [catboost_learning['Test Mean'].iloc[-1], 
                 lightgbm_learning['Test Mean'].iloc[-1], 
                 xgboost_learning['Test Mean'].iloc[-1]].index(max(
                    catboost_learning['Test Mean'].iloc[-1], 
                    lightgbm_learning['Test Mean'].iloc[-1], 
                    xgboost_learning['Test Mean'].iloc[-1]
                ))
            ]
            
            # Modèle avec la meilleure progression finale
            best_progress = ['CatBoost', 'LightGBM', 'XGBoost'][
                [catboost_last_diff, lightgbm_last_diff, xgboost_last_diff].index(max(
                    catboost_last_diff, lightgbm_last_diff, xgboost_last_diff
                ))
            ]
            
            # Modèle avec le plus petit écart train-test (moins de surapprentissage)
            best_gap = ['CatBoost', 'LightGBM', 'XGBoost'][
                [abs(catboost_gap), abs(lightgbm_gap), abs(xgboost_gap)].index(min(
                    abs(catboost_gap), abs(lightgbm_gap), abs(xgboost_gap)
                ))
            ]
            
            st.markdown(f"""
            <div class="highlight">
            <p><strong>Meilleur score final:</strong> <span style="color: #1E88E5;">{best_final}</span></p>
            <p><strong>Meilleure progression finale:</strong> <span style="color: #1E88E5;">{best_progress}</span></p>
            <p><strong>Modèle avec le moins de surapprentissage:</strong> <span style="color: #1E88E5;">{best_gap}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Observations:")
            
            st.markdown("""
            Le modèle CatBoost présente des signes de surapprentissage.
            """)
        
    st.markdown('</div>', unsafe_allow_html=True)

# Barre latérale avec des informations supplémentaires
with st.sidebar:
    st.markdown("## Informations sur le projet")
    
    st.markdown("""
    
    ### ⭐ Résultats finaux
    - Meilleur score final: LightGBM
    - Meilleure progression finale: CatBoost
    - Modèle avec le moins de surapprentissage: XGBoost
    
    ### 📂 Données utilisées
    - predictions_catboost.csv
    - predictions_lightgbm.csv
    - predictions_xgboost.csv
    - catboost_robustesse.csv
    - lightgbm_robustesse.csv
    - xgboost_robustesse.csv
    - catboost_feature_importance.csv
    - lightgbm_feature_importance.csv
    - xgboost_feature_importance.csv
    - catboost_learning_curve.csv
    - lightgbm_learning_curve.csv
    - xgboost_learning_curve.csv
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📊 Méthodologie
    1. Interprétation des erreurs de classification
    2. Évaluation de l'importance des features par modèle
    3. Tests de robustesse face aux variations des données
    4. Analyse des courbes d'apprentissage (learning curves)
    """)
    
# Footer
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Développé avec Streamlit • Dernière mise à jour: Février 2025")
url = "https://github.com/heretounderstand/football_league_classification"
st.markdown("Github Repo : [link](%s)" % url)
st.markdown("📧 kadrilud@gmail.com")
st.markdown("</div>", unsafe_allow_html=True)

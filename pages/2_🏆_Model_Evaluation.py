import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Football League Model Evaluation",
    page_icon="🏆",
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
        background-color: #e3f2fd;
        padding: 2px 5px;
        border-radius: 4px;
        font-weight: 500;
    }
    .model-card {
        border-left: 5px solid;
        padding-left: 15px;
        margin-bottom: 15px;
    }
    .lightgbm {
        border-color: #4CAF50;
    }
    .xgboost {
        border-color: #2196F3;
    }
    .catboost {
        border-color: #FF9800;
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

# Chargement des différents datasets
cv_results_summary = load_data("data/cv_results_summary.csv")
evaluation_summary = load_data("data/evaluation_summary.csv")
predictions_catboost = load_data("data/predictions_catboost.csv")
predictions_lightgbm = load_data("data/predictions_lightgbm.csv")
predictions_xgboost = load_data("data/predictions_xgboost.csv")

def plot_confusion_matrix(y_true, y_pred, model_name, class_labels=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum(axis=1)[:, np.newaxis]

    fig = make_subplots(
        rows=2, cols=1,  
        subplot_titles=("Confusion Matrix - Counts", "Confusion Matrix - Percentages"),
        vertical_spacing=0.35,  
        specs=[[{"type": "heatmap"}], [{"type": "heatmap"}]]
    )

    # Définir les labels des classes
    if class_labels is None:
        class_labels = np.unique(y_true)  # Utilise les classes uniques si non fournies
    
    # Matrice des comptes
    fig.add_trace(
        go.Heatmap(z=cm, colorscale="Blues", 
                   showscale=False, text=cm, texttemplate="%{text}",
                   x=class_labels, y=class_labels,
                   hovertemplate="Prédiction: %{x}<br>Réel: %{y}<br>Nombre: %{text}<extra></extra>"),
        row=1, col=1
    )

    # Matrice des pourcentages
    fig.add_trace(
        go.Heatmap(z=cm_percent, colorscale="Viridis", 
                   showscale=False, text=cm_percent, texttemplate="%{text:.1%}",
                   x=class_labels, y=class_labels,
                   hovertemplate="Prédiction: %{x}<br>Réel: %{y}<br>Pourcentage: %{text:.1%}<extra></extra>"),
        row=2, col=1
    )

    fig.update_layout(
        title=f"Matrices de Confusion - {model_name}",
        height=700,  
        margin=dict(l=0, r=0, t=60, b=40),
    )

    # Axes labels avec les classes
    fig.update_xaxes(title_text="Prédiction", tickvals=list(range(len(class_labels))), ticktext=class_labels, row=1, col=1)
    fig.update_yaxes(title_text="Valeur réelle", tickvals=list(range(len(class_labels))), ticktext=class_labels, row=1, col=1)
    fig.update_xaxes(title_text="Prédiction", tickvals=list(range(len(class_labels))), ticktext=class_labels, row=2, col=1)
    fig.update_yaxes(title_text="Valeur réelle", tickvals=list(range(len(class_labels))), ticktext=class_labels, row=2, col=1)

    return fig


def plot_roc_curve(df, title="Courbe ROC"):
    
    # Détecter les colonnes de probabilités
    classes = df['y_true'].unique()
    
    # Créer la figure
    fig = make_subplots(rows=1, cols=1)
    
    # Tracer la ligne de référence (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1], 
            y=[0, 1], 
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Aléatoire (AUC = 0.5)'
        )
    )
    
    # Tracer les courbes ROC pour chaque classe
    for i, class_name in enumerate(classes):
        prob_col = f"prob_class_{class_name}"
        
        # Créer des labels binaires pour cette classe
        y_true_binary = (df['y_true'] == class_name).astype(int)
        
        # Calculer la courbe ROC
        fpr, tpr, _ = roc_curve(y_true_binary, df[prob_col])
        roc_auc = auc(fpr, tpr)
        
        # Ajouter la courbe au graphique
        fig.add_trace(
            go.Scatter(
                x=fpr, 
                y=tpr,
                mode='lines',
                name=f'{class_name} (AUC = {roc_auc:.3f})'
            )
        )
    
    # Mise en forme du graphique
    fig.update_layout(
        title=title,
        xaxis_title='Taux de faux positifs',
        yaxis_title='Taux de vrais positifs',
        plot_bgcolor='white',
        hovermode='closest',
    )
    
    return fig

# Données des modèles optimisés
model_optimizations = {
    "LightGBM": {
        "hyperparameters": {
            'colsample_bytree': 0.8925879806965068, 
            'learning_rate': 0.06990213464750791, 
            'max_depth': 9, 
            'n_estimators': 293, 
            'num_leaves': 83, 
            'subsample': 0.73338144662399
        },
        "best_cv_score": 0.9650,
        "execution_time": 1448.83,
        "stats": {
            "mean": 0.9478,
            "max": 0.9650,
            "min": 0.8569,
            "std": 0.0294
        }
    },
    "XGBoost": {
        "hyperparameters": {
            'colsample_bytree': 0.5779972601681014, 
            'gamma': 0.05808361216819946, 
            'learning_rate': 0.26985284373248053, 
            'max_depth': 6, 
            'n_estimators': 153, 
            'subsample': 0.8540362888980227
        },
        "best_cv_score": 0.9610,
        "execution_time": 1292.19,
        "stats": {
            "mean": 0.9149,
            "max": 0.9610,
            "min": 0.6929,
            "std": 0.0658
        }
    },
    "CatBoost": {
        "hyperparameters": {
            'colsample_bylevel': 0.5370223258670452, 
            'depth': 9, 
            'iterations': 299, 
            'l2_leaf_reg': 2.158690595251297, 
            'learning_rate': 0.2689310277626781, 
            'subsample': 0.811649063413779
        },
        "best_cv_score": 0.9148,
        "execution_time": 2259.50,
        "stats": {
            "mean": 0.8379,
            "max": 0.9148,
            "min": 0.7324,
            "std": 0.0615
        }
    }
}
# En-tête principal avec logo et titre
col_logo, col_title = st.columns([2, 5])
with col_logo:
    st.image("image/top-5.jpg", width=275)
with col_title:
    st.markdown('<h1 class="main-header">Évaluation des Modèles de Classification</h1>', unsafe_allow_html=True)

# Barre de navigation en tabs
tabs = st.tabs(["📊 Vue d'ensemble", "🔍 Détails des Modèles", "⭐ Meilleurs Modèles", "📈 Matrices de Confusion", "📉 Courbes ROC"])

# Tab 1: Vue d'ensemble
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ## 🎯 Présentation de l'analyse
    
    Cette page présente les résultats des différents modèles de classification 
    qui ont été optimisés pour prédire les ligues de football. Vous pouvez explorer les performances des modèles, 
    les matrices de confusion et les courbes ROC.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Création d'un graphique comparatif pour les métriques
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    comparison_df = evaluation_summary.melt(id_vars=["Modèle"], value_vars=metrics, var_name="Métrique", value_name="Valeur")
    
    fig = px.bar(
        comparison_df, 
        x="Métrique", 
        y="Valeur", 
        color="Modèle",
        barmode="group",
        title="Comparaison des performances des meilleurs modèles",
        color_discrete_map={"LightGBM": "#4CAF50", "XGBoost": "#2196F3", "CatBoost": "#FF9800"}
    )
    
    fig.update_layout(
        yaxis_title="Score",
        yaxis=dict(range=[0.8, 1.0]),
        legend_title_text="",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Temps d'exécution et score CV
    cv_times = []
    cv_scores = []
    cv_stds = []
    
    for model in ["LightGBM", "XGBoost", "CatBoost"]:
        cv_times.append(model_optimizations[model]["execution_time"])
        cv_scores.append(model_optimizations[model]["best_cv_score"])
        cv_stds.append(model_optimizations[model]["stats"]["std"])
    
    cv_df = pd.DataFrame({
        "Modèle": ["LightGBM", "XGBoost", "CatBoost"],
        "Temps d'exécution (s)": cv_times,
        "Score CV": cv_scores,
        "Écart-type": cv_stds
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = px.bar(
            cv_df,
            x="Modèle",
            y="Temps d'exécution (s)",
            color="Modèle",
            title="Temps d'optimisation des hyperparamètres",
            color_discrete_map={"LightGBM": "#4CAF50", "XGBoost": "#2196F3", "CatBoost": "#FF9800"}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig = px.bar(
            cv_df,
            x="Modèle",
            y="Score CV",
            error_y="Écart-type",
            color="Modèle",
            title="Meilleur score de validation croisée",
            color_discrete_map={"LightGBM": "#4CAF50", "XGBoost": "#2196F3", "CatBoost": "#FF9800"}
        )
        fig.update_layout(yaxis=dict(range=[0.8, 1.0]))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
# Tab 2: Détails des Modèles
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Création d'un graphique comparatif pour les métriques
    
    fig = px.bar(
            cv_results_summary,
            x="Modèle",
            y="Temps d'exécution (s)",
            color="Modèle",
            title="Temps d'execution",
        )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="card">', unsafe_allow_html=True)
    # Création d'un graphique comparatif pour les métriques
    fig = px.bar(
        cv_results_summary,
        x="Modèle",
        y="Précision CV Moyenne",
        error_y="Écart-type",
        color="Modèle",
        title="Score de validation croisée",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 3: Meilleurs Modèles
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Hyperparamètres Optimisés</h3>', unsafe_allow_html=True)
    
    best_model = "LightGBM"  # Le modèle avec le meilleur score
    
    # LightGBM
    st.markdown(f"""
    <div class="model-card lightgbm">
        <h4>LightGBM <span class="highlight">{'(Meilleur modèle)' if best_model == 'LightGBM' else ''}</span></h4>
        
        Meilleur score CV: {model_optimizations["LightGBM"]["best_cv_score"]:.4f}
        Temps d'exécution: {model_optimizations["LightGBM"]["execution_time"]:.2f} secondes
        
        Meilleurs hyperparamètres:
        - colsample_bytree: {model_optimizations["LightGBM"]["hyperparameters"]["colsample_bytree"]:.4f}
        - learning_rate: {model_optimizations["LightGBM"]["hyperparameters"]["learning_rate"]:.4f}
        - max_depth: {model_optimizations["LightGBM"]["hyperparameters"]["max_depth"]}
        - n_estimators: {model_optimizations["LightGBM"]["hyperparameters"]["n_estimators"]}
        - num_leaves: {model_optimizations["LightGBM"]["hyperparameters"]["num_leaves"]}
        - subsample: {model_optimizations["LightGBM"]["hyperparameters"]["subsample"]:.4f}
        
        Statistiques d'optimisation:
        - Score moyen: {model_optimizations["LightGBM"]["stats"]["mean"]:.4f}
        - Score max: {model_optimizations["LightGBM"]["stats"]["max"]:.4f}
        - Score min: {model_optimizations["LightGBM"]["stats"]["min"]:.4f}
        - Écart-type: {model_optimizations["LightGBM"]["stats"]["std"]:.4f}
    </div>
    """, unsafe_allow_html=True)
    
    # XGBoost
    st.markdown(f"""
    <div class="model-card xgboost">
        <h4>XGBoost <span class="highlight">{'(Meilleur modèle)' if best_model == 'XGBoost' else ''}</span></h4>
        
        Meilleur score CV: {model_optimizations["XGBoost"]["best_cv_score"]:.4f}
        Temps d'exécution: {model_optimizations["XGBoost"]["execution_time"]:.2f} secondes
        
        Meilleurs hyperparamètres:
        - colsample_bytree: {model_optimizations["XGBoost"]["hyperparameters"]["colsample_bytree"]:.4f}
        - gamma: {model_optimizations["XGBoost"]["hyperparameters"]["gamma"]:.4f}
        - learning_rate: {model_optimizations["XGBoost"]["hyperparameters"]["learning_rate"]:.4f}
        - max_depth: {model_optimizations["XGBoost"]["hyperparameters"]["max_depth"]}
        - n_estimators: {model_optimizations["XGBoost"]["hyperparameters"]["n_estimators"]}
        - subsample: {model_optimizations["XGBoost"]["hyperparameters"]["subsample"]:.4f}
        
        Statistiques d'optimisation:
        - Score moyen: {model_optimizations["XGBoost"]["stats"]["mean"]:.4f}
        - Score max: {model_optimizations["XGBoost"]["stats"]["max"]:.4f}
        - Score min: {model_optimizations["XGBoost"]["stats"]["min"]:.4f}
        - Écart-type: {model_optimizations["XGBoost"]["stats"]["std"]:.4f}
    </div>
    """, unsafe_allow_html=True)
    
    # CatBoost
    st.markdown(f"""
    <div class="model-card catboost">
        <h4>CatBoost <span class="highlight">{'(Meilleur modèle)' if best_model == 'CatBoost' else ''}</span></h4>
        
        Meilleur score CV: {model_optimizations["CatBoost"]["best_cv_score"]:.4f}
        Temps d'exécution: {model_optimizations["CatBoost"]["execution_time"]:.2f} secondes
        
        Meilleurs hyperparamètres:
        - colsample_bylevel: {model_optimizations["CatBoost"]["hyperparameters"]["colsample_bylevel"]:.4f}
        - depth: {model_optimizations["CatBoost"]["hyperparameters"]["depth"]}
        - iterations: {model_optimizations["CatBoost"]["hyperparameters"]["iterations"]}
        - l2_leaf_reg: {model_optimizations["CatBoost"]["hyperparameters"]["l2_leaf_reg"]:.4f}
        - learning_rate: {model_optimizations["CatBoost"]["hyperparameters"]["learning_rate"]:.4f}
        - subsample: {model_optimizations["CatBoost"]["hyperparameters"]["subsample"]:.4f}
        
        Statistiques d'optimisation:
        - Score moyen: {model_optimizations["CatBoost"]["stats"]["mean"]:.4f}
        - Score max: {model_optimizations["CatBoost"]["stats"]["max"]:.4f}
        - Score min: {model_optimizations["CatBoost"]["stats"]["min"]:.4f}
        - Écart-type: {model_optimizations["CatBoost"]["stats"]["std"]:.4f}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Comparaison des métriques d\'évaluation</h3>', unsafe_allow_html=True)
    
    # Afficher le tableau de comparaison
    st.dataframe(evaluation_summary, use_container_width=True)
    
    # Créer un graphique radar pour comparer les métriques
    categories = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    
    fig = go.Figure()
    
    for i, model in enumerate(evaluation_summary["Modèle"]):
        values = evaluation_summary.iloc[i, 1:].tolist()
        values.append(values[0])  # Fermer le polygone
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=model,
            line_color=["#4CAF50", "#2196F3", "#FF9800"][i]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.9, 1.0]
            )),
        showlegend=True,
        title="Comparaison des métriques d'évaluation"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 4: Matrices de Confusion
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Matrices de Confusion</h3>', unsafe_allow_html=True)
    
    # Sélection du modèle à afficher
    selected_model_confusion = st.selectbox(
        "Sélectionnez un modèle pour voir sa matrice de confusion:",
        ["LightGBM", "XGBoost", "CatBoost"],
        key="confusion_model"
    )
    
    if selected_model_confusion == "LightGBM":
        y = predictions_lightgbm
    elif selected_model_confusion == "XGBoost":
        y = predictions_xgboost
    else:  # CatBoost
        y = predictions_catboost
    
    # Afficher la matrice de confusion
    cm_fig = plot_confusion_matrix(y["y_true"], y["y_pred"], selected_model_confusion)
    st.plotly_chart(cm_fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Tab 5: Courbes ROC
with tabs[4]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="sub-header">Courbes ROC</h3>', unsafe_allow_html=True)
    
    # Sélection du modèle à afficher
    selected_model_roc = st.selectbox(
        "Sélectionnez un modèle pour voir sa courbe ROC:",
        ["LightGBM", "XGBoost", "CatBoost"],
        key="roc_model"
    )
    
    if selected_model_roc == "LightGBM":
        y = predictions_lightgbm
    elif selected_model_roc == "XGBoost":
        y = predictions_xgboost
    else:  # CatBoost
        y = predictions_catboost
    
    # Afficher les courbes ROC
    roc_fig = plot_roc_curve(y, title=f"Courbes ROC - {selected_model_roc}")
    st.plotly_chart(roc_fig, use_container_width=True)
    
    # Explication des courbes ROC
    st.markdown("""
    ### Interprétation des courbes ROC
    
    - L'axe X représente le taux de faux positifs (1 - spécificité)
    - L'axe Y représente le taux de vrais positifs (sensibilité)
    - Une courbe plus proche du coin supérieur gauche indique une meilleure performance
    - L'AUC (Area Under the Curve) est une mesure globale de la performance:
        - AUC = 1.0: classification parfaite
        - AUC = 0.5: classification aléatoire (ligne diagonale)
    
    Les courbes montrent que tous les modèles ont d'excellentes performances, avec LightGBM légèrement supérieur dans la plupart des classes.
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Barre latérale avec des informations supplémentaires
with st.sidebar:
    st.markdown("## Informations sur le projet")
    
    st.markdown("""
    
    ### 🏆 Meilleur modèle
    **LightGBM** a obtenu les meilleures performances globales avec:
    - Accuracy: 0.965
    - F1 Score: 0.965
    - ROC AUC: 0.998
    
    ### 📂 Données utilisées
    - cv_results_summary.csv
    - evaluation_summary.csv
    - predictions_lightgbm.csv
    - predictions_xgboost.csv
    - predictions_catboost.csv
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 📊 Méthodologie
    1. Optimisation des hyperparamètres avec validation croisée
    2. Entraînement des modèles avec les meilleurs hyperparamètres
    3. Évaluation sur un ensemble de test indépendant
    4. Analyse comparative des performances
    """)

# Footer
st.markdown("---")
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Développé avec Streamlit • Dernière mise à jour: Février 2025")
url = "https://github.com/heretounderstand/football_league_classification"
st.markdown("Github Repo : [link](%s)" % url)
st.markdown("📧 kadrilud@gmail.com")
st.markdown("</div>", unsafe_allow_html=True)
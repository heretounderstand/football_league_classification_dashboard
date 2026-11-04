# Quelle ligue ? Dashboard

Dashboard Streamlit du projet [`football_league_classification`](https://github.com/heretounderstand/football_league_classification) :
peut-on reconnaître la ligue d'un match (10 premières et deuxièmes divisions européennes) à partir de ses statistiques ?

## Pages

| Page | Contenu |
|---|---|
| **Le projet** | La question, les résultats clés et l'audit chiffré de l'ancienne version (96 % → fuites de données) |
| **Données & styles de jeu** | Couverture des statistiques par ligue et saison, carte d'identité statistique de chaque ligue, distributions, évolution dans le temps |
| **Performance des modèles** | Comparaison avec IC à 95 %, matrice de confusion, rappel et précision par ligue, accuracy par saison, calibration |
| **Ce qui distingue les ligues** | Importance par permutation (par famille et par variable), confusions les plus fréquentes |
| **Explorer les prédictions** | Parcourir les 7 809 matchs de test, filtrer, voir les probabilités d'un match |
| **Simulateur** | Saisir les statistiques d'un match et obtenir les probabilités par ligue |

Encodage visuel : la couleur indique le pays (palette vérifiée pour le daltonisme), les hachures ou les
tirets indiquent la division 2.

## Lancer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Données

`data/` contient uniquement les artefacts (environ 4,5 Mo) produits par `python -m flc.pipeline` dans le
dépôt de classification. Pour les mettre à jour :

```bash
cp ../football_league_classification/artifacts/* data/
```

`core/features.py` reprend le calcul des variables « match » et « contexte » du dépôt de classification
pour le simulateur. Gardez les deux fichiers synchronisés.

## Structure

```
app.py               navigation
core/data.py         chargement (mis en cache) des artefacts
core/style.py        palette, mise en forme Plotly
core/features.py     variables du simulateur
views/*.py           une page par fichier
data/                artefacts du pipeline
```

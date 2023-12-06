# Classification des radiographies pulmonaires avec un réseau de neurones convolutifs
Dans le cadre du cours MMD6020, par Gilbert Jabbour et Fayçal Zine-Eddine.

## Installation des dépendances
Le fichier environment.yml peut être utilisé pour créer l'environnement de travail et installer les librairies nécessaires.

## Visualisation des données et ingénierie des caractéristiques (Feature Engineering)
Le pre-processing et la visualisation ont lieu dans un Jupyter Notebook accessible directement sur Kaggle via ce lien : https://www.kaggle.com/code/faycalzineeddine/chest-x-ray-detection

Comme le dataset est très grand, les images ont été extraites de façon séquentielle, transformées, puis les différents jeux de données utilisés pour l'entraînement, la validation et le test ont été enregistré directement de façon numérique en format parquet. Ces donnnées sont accessibles dans le dossier data.

## Entraînement du modèle
Le ficher main.py est celui qui contient la principale loop d'entraînement. Il extrait ses fonctions principales et ses paramètres des fichiers ulils.py et des yaml.
L'entraînement du modèle a été enregistré sur weights and biases. 

## Résultats
Les résultats sont disponibles dans le dossier results.

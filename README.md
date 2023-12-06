# Classification des radiographies pulmonaires avec un réseau de neurones convolutifs
Dans le cadre du cours MMD6020, par Gilbert Jabbour et Fayçal Zine-Eddine.

## Installation des dépendances
Le fichier environment.yml peut être utilisé pour créer l'environnement de travail et installer les librairies nécessaires.

## Visualisation des données et ingénierie des caractéristiques (Feature Engineering)
Le pre-processing et la visualisation ont lieu dans un Jupyter Notebook accessible directement sur Kaggle via ce lien : https://www.kaggle.com/code/faycalzineeddine/chest-x-ray-detection
Si le notebook est utilisé localement, il faudra télécharger le jeu de données https://www.kaggle.com/datasets/nih-chest-xrays/data (45 Go) et changer le filepath dans le notebook.

Comme le dataset est très grand, les images ont été extraites de façon séquentielle, transformées, puis les différents jeux de données utilisés pour l'entraînement, la validation et le test ont été enregistré directement de façon numérique en format parquet. Ces donnnées sont accessibles via le lien suivant : https://www.dropbox.com/scl/fo/8p6gxxp4lwm26s9qzmw5u/h?rlkey=lu7b5imzk0846mhg2028mr3jr&dl=0
Les fichiers restent volumineux, donc nous avons préféré ne pas les stocker sur Github en plus pour éviter de gaspiller du stockage.

## Entraînement du modèle
Le ficher main.py est celui qui contient la principale loop d'entraînement. Il extrait ses fonctions principales et ses paramètres des fichiers ulils.py et des yaml.
L'entraînement du modèle a été enregistré sur weights and biases. 

## Résultats
Les résultats sont disponibles dans le dossier results.

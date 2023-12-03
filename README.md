# NIH-Chest-XRay
Repo for the class MMD 6020

Dépôt temporaire, code à rebaser, j'ai surtout travaillé sur des notebook pour l'instant pour pouvoir utiliser des GPU sur Colab/Kaggle.

Tentative initiale de classification multiclasse avec 5 catégories ['Pneumonie', 'Edema', 'Both', 'Normal', 'Others'], mais trop gros class imbalance. Classificateurs qui performent très mal, malgré tentative d'équilibrer les classes et d'utiliser un weighted loss.

Je pense donc qu'il est préférable d'utiliser un classificateur de pneumonie vs oedème pulmonaire, à utiliser quand on voit une anomalie sur la radiographie pulmonaire et qu'on se questionne sur sa nature. 


# Cahier des Charges : Bibliothèque d'Arbres de Décision en TypeScript

## 1. Objectif Général

Développer une bibliothèque TypeScript robuste, modulaire, performante et facile à utiliser pour la création, l'entraînement et l'utilisation de modèles d'arbres de décision pour des tâches de classification et de régression. La bibliothèque devra offrir une API claire et des fonctionnalités comparables en termes de complétude à une bibliothèque de réseaux de neurones moderne.

## 2. Fonctionnalités Clés

2.1. Types d'Arbres de Décision * Arbre de Classification: * Prédire des étiquettes de classe discrètes. * Calculer les probabilités d'appartenance à chaque classe. * Arbre de Régression: * Prédire des valeurs continues.

2.2. Construction de l'Arbre (Apprentissage / fit) * Algorithmes de Construction: * Implémentation d'un algorithme de base (ex: variante de CART). * Critères de Division (Split Criteria): * Pour la classification: Indice de Gini, Entropie (Gain d'information). * Pour la régression: Erreur Quadratique Moyenne (MSE), Erreur Absolue Moyenne (MAE). * Gestion des Types de Features: * Features numériques (continues ou discrètes). * Features catégorielles (avec gestion appropriée, ex: one-hot encoding implicite ou gestion native si possible). * Paramètres de Contrôle de la Croissance: * max_depth: Profondeur maximale de l'arbre. * min_samples_split: Nombre minimum d'échantillons requis pour diviser un nœud interne. * min_samples_leaf: Nombre minimum d'échantillons requis dans un nœud feuille. * min_impurity_decrease: Seuil minimal de réduction de l'impureté pour effectuer une division. * Gestion des Valeurs Manquantes (Optionnel, pour complétude avancée): * Stratégies de base pour gérer les NaN (ex: imputation simple, envoi des échantillons dans les deux branches avec pondération).

2.3. Prédiction (predict, predict_proba) * Traversée efficace de l'arbre pour de nouvelles instances. * predict(X): Retourne les prédictions (classes ou valeurs). * predict_proba(X) (pour classification): Retourne les probabilités des classes.

2.4. Élagage (Pruning) * Mécanismes pour réduire le surapprentissage. * Exemple: Élagage basé sur la complexité du coût (Cost-Complexity Pruning) minimal. * Paramètres configurables pour l'élagage.

2.5. Évaluation du Modèle * Fournir des fonctions ou s'intégrer avec des utilitaires pour calculer des métriques de performance courantes : * Classification: Accuracy, Précision, Rappel, Score F1, Matrice de Confusion. * Régression: MSE, MAE, R².

2.6. Importance des Features * Calculer et exposer l'importance de chaque feature dans le modèle entraîné (ex: basée sur la réduction moyenne de l'impureté).

2.7. Sérialisation et Désérialisation * Sauvegarder la structure et les paramètres d'un arbre entraîné (ex: au format JSON). * Charger un modèle sauvegardé pour réutilisation.

2.8. Visualisation (Fortement Recommandé) * Capacité à exporter la structure de l'arbre dans un format lisible ou visualisable (ex: format DOT pour Graphviz, ou une structure JSON simple pour des rendus personnalisés).

2.9. (Avancé) Ensembles d'Arbres - Pour une Complétude Maximale * Random Forest (Forêt Aléatoire): * Pour la classification et la régression. * Construction de multiples arbres sur des sous-ensembles d'échantillons (bagging) et de features. * Agrégation des prédictions (vote majoritaire pour classification, moyenne pour régression). * Paramètres spécifiques: n_estimators (nombre d'arbres), max_features (nombre de features à considérer pour chaque split).

## 3. Architecture et Conception

3.1. Modularité: * Séparation claire des responsabilités: * Structure de Nœud (Node). * Structure d'Arbre (Tree). * Logique des critères de division. * Algorithmes d'apprentissage. * Fonctions de prédiction. 3.2. API: * Interface utilisateur intuitive et cohérente, s'inspirant potentiellement d'APIs populaires (ex: scikit-learn). * Méthodes principales: fit(X, y), predict(X), predict_proba(X). * Configuration du modèle via des paramètres au constructeur ou des méthodes dédiées. 3.3. Typage: * Utilisation rigoureuse de TypeScript pour la robustesse et la clarté du code. * Définition de types clairs pour les données d'entrée, les paramètres et les sorties. 3.4. Performance: * Optimisation des algorithmes de construction et de prédiction pour une bonne performance, notamment avec de grands datasets. * Utilisation de structures de données efficaces. 3.5. Extensibilité: * Conception permettant d'ajouter facilement de nouveaux critères de division, stratégies d'élagage ou même de nouveaux types d'arbres/ensembles à l'avenir.

## 4. Données d'Entrée et de Sortie

Entrée (X): Accepter des tableaux 2D (ou structures similaires) de nombres pour les features. Pour les features catégorielles, définir une convention (ex: pré-encodées numériquement ou gestion interne).
Cibles (y): Accepter des tableaux 1D de nombres (pour régression) ou d'étiquettes (numériques ou chaînes pour classification).
Sortie de predict: Tableau 1D des prédictions.
Sortie de predict_proba: Tableau 2D des probabilités par classe.

## 5. Documentation et Tests

5.1. Documentation: * Documentation complète de l'API (chaque classe, méthode, paramètre). * Tutoriels et exemples d'utilisation pour la classification et la régression. * Explication des concepts clés et des algorithmes implémentés. 5.2. Tests: * Couverture de tests unitaires exhaustive pour tous les modules. * Tests d'intégration pour valider le flux complet (entraînement, prédiction, évaluation). * Tests de non-régression.

## 6. 

```txt
am-decisiontree
├── .gitattributes
├── .gitignore
├── .nvmrc
├── biome.json
├── jsr.json
├── LICENSE
├── package.json
├── README.md
├── tsconfig.json
├── .github/
│   └── workflows/  // Optionnel
│       ├── ci.yml
│       └── ...
├── .vscode/
│   └── settings.json
├── examples/
│   ├── iris_classification.ts
│   ├── simple_classifier.ts
│   └── regression_tree_example.ts
└── src/
    ├── core/
    │   ├── mod.ts
    │   ├── decision_tree.ts
    │   └── node.ts
    ├── criteria/
    │   ├── mod.ts
    │   ├── gini_impurity.ts
    │   ├── entropy.ts
    │   └── mse_criterion.ts
    └── utils/      // Optionnel
        └── mod.ts
```


# Projet Titanic

Ce projet utilise un modèle de machine learning pour prédire la survie des passagers du Titanic. Les étapes incluent le chargement des données, la prétraitement, l'entraînement du modèle, et l'évaluation.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les éléments suivants :

- Python 3.12
- pip (Python package installer)
- Poetry (gestionnaire de dépendances pour Python)

## Installation

1. Clonez le dépôt sur votre machine locale :

```bash
git clone https://github.com/KillianB94430/Titanic.git
cd Titanic
```

2. Installez les dépendances avec Poetry :

```bash
poetry install
```

## Structure du Projet

```
.
├── Data
│   └── train.csv          # Jeu de données d'entraînement
├── Results
│   └── submission.csv     # Fichier de soumission
├── src
│   ├── titanic_train.py   # Script d'entraînement du modèle
│   └── titanic_pred.py    # Script de prédiction
└── README.md              # Documentation du projet
```

## Utilisation

### Entraînement du Modèle

Pour entraîner le modèle, exécutez le script `titanic_train.py` :

```bash
poetry run python src/titanic_train.py
```

### Faire des Prédictions

Pour faire des prédictions sur un jeu de données de test, exécutez le script `titanic_pred.py` :

```bash
poetry run python src/titanic_pred.py
```

### Pour lancer les deux script en même temps
```bash
poetry run python src/main.py
```

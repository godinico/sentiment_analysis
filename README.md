# Projet sur l'Analyse de Sentiments de Tweets

## Description du Projet
Ce projet a été développé pour Air Paradis, une compagnie aérienne cherchant à anticiper les bad buzz sur les réseaux sociaux. L'objectif est de créer un système de prédiction de sentiment basé sur le contenu des tweets, permettant d'identifier les messages négatifs nécessitant une attention particulière.

## Fonctionnalités
- Prédiction du sentiment (positif/négatif) associé à un tweet
- API de prédiction déployée sur Azure
- Suivi des performances du modèle via Azure Application Insights
- Alertes automatiques en cas de prédictions incorrectes fréquentes

## Structure du Projet
```

├── mlruns/                           # Tracking et enregistrement des modèles via MLflow
├── .github/
│   └── workflows/
│       └── azure_deploy.yml          # Pipeline CI/CD pour le déploiement sur Azure via GitHub Actions
├── notebook_modeles_sur_mesure.ipynb # Notebook d'entraînement et évaluation des modèles
├── app.py                            # Code de l'API Flask
├── modele LSTM.h5                    # Modèle LSTM entraîné utilisé par l'API
├── tokenizer.pkl                     # Tokenizer entraîné pour le prétraitement du texte
├── tests.py                          # Tests unitaires pour l'API
├── api_test_interface.py             # Interface Streamlit pour tester l'API déployée sur Azure
└── requirements.txt                  # Liste des dépendances de l'API
```

## Approche MLOps
Ce projet suit une approche MLOps afin d'assurer la qualité, la reproductibilité et le déploiement continu du modèle :

1. **Gestion des expérimentations** : Utilisation de MLflow pour suivre les différentes expériences, les paramètres et les métriques des modèles testés.
2. **Versioning** : Gestion du code source via Git/GitHub pour suivre les modifications et faciliter la collaboration.
3. **Tests automatisés** : Intégration de tests unitaires pour valider le fonctionnement de l'API et la qualité des prédictions.
4. **Déploiement continu** : Pipeline CI/CD via GitHub Actions pour déployer automatiquement l'API sur Azure à chaque modification validée.
5. **Monitoring** : Intégration d'Azure Application Insights pour suivre les performances du modèle en production et identifier les prédictions erronées.

## Modèles Développés
Plusieurs approches ont été testées pour la prédiction de sentiment :
- Modèle sur mesure simple (régression logistique)
- Modèle sur mesure avancés (LSTM et CNN)
- Modèle avancé BERT

Le modèle final retenu est un réseau de neurones LSTM, offrant le meilleur compromis entre performance et rapidité d'inférence.

## Installation et Utilisation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installation
```bash
pip install -r requirements.txt
```

### Lancement de l'API en local
```bash
python app.py
```

### Exemple d'utilisation
Pour une utilisation de l'API en local : une fois l'API exécutée (commande `python app.py`), celle-ci peut-être appelée comme suit pour prédire le sentiment associé à un tweet :

```python
import requests

url = "http://localhost:5000/predict"
data = {"text": "The flight was cancelled without explanation and the customer service was unreachable"}
response = requests.post(url, json=data)
print(response.json())
```

Une interface de test Streamlit est également disponible afin d'envoyer des requêtes à la version de l'API déployée sur le Cloud (voir section Déploiement ci-après). Cette interface se lance via la commande `streamlit run .\api_test_interface.py`. L'interface de test devrait ensuite se lancer automatiquement dans une page de votre navigateur web.

## Déploiement
L'API est automatiquement déployée sur Azure Web App via le pipeline CI/CD configuré dans le fichier `.github/workflows/azure_deploy.yml`. Chaque push sur la branche principale déclenche l'installation des dépendances, une série de tests unitaires puis le déploiement si les tests sont réussis.

## Monitoring et Amélioration Continue
Le modèle en production est suivi via Azure Application Insights pour :
- Identifier les tweets mal classifiés
- Déclencher des alertes en cas d'augmentation des erreurs de prédiction
- Collecter des données pour l'amélioration future du modèle

## Auteur
Projet réalisé par Nicolas G. dans le cadre de la formation AI Engineer d'OpenClassrooms

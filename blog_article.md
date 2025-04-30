# Détection de Sentiments dans les Tweets : Comparaison de Modèles et Mise en Œuvre d'une Démarche MLOps

## 1. Introduction

<p align="justify">
Dans un monde où les réseaux sociaux jouent un rôle clé dans la communication, l'analyse des sentiments exprimés dans les tweets est devenue une tâche cruciale pour les entreprises, les chercheurs et les décideurs. C'est notamment le cas de la compagnie aérienne Air Paradis, qui souhaite acquérir une solution d'intelligence artificielle capable de prédire le sentiment associé à un tweet, afin d'anticiper et de gérer les éventuels bad buzz pouvant l'affecter.
</p>

<p align="justify">
Cet article décrit l'approche adoptée pour répondre au besoin d'Air Paradis, en comparant différents modèles de prédiction de sentiment et en intégrant une méthodologie MLOps (Machine Learning Operations) tout au long des différentes étapes.
</p>

<p align="justify">
Au cours de ce projet, nous avons testé et comparé trois approches pour la détection de sentiments dans les tweets : un modèle sur mesure simple, des modèles sur mesure avancés basés sur des réseaux de neurones profonds (Deep Learning), et un modèle avancé <a href="https://huggingface.co/docs/transformers/model_doc/bert">BERT</a>. En parallèle, une démarche MLOps a été mise en place pour garantir une gestion efficace du cycle de vie du modèle, de son développement à son déploiement en production.
</p>

<p align="justify">
Nous aborderons également le suivi des performances en production grâce à
   <a href="https://learn.microsoft.com/fr-fr/azure/azure-monitor/app/app-insights-overview">Azure Application Insights</a> et proposerons une démarche pour analyser les statistiques et améliorer le modèle dans le temps.
</p>

## 2. Les Modèles de Prédiction de Sentiments

Afin de développer différents modèles pour prédire les sentiments associés à des tweets, nous disposions du jeu de données [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) qui regroupe un ensemble de 1.6 millions de tweets rédigés en anglais. Chaque tweet est associé à un label binaire indiquant si le sentiment exprimé est positif ou négatif. Voici un exemple de deux tweets présents parmi les données :

   Tweet                                  | Sentiment   |
 |-----------------------------------------|-------------|
 | @machineplay I'm so sorry you're having to go through this. Again.  #therapyfail | Négatif     |
 | @esmeeworld  OMG that was so cool  you gave a shout out to me !! ? lovee yaa ! | Positif     |
 
Ces données labellisées ont ainsi permis d'entraîner différents types de modèles d'apprentissage supervisé, qui sont présentés dans la partie suivante. Avant de pouvoir entrainer les modèles sur cet ensemble de tweets, une étape de nettoyage des textes est nécéssaire. Celle-ci se décompose en différentes opérations :

 - conversion des textes en minuscules,
 - suppresion des mentions @users,
 - suppresion du caractère # pour les "hashtags" ("#therapyfail" devient "therapyfail" ),
 - suppression des caractères spéciaux.

### 2.1. Le Modèle sur Mesure Simple

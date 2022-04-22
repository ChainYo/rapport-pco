# Introduction

## Contexte

Aujourd'hui, il existe plusieurs solutions de surveillance de portefeuilles financiers en ligne, qui permettent le suivi
de l'évolution du cours de différents actifs. Dans le cas des crypto-monnaies, ces outils offrent à l'utilisateur une
interface gratuite et simple lui permettant d'ajouter manuellement ses actifs et de les suivre en temps réel. Aucun à ma
connaissance n'intègre des outils d'analyses et de prédictions des cours des crypto-monnaies.

## Enoncé du problème

Puisqu'aucun outil, disponible gratuitement, n'offre la possibilité de simuler une prédiction de l'évolution des cours
des crypto-monnaies, je souhaite créer un outil permettant de façon automatisée et transparente pour l'utilisateur de
suivre l'évolution des cours des crypto-monnaies. Comment permettra à l'utilisateur de disposer d'informations de
prédictions automatiques sur ses actifs qui soient un minimum fiables ?

## Objectifs

Développer une architecture permettant l'entraînement automatique de modèles de prédiction de l'évolution des cours
des crypto-monnaies, le service de ces modèles via une API REST ainsi qu'une interface web permettant de visualiser
les prédictions et l'évolution des cours des crypto-monnaies par l'utilisateur.

## Approche de la solution

Le projet se décompose en trois composants distincts et qui interagissent entre eux pour former la solution :

* **Interface** : l'interface web permettant à l'utilisateur de visualiser les prédictions et l'évolution des cours
des crypto-monnaies.
* **Serving** : le serveur web qui met à disposition les modèles de prédiction via une API REST.
* **Training** : pipeline automatisé d'entraînement des modèles, de leur validation et de leur stockage.

Voici un schéma de l'architecture du projet :

![Architecture du projet Make Us Rich \label {fig:1.1}](/content/assets/project-architecture.png)

Ce schéma très complet reprend tous les composants du projet et présente les différents liens entre eux. Nous détaillerons
les différents composants et leurs fonctions dans les sections suivantes.

## Contributions et réalisations

Tout ce qui est présenté dans le schéma d'architecture ci-dessus est fonctionnel et déployé. Les composants sont développés
en utilisant le langage de programmation Python et différents outils et librairies très utiles comme *FastAPI*, 
*Pytorch-Lightning*, *Scikit-learn*, etc.

Le projet est open-source et est accessible sur GitHub : [Make Us Rich](https://github.com/ChainYo/make-us-rich).

Il est possible d'y contribuer et de l'améliorer. Il est également possible de simplement l'utiliser et déployer localement
tous les composants du projet. Toutes les étapes de déploiement sont détaillées dans la documentation associée qui est
également disponible sur GitHub : [Documentation](https://chainyo.github.io/make-us-rich/).

## Organisation du rapport

Le rapport s'organise en trois sections :

* **Section 1 - Etat de l'art** : présentation des avancées des modélisations IA et des algorithmes de prédiction sur
des séries temporelles.
* **Section 2 - Make Us Rich** : présentation et détails de la solution et de son architecture.
* **Section 3 - Ouverture** : retour sur le projet et ouverture sur les différents axes d'amélioration du projet.

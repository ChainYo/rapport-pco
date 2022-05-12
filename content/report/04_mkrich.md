\newpage
# Make Us Rich

Il existe beaucoup de ressources, notamment sur les modèles de prédiction appliqués à des séries temporelles, qui sont
très utiles sur des données stationnaires telles que les données météorologiques ou les vols d'avions. En revanche, en 
ce qui concerne les marchés financiers et les crypto-monnaies, il est difficile de trouver des méthodes dites *classiques*
qui soient aussi efficaces et fiables.

Étant passionné par le *Deep Learning* et surveillant d'un oeil positif l'évolution des crypto-monnaies, j'ai souhaité
créer un outil permettant de lier les deux. En plus de créer cet outil de prédiction, j'ai voulu aller plus loin et 
développer le projet pour un faire une base solide à déployer pour quiconque souhaiterait comprendre les enjeux du 
déploiement de modèles de *Deep Learning* et les outils d'automatisation et de supervision des modèles.

C'est donc avant tout un projet de passioné, mais aussi un formidable outil (en tous cas je l'espère) pour apprendre à
automatiser, déployer et mettre à disposition des modèles de *Deep Learning*. Sans plus attendre, je vous invite à 
découvrir plus en détails le projet et son architecture présentée en introduction de ce dossier.

## Les données

TODO : 
- récupération des données
- description des données
- préparation des données (= feature engineering)
- formatage des données (= model inputs)

### Récupération des données

Cette étape de récupération des séries temporelles correspond à l'étape de *fetching* sur le schéma présenté en introduction.
Les données que j'utilise sont des données publiques collectées par beaucoup de plateformes de marchés financiers. Pour
ce projet, j'ai choisi de récupérer les données de la plateforme *Binance*. *Binance* est une plateforme très connue dans
le monde de l'échange de crypto-monnaies et elle dispose d'une API de trading qui permet de récupérer des données très 
simplement et rapidement. Il suffit de disposer d'un compte sur *Binance* et de se générer un token d'accès pour pouvoir
utiliser cette API de façon permanente et sans frais.

*Binance* dispose également d'un package Python (`python-binance`) qui facilite les appels à son API. J'ai donc coder une classe, `BinanceClient`[(1)],
qui permet de gérer les interactions avec l'API de *Binance* et qui inclut des méthodes telles que la récupération de données
sur cinq jours, un an ou une période à définir par l'utilisateur. Ces méthodes requièrent en argument un symbole de crypto-
monnaie et une monnaie de comparaison dans tous les cas et renvoient les données sous forme de `pandas.DataFrame`.

[(1)]: #annexe-1

### Description des données

Les données récupérées sont des séries temporelles de la crypto-monnaie cible comparée à une monnaie. Par exemple, je peux
récupérer les données de la crypto-monnaie *BTC* par rapport à la monnaie *EUR* sur les 5 dernières jours avec une intervalle
de 1 heure. 

On compte 12 colonnes de données :

* `timestamp` : date et heure de la série temporelle.
* `open` : valeur d'ouverture de la crypto-monnaie cible sur l'intervalle.
* `high` : valeur haute de la crypto-monnaie cible sur l'intervalle.
* `low` : valeur basse de la crypto-monnaie cible sur l'intervalle.
* `close` : valeur de clôture de la crypto-monnaie cible sur l'intervalle.
* `volume` : volume d'échange de la crypto-monnaie cible sur l'intervalle.
* `close_time` : date et heure de la clôture de la crypto-monnaie cible sur l'intervalle.
* `quote_av` (quote asset volume) : correspond au volume d'échange de la monnaie cible sur l'intervalle.
* `trades` : correspond au nombre de trades effectués sur l'intervalle.
* `tb_base_av` (taker buy base asset volume) : correspond au volume d'acheteur de la crypto-monnaie cible sur l'intervalle.
* `tb_quote_av` (taker buy quote asset volume) : correspond au volume d'acheteur de la monnaie cible sur l'intervalle.
* `ignore` : correspond à une colonne qui ne sera pas utilisée.

Ce sont des informations classiques que l'on retrouve souvent sur les plateformes de marchés financiers. Nous allons voir
que pour notre cas d'usage, toutes ces données ne seront pas utilisées. De plus, ces données ne seront pas stockées 
puisqu'elles ne sont plus valables après la fin de l'intervalle de récupération et que le projet prévoit un ré-entrainement
toutes les heures des modèles. Ainsi nous n'avons pas besoin de stocker les données pour une utilisation ultérieure, un 
simple appel à l'API suffit pour récupérer les nouvelles données utiles à un entraînement de modèle.

### Préparation des données

Il est nécessaire de préparer les données pour qu'elles soient utilisables par le modèle. Pour cela, nous allons utiliser
plusieurs fonctions définies dans l'étape de `preprocessing` de ce projet.

#### Extraction des données utiles

#### Séparation des jeux de données

#### Mise à l'échelle des données

#### Préparation des séquences de données

## Modélisation

TODO :
- présentation du modèle
- description du modèle (avec code)
- choix des hyperparamètres
- monitoring des entraînements
- validation du modèle
- conversion vers ONNX
- stockage objet des modèles + features engineering

## Service des modèles

TODO :
- contexte global sur serving de modèles (docker, kubernetes, ...)
- présentation API REST
- détails des endpoints de l'API REST

## Interface utilisateur

TODO :
- présentation de l'interface utilisateur
- base de données relationnelle pour authentification
- tokens pour appels API

## Packaging du projet

TODO :
- ETL - Prefect
- Déploiement - Docker, Cloud ou Local
- Documentation - Mkdocs material
- Alerting


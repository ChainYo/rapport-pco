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

Pour conserver une certaine lisibilité, les différents blocs de code sont numérotés entre paranthèses et leur détails 
sont disponibles dans les annexes de ce rapport.

## Les données

La récupération de données et la création d'un jeu de données est la première étape d'un projet de *Machine Learning*, voir
même de *Data Science* en général. C'est une étape qui n'est pas à sous-estimer, car elle va déterminer la réussite de votre projet.

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
plusieurs fonctions définies dans l'étape de `preprocessing` de ce projet. C'est à cette étape qu'intervient un choix des
*features* à utiliser pour l'entraînement du modèle leur *engineering*. Le terme *feature engineering* est un terme 
désignant les différentes étapes de "rafinement" des données pour qu'elles soient utilisables par le modèle.

#### Extraction des données utiles

Tout d'abord, nous allons extraire les données utiles à partir de la série temporelle grâce à la fonction 
`extract_features_from_dataset`[(2)]. Pour cela, nous allons utiliser uniquement les colonnes `open`, `high`, `low`, `close` 
et `timestamp`. Nous stockons également la différence entre la valeur de clôture et la valeur d'ouverture pour chaque 
intervalle sous le nom `close_change`.

Ainsi à l'issu de cette étape, nous obtenons un nouveau `pandas.DataFrame` qui contient les features spécialement
sélectionnées pour l'entraînement. Nous n'incluons pas les colonnes relatives aux volumes d'échange et aux trades, car
ce qui nous intéressera sera la prédiction de la valeur de clôture sur l'intervalle suivant. Il serait néanmoins possible
d'inclure les notions de volumes dans l'entraînement, mais cela complexifierait le modèle et l'alourdirait pour un gain
probablement peu significatif.

[(2)]: #annexe-2

#### Séparation des jeux de données

Une fois nos *features* sélectionnées, nous allons séparer les données en trois jeux de données distincts grâce à la 
fonction `split_data`[(3)]. La séparation consiste à diviser les données en deux jeux de données : 

* `training_set` : 90%
* `test_set` : 10%

Pour des données temporelles il est important de ne pas mélanger la chronologie des données puisque cela peut créer des 
problèmes de cohérence. En effet, nous voulons que le modèle puisse prédire les valeurs de clôture sur l'intervalle suivant 
et non pas sur des intervalles passés.

[(3)]: #annexe-3

#### Mise à l'échelle des données

Il est important de mettre à l'échelle les données pour que le modèle puisse les utiliser correctement. Pour cela, nous
allons utiliser la fonction `scale_data`[(4)]. Cette fonction va permettre de normaliser les données pour que les données 
de nos deux jeux de données soient comprises entre -1 et 1. C'est une technique de normalisation qui permet de réduire
les écarts entre les données et ainsi les rendre plus facile à manipuler par le modèle lors de l'entraînement.

On utilise ici la méthode de normalisation `MinMaxScaler` de la librairie `sklearn`. Il est important de noter que nous
sauvegardons également cet objet de normalisation dans un fichier pickle pour pouvoir l'utiliser plus tard lors de
l'inférence via l'API. En effet, puisque le modèle est entraîné sur des données normalisées, il est primordial qu'elles 
le soient également lors des prédictions postérieures. De plus, nous avons besoin de cet objet pour pouvoir inverser la
normalisation des données prédites et obtenir des valeurs de clôture réelles, c'est-à-dire des valeurs de clôture non
normalisées.

[(4)]: #annexe-4

#### Préparation des séquences de données

Il ne reste plus qu'à préparer les données pour qu'elles soient utilisables par le modèle. Pour cela, nous allons devoir
créer des séquences de données. Pour cela, nous allons utiliser la fonction `create_sequences`[(5)]. Cette fonction va
utiliser les données préalablement normalisées pour créer des séquences de données de taille `sequence_length`.

C'est à cette étape que nous construisons les *features* d'entrée du modèle et la *target* de sortie, aussi appelé *label*.
Dans notre cas, nous utiliserons la colonne `close` comme *target* et le reste des colonnes comme *features*.

Il est à noter que nous alons séparer les données du `training_set` en deux séquences de données distinctes pour avoir
également des séquences de données pour la validation du modèle, grâce à la fonction `split_sequences`[(6)]. Nous utiliserons
comme taille `val_size=0.2` pour la validation du modèle, ce qui représente 18% des données totales attribuées pour la 
validation du modèle.

[(5)]: #annexe-5
[(6)]: #annexe-6

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


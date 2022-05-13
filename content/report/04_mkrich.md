\newpage
# Make Us Rich

Il existe beaucoup de ressources, notamment sur les modèles de prédiction appliqués à des séries temporelles, qui sont
très utiles sur des données stationnaires telles que les données météorologiques ou les vols d'avions. En revanche, en 
ce qui concerne les marchés financiers et les crypto-monnaies, il est difficile de trouver des méthodes dites *classiques*
qui soient aussi efficaces et fiables.

Étant passionné par le *Deep Learning* et surveillant d'un oeil positif l'évolution des crypto-monnaies, j'ai souhaité
créer un outil permettant de lier les deux. En plus de créer cet outil de prédiction, j'ai voulu aller plus loin et 
développer le projet pour en faire une base solide à déployer pour quiconque souhaiterait comprendre les enjeux du 
déploiement de modèles de *Deep Learning* et les outils d'automatisation et de supervision des modèles.

C'est donc avant tout un projet de passioné, mais aussi un formidable outil (en tous cas je l'espère) pour apprendre à
automatiser, déployer et mettre à disposition des modèles de *Deep Learning*. Sans plus attendre, je vous invite à 
découvrir plus en détails le projet et son architecture présentée en introduction de ce dossier.

Pour conserver une certaine lisibilité, les différents blocs de code sont numérotés entre paranthèses et leur détails 
sont disponibles dans les annexes de ce rapport.

## Les données

La récupération de données et la création d'un jeu de données est la première étape d'un projet de *Machine Learning*, 
voir même de *Data Science* en général. C'est une étape qui n'est pas à sous-estimer, car elle va déterminer la réussite 
de votre projet.

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
récupérer les données de la crypto-monnaie *BTC* par rapport à la monnaie *EUR* sur les 5 dernières jours avec un intervalle
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
*features* à utiliser pour l'entraînement du modèle et leur *engineering*. Le terme *feature engineering* est un terme 
désignant les différentes étapes de *raffinage* des données pour qu'elles soient utilisables par le modèle.

#### Extraction des données utiles

Tout d'abord, nous allons extraire les données utiles à partir de la série temporelle grâce à la fonction 
`extract_features_from_dataset()`[(2)]. Pour cela, nous allons utiliser uniquement les colonnes `open`, `high`, `low`, `close` 
et `timestamp`. Nous stockons également la différence entre la valeur de clôture et la valeur d'ouverture pour chaque 
intervalle sous le nom `close_change`.

Ainsi à l'issu de cette étape, nous obtenons un nouveau `pandas.DataFrame` qui contient les features spécialement
sélectionnées pour l'entraînement. Nous n'incluons pas les colonnes relatives aux volumes d'échange et aux trades, car
c'est la prédiction de la valeur de clôture sur l'intervalle suivant qui nous intéresse ici. Il serait néanmoins possible
d'inclure les notions de volumes dans l'entraînement, mais cela complexifierait le modèle et l'alourdirait pour un gain
potentiel à déterminer.

[(2)]: #annexe-2

#### Séparation des jeux de données

Une fois nos *features* sélectionnées, nous allons séparer les données en trois jeux de données distincts grâce à la 
fonction `split_data()`[(3)]. La séparation consiste à diviser les données en deux jeux de données : 

* `training_set` : 90%
* `test_set` : 10%

Pour des données temporelles il est important de ne pas mélanger la chronologie des données puisque cela peut créer des 
problèmes de cohérence. En effet, nous voulons que le modèle puisse prédire les valeurs de clôture sur l'intervalle suivant 
et non pas sur des intervalles passés.

[(3)]: #annexe-3

#### Mise à l'échelle des données

Il est important de mettre à l'échelle les données pour que le modèle puisse les utiliser correctement. Pour cela, nous
allons utiliser la fonction `scale_data()`[(4)]. Cette fonction va permettre de normaliser les données pour qu'elles soient
comprises entre -1 et 1 pour nos deux jeux de données. C'est une technique de normalisation qui permet de réduire
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
créer des séquences de données. Pour cela, nous allons utiliser la fonction `create_sequences()`[(5)]. Cette fonction va
utiliser les données préalablement normalisées pour créer des séquences de données de taille `sequence_length`.

C'est à cette étape que nous construisons les *features* d'entrée du modèle et la *target* de sortie, aussi appelé *label*.
Dans notre cas, nous utiliserons la colonne `close` comme *target* et le reste des colonnes comme *features*.

Il est à noter que nous allons séparer les données du `training_set` en deux séquences de données distinctes pour avoir
également des séquences de données pour la validation du modèle, grâce à la fonction `split_train_and_val_sequences()`[(6)]. 
Nous utiliserons comme taille `val_size=0.2` pour la validation du modèle, ce qui représente 18% des données totales 
attribuées pour la validation du modèle.

[(5)]: #annexe-5
[(6)]: #annexe-6

## Modélisation

Pour ce projet, nous avons choisi d'utiliser un modèle de type *LSTM* pour prédire les valeurs de clôture de la monnaie
sur un intervalle de temps de 1 heure. Pour le chargement des données, la définition de l'architecture du modèle ainsi
que son entraînement, nous avons choisi d'utiliser la librairie [*PyTorch-Lightning*](https://pytorchlightning.ai/) qui 
est une sur-couche de l'excellente librairie *PyTorch*. Cette librairie permet de packager plus simplement et rapidement 
du code *PyTorch*, ce qui va nous aider pour le déploiement et le service de nos modèles via notre API dans un second temps.

### Définition de l'architecture du modèle

Nous allons commencer par décrire l'architecture du modèle qui se compose de deux parties complémentaires :

* Un premier module `LSTMRegressor`[(7)] qui définit la structure du modèle, les hyperparamètres, ainsi que les différentes
étapes d'inférence via un `nn.Module` de *PyTorch*.
* Un second module `PricePredictor`[(8)] qui hérite de l'architecture du premier module et qui va permettre de définir
les étapes d'entraînement, de validation, de test, le *learning rate* et la fonction de *loss* du modèle.

La fonction de *loss* du modèle est la fonction de coût qui va permettre de déterminer la qualité du modèle. Nous utilisons
la fonction de coût `nn.MSELoss()` de la librairie *PyTorch* qui va nous permettre de calculer l'erreur au carré (*mean
squared error*, en anglais) entre la valeur prédite et la valeur réelle.

Pour l'entraînement du modèle, nous utiliserons un *dataloader* qui va permettre de charger les données en batchs. C'est
la classe `LSTMDataLoader`[(9)] qui hérite de `CryptoDataset`[(10)] qui va s'occuper de charger et de distribuer les
batchs de données lors des différentes phases d'entraînement, validation et test.

[(7)]: #annexe-7
[(8)]: #annexe-8
[(9)]: #annexe-9
[(10)]: #annexe-10

### Choix des hyperparamètres

Les hyperparamètres utilisés pour l'entraînement de notre modèle ne sont pas définis dans le code, mais dans un fichier
de configuration à part. Cela permet de faciliter la modification des hyperparamètres du modèle et de faciliter le
re-entraînement du modèle si besoin. Ils sont donc définis dans un fichier `/conf/base/parameters.yaml` où se trouvent
également les paramètres de *fetching* et *preprocessing* des données. 

Ce fonctionnement est important puisqu'il permet d'harmoniser le déroulement du pipeline complet. Ainsi, nous n'avons plus
besoin de toucher aux fichiers de code pour tester des nouveaux hyperparamètres, de même si nous voulons augmenter la 
taille des séquences de données.

Voici la liste des hyperparamètres retenus et utilisés pour l'entraînement des modèles :

```yml
training:
  train_batch_size: 64
  val_batch_size: 1
  train_workers: 2
  val_workers: 1
  max_epochs: 100
  hidden_size: 128
  number_of_features: 9
  number_of_layers: 2
  dropout_rate: 0.2
  learning_rate: 1e-4
  log_n_steps: 2
  run_on_gpu: True # False if running on CPU
  wandb_project: "make-us-rich"
```

### Entraînements et monitoring

L'entraînement du modèle se fait via la méthode `training_loop()`[(11)] qui instancie les classes : `LSTMDataLoader`, 
`PricePredictor` utilisées par `Trainer` qui est la classe `Trainer` de *PyTorch-Lightning* qui gère l'entraînement.

Nous utilisons une *seed* pour figer l'aléatoire du modèle, via la fonction `seed_everything` de la librairie 
*PyTorch-Lightning*, afin de pouvoir reproduire les résultats du modèle si besoin. Nous définissons également deux
*callbacks* :

* `ModelCheckpoint()` qui va permettre de sauvegarder les poids du modèle à chaque *epoch* et de conserver uniquement
    les poids les plus performants.
* `EarlyStopping()` qui va permettre de stopper l'entraînement du modèle si le modèle n'a pas progressé depuis un certain
    nombre d'*epochs*. Ici, nous utilisons un *patience* de 2 *epochs*.

Dans les deux cas, nous utilisons les valeurs de *loss* de validation, que l'on cherche à minimiser, pour déterminer les
poids les plus performants et s'il faut continuer ou stopper l'entraînement.

En ce qui concerne le *monitoring* et le *logging*, nous utilisons la classe `WandbLogger` de *Wandb* incluse dans la
librairie *PyTorch-Lightning* qui va nous permettre de stocker les hyperparamètres, l'environnement et toutes les métriques
de notre modèle directement sur *Wandb*.

*Wandb* est un outil de monitoring qui permet de stocker l'historique des entraînements de nos modèles et de comparer les
différents modèles. C'est cette plateforme que nous avons privilégié pour l'expérimentation et le monitoring de nos modèles.
J'ai donc créé un projet sur *Wandb* pour *Make-Us-Rich* et connecté le pipeline d'entraînement pour que tout soit stocké
directement sur *Wandb*[(12)].

[(11)]: #annexe-11
[(12)]: #annexe-12

### Validation du modèle

La validation du modèle se fait en vérifiant que la moyenne de la valeur de *loss* de validation et celle de test est bien
inférieure à une certaine valeur. Cette partie est assez légère et mériterait un ajustement dans notre pipeline d'entraînement
automatique. L'architecture du modèle permet néanmoins d'obtenir des résultats satisfaisants qui sont directement observables
et comparables sur l'interface de *Wandb*. 

On peut constater tout de même que le taux d'erreur sur les données de validation est vraiment très faible, et il est 
meilleur que le taux d'erreur sur les données de test. En effet, plus on s'éloigne dans le temps des données d'entraînement,
et plus la précision du modèle diminue et donc plus le taux d'erreur, la *loss*, augmente.

### Conversion vers ONNX

Nous avons fait le choix d'inclure une étape de conversion automatique du modèle en *ONNX* afin de faciliter la prise en
charge de ce modèle par d'autres applications et également un optimisation du temps d'inférence lors du service des modèles
via *API*.

En effet le format *ONNX* (*Open Neural Network Exchange*) est un format de représentation de modèle standardisé qui permet, 
notamment sur *CPU*, de réduire les temps de calcul des modèles [@chaigneau_2022]. C'est via deux fonctions que nous allons 
pouvoir convertir le modèle en *ONNX* et valider que le modèle converti est conforme au modèle original, surtout au 
niveau de la précision des prédictions. 

La première fonction `convert_model()`[(13)] permet la conversion du modèle en *ONNX* et son stockage avant validation. 
La seconde fonction `validate_model()`[(14)] assure que le modèle converti est valable d'un point de vue architecture et 
noeuds des graphiques, ainsi qu'au niveau de la précision par rapport au modèle *PyTorch* original. La différence entre 
les deux prédictions doit respecter une tolérance absolue de $10^{-5}$ et une tolérance relative de $10^{-3}$.

[(13)]: #annexe-13
[(14)]: #annexe-14

### Stockage des modèles et des features engineering

Il ne nous reste plus qu'à stocker les modèles et les features engineering dans une base de données. Vu les données que
nous souhaitons conserver, c'est une base de données orientée vers le stockage objet que nous utiliserons, tel que *AWS S3*,
*Google Cloud Storage* ou *Azure Blob Storage*. Dans notre cas, nous utilisons *Minio* pour stocker nos données, car c'est
l'équivalent de *AWS S3* mais hébergeable n'importe où sur le web ou en local.

C'est grâce à la fonction `upload_files()`[(15)] que nous allons pouvoir stocker nos modèles et les features engineering qui
sont associées dans un répertoire unique de notre base de données. Ainsi, ils seront accessibles par la suite par l'API
pour leur utilisation.

Enfin, afin de s'assurer que les fichiers générés par l'entraînement d'un modèle et permettre de conserver de l'espace 
disque sur la machine qui réalise l'entraînement, nous utilisons une fonction de nettoyage de tous les fichiers locaux 
qui ne sont plus utilisés. Ceci est réalisé par la dernière fonction du pipeline nommée `clean_files()`[(16)].

[(15)]: #annexe-15

## Service des modèles

...

### Qu'est-ce-que servir des modèles

...

### 🐳 Docker

...

### Présentation de l'API

...

#### Gestion des modèles

...

#### Les endpoints de l'API

...

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


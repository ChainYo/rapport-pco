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

Pour conserver une certaine lisibilité, les différents blocs de code sont numérotés entre paranthèses et leurs détails 
sont disponibles dans les annexes de ce rapport.

## Les données

La récupération de données et la création d'un jeu de données est la première étape d'un projet de *Machine Learning*, 
voir même de *Data Science* en général. C'est une étape qui n'est pas à sous-estimer, car elle va déterminer la réussite 
de votre projet.

### Récupération des données

Cette étape de récupération des séries temporelles correspond à l'étape de *fetching* sur le schéma présenté en introduction.
Les données que j'utilise sont des données publiques collectées par beaucoup de plateformes de marchés financiers. Pour
ce projet, j'ai choisi de récupérer les données de la plateforme *Binance*. *Binance* est une plateforme très connue dans
le monde de l'échange de crypto-monnaies et elle dispose d'une *API* qui permet de récupérer des données très 
simplement et rapidement. Il suffit de disposer d'un compte sur *Binance* et de se générer un *token* d'accès pour pouvoir
utiliser cette *API* de façon permanente et sans frais.

*Binance* dispose également d'un package Python (`python-binance`) qui facilite les appels à son *API*. La classe `BinanceClient`$^{(1)}$
permet de gérer les interactions avec l'*API* de *Binance* et inclut des méthodes telles que la récupération de données
sur cinq jours, un an ou une période à définir par l'utilisateur. Ces méthodes requièrent en argument un symbole de crypto-
monnaie ainsi qu'une monnaie de comparaison et renvoient les données sous forme de `pandas.DataFrame`.

### Description des données

Les données récupérées sont des séries temporelles de la crypto-monnaie cible comparée à une monnaie. Par exemple, je peux
récupérer les données de la crypto-monnaie *BTC* par rapport à la monnaie *EUR* sur les 5 derniers jours avec un intervalle
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
simple appel à l'*API* suffit pour récupérer les nouvelles données utiles à un entraînement de modèle.

### Préparation des données

Il est nécessaire de préparer les données pour qu'elles soient utilisables par le modèle. Pour cela, nous allons utiliser
plusieurs fonctions définies dans l'étape de `preprocessing` de ce projet. C'est à cette étape qu'intervient un choix des
*features* à utiliser pour l'entraînement du modèle et leur *engineering*. Le terme *feature engineering* est un terme 
désignant les différentes étapes de *raffinage* des données pour qu'elles soient utilisables par le modèle.

#### Extraction des données utiles

Tout d'abord, nous allons extraire les données utiles à partir de la série temporelle grâce à la fonction 
`extract_features_from_dataset()`$^{(2)}$. Pour cela, nous allons utiliser uniquement les colonnes `open`, `high`, `low`, `close` 
et `timestamp`. Nous stockons également la différence entre la valeur de clôture et la valeur d'ouverture pour chaque 
intervalle sous le nom `close_change`.

Ainsi à l'issue de cette étape, nous obtenons un nouveau `pandas.DataFrame` qui contient les *features* spécialement
sélectionnées pour l'entraînement. Nous n'incluons pas les colonnes relatives aux volumes d'échange et aux trades, car
c'est la prédiction de la valeur de clôture sur l'intervalle suivant qui nous intéresse ici. Il serait néanmoins possible
d'inclure les notions de volumes dans l'entraînement, mais cela complexifierait le modèle et l'alourdirait pour un gain
potentiel à déterminer.

#### Séparation des jeux de données

Une fois nos *features* sélectionnées, nous allons séparer les données en trois jeux de données distincts grâce à la 
fonction `split_data()`$^{(3)}$. La séparation consiste à diviser les données en deux jeux de données : 

* `training_set` : 90%
* `test_set` : 10%

Pour des données temporelles il est impératif de ne pas mélanger la chronologie des données puisque cela peut créer des 
problèmes de cohérence. En effet, nous voulons que le modèle puisse prédire les valeurs de clôture sur l'intervalle suivant 
et non pas sur des intervalles passés.

#### Mise à l'échelle des données

Il est primordial de mettre à l'échelle les données pour que le modèle puisse les utiliser correctement. Pour cela, nous
allons utiliser la fonction `scale_data()`$^{(4)}$. Cette fonction va permettre de normaliser les données pour qu'elles soient
comprises entre -1 et 1 pour nos deux jeux de données. C'est une technique de normalisation qui permet de réduire
les écarts entre les données et ainsi les rendre plus facile à manipuler par le modèle lors de l'entraînement.

On utilise ici la méthode de normalisation `MinMaxScaler` de la librairie `scikit-learn`. Il est important de noter que nous
sauvegardons également cet objet de normalisation dans un fichier *pickle* pour pouvoir l'utiliser plus tard lors de
l'inférence via l'*API*. En effet, puisque le modèle est entraîné sur des données normalisées, il est primordial qu'elles 
le soient également lors des prédictions ultérieures. De plus, nous avons besoin de cet objet pour pouvoir inverser la
normalisation des données prédites et obtenir des valeurs de clôture réelles, c'est-à-dire des valeurs de clôture non
normalisées.

#### Préparation des séquences de données

Il ne reste plus qu'à préparer les données pour qu'elles soient utilisables par le modèle. Pour cela, nous allons devoir
créer des séquences de données à l'aide de la fonction `create_sequences()`$^{(5)}$. Cette fonction va
utiliser les données préalablement normalisées pour créer des séquences de données de taille `sequence_length`.

C'est à cette étape que nous construisons les *features* d'entrée du modèle et la *target* de sortie, aussi appelée *label*.
Dans notre cas, nous utiliserons la colonne `close` comme *target* et le reste des colonnes comme *features*.

Il est à noter que nous allons séparer les données du `training_set` en deux séquences de données distinctes pour avoir
également des séquences de données pour la validation du modèle, grâce à la fonction `split_train_and_val_sequences()`$^{(6)}$. 
Nous utiliserons comme taille `val_size=0.2` pour la validation du modèle, ce qui représente 18% des données totales 
attribuées pour la validation du modèle.

## Modélisation

Pour ce projet, nous avons choisi d'utiliser un modèle de type *LSTM* pour prédire les valeurs de clôture de la monnaie
sur un intervalle de temps de 1 heure. Pour le chargement des données, la définition de l'architecture du modèle ainsi
que son entraînement, nous avons choisi d'utiliser la librairie [*PyTorch-Lightning*](https://pytorchlightning.ai/) qui 
est une sur-couche de l'excellente librairie *PyTorch*. Cette librairie permet de packager plus simplement et rapidement 
du code *PyTorch*, ce qui va nous aider pour le déploiement et le service de nos modèles via notre *API* dans un second temps.

### Définition de l'architecture du modèle

Nous allons commencer par décrire l'architecture du modèle qui se compose de deux parties complémentaires :

* Un premier module `LSTMRegressor`$^{(7)}$ qui définit la structure du modèle, les hyperparamètres, ainsi que les différentes
étapes d'inférence via un `nn.Module` de *PyTorch*.
* Un second module `PricePredictor`$^{(8)}$ qui hérite de l'architecture du premier module et qui va permettre de définir
les étapes d'entraînement, de validation, de test, le *learning rate* et la fonction de *loss* du modèle.

La fonction de *loss* du modèle est la fonction de coût qui va permettre de déterminer la qualité du modèle. Nous utilisons
la fonction de coût `nn.MSELoss()` de la librairie *PyTorch* qui va nous permettre de calculer l'erreur au carré (*mean
squared error*, en anglais) entre la valeur prédite et la valeur réelle.

Pour l'entraînement du modèle, nous utiliserons un *Dataloader* qui va permettre de charger les données en batchs. C'est
la classe `LSTMDataLoader`$^{(9)}$ qui hérite de `CryptoDataset`$^{(10)}$ qui va s'occuper de charger et de distribuer les
batchs de données lors des différentes phases d'entraînement, validation et test.

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

L'entraînement du modèle se fait via la méthode `training_loop()`$^{(11)}$ qui instancie les classes : `LSTMDataLoader`, 
`PricePredictor` utilisées par `Trainer` qui est la classe `Trainer` de *PyTorch-Lightning* qui gère l'entraînement.

Nous utilisons une *seed* pour figer l'aléatoire du modèle, via la fonction `seed_everything` de la librairie 
*PyTorch-Lightning*, afin de pouvoir reproduire les résultats du modèle si besoin. Nous définissons également deux
*callbacks* :

* `ModelCheckpoint()` qui va permettre de sauvegarder les poids du modèle à chaque *epoch* et de conserver uniquement
    les poids les plus performants.
* `EarlyStopping()` qui va permettre de stopper l'entraînement du modèle si le modèle n'a pas progressé depuis un certain
    nombre d'*epochs*. Ici, nous utilisons une *patience* de 2 *epochs*.

Dans les deux cas, nous utilisons les valeurs de *loss* de validation, que l'on cherche à minimiser, pour déterminer les
poids les plus performants et savoir s'il faut continuer ou stopper l'entraînement.

En ce qui concerne le *monitoring* et le *logging*, nous utilisons la classe `WandbLogger` de *Wandb* incluse dans la
librairie *PyTorch-Lightning* qui va nous permettre de stocker les hyperparamètres, l'environnement et toutes les métriques
de notre modèle directement sur *Wandb*.

*Wandb* est un outil de monitoring qui permet de stocker l'historique des entraînements de nos modèles et de comparer les
différents modèles. C'est cette plateforme que nous avons privilégiée pour l'expérimentation et le monitoring de nos modèles.
J'ai donc créé un projet sur *Wandb* pour *Make-Us-Rich* et connecté le pipeline d'entraînement pour que tout soit stocké
directement sur *Wandb*$^{(12)}$.

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
charge de ce modèle par d'autres applications et également une optimisation du temps d'inférence lors du service des modèles
via *API*.

En effet le format *ONNX* (*Open Neural Network Exchange*) est un format de représentation de modèle standardisé qui permet, 
notamment sur *CPU*, de réduire les temps de calcul des modèles [@chaigneau_2022]. C'est via deux fonctions que nous allons 
pouvoir convertir le modèle en *ONNX* et valider que le modèle converti est conforme au modèle original, surtout au 
niveau de la précision des prédictions. 

La première fonction `convert_model()`$^{(13)}$ permet la conversion du modèle en *ONNX* et son stockage avant validation. 
La seconde fonction `validate_model()`$^{(14)}$ assure que le modèle converti est valable d'un point de vue architecture et 
noeuds des graphiques, ainsi qu'au niveau de la précision par rapport au modèle *PyTorch* original. La différence entre 
les deux prédictions doit respecter une tolérance absolue de $10^{-5}$ et une tolérance relative de $10^{-3}$.

### Stockage des modèles et des *features engineering*

Il ne nous reste plus qu'à stocker les modèles et les *features engineering* dans une base de données. Vu les données que
nous souhaitons conserver, c'est une base de données orientée vers le stockage objet que nous utiliserons, tel que *AWS S3*,
*Google Cloud Storage* ou *Azure Blob Storage*. Dans notre cas, nous utilisons *Minio* pour stocker nos données, car c'est
l'équivalent de *AWS S3* mais hébergeable n'importe où sur le web ou en local.

C'est grâce à la fonction `upload_files()`$^{(15)}$ que nous allons pouvoir stocker nos modèles et les *features engineering* qui
sont associées dans un répertoire unique de notre base de données. Ainsi, ils seront accessibles par la suite par l'*API*
pour leur utilisation. Dans notre, cas c'est uniquement le `MinMaxScaler` qui est stocké dans notre base de données au 
format *pickle*.

Enfin, afin de s'assurer que les fichiers générés par l'entraînement d'un modèle sont supprimés et ainsi permettre de 
conserver de l'espace disque sur la machine qui réalise l'entraînement, nous utilisons une fonction de nettoyage de tous 
les fichiers locaux qui ne sont plus utilisés. Ceci est réalisé par la dernière fonction du pipeline nommée `clean_files()`$^{(16)}$.

## Service des modèles

Maintenant que nous avons terminé l'entraînement du modèle et qu'il est prêt à être utilisé, nous allons créer un service
pour le stocker et le rendre accessible via une *API*. C'est exactement la définition du service d'un modèle de *Machine
Learning* (*model serving*, en anglais) et c'est une étape cruciale pour permettre à des utilisateurs de bénéficier du 
produit de *Machine Learning* qu'offre notre service.

### Qu'est-ce-que servir des modèles

Servir des modèles de *Machine Learning* est une tâche qui implique la gestion de ressources, de données et le monitoring
permanent des performances du modèle. C'est donc une étape à ne pas sous-estimer et qui implique une bonne préparation et
une bonne organisation pour réussir.

À cela s'ajoute un facteur d'échelle, qui est la capacité de répondre aux besoins des utilisateurs. Ce n'est pas pareil
d'avoir un modèle disponible pour 5 personnes ou pour 50.000 personnes. C'est pourquoi notre approche quant au service de nos
modèles de prédictions a été de prévoir la mise à l'échelle en faisant en sorte que le déploiement d'un seul modèle soit
identique et répétable pour $N$ modèles et $N$ utilisateurs.

### Dockerisation

Pour que le déploiement soit répétable et identique pour chaque modèle et chaque utilisateur, nous utilisons *Docker* 
comme outil de déploiement. C'est un outil de gestion de conteneurs qui permet de déployer des applications isolées en local ou
sur un serveur cloud. On définit une série d'instructions pour la création et le déploiement du ou des différents containers
via des fichiers *Dockerfile* et *Docker Compose*.

Le *Dockerfile* est un fichier de configuration qui permet de définir les instructions de création d'un container spécifique,
et le *Docker Compose* est un fichier de configuration qui permet de définir les instructions de déploiement de plusieurs
containers. Nous avons donc plusieurs fichiers pour l'interface utilisateur $^{(17)}$ et un *Dockerfile* pour l'*API*$^{(18)}$.

### Présentation de l'API

Nous allons maintenant décrire l'*API*, ses différents *endpoints* et leurs rôles. L'avantage de l'*API* est de pouvoir
s'adapter à toutes les exigences de nos utilisateurs. Ainsi, un utilisateur mobile peut demander une prédiction à l'*API*, 
tout comme un utilisateur de bureau peut demander une prédiction à l'*API* via notre interface web ou un script Python.
De cette manière, nous pouvons rendre accessible le modèle de prédiction à tous les utilisateurs.

L'*API* est composée de plusieurs *endpoints*. Chaque *endpoint* est défini par une URL et une méthode HTTP. Lorsque l'on
souhaite accèder à l'*API*, nous arrivons directement sur la documentation des différents *endpoints*$^{(19)}$.

#### Gestion des modèles

Les différents modèles de prédictions sont chargés par l'*API* grâce à la classe qui les gère, `ModelLoader`$^{(20)}$. Cette
classe de gestion du chargement des modèles et de leur prédiction va permettre une flexibilité totale au niveau du nombre 
de modèles disponibles, leurs *features engineering* spécifiques et leurs informations respectives.

La classe `ModelLoader` se base sur une autre classe `ONNXModel`$^{(21)}$ qui va être le squelette de base pour chacun des 
modèles de prédictions. Ainsi, cette base permet à chaque modèle de fonctionner de la même manière peu importe la crypto-monnaie
sur laquelle il est basé.

Nous avons donc une classe qui permet le fonctionnement de chaque modèle de façon identique et une autre classe qui s'occupe
d'orchestrer l'ensemble des modèles de prédictions pour qu'ils soient mis à jour et disponibles pour tous les utilisateurs
via les différents endpoints de l'*API*.

#### Les endpoints de l'API

Les endpoints de l'*API* sont au nombre de six, dont trois endpoints avec une méthode *PUT* et trois endpoints plus 
utilitaires disposant d'une méthode *GET*.

* Endpoints de *serving* $^{(22)}$:
    * `/predict` : permet de récupérer la prédiction d'un modèle d'une crypto-monnaie comparée à une monnaie.
    * `/update_models` : permet de mettre à jour les modèles de prédiction avec les derniers fichiers disponibles dans la
    base de données.
    * `/update_date` : permet de mettre à jour la date de la dernière mise à jour des modèles, important pour assurer que
    l'*API* met toujours à disposition les derniers modèles de prédiction.

* Endpoints de *monitoring* $^{(23)}$:
    * `/check_models_number` : permet de vérifier le nombre de modèles disponibles sur l'*API*.
    * `/healthz` : permet de vérifier le bon fonctionnement de l'*API*. Indispensable si orchestration via *Kubernetes*.
    * `/readyz` : permet de vérifier la disponibilité de l'*API*. Indispensable si orchestration via *Kubernetes*.

## Interface utilisateur

Le dernier composant du projet est l'interface utilisateur. Cette interface permet de choisir la crypto-monnaie à comparer
et d'accéder aux prédictions des différents modèles de prédiction. L'interface utilisateur est un démonstrateur *Streamlit*
combiné avec une base de données relationnelles *PostgreSQL* pour assurer l'authentification des utilisateurs.

### Présentation de l'interface utilisateur

Pour pouvoir accéder à l'interface utilisateur, l'utilisateur doit être authentifié $^{(24)}$. Pour cela, il doit être enregistré
dans la base de données. Si l'utilisateur se connecte pour la première fois, il sera automatiquement enregistré et un 
jeton d'authentification sera créé avec une validité de dix jours.

Une fois authentifié, l'utilisateur peut avoir accès à l'application *Streamlit* sur laquelle il peut afficher les courbes des 
différentes crypto-monnaies comparées à une monnaie. Il peut également choisir la crypto-monnaie à comparer et les modèles de 
prédiction à afficher $^{(25)}$. 

Si l'utilisateur le souhaite, il peut également récupérer un token qui lui permettra d'utiliser l'*API* pour récupérer les
prédictions directement sans passer par l'interface utilisateur $^{(26)}$.

### Présentation de la base de données relationnelles

La base de données regroupe les différentes données des utilisateurs de l'interface et de l'*API*. Elle est composée de
cinq tables :

* `users` : table qui stocke les identifiants et les mots de passe hashés des utilisateurs.
* `roles` : table qui stocke les différents rôles des utilisateurs parmi *admin* et *member*.
* `user_roles` : table qui stocke les rôles des utilisateurs.
* `api_tokens` : table qui stocke les jetons d'authentification à l'*API* pour les utilisateurs.
* `user_api_consumptions` : table qui stocke les consommations des utilisateurs de l'*API*.

![Schéma de la base de données relationnelle des utilisateurs de l'application \label {fig:3.1}](./content/assets/bdd-postgresql.png){ width=320px, height=300px }

La base de données *PostgreSQL* est initialisée au déploiement avec toutes ces tables via un fichier `init.sql`$^{(27)}$.
Les insertions dans cette base de données ainsi que les requêtes sont gérées par l'interface grâce à la classe 
`DatabaseHandler`$^{(28)}$.

### Explication du fonctionnement des tokens

Les tokens des utilisateurs sont créés à l'enregistrement de l'utilisateur dans la base de données. Ils sont ensuite
utilisés à chaque requête à l'*API* qui sert les modèles de prédictions. Ces tokens servent à limiter le nombre de requêtes
effectuées par l'utilisateur sur une période de temps limitée. La limite est très large pour seulement trois modèles, mais
dans une optique d'ajout de modèles, il serait possible d'inclure une forme de monétisation du service qui passerait par une 
gestion de la consommation.

## Packaging du projet

Le projet entier a été réfléchi pour être déployable sur n'importe quel système, avec les composants au même endroit ou
sur des serveurs différents. Nous avons déjà évoqué le fait que les composants de service et d'interface utilisateur 
utilisent la puissance de *Docker* pour leur déploiement, mais il y a également la partie *training* qui se veut automatisée
grâce à *Prefect*.

L'ensemble du code est packagé grâce à *pypi* ce qui permet à n'importe qui d'installer le projet via la commande
`pip install make-us-rich` et de commencer à déployer les composants du projet selon ses besoins.

### Utilisation de Prefect

Nous avons décidé d'utiliser la librairie *Prefect*$^{(29)}$ pour l'automatisation du pipeline d'entraînement des modèles de 
prédiction. Cette librairie est un *ETL (Extract Transform Load)* qui permet de définir des *flows* et des *tasks* qui seront 
automatiquement exécutés selon les instructions définies $^{(30)}$. Dans notre cas, nous avons défini un *flow* pour chaque crypto-monnaie
pour laquelle nous souhaitons entraîner un modèle de prédiction $^{(31)}$.

L'avantage c'est que tout cet enchaînement d'actions bénéficie d'une interface de visualisation des fonctionnalités
pour relancer une tâche qui aurait planté ou pour tout simplement voir le détail d'une tâche et son avancement.

Le `scheduler` de *Prefect* est aussi très utile puisque nous avons besoin de définir des tâches qui seront exécutées
à intervalles réguliers (toutes les heures), ces intervalles peuvent être modifiés très simplement sans avoir besoin de 
tout changer.

### Documentation du projet

L'intégralité du projet bénéficie d'un documentation qui lui est propre, il est possible de la consulter via le lien présent
sur le *GitHub* du projet. Cette documentation permet de présenter le projet, ses différents composants et surtout
d'expliquer comment les déployer pour l'utilisateur. Elle permet également de décrire les différentes fonctions et classes
du package *Python* du projet.

Pour rédiger la documentation, nous avons utilisé la librairie *Mkdocs-material* qui permet de générer un site web à partir
d'un fichier *yaml*$^{(32)}$ décrivant la configuration du projet et des fichiers markdown qui composeront le contenu de chaque page.

Le déploiement de la documentation est également automatisé via *GitHub-Action* qui permet de définir des actions de 
déploiement automatiques à chaque fois qu'un changement est apporté au code du projet. Les instructions doivent être mises
dans un dossier `.github/workflows/`$^{(33)}$ du projet.

### Alerting

Enfin pour rendre le projet plus complet, un système d'alerting a été mis en place pour notifier automatiquement l'utilisateur
en cas de problèmes avec l'*API* ou les *flows* de *Prefect* qui pourraient ne pas fonctionner correctement. C'est aussi
une des raisons pour laquelle des endpoints de *monitoring* ont été mis en place dans l'*API*.

Les alertes émises par les différents composants sont envoyées directement sur un serveur *Discord* accessibles à l'équipe 
qui se charge de la maintenance du projet. C'est un service gratuit de messagerie qu'il faut mettre en place, et via un 
*Webhook* toutes les alertes seront transmises directement dans un salon prédéfini sur le serveur.

### Méthodologie et organisation

Pour mener à bien ce projet, nous avons décidé d'utiliser l'outil de méthodologie agile intégré à *GitHub* qui est un tableau 
*Kanban*$^{(34)}$ sur lequel nous avons différencié les tâches selon trois status : *To Do*, *In Progress* et *Done*. Les 
tâches ont été définies en amont par rapport aux objectifs du projet et également au fil du projet pour venir combler des 
manques ou des besoins qui n'avaient pas été pensés en amont.

Les tâches ont été différenciées par des *tags* qui visent à les classer dans des catégories : *documentation*, *AI feature*,
*bug*, *dev*, *feature*, *front-end*, *main feature*. Ces différenciations permettent de rapidement voir à quoi va servir la tâche
et aussi se donner une idée sur les tâches prioritaires par rapport à l'ensemble des tâches à faire.

# Section 1: Etat de l'art

Dans cette première section, nous allons décrire les avancées de la modélisation IA et des algorithmes de prédiction
sur des données temporelles. Ce ne sera malheureusement pas une liste exhaustive de toutes les options disponibles, ni
un historique complet des différentes évolution des algorithmes de prédiction sur des séries temporelles, car cela est
trop riche pour tenir dans ce rapport. 

Nous allons donc nous concentrer sur les avancées les plus récentes et celles qui ont un rapport direct avec la solution
envisagée dans ce projet. Commençons par un bref aperçu et historique de la tâche de prédiction à l'aide de données
temporelles.

## Aperçu et historique rapide

Les séries temporelles, ainsi que leur analyse, sont de plus en plus importantes en raison de la production massive de
données dans le monde. Il y a donc un besoin, en constante augmentation, dans l'analyse de ces séries chronologiques
avec des techniques statistiques et plus récemment d'apprentissage automatique.

L'analyse des séries chronologiques consiste à extraire des informations récapitulatives et statistiques significatives
à partir de points classés par ordre chronologique. L'intérêt étant de diagnostiquer le comportement passé pour prédire
le comportement futur.

Aucune des techniques ne s'est développée dans le vide ou par intérêt purement théorique. Les innovations dans l'analyse
des séries chronologiques résultent de nouvelles méthodes de collecte, d'enregistrement et de visualisation des données.

Il existe énormément de domaine d'application tels que la médecine, la météorologie, l'astronomie ou encore ce qui va 
nous intéresser ici, les marchés financiers et notamment celui des crypto-monnaies.

Pour revenir à l'aspect historique, les organisations privées et notamment bancaires ont commencé à collecter des données
par imitation du gouvernement américain qui collectait des données économiques publiques. Les premiers pionniers de
l'analyse des données chronologiques des cours de la bourse ont fait ce travail mathématique à la main, alors que de nos
jours ce travail est réalisé avec l'assistance de méthodes analytiques et des algorithmes de machine learning.

Richard Dennis, dans les années 80,  a été le premier à développer un algorithme de prédiction des cours de la bourse qui
ne comprenait que quelques règles de base, permettant à quiconque les connaissant de prévoir le prix d'une action et d'en
retirer des bénéfices par spéculation.

Progressivement, avec l'accumulation de personnes utilisant ces règles, elles sont devenues de plus en plus inefficaces.
Il aura donc fallu développer de nouvelles méthodes, notamment statistiques, plus complexes pour toujours mieux prévoir
l'évolution des cours des marchés financiers.

C'est ainsi que des méthodes historiques telles que *SARIMA* ou *ARIMA* se sont démocratisées. Elles présentent néanmoins un
inconvénient : elles nécessitent des données stationnaires pour fonctionner. De plus, ces techniques statistiques dites
historiques ont des résultats médiocres sur le long terme, et ainsi se sont développés d'autres méthodes d'apprentissage
automatique utilisant la puissance des réseaux de neurones, comme *RNN* (Recurrent Neural Network).

## Fondamentaux des séries temporelles

Comme évoqué précédemment, la stationnarité d'une série est une propriété essentielle pour l'analyse statistique. Une
série chronologique est dite stationnaire si ces propriétés telles que la moyenne, la variance ou la covariance sont
constantes au cours du temps. Or, cela n'est pas vrai pour toutes les séries temporelles et notamment les données
issues des marchés financiers, qui ne sont stationnaires que sur une période de temps fixée (souvent courte).

Il existe troise composantes qui constituent une série temporelle :

* **Tendance (T = Trend)** : correspond à une augmentation ou à une diminution sur le long terme des données et qui peut 
assumer une grande variété de modèles. Nous utilisons la tendance pour estimer le niveau, c'est-à-dire la valeur ou la
plage typique de valeurs, que la variable doit avoir au cours du temps. On parle de tendance à la hausse ou à la baisse.
* **Saisonnalité (S = Seasonal)** : est l'apparition de schémas de variations cycliques qui se répètent à des taux de 
fréquence relativement constants.
* **Résidu (R = Remainder)** : correspondent aux fluctuations à court terme qui ne sont ni systématiques ni prévisibles.
Au quotidien, des événements imprévus provoquent de telles instabilités. Concrètement, la composante résiduelle est ce 
qui reste après l'estimation de la tendance et de la saisonnalité, et leur suppression d'une série chronologique.

Voici une représentation de la décomposition des composantes d'une série temporelle :

![Graphique présentant la décomposition classique d'un modèle additif d'une série temporelle \label {fig:2.1}](/content/assets/timeseries-decomposition.png)

## Différents modèles de prédiction

Nous alons voir ici les différentes techniques et modèles qui existent pour la prédiction à l'aide de données temporelles.

Nous pouvons d'ores et déjà distinguer deux catégories de modèles de prédiction :

* **Modèles statistiques** : dits traditionnels, ils regroupent les modèles univariés et multivariés comprenant
respectivement *ARIMA*, *SARIMA* et *VAR*.
* **Modèles d'apprentissage automatique** : ils regroupent les modèles de régression par amplification de gradient et
les modèles par apprentissage profond comprenant les réseaux de neurones récurrents et convolutionnels.


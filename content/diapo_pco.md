---
title: "Make Us Rich"
description: "Cryptocurrency forecasting: training and serving models made automatic"
author: Thomas Chaigneau
keywords: deep learning, AI, forecasting
marp: true
paginate: true
theme: uncover
class: invert
style: |
  video::-webkit-media-controls {
    will-change: transform;
  }
---

# Door Detector 🚪

[![Door Detector Demo](http://img.youtube.com/vi/pPVfg5bXbA0/0.jpg)](http://www.youtube.com/watch?v=pPVfg5bXbA0 "Door Detector Demo")

---

# Make Us Rich

Cryptocurrency forecasting 📈 
training and serving models made automatic

###### *Thomas Chaigneau - Mai 2022*

---
<!-- _class: lead -->

## Sommaire

1. Qu'est-ce que `Make Us Rich` ❓
2. Démonstration 🎥
3. Crypto-monnaies et données
4. Les RNNs: GRU & LSTM
5. Entraînement 🏋️‍♀️
6. Stockage et base de données
7. API et serving
8. Interface utilisateur
9. Conclusion

---

![bg left](./assets/forecasting-charts.jpg)

Qu'est ce que 
`Make Us Rich` ❓

---
<!-- _class: lead -->
##### Architecture du projet
![width:800px](./assets/project-architecture.png)

---
# Démonstration 🎥

[![Make Us Rich Demo](http://img.youtube.com/vi/u0jEt9UfI0I/0.jpg)](http://www.youtube.com/watch?v=u0jEt9UfI0I "Make Us Rich Demo")

---
<!-- _class: lead -->

![height:700px](./assets/training-pipeline.png)

---
<!-- _class: lead -->

![bg right](./assets/cryptos.jpg)

### <!--fit--> Crypto-monnaies et données

**Binance API:**
timestamp, open, high, low, close, volume, close_time, quote_av, trades, tb_base_av, tb_quote_av, ignore

---
<!-- _class: lead -->

![bg right](./assets/cryptos.jpg)

### <!--fit--> Crypto-monnaies et données

**Binance API:**
* *timestamp*
* *open*
* *high*
* *low*
* **close**
* *close_change* $_{(close - open)}$


---
<!-- _class: lead -->

# Normalisation des données

![height:200px](./assets/sklearn-minmaxscaler.png)

$$
x' = 2\frac{x - min(x)}{max(x) - min(x)} - 1
$$

---
<!-- _class: lead -->

#### Création de séquences à partir des données normalisées

![bg left](./assets/create-sequences.png)

---
<!-- _class: lead -->

# Les RNNs: GRU & LSTM

![width:1400px](./assets/rnn-architecture.png)

Architecture d'un RNN

---
<!-- _class: lead -->
# Les RNNs: GRU & LSTM

![width:950px](./assets/lstm-vs-gru.png)

---
<!-- _class: lead -->

## Pourquoi LSTM ❓

![width:1200px](./assets/validation-lstm-model.png)

---
<!-- _class: lead -->

## Entraînement

![](./assets/pytorch-lightning-logo.png) 
## +
![](./assets/wandb-logo.png)

![bg right](./assets/training.jpg)

---
<!-- _class: lead -->

## Fonction coût à minimiser

`torch.nn.MSELoss()`

$$
l_{n} = (x_{n} - y_{n})^{2}
$$
---

### <!--fit--> Répartition des données

![width:1000px](./assets/dataset_split.png)

(8.64 mois, 2.16 mois et 1.2 mois)


---
<!-- _class: lead -->

## Hyperparamètres

```yaml
train_batch_size: 64
val_batch_size: 1
train_workers: 2
val_workers: 1
max_epochs: 100
hidden_size: 128
number_of_features: 9
number_of_layers: 2
dropout_rate: 0.2
learning_rate: 0.0001
log_n_steps: 2
run_on_gpu: True # False if running on CPU
wandb_project: "make-us-rich"
```

---
<!-- _class: lead -->

![width:120px](./assets/pytorch-lightning-logo.png)
![bg left](./assets/training-loop.png)

## Training Loop


* seed + logger
* model
* data module
* trainer

---
<!-- _class: lead -->

![](./assets/wandb-logo.png)
![bg right](./assets/wandb-summary.png)
## Métriques

Monitoring des métriques d'entraînement

---
<!-- _class: lead -->

## Stockage et base de données

* Fichiers modèles et features engineering
![width:300px](./assets/minio-logo.png)
* Utilisateurs de l'interface et API
![width:400px](./assets/postgresql-logo.png)

---
<!-- _class: lead -->

## Base de données relationnelle

![bg left width:720px](./assets/bdd-postgresql.png)

---
<!-- _class: lead -->

## Serving via API

![bg right height:700px](./assets/serving-part.png)

---
<!-- _class: lead -->

## Gestion des modèles

![width:1000px](./assets/model-classes.png)

---
<!-- _class: lead -->

## Les Endpoints

![width:1200px](./appendix/19.png)

---

## L'interface utilisateur

![bg left width:650px](./assets/interface-part.png)

---
<!-- _class: lead -->

### Authentification

![bg right height:600px](./appendix/24.png)

---
<!-- _class: lead -->

### Graphiques avec prédictions

![bg left height:800px](./appendix/25.png)

---
<!-- _class: lead -->

![bg height:900px](./appendix/26.png)

---
<!-- _class: lead -->

## Organisation

![width:900px](./appendix/34.png)

---

## Conclusion

![bg left](./assets/conclusion-snapshot.jpg)

---
<!-- _class: lead -->

# Questions ❓

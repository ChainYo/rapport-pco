---
title: "Make Us Rich"
description: "Cryptocurrency forecasting: training and serving models made automatic"
author: Thomas Chaigneau
keywords: deep learning, AI, forecasting
marp: true
paginate: true
theme: uncover
class: invert
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
* *close_change* $_{(close - open)}$
* **close**

---

### <!--fit--> Répartition des données

![width:1000px](./assets/dataset_split.png)

---
<!-- _class: lead -->

# Normalisation des données

```python
sklearn.preprocessing.MinMaxScaler(
    feature_range=(-1, 1)
    )

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
# où min = -1 et max = 1
```

---
<!-- _class: lead -->

Création de séquences à partir des données normalisées

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

---
<!-- _class: lead -->

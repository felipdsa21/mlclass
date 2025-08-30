#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar a metodologia e as técnicas para a avaliação de classificadores.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests

print("- Lendo o arquivo com o dataset")
data = pd.read_csv("abalone_dataset.csv")

# Criando X e y para o algoritmo de aprendizagem de máquina
print("- Criando X e y para o algoritmo de aprendizagem a partir do dataset")

feature_cols = [
    "sex",
    "length",
    "diameter",
    "height",
    "whole_weight",
    "shucked_weight",
    "viscera_weight",
    "shell_weight",
]
X = data[feature_cols]
X.loc[:, "sex"] = X.loc[:, "sex"].map({"M": 0, "F": 1, "I": 2})
y = data.type

# Criando o modelo preditivo para o dataset
print("- Criando modelo preditivo")
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# Realizando previsões com o arquivo de testes
print("- Aplicando modelo e enviando para o servidor")
data_app = pd.read_csv("abalone_app.csv")
data_app = data_app[feature_cols]
data_app.loc[:, "sex"] = data_app.loc[:, "sex"].map({"M": 0, "F": 1, "I": 2})
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"
DEV_KEY = "MLgiga"

# JSON a ser enviado para o servidor
data = {"dev_key": DEV_KEY, "predictions": pd.Series(y_pred).to_json(orient="values")}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url=URL, data=data)

# Extraindo e imprimindo o texto da resposta
print("- Resposta do servidor:\n", r.text)

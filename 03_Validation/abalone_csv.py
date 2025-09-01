#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar a metodologia e as técnicas para a avaliação de classificadores.
"""

import sys

import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import requests


def prepare_data_tree(df: pd.DataFrame) -> pd.DataFrame:
    return df


def create_model_tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(criterion="entropy", max_depth="4", random_state=42)


def prepare_data_svm(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df[feature_cols], columns=["sex"])


def create_model_svm() -> DecisionTreeClassifier:
    return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)


prepare_data = prepare_data_svm
create_model = create_model_svm


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
X = prepare_data(X)
y = data.type

# Criando o modelo preditivo para o dataset
print("- Criando modelo preditivo")
model = create_model()
model.fit(X, y)

# Realizando previsões com o arquivo de testes
print("- Aplicando modelo e salvando previsões")
data_app = pd.read_csv("abalone_app.csv")
data_app = data_app[feature_cols]
data_app = prepare_data(data_app)
y_pred = model.predict(data_app)

predictions = pd.Series(y_pred).to_json(orient="values")

with open("predictions.json", "w", encoding="utf-8") as file:
    file.write(predictions)

# Enviando previsões realizadas com o modelo para o servidor
if input("- Enviar previsões para o servidor? (s/n) ").lower() != "s":
    sys.exit()

URL = "https://aydanomachado.com/mlclass/03_Validation.php"
DEV_KEY = "MLgiga"

# JSON a ser enviado para o servidor
data = {"dev_key": DEV_KEY, "predictions": predictions}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url=URL, data=data)

# Extraindo e imprimindo o texto da resposta
print("- Resposta do servidor:\n", r.text)

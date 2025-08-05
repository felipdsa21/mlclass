import os

import numpy as np
import pandas as pd

training_ds = "diabetes_dataset.csv"
orig_training_ds = "orig_" + training_ds

test_ds = "diabetes_app.csv"
orig_test_ds = "orig_" + test_ds

# Renomeia os arquivos originais
if not os.path.exists(orig_training_ds):
    os.rename(training_ds, orig_training_ds)

if not os.path.exists(orig_test_ds):
    os.rename(test_ds, orig_test_ds)

# Carrega datasets originais
training_df = pd.read_csv(orig_training_ds)
test_df = pd.read_csv(orig_test_ds)

# Substitui "" por NaN e converte strings para números
training_df = training_df.replace("", pd.NA).apply(pd.to_numeric)
test_df = test_df.replace("", pd.NA).apply(pd.to_numeric)

# Remove coluna de insulina (faltando em muitas linhas)
training_df = training_df.drop("Insulin", axis="columns")

# Remove linhas com muitas colunas faltando
training_df = training_df.dropna(axis="index", thresh=len(training_df.columns) - 2)

# Normaliza os valores para [0, 1]
df_max = training_df.max()
df_min = training_df.min()

training_df = (training_df - df_min) / (df_max - df_min)
test_df = (test_df - df_min) / (df_max - df_min)

# Preenche entradas faltando no dataset de treinamento
for col in training_df.columns:
    if col != "Outcome":
        series = training_df[col]
        nans = series.isna()
        # Gera valores aleatórios a partir da distribuição normal
        placeholders = np.random.normal(series.mean(), series.std(), nans.sum())
        training_df.loc[nans, col] = placeholders

# Salva datasets tratados
training_df.to_csv(training_ds, index=False)
test_df.to_csv(test_ds, index=False)

import os

import numpy as np
import pandas as pd

filename = "diabetes_dataset.csv"
orig_filename = "orig_" + filename

# Renomeia o arquivo original
if not os.path.exists(orig_filename):
    os.rename(filename, orig_filename)

# Lê o dataset original, transforma e salva as mudanças
df = pd.read_csv(orig_filename)

# Substitui "" por NaN e converte as strings para números
df = df.replace("", pd.NA).apply(pd.to_numeric)

# Normaliza os valores para [0, 1]
df = (df - df.min()) / (df.max() - df.min())

for col in df.columns:
    if col != "Outcome":
        nans = df[col].isna()
        # Gera valores aleatórios a partir da distribuição normal
        placeholders = np.random.normal(df[col].mean(), df[col].std(), nans.sum())
        df.loc[nans, col] = placeholders

df.to_csv(filename, index=False)

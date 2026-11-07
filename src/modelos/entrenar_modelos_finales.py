"""Entrena los dos modelos elegidos y los guarda en la carpeta models/ para el tablero.

Uso (desde la carpeta raíz del repositorio):
    python src/modelos/entrenar_modelos_finales.py

- Marítimo: configuración del Experimento 1 (MLflow "Cotizador-Flete-Maritimo").
- Terrestre: mejor configuración de "Cotizador-Flete-Terrestre".
Se entrenan con 2024-2025 y se evalúan en 2026; el error de prueba (MAE) se guarda
para mostrar un rango probable en el tablero.
"""
import json
import os
import sys

import numpy as np
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(__file__))
from comun import (LAND_CAT, LAND_NUM, OCEAN_CAT, OCEAN_NUM, OCEAN_PARAMS,  # noqa: E402
                   cargar, mae, matriz, rmse, separar_por_tiempo)

LAND_PARAMS = dict(n_estimators=500, max_depth=8, learning_rate=0.03)
os.makedirs("models", exist_ok=True)

df = cargar()
info = {}

for nombre, datos, objetivo, num, cat, params in [
    ("ocean", df, "ocean_cost_per_skid", OCEAN_NUM, OCEAN_CAT, OCEAN_PARAMS),
    ("land", df[df["land_cost"] > 0], "land_cost_per_skid", LAND_NUM, LAND_CAT, LAND_PARAMS),
]:
    train, test = separar_por_tiempo(datos)
    X_train = matriz(train, num, cat)
    X_test = matriz(test, num, cat, columnas=X_train.columns)
    modelo = XGBRegressor(random_state=42, **params)
    modelo.fit(X_train, np.log1p(train[objetivo]))
    pred = np.expm1(modelo.predict(X_test))
    modelo.save_model(f"models/modelo_{nombre}.json")
    info[nombre] = {
        "objetivo": objetivo, "num": num, "cat": cat, "params": params,
        "columnas": list(X_train.columns),
        "mae_prueba": mae(test[objetivo], pred), "rmse_prueba": rmse(test[objetivo], pred),
    }
    print(f"{nombre}: MAE={info[nombre]['mae_prueba']:.2f}  RMSE={info[nombre]['rmse_prueba']:.2f}")

with open("models/info_modelos.json", "w") as f:
    json.dump(info, f, indent=2)
print("Modelos guardados en models/")

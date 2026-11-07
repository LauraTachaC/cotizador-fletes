"""Experimentos del modelo de flete terrestre (land freight por skid) en MLflow.

Uso (desde la carpeta raíz del repositorio):
    export MLFLOW_TRACKING_URI=http://localhost:5000
    python src/modelos/experimentos_land_mlflow.py

Se entrena solo con órdenes que tuvieron flete terrestre (land_cost > 0).
Cuando el cliente recoge en el puerto, el land freight es 0 por regla de negocio.
"""
import os
import sys

import mlflow
import mlflow.xgboost
import numpy as np
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(__file__))
from comun import LAND_CAT, LAND_NUM, NUM_BASE, cargar, mae, matriz, rmse, separar_por_tiempo  # noqa: E402

OBJETIVO = "land_cost_per_skid"

df = cargar()
df = df[df["land_cost"] > 0]
train, test = separar_por_tiempo(df)
y_train, y_test = train[OBJETIVO], test[OBJETIVO]

mlflow.set_experiment("Cotizador-Flete-Terrestre")

# 1. Línea base sin modelo: mediana histórica por puerto y estado de entrega
with mlflow.start_run(run_name="linea_base_mediana_ruta"):
    med = train.groupby(["destination_port", "delivery_state"])[OBJETIVO].median()
    pred = [med.get((p, e), np.nan) for p, e in zip(test["destination_port"], test["delivery_state"])]
    pred = np.where(np.isnan(pred), y_train.median(), pred)
    mlflow.log_param("modelo", "mediana por puerto y estado")
    mlflow.log_metric("mae", mae(y_test, pred))
    mlflow.log_metric("rmse", rmse(y_test, pred))

# 2. Experimentos con XGBoost
experimentos = [
    ("xgb_sin_geografia", NUM_BASE, ["container_type"], dict(n_estimators=150, max_depth=6, learning_rate=0.05)),
    ("xgb_con_geografia", LAND_NUM, LAND_CAT, dict(n_estimators=150, max_depth=6, learning_rate=0.05)),
    ("xgb_arboles_poco_profundos", LAND_NUM, LAND_CAT, dict(n_estimators=300, max_depth=4, learning_rate=0.05)),
    ("xgb_arboles_profundos", LAND_NUM, LAND_CAT, dict(n_estimators=500, max_depth=8, learning_rate=0.03)),
    ("xgb_min_child_weight", LAND_NUM, LAND_CAT, dict(n_estimators=300, max_depth=6, learning_rate=0.05, min_child_weight=5)),
]

for nombre, num, cat, params in experimentos:
    with mlflow.start_run(run_name=nombre):
        X_train = matriz(train, num, cat)
        X_test = matriz(test, num, cat, columnas=X_train.columns)
        modelo = XGBRegressor(random_state=42, **params)
        modelo.fit(X_train, np.log1p(y_train))
        pred = np.expm1(modelo.predict(X_test))

        mlflow.log_param("modelo", "XGBoost")
        mlflow.log_param("variables_categoricas", ", ".join(cat))
        mlflow.log_param("num_trees", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("learning_rate", params["learning_rate"])
        mlflow.log_param("min_child_weight", params.get("min_child_weight", 1))
        mlflow.log_metric("mae", mae(y_test, pred))
        mlflow.log_metric("rmse", rmse(y_test, pred))
        mlflow.xgboost.log_model(modelo, name="xgboost-land-model")
        print(f"{nombre}: MAE={mae(y_test, pred):.2f}  RMSE={rmse(y_test, pred):.2f}")

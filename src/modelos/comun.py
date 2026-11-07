"""Funciones comunes para entrenar y usar los modelos del cotizador.

Las variables que se usan son solo las que ventas conoce al momento de cotizar.
"""
import numpy as np
import pandas as pd

RUTA_DATOS = "data/ordenes_modelado.csv"

# Variables numéricas comunes (creadas en src/preprocesamiento_datos.py)
NUM_BASE = [
    "total_skids", "lead_time_days", "mes_seno", "mes_coseno",
    "anticipacion_semanas", "es_entrada_oeste", "es_entrada_este",
    "es_destino_interior",
]

# Modelo marítimo (configuración del Experimento 1 de Diego)
OCEAN_NUM = NUM_BASE + ["recogida_en_puerto"]
OCEAN_CAT = ["container_type"]
OCEAN_PARAMS = dict(n_estimators=150, max_depth=6, learning_rate=0.05)

# Modelo terrestre: agrega puerto de llegada, estado de entrega, compañía y moneda
LAND_NUM = NUM_BASE
LAND_CAT = ["container_type", "destination_port", "delivery_state", "company", "currency"]


def cargar(ruta=RUTA_DATOS):
    df = pd.read_csv(ruta)
    df["delivery_state"] = df["delivery_state"].fillna("Desconocido")
    df["company"] = df["company"].astype(str)
    return df


def separar_por_tiempo(df):
    """Entrenamiento: 2024-2025. Prueba: 2026 (igual que el modelo marítimo)."""
    return df[df["order_year"] <= 2025].copy(), df[df["order_year"] == 2026].copy()


def matriz(df, num, cat, columnas=None):
    """One-hot de las categóricas. Si se pasan columnas, se alinea a ellas."""
    X = pd.get_dummies(df[num + cat].copy(), columns=cat)
    if columnas is not None:
        X = X.reindex(columns=columnas, fill_value=0)
    return X.astype(float)


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))

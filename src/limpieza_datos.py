"""
Limpieza y unión de datos - Proyecto Cotizador de costos logísticos
-------------------------------------------------------------------
Entradas : reporte IMCS y reporte Shipping Schedule (ya anonimizados por la empresa)
Salida   : data/ordenes_limpias.csv  (una fila por orden/waybill)

Uso:
    python limpieza_datos.py <ruta_IMCS.xlsx> <ruta_Shipping_Schedule.xlsx> [salida.csv]

Pasos:
  1. Unir los dos reportes por Waybill.
  2. Quitar columnas que no se usan o que podrían identificar a la empresa.
  3. Filtrar el alcance: envíos marítimos desde China, desde 2024, ya facturados,
     con costos reales mayores que cero y sin las órdenes de muestras (Samples).
  4. Crear variables (mes, días de anticipación, tipo de contenedor...) y los
     objetivos (costo real ocean y land, total y por skid).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Sufijos que NO son envíos marítimos normales (mensajería, muestras, aéreo)
SUFIJOS_FUERA_ALCANCE = {"DHL", "PARCEL", "FEDEX", "UPS", "PREPRODUCTION", "AIR"}
CARRIERS_FUERA_ALCANCE = {"TEST OCEAN"}
# Clientes internos de muestras (Samples): no son órdenes comerciales
CLIENTES_MUESTRAS = {51, 535}


def leer(imcs_path, ss_path):
    imcs = pd.read_excel(imcs_path)
    ss = pd.read_excel(ss_path)
    # Columnas del Shipping Schedule que aportan información nueva
    cols_ss = [
        "Waybill", "Prod Source", "Origin", "Destination Port", "Delivery Dest",
        "Req Delivery", "Req Ship Date", "Ocean Carrier", "Land Carrier",
        "CUT OFF", "Original Destination", "Original Ocean Value",
    ]
    return imcs.merge(ss[cols_ss], on="Waybill", how="inner")


def limpiar(df):
    df = df.copy()
    df["Suffix"] = df["Suffix"].astype(str).str.upper().str.strip().replace("NAN", "")
    df["Ocean Carrier"] = df["Ocean Carrier"].astype(str).str.strip()
    df["Land Carrier"] = df["Land Carrier"].astype(str).str.strip()
    df["order_date"] = pd.to_datetime(df["Order Creation Date"], format="%m/%d/%Y", errors="coerce")
    df["invoiced_date"] = pd.to_datetime(df["Sage Invoiced At"], format="%m/%d/%Y", errors="coerce")

    # --- Filtros de alcance ---
    filtro = (
        (df["Prod Source"] == "China")
        & (df["order_date"] >= "2024-01-01")
        & df["invoiced_date"].notna()                      # orden cerrada: costos reales completos
        & (df["Total Skids"] > 0)
        & (df["Actual Total Ocean Freight with Misc"] > 0)
        & ~df["Suffix"].isin(SUFIJOS_FUERA_ALCANCE)
        & ~df["Ocean Carrier"].str.upper().isin(CARRIERS_FUERA_ALCANCE)
        & ~df["Customer ID"].isin(CLIENTES_MUESTRAS)
    )
    df = df[filtro]

    # --- Variables (se conocen al cotizar) ---
    out = pd.DataFrame({
        "waybill": df["Waybill"],
        "company": df["Company"],
        "customer_id": df["Customer ID"],
        "order_date": df["order_date"].dt.date,
        "order_year": df["order_date"].dt.year,
        "order_month": df["order_date"].dt.month,
        "origin": df["Origin"],
        "destination_port": df["Destination Port"].fillna(df["Port Landed"]),
        "delivery_dest": df["Delivery Dest"],
        "currency": df["Currency"],
        "container_type": np.select(
            [df["Suffix"].str.contains("LCL"), df["Suffix"].str.contains("20")],
            ["LCL", "20FT"], default="40FT"),
        "total_skids": df["Total Skids"],
        "lead_time_days": (df["Req Delivery"] - df["order_date"]).dt.days,
        # Asignados durante la logística: revisar si se conocen al cotizar (posible leakage)
        "ocean_carrier": df["Ocean Carrier"].replace("nan", np.nan),
        "land_carrier": df["Land Carrier"].replace("nan", np.nan),
        # --- Reglas de negocio ---
        "base_ocean_rate": df["Base Ocean Rate"],
        "invoice_rate_markup": df["Invoice Rate"],
        # --- Objetivos (costos reales, solo se conocen después) ---
        "ocean_cost": df["Actual Total Ocean Freight with Misc"],
        "land_cost": df["Actual Land Freight incl. Drayage"],
        # --- Para verificar las reglas (lo facturado) ---
        "ocean_surcharge": df["Ocean Surcharge"],
        "final_surcharge": df["Final Surcharge"],
        "land_freight_invoiced": df["Land Freight (Invoice Rate)"],
        "surcharge_discount": df["Freight Surcharge Discount"],
        "land_freight_discount": df["Land Freight Discount"],
        # --- Cambios logísticos (parte descriptiva) ---
        "ocean_value_change": df["Ocean Value Change"].fillna(df["CUT OFF"]),
        "original_destination": df["Original Destination"],
        "original_ocean_value": df["Original Ocean Value"],
    })
    out.loc[out["lead_time_days"] < 0, "lead_time_days"] = np.nan   # fechas inconsistentes
    out["delivery_state"] = out["delivery_dest"].astype(str).str.split(",").str[-1].str.strip()
    out["ocean_cost_per_skid"] = out["ocean_cost"] / out["total_skids"]
    out["land_cost_per_skid"] = out["land_cost"] / out["total_skids"]
    return out.sort_values("order_date").reset_index(drop=True)


if __name__ == "__main__":
    imcs_path, ss_path = sys.argv[1], sys.argv[2]
    salida = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/ordenes_limpias.csv")
    salida.parent.mkdir(parents=True, exist_ok=True)
    datos = limpiar(leer(imcs_path, ss_path))
    datos.to_csv(salida, index=False)
    print(f"{len(datos)} órdenes guardadas en {salida}")

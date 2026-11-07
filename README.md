# Cotizador de Costos Logísticos

Proyecto del curso Despliegue de Soluciones Analíticas (MIAD, Uniandes). Equipo: Isabella Navarro, Diego Triana y Laura Tacha.

Estima, al momento de cotizar una orden importada desde China, el **surcharge** y el **land freight** que se le cobrarán al cliente, y qué porcentaje representan sobre el valor de la orden.

## Estructura
| Carpeta | Contenido |
|---|---|
| `data/` | Datos versionados con DVC (`dvc pull`). No se suben a GitHub. |
| `src/limpieza_datos.py` | Une y limpia los reportes IMCS y Shipping Schedule. |
| `src/preprocesamiento_datos.py` | Prepara los datos para los modelos (`data/ordenes_modelado.csv`). |
| `notebooks/mlflow_proyecto1.py` | Experimentos del modelo de flete marítimo en MLflow. |
| `src/modelos/experimentos_land_mlflow.py` | Experimentos del modelo de flete terrestre en MLflow. |
| `src/modelos/entrenar_modelos_finales.py` | Entrena los dos modelos elegidos y los guarda en `models/`. |
| `tablero/app.py` | Tablero en Dash (cotizar orden, histórico y cambios logísticos). |
| `docs/` | Maqueta del tablero. |

## Cómo ejecutarlo
```bash
pip install -r requirements.txt
dvc pull                                   # o copiar data/ordenes_limpias.csv
python src/preprocesamiento_datos.py
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/modelos/experimentos_land_mlflow.py
python src/modelos/entrenar_modelos_finales.py
python tablero/app.py                      # http://IP:8050
```

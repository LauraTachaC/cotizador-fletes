# Resumen del proyecto: Cotizador de Costos Logísticos

**Equipo G26:** Laura Tacha · Diego Triana · Isabella Navarro

## Problema
La empresa importa materas plásticas desde China (Shanghai y Nanjing) y las distribuye en Estados Unidos y Canadá. Además del precio del producto, cobra dos cargos logísticos: el **surcharge** (flete marítimo) y el **land freight** (flete terrestre). Hoy ambos se conocen solo al facturar, tres o cuatro meses después de cotizar, lo que genera reclamos y descuentos no planeados.

## Pregunta de negocio
Al cotizar una orden, ¿cuánto se le cobrará al cliente por surcharge y land freight, y qué porcentaje representan sobre el valor de la orden?

**Usuarios:** Ventas (cotización transparente) y Logística (análisis de rutas).

## Datos
- Reportes internos IMCS y Shipping Schedule, unidos por waybill.
- Órdenes producidas en China con envío marítimo, de enero de 2024 a julio de 2026, ya facturadas.
- 5.608 órdenes y 30 columnas; 5.558 órdenes después de la preparación.
- Los datos se versionan con DVC en S3 y no se suben a GitHub.

## Solución
1. **Modelos (XGBoost):** estiman el costo marítimo y el terrestre por skid con la información disponible al cotizar. Se entrenan con 2024–2025 y se evalúan con 2026.
2. **Reglas de negocio:** a partir de las predicciones se calculan el surcharge, el land freight y el porcentaje sobre el valor de la orden.
3. **Tablero (Dash):** pestañas *Cotizar orden*, *Histórico* y *Cambios logísticos*.

## Resultados de la Entrega 2
| Modelo | Línea base (MAE) | Modelo elegido (MAE) | Mejora |
|---|---|---|---|
| Flete marítimo | 91,8 USD/skid | 68,6 USD/skid | ~25 % |
| Flete terrestre | 62,1 USD/skid | 44,0 USD/skid | ~29 % |

Los experimentos están registrados en MLflow sobre máquinas AWS EC2.

## Estructura del repositorio
- `src/preprocesamiento_datos.py`: limpieza y creación de variables.
- `tablero/app.py`: tablero en Dash.
- `docs/`: documentación y reportes del proyecto.

## Próximos pasos
Empaquetar los modelos, servirlos con una API, conectar el tablero a la API y desplegar todo con Docker.

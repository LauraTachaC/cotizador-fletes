"""Tablero del Cotizador de Costos Logísticos (Dash).

Uso (desde la carpeta raíz del repositorio, con los modelos ya entrenados):
    python tablero/app.py
y abrir http://IP:8050

Pestañas (según la maqueta de la Entrega 1):
 1. Cotizar orden: estima ocean y land freight con los modelos y aplica las reglas de negocio.
 2. Histórico: costos por puerto, por mes y surcharge sobre el base rate por ruta.
 3. Cambios logísticos: cut offs, cambios a 20FT/LCL y rutas desviadas.

En esta versión el tablero carga los modelos directamente desde models/.
En la siguiente entrega las estimaciones se pedirán a la API.
"""
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dash_table, dcc, html
from xgboost import XGBRegressor

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "src"))
sys.path.insert(0, os.path.join(RAIZ, "src", "modelos"))
from comun import matriz  # noqa: E402
from preprocesamiento_datos import ingenieria_caracteristicas  # noqa: E402

TASA_CAD_USD = 0.74
VERDE, VERDE_CLARO, AZUL, NARANJA, ROJO, GRIS = "#2E5E2E", "#9BBB59", "#4E6E8E", "#E8A33D", "#B5473A", "#F2F2F2"

# ---------- Datos y modelos ----------
info = json.load(open(os.path.join(RAIZ, "models", "info_modelos.json")))
modelos = {}
for k in ("ocean", "land"):
    m = XGBRegressor()
    m.load_model(os.path.join(RAIZ, "models", f"modelo_{k}.json"))
    modelos[k] = m

hist = pd.read_csv(os.path.join(RAIZ, "data", "ordenes_limpias.csv"), parse_dates=["order_date"])
hist = hist[hist["total_skids"] >= 1].copy()
cad = hist["currency"] == "CAD"
for c in ["ocean_cost_per_skid", "land_cost_per_skid"]:
    hist[c + "_usd"] = np.where(cad, hist[c] * TASA_CAD_USD, hist[c])
hist["ruta"] = hist["origin"] + " → " + hist["destination_port"].str.split(",").str[0]
hist["surcharge_pct"] = hist["ocean_surcharge"] / hist["base_ocean_rate"].replace(0, np.nan)
# Cargos misceláneos (misc / demurrage): diferencia entre el surcharge histórico y (ocean_cost - base rate).
# No los predice el modelo; se usa la mediana del último año por moneda.
hist["misc"] = hist["ocean_surcharge"] - (hist["ocean_cost"] - hist["base_ocean_rate"])
ultimo_anio = hist[hist["order_date"] >= hist["order_date"].max() - pd.DateOffset(years=1)]
MISC = ultimo_anio.groupby("currency")["misc"].median().to_dict()
hist["company"] = hist["company"].astype(str)

puertos = sorted(hist["destination_port"].dropna().unique())
estados = sorted(hist["delivery_state"].dropna().unique())
ciudades_por_puerto = hist.dropna(subset=["delivery_state"]).groupby("destination_port")["delivery_state"].agg(lambda s: s.mode().iloc[0])


def estimar(entrada):
    """Devuelve el costo estimado por skid (en USD) de los dos modelos."""
    fila = pd.DataFrame([entrada])
    fila["land_cost"] = 0.0 if entrada["recoge"] else 1.0
    fila = ingenieria_caracteristicas(fila)
    out = {}
    for k in ("ocean", "land"):
        X = matriz(fila, info[k]["num"], info[k]["cat"], columnas=info[k]["columnas"])
        out[k] = float(np.expm1(modelos[k].predict(X))[0])
    if entrada["recoge"]:
        out["land"] = 0.0
    return out


# ---------- Componentes de la interfaz ----------
def campo(etiqueta, control):
    return html.Div([html.Label(etiqueta, className="etq"), control], className="campo")


def tarjeta(titulo, id_valor, color=AZUL):
    return html.Div([html.Div(titulo, className="tit"), html.Div("$ —", id=id_valor, className="val", style={"color": color})],
                    className="tarjeta")


def kpi(valor, texto, color=VERDE):
    return html.Div([html.Div(valor, className="val", style={"color": color}), html.Div(texto, className="tit")], className="tarjeta")


def fig_base(fig, titulo):
    fig.update_layout(title=dict(text=titulo, x=0.5, font=dict(size=14)), template="simple_white",
                      margin=dict(l=10, r=10, t=50, b=10), height=330, legend=dict(orientation="h", y=-0.15),
                      font=dict(family="Calibri, Arial", size=12))
    return fig


tab_cotizar = html.Div(className="fila", children=[
    html.Div(className="panel", style={"flex": "1"}, children=[
        html.H4("Datos de la orden"),
        campo("Compañía", dcc.Dropdown(["1", "2"], "1", id="company", clearable=False)),
        campo("Moneda", dcc.Dropdown(["USD", "CAD"], "USD", id="currency", clearable=False)),
        campo("Puerto de origen", dcc.Dropdown(["Shanghai", "Nanjing"], "Shanghai", id="origin", clearable=False)),
        campo("Puerto de destino", dcc.Dropdown(puertos, "Toronto, ON", id="port", clearable=False)),
        campo("Estado / provincia de entrega", dcc.Dropdown(estados, "ON", id="state", clearable=False)),
        campo("Tipo de contenedor", dcc.Dropdown(["40FT", "20FT", "LCL"], "40FT", id="container", clearable=False)),
        campo("Skids delivered", dcc.Input(id="sk_d", type="number", value=12, min=0)),
        campo("Skids undelivered", dcc.Input(id="sk_u", type="number", value=8, min=0)),
        campo("Fecha de la orden", dcc.DatePickerSingle(id="f_orden", date=date.today(), display_format="DD/MM/YYYY")),
        campo("Fecha pedida por el cliente", dcc.DatePickerSingle(id="f_pedida", date=date(date.today().year + 1, 2, 15), display_format="DD/MM/YYYY")),
        campo("Base rate ($)", dcc.Input(id="base", type="number", value=4600, min=0)),
        campo("Mark up", dcc.Dropdown([1.1, 1.2, 1.3], 1.3, id="markup", clearable=False)),
        campo("Valor de la orden ($)", dcc.Input(id="valor", type="number", value=25000, min=1)),
        dcc.Checklist([{"label": " El cliente recoge en el puerto", "value": "si"}], [], id="recoge"),
        html.Button("ESTIMAR", id="btn", className="boton"),
    ]),
    html.Div(className="panel", style={"flex": "1.3"}, children=[
        html.H4("Resultado estimado"),
        html.Div(className="rejilla", children=[
            tarjeta("Ocean freight estimado", "o_ocean"), tarjeta("Land freight estimado", "o_land"),
            tarjeta("Surcharge para el cliente", "o_sur", VERDE), tarjeta("Land freight para el cliente", "o_lf", VERDE)]),
        html.Div(id="o_total", className="total"),
        dcc.Graph(id="g_comp", config={"displayModeBar": False}),
        html.Div("Estimación con modelos XGBoost entrenados con órdenes de 2024-2025 y evaluados en 2026. "
                 "El surcharge incluye cargos misceláneos estimados con la mediana del último año. "
                 "El rango probable usa el error promedio (MAE) de prueba de cada modelo.", className="nota"),
    ]),
])

filtros = html.Div(className="fila", children=[
    html.Div(dcc.Dropdown(puertos, None, id="h_port", placeholder="Puerto de destino (todos)", multi=True), style={"flex": "2"}),
    html.Div(dcc.Dropdown(["1", "2"], None, id="h_company", placeholder="Compañía (todas)"), style={"flex": "1"}),
    html.Div(dcc.DatePickerRange(id="h_fechas", start_date=hist.order_date.min().date(), end_date=hist.order_date.max().date(),
                                 display_format="MMM YYYY"), style={"flex": "2"}),
])

tab_hist = html.Div([
    filtros,
    html.Div(className="fila", children=[
        html.Div(dcc.Graph(id="g_puertos"), className="panel", style={"flex": "1"}),
        html.Div(dcc.Graph(id="g_mes"), className="panel", style={"flex": "1"}),
    ]),
    html.Div(dcc.Graph(id="g_ruta"), className="panel"),
])

co = hist[hist["ocean_value_change"] == "CutOff"]
sin = hist[hist["ocean_value_change"].isna()]
sobrecosto_land = co["land_cost_per_skid_usd"].median() - sin["land_cost_per_skid_usd"].median()
desvios = (co.dropna(subset=["original_destination"])
           .groupby(["original_destination", "destination_port"])
           .agg(ordenes=("waybill", "count"), land=("land_cost_per_skid_usd", "median"), ocean=("ocean_cost_per_skid_usd", "median"))
           .reset_index().sort_values("ordenes", ascending=False).head(10))
desvios["land"] = desvios["land"].map("${:,.0f}".format)
desvios["ocean"] = desvios["ocean"].map("${:,.0f}".format)
comp = pd.DataFrame({"tipo": ["Sin cambio", "Cut off"],
                     "Ocean": [sin["ocean_cost_per_skid_usd"].median(), co["ocean_cost_per_skid_usd"].median()],
                     "Land": [sin["land_cost_per_skid_usd"].median(), co["land_cost_per_skid_usd"].median()]})
fig_co = go.Figure([go.Bar(name="Ocean", x=comp.tipo, y=comp.Ocean, marker_color=AZUL, text=comp.Ocean.round(0), textposition="outside"),
                    go.Bar(name="Land", x=comp.tipo, y=comp.Land, marker_color=VERDE_CLARO, text=comp.Land.round(0), textposition="outside")])
fig_base(fig_co, "Costo mediano por skid (USD): órdenes con y sin cut off")
anual = co.groupby(co.order_date.dt.year).size()
fig_anual = fig_base(go.Figure(go.Bar(x=anual.index.astype(str), y=anual.values, marker_color=ROJO, text=anual.values, textposition="outside")),
                     "Cut offs por año")

tab_cambios = html.Div([
    html.Div(className="fila", children=[
        kpi(f"{len(co)}", "Cut offs"), kpi(f"{(hist.ocean_value_change == '20FT').sum()}", "Cambios a 20FT"),
        kpi(f"{(hist.ocean_value_change == 'LCL').sum()}", "Cambios a LCL"),
        kpi(f"+${sobrecosto_land:,.0f}", "Land freight adicional por skid en un cut off (mediana, USD)", ROJO)]),
    html.Div(className="fila", children=[
        html.Div(dcc.Graph(figure=fig_co), className="panel", style={"flex": "1"}),
        html.Div(dcc.Graph(figure=fig_anual), className="panel", style={"flex": "1"}),
    ]),
    html.Div(className="panel", children=[
        html.H4("Rutas desviadas más frecuentes"),
        dash_table.DataTable(
            data=desvios.to_dict("records"),
            columns=[{"name": n, "id": i} for n, i in [("Destino original", "original_destination"), ("Destino nuevo", "destination_port"),
                                                       ("Órdenes", "ordenes"), ("Land por skid", "land"), ("Ocean por skid", "ocean")]],
            style_header={"backgroundColor": VERDE, "color": "white", "fontWeight": "bold"},
            style_cell={"fontFamily": "Calibri, Arial", "padding": "6px", "textAlign": "left"},
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": GRIS}]),
        html.Div("El land freight desde el puerto desviado suele multiplicar el costo total, aunque el ocean sea más barato.", className="nota"),
    ]),
])

app = Dash(__name__, title="Cotizador de fletes")
app.layout = html.Div(className="contenedor", children=[
    html.H2("Cotizador de Costos Logísticos", style={"color": VERDE}),
    dcc.Tabs(id="tabs", value="t1", children=[
        dcc.Tab(label="🧮 Cotizar orden", value="t1", children=tab_cotizar, selected_style={"backgroundColor": VERDE, "color": "white"}),
        dcc.Tab(label="📈 Histórico", value="t2", children=tab_hist, selected_style={"backgroundColor": VERDE, "color": "white"}),
        dcc.Tab(label="🚚 Cambios logísticos", value="t3", children=tab_cambios, selected_style={"backgroundColor": VERDE, "color": "white"}),
    ]),
])


@app.callback(Output("state", "value"), Input("port", "value"))
def estado_sugerido(port):
    return ciudades_por_puerto.get(port, "ON")


@app.callback(
    [Output(i, "children") for i in ("o_ocean", "o_land", "o_sur", "o_lf", "o_total")] + [Output("g_comp", "figure")],
    Input("btn", "n_clicks"),
    [State(i, "value") for i in ("company", "currency", "origin", "port", "state", "container", "sk_d", "sk_u",
                                 "base", "markup", "valor", "recoge")] + [State("f_orden", "date"), State("f_pedida", "date")],
)
def cotizar(_, company, currency, origin, port, state, container, sk_d, sk_u, base, markup, valor, recoge, f_orden, f_pedida):
    sk_d, sk_u = float(sk_d or 0), float(sk_u or 0)
    skids = max(sk_d + sk_u, 1.0)
    f1, f2 = pd.to_datetime(f_orden), pd.to_datetime(f_pedida)
    entrada = dict(company=str(company), currency=currency, origin=origin, destination_port=port, delivery_state=state,
                   container_type=container, total_skids=skids, lead_time_days=max((f2 - f1).days, 0),
                   order_month=f1.month, recoge=bool(recoge))
    est = estimar(entrada)
    a_moneda = (1 / TASA_CAD_USD) if currency == "CAD" else 1.0   # los modelos estiman en USD
    ocean = est["ocean"] * skids * a_moneda
    land = est["land"] * skids * a_moneda

    # Reglas de negocio
    misc = MISC.get(currency, 0.0)
    surcharge = ocean + misc - float(base or 0)
    land_cliente = land * float(markup)
    parte_delivered = land_cliente * sk_d / skids
    surcharge_cliente = surcharge + parte_delivered       # delivered: se suma al surcharge
    land_facturado = land_cliente - parte_delivered      # undelivered: se factura como freight
    total = surcharge_cliente + land_facturado
    err = (info["ocean"]["mae_prueba"] + info["land"]["mae_prueba"] * float(markup)) * skids * a_moneda
    m = "C$" if currency == "CAD" else "$"

    total_div = [html.Div(f"Total en fletes: {m}{total:,.0f}", style={"fontSize": "22px", "fontWeight": "bold"}),
                 html.Div(f"+{total / float(valor or 1):.0%} sobre el valor de la orden"),
                 html.Div(f"Rango probable: {m}{max(total - err, 0):,.0f} – {m}{total + err:,.0f}", style={"fontSize": "13px"})]
    fig = go.Figure(go.Bar(x=[surcharge_cliente, land_facturado, total], y=["Surcharge", "Land freight", "Total"], orientation="h",
                           marker_color=[VERDE_CLARO, VERDE_CLARO, VERDE], text=[f"{m}{v:,.0f}" for v in [surcharge_cliente, land_facturado, total]],
                           textposition="outside"))
    fig_base(fig, "Composición del cobro estimado").update_layout(height=260)
    return (f"{m}{ocean:,.0f}", f"{m}{land:,.0f}", f"{m}{surcharge_cliente:,.0f}", f"{m}{land_facturado:,.0f}", total_div, fig)


@app.callback([Output("g_puertos", "figure"), Output("g_mes", "figure"), Output("g_ruta", "figure")],
              [Input("h_port", "value"), Input("h_company", "value"), Input("h_fechas", "start_date"), Input("h_fechas", "end_date")])
def historico(ports, company, ini, fin):
    d = hist[(hist.order_date >= ini) & (hist.order_date <= fin)]
    if company:
        d = d[d.company == company]
    if ports:
        d = d[d.destination_port.isin(ports)]
    top = d.destination_port.value_counts().head(8).index
    g = d[d.destination_port.isin(top)].groupby("destination_port")[["ocean_cost_per_skid_usd", "land_cost_per_skid_usd"]].median()
    g = g.sort_values("ocean_cost_per_skid_usd", ascending=False)
    x = [p.split(",")[0] for p in g.index]
    f1 = go.Figure([go.Bar(name="Ocean", x=x, y=g.ocean_cost_per_skid_usd, marker_color=AZUL),
                    go.Bar(name="Land", x=x, y=g.land_cost_per_skid_usd, marker_color=VERDE_CLARO)])
    fig_base(f1, "Costo mediano por skid según puerto (USD)")
    mm = d.groupby(d.order_date.dt.to_period("M"))[["ocean_cost_per_skid_usd", "land_cost_per_skid_usd"]].median()
    f2 = go.Figure([go.Scatter(name="Ocean", x=mm.index.astype(str), y=mm.ocean_cost_per_skid_usd, line=dict(color=AZUL, width=2)),
                    go.Scatter(name="Land", x=mm.index.astype(str), y=mm.land_cost_per_skid_usd, line=dict(color=VERDE_CLARO, width=2))])
    fig_base(f2, "Costo mediano por skid según mes (USD)")
    r = d[d.base_ocean_rate > 0].groupby("ruta").agg(n=("waybill", "count"), pct=("surcharge_pct", "median"))
    r = r[r.n >= 15].sort_values("n", ascending=False).head(8).sort_values("pct")
    f3 = go.Figure(go.Bar(x=r.pct, y=r.index, orientation="h", marker_color=NARANJA, text=[f"{v:.0%}" for v in r.pct], textposition="outside"))
    fig_base(f3, "Surcharge real sobre el base rate por ruta (mediana, rutas con más órdenes)").update_xaxes(tickformat=".0%")
    return f1, f2, f3


app.index_string = app.index_string.replace("</head>", """<style>
body{font-family:Calibri,Arial,sans-serif;background:#FAFAF7;margin:0}
.contenedor{max-width:1250px;margin:auto;padding:12px 20px}
.fila{display:flex;gap:14px;margin:12px 0;flex-wrap:wrap}
.panel{background:white;border:1px solid #DDD;border-radius:10px;padding:12px 16px;margin-bottom:12px}
.panel h4{color:#2E5E2E;margin:4px 0 10px}
.campo{display:flex;align-items:center;margin:5px 0}.etq{width:45%;font-size:14px}.campo>div,.campo>input{width:55%}
.boton{background:#2E5E2E;color:white;border:none;border-radius:8px;padding:10px 40px;font-weight:bold;margin:12px auto;display:block;cursor:pointer}
.rejilla{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.tarjeta{background:#F2F2F2;border-radius:8px;padding:10px 14px;flex:1;min-width:180px}.tarjeta .tit{font-size:13px;color:#555}.tarjeta .val{font-size:24px;font-weight:bold}
.total{background:#2E5E2E;color:white;border-radius:8px;padding:12px 16px;margin:12px 0}
.nota{font-size:12px;color:#777;font-style:italic;margin-top:6px}
</style></head>""")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)

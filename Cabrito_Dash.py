# dash2.py

import streamlit as st
import pandas as pd
import zipfile
import io
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu  # Asegúrate de importar esto arriba

# ------------------ Definiciones de clases/funciones personalizadas ------------------

class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

# ---------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import joblib

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

st.markdown("""
    <style>
        /* Fondo general sobrio y elegante */
        .main {
            background-color: #f5f7fa !important;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Texto principal en gris oscuro */
        .main > div {
            color: #1e2022 !important;
        }

        /* Estilo de métricas */
        [data-testid="stMetricLabel"] {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a73e8 !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #202124 !important;
        }

        [data-testid="stMetricDelta"] {
            font-weight: bold;
            color: #34a853 !important;
        }

        /* Sidebar limpio con acento azul petróleo */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e0e0e0;
        }

        [data-testid="stSidebar"] * {
            color: #1a3c5a !important;
        }

        /* Encabezados de expander elegantes */
        .stExpander > summary {
            font-weight: 600;
            color: #1a3c5a !important;
        }

        /* Tabs refinadas */
        .stTabs [data-baseweb="tab"] {
            font-size: 15px;
            padding: 12px;
            border-bottom: 2px solid transparent;
            color: #5f6368;
        }

        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #1a73e8;
            color: #1a73e8;
            font-weight: 600;
        }

        /* Selector multiple compacto */
        .stMultiSelect .css-12w0qpk {
            max-height: 0px !important;
            overflow: hidden !important;
        }

        .stMultiSelect {
            height: 38px !important;
        }

        /* Ocultar watermark de Streamlit */
        .css-1wa3eu0 {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)


def clasificar_zonas(df, estado_sel):
    if estado_sel == "Nacional":
        principales = ['Ciudad de México', 'Nuevo León', 'Jalisco']
        return df['estado_del_cliente'].apply(lambda x: x if x in principales else 'Provincia')
    else:
        top_ciudades = (
            df[df['estado_del_cliente'] == estado_sel]['ciudad_cliente']
            .value_counts()
            .nlargest(3)
            .index
            .tolist()
        )
        return df['ciudad_cliente'].apply(lambda x: x if x in top_ciudades else 'Otras')


tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "App Danu 📈"])

with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if not archivo_zip:
    st.stop()

# ===================== CARGA Y PROCESAMIENTO DE DATOS =====================
with zipfile.ZipFile(archivo_zip) as z:
    requeridos = [
        'DF.csv', 'DF2.csv',
        'modelo_costoflete.sav',
        'modelo_dias_pipeline.joblib',
        'label_encoder_dias.joblib'
    ]
    faltantes = [r for r in requeridos if r not in z.namelist()]
    if faltantes:
        st.error(f"❌ Faltan archivos en el ZIP: {faltantes}")
        st.stop()

    df  = pd.read_csv(z.open('DF.csv'))
    df2 = pd.read_csv(z.open('DF2.csv'))
    modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
    modelo_dias   = joblib.load(z.open('modelo_dias_pipeline.joblib'))
    label_encoder = joblib.load(z.open('label_encoder_dias.joblib'))

with st.sidebar:
    st.subheader("🎛️ Filtro de Estado")
    estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
    estado_sel = option_menu(
        menu_title="Selecciona un estado",
        options=estados,
        icons=["globe"] + ["geo"] * (len(estados) - 1),
        default_index=0
    )

df_filtrado = df.copy() if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel]

# ========================= 📊 RESUMEN NACIONAL =========================
with tabs[0]:
    zona_display = estado_sel if estado_sel != "Nacional" else "Resumen Nacional"
    st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {zona_display}")
    col1, col2 = st.columns(2)
    col1.metric("Pedidos", f"{len(df_filtrado):,}")
    col2.metric("Llegadas muy adelantadas (≥10 días)", f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean() * 100:.1f}%")
    if 'dias_entrega' in df_filtrado.columns:
        # ... (gráficos existentes sin cambios) ...

# ========================= PESTAÑA 🏠 Costo de Envío =========================
with tabs[1]:
    col1, col2 = st.columns(2)
    col1.metric("📦 Total de Pedidos", f"{len(df_filtrado):,}")
    col2.metric("💰 Flete Alto vs Precio", f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%")
    st.subheader("💸 Relación Envío–Precio: ¿Gasto Justificado?")
    # ... (gráficas de envío/precio sin cambios) ...

# ========================= PESTAÑA 🧮 Calculadora =========================
with tabs[2]:
    st.header("🧮 Calculadora de Predicción de Flete y Entrega")

    # --- Preprocesamiento de fechas ---
    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    # --- Selección de Estado y Categoría ---
    col1, col2 = st.columns(2)
    estado2    = col1.selectbox("Estado", sorted(df2['estado_del_cliente'].dropna().unique()))
    categoria2 = col2.selectbox("Categoría", sorted(df2['Categoría'].dropna().unique()))

    # --- Selección de Meses ---
    meses_dict = {
        1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril",
        5:"Mayo", 6:"Junio", 7:"Julio", 8:"Agosto",
        9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
    }
    mcol1, mcol2 = st.columns(2)
    mes1_nombre = mcol1.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2_nombre = mcol2.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1 = [k for k,v in meses_dict.items() if v == mes1_nombre][0]
    mes2 = [k for k,v in meses_dict.items() if v == mes2_nombre][0]

    # --- Filtrar dataframes ---
    dfm1 = df2[
        (df2['estado_del_cliente'] == estado2) &
        (df2['Categoría'] == categoria2) &
        (df2['mes'] == mes1)
    ].copy()
    dfm2 = df2[
        (df2['estado_del_cliente'] == estado2) &
        (df2['Categoría'] == categoria2) &
        (df2['mes'] == mes2)
    ].copy()

    # --- Función de predicción ---
    def predecir(df_in):
        if df_in.empty:
            return df_in
        cols_f = [
            'total_peso_g','precio','#_deproductos','duracion_estimada_min',
            'ciudad_cliente','nombre_dc','hora_compra','año','mes',
            'datetime_origen','region','dias_promedio_ciudad',
            'Categoría','tipo_de_pago'
        ]
        Xf = pd.get_dummies(df_in[cols_f])
        feats = modelo_flete.get_booster().feature_names
        Xf = Xf.reindex(columns=feats, fill_value=0)
        df_in['costo_estimado'] = modelo_flete.predict(Xf).round(2)
        df_in['costo_de_flete'] = df_in['costo_estimado']
        dias_feats = [c for c in df_in.columns if c in modelo_dias.feature_names_in_]
        if dias_feats:
            preds = modelo_dias.predict(df_in[dias_feats])
            df_in['clase_entrega'] = label_encoder.inverse_transform(preds)
        return df_in

    # --- Función de agrupación por Ciudad ---
    def agrupar(df_p, nombre):
        if 'costo_estimado' not in df_p.columns:
            return pd.DataFrame(columns=['Ciudad', nombre])
        out = df_p.groupby('ciudad_cliente').agg({
            nombre:('costo_estimado','mean'),
            f"Entrega {nombre}":(
                'clase_entrega',
                lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
            )
        }).reset_index().rename(columns={'ciudad_cliente':'Ciudad'})
        return out

    # --- Ejecutar predicción y agrupación ---
    dfp1 = predecir(dfm1)
    dfp2 = predecir(dfm2)
    r1 = agrupar(dfp1, mes1_nombre)
    r2 = agrupar(dfp2, mes2_nombre)

    comp = pd.merge(r1, r2, on='Ciudad', how='outer')
    for c in [mes1_nombre, mes2_nombre]:
        comp[c] = pd.to_numeric(comp.get(c), errors='coerce')
    comp['Diferencia'] = (comp[mes2_nombre] - comp[mes1_nombre]).round(2)

    # --- KPIs comparativos ---
    k1, k2, k3 = st.columns(3)
    avg1 = dfp1['costo_estimado'].mean() if not dfp1.empty else np.nan
    avg2 = dfp2['costo_estimado'].mean() if not dfp2.empty else np.nan
    pct  = ((avg2 - avg1) / avg1 * 100) if avg1 else 0
    k1.metric(f"Avg {mes1_nombre}", f"{avg1:.2f}")
    k2.metric("% Cambio", f"{pct:.1f}%")
    k3.metric(f"Avg {mes2_nombre}", f"{avg2:.2f}")

    # --- Tabla de comparación ---
    def style_diff(val):
        return 'color: green' if val > 0 else ('color: red' if val < 0 else '')
    st.subheader(f"Comparación {mes1_nombre} vs {mes2_nombre} — {estado2}/{categoria2}")
    st.dataframe(
        comp.style
            .applymap(style_diff, subset=['Diferencia'])
            .format(precision=2)
    )
    st.download_button(
        "⬇️ Descargar CSV",
        comp.to_csv(index=False),
        file_name="calculadora_comparacion.csv"
    )

# ========================= PESTAÑA App Danu 📈 =========================
with tabs[3]:
    st.header("App Danu – Insights 📈")
    st.write("Aquí va contenido adicional de App Danu...")

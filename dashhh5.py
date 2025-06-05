# dashboard_app.py

import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import joblib
import os
import numpy as np
from PIL import Image

# ======================= CONFIGURACIÓN =======================
st.set_page_config(
    page_title="Cabrito Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================= ESTILO PERSONALIZADO =======================
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #002244;
        }
        [data-testid="stSidebar"] label, .st-cb {
            color: white;
        }
        .stSlider > div[data-testid="stTickBar"] {
            background-color: #ffffff11;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            padding: 10px;
            border-bottom: 3px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #004b8d;
            color: #004b8d;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ======================= LOGO Y TÍTULO =======================
with st.sidebar:
    logo = Image.open("danu_logo.png")
    st.image(logo, use_container_width=True)
    st.title("Cabrito Analytics")

# ======================= TABS =======================
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ======================= FUNCIONES CACHEADAS =======================

@st.cache_data
def cargar_datos(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open("DF.csv") as df_file:
            df = pd.read_csv(df_file)
        with zip_ref.open("DF2.csv") as df2_file:
            df2 = pd.read_csv(df2_file)
    return df, df2

@st.cache_resource
def cargar_modelos():
    modelo_flete = joblib.load("modelo_costoflete.sav")
    modelo_dias = joblib.load("modelo_dias_pipeline.joblib")
    label_encoder = joblib.load("label_encoder_dias.joblib")
    return modelo_flete, modelo_dias, label_encoder

# ======================= SUBIDA Y CARGA DE ARCHIVOS =======================

st.sidebar.markdown("## 📁 Sube tu archivo ZIP")
zip_file = st.sidebar.file_uploader("Archivo ZIP que contenga DF.csv y DF2.csv", type="zip")

if not zip_file:
    st.warning("📦 Por favor sube un archivo .zip con DF.csv y DF2.csv")
    st.stop()

df, df2 = cargar_datos(zip_file)
modelo_flete, modelo_dias, label_encoder = cargar_modelos()

# ======================= PESTAÑA 1: DASHBOARD =======================

with tabs[0]:
    st.header("📊 Panel Logístico")

    # ---------- FILTROS EN SIDEBAR ----------
    with st.sidebar.form("form_filtros"):
        with st.expander("📂 Categoría"):
            cat_sel = st.multiselect("Selecciona categoría", sorted(df['Categoría'].dropna().unique()))
        with st.expander("🗺️ Estado"):
            est_sel = st.multiselect("Selecciona estado", sorted(df['estado_del_cliente'].dropna().unique()))
        with st.expander("📅 Año"):
            año_sel = st.multiselect("Selecciona año", sorted(df['año'].dropna().unique()))
        with st.expander("🗓️ Mes"):
            mes_sel = st.multiselect("Selecciona mes", sorted(df['mes'].dropna().unique()))
        aplicar = st.form_submit_button("✅ Aplicar filtros")

    # ---------- CONTENIDO SI SE APLICAN FILTROS ----------
    if aplicar:
        df_filt = df[
            (df['Categoría'].isin(cat_sel)) &
            (df['estado_del_cliente'].isin(est_sel)) &
            (df['año'].isin(año_sel)) &
            (df['mes'].isin(mes_sel))
        ]

        # ---------- KPIs ----------
        st.subheader("📌 Indicadores principales")
        col1, col2, col3 = st.columns(3)
        col1.metric("📦 Total pedidos", f"{len(df_filt):,}")
        pct_flete = (df_filt['costo_de_flete'] / df_filt['precio'] > 0.5).mean() * 100
        col2.metric("🚚 Flete > 50%", f"{pct_flete:.1f}%")
        anticipadas = (df_filt['desviacion_vs_promesa'] < -7).mean() * 100
        col3.metric("⏱ Entregas ≥7 días antes", f"{anticipadas:.1f}%")

        # ---------- VISUALIZACIONES ----------
        st.subheader("📊 Visualizaciones")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("*Treemap por Categoría*")
            fig = px.treemap(df_filt, path=['Categoría'], values='precio', color='Categoría')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("*Mapa de entregas*")
            df_mapa = df_filt.dropna(subset=['lat_cliente', 'lon_cliente'])
            if not df_mapa.empty:
                st.map(df_mapa.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
            else:
                st.info("Sin ubicaciones disponibles")

        with col3:
            st.markdown("*Entrega vs Colchón*")
            promedio = df_filt.groupby('estado_del_cliente')[['dias_entrega', 'colchon_dias']].mean().reset_index()
            fig = px.bar(promedio, x='estado_del_cliente', y=['dias_entrega', 'colchon_dias'], barmode='group')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)


# ======================= PESTAÑA 2: CALCULADORA =======================
with tabs[1]:
    st.header("📈 Calculadora con ML")

    # --- Selección de filtros ---
    estados_calc = sorted(df2['estado_del_cliente'].dropna().unique())
    categorias_calc = sorted(df2['Categoría'].dropna().unique())

    estado = st.selectbox("Estado", estados_calc)
    categoria = st.selectbox("Categoría", categorias_calc)

    meses_dict = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre",
        11: "Noviembre", 12: "Diciembre"
    }

    col1, col2 = st.columns(2)
    mes1_nombre = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2_nombre = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1 = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
    mes2 = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

    # --- Preprocesamiento ---
    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    filtro = (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)
    df_mes1 = df2[(df2['mes'] == mes1) & filtro].copy()
    df_mes2 = df2[(df2['mes'] == mes2) & filtro].copy()

    # --- Función de predicción ---
    def predecir(df_input):
        df_input = df_input.copy()
        features = ['frecuencia_cliente', 'precio', 'colchon_dias']
        if all(f in df_input.columns for f in features):
            df_input['costo_estimado'] = modelo_flete.predict(df_input[features])
            pred_labels = modelo_dias.predict(df_input[features])
            try:
                df_input['clase_entrega'] = label_encoder.inverse_transform(pred_labels)
            except Exception as e:
                df_input['clase_entrega'] = 'N/A'
                st.warning(f"⚠️ Error al decodificar etiquetas de entrega: {e}")
        else:
            df_input['costo_estimado'] = np.nan
            df_input['clase_entrega'] = 'N/A'
        return df_input

    df_mes1 = predecir(df_mes1)
    df_mes2 = predecir(df_mes2)

    # --- Función resumen por ciudad ---
    def resumen(df, nombre_mes):
        if 'ciudad_cliente' in df.columns:
            return df.groupby('ciudad_cliente').agg({
                'costo_estimado': 'mean',
                'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
            }).reset_index().rename(columns={
                'costo_estimado': nombre_mes,
                'clase_entrega': f"Entrega {nombre_mes}"
            })
        return pd.DataFrame()

    # --- Comparación de resultados ---
    res1 = resumen(df_mes1, mes1_nombre)
    res2 = resumen(df_mes2, mes2_nombre)
    comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
    comparacion['Diferencia Costo'] = comparacion[mes2_nombre] - comparacion[mes1_nombre]

    # --- Visualización y descarga ---
    st.dataframe(comparacion)
    st.download_button(
        "⬇ Descargar comparación",
        comparacion.to_csv(index=False),
        "comparacion.csv",
        "text/csv"
    )


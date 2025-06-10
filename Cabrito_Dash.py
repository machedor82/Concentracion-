# Cabrito Dash v2 (10/06/2025)

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

# ===================== ESTILOS =====================
st.markdown("""
    <style>
        .main { background-color: #f5f7fa !important; font-family: 'Segoe UI', sans-serif; }
        .main > div { color: #1e2022 !important; }
        [data-testid="stMetricLabel"] { font-size: 1.5rem; font-weight: 600; color: #1a73e8 !important; }
        [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; color: #202124 !important; }
        [data-testid="stMetricDelta"] { font-weight: bold; color: #34a853 !important; }
        [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 1px solid #e0e0e0; }
        [data-testid="stSidebar"] * { color: #1a3c5a !important; }
        .stExpander > summary { font-weight: 600; color: #1a3c5a !important; }
        .stTabs [data-baseweb="tab"] { font-size: 15px; padding: 12px; border-bottom: 2px solid transparent; color: #5f6368; }
        .stTabs [aria-selected="true"] { border-bottom: 3px solid #1a73e8; color: #1a73e8; font-weight: 600; }
        .css-1wa3eu0 { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# ===================== FUNCIONES AUXILIARES =====================
def clasificar_zonas(df, estado_sel):
    if estado_sel == "Nacional":
        principales = ['Ciudad de México', 'Nuevo León', 'Jalisco']
        return df['estado_del_cliente'].apply(lambda x: x if x in principales else 'Provincia')
    else:
        top_ciudades = (
            df[df['estado_del_cliente'] == estado_sel]['ciudad_cliente']
            .value_counts().nlargest(3).index.tolist()
        )
        return df['ciudad_cliente'].apply(lambda x: x if x in top_ciudades else 'Otras')

# ===================== CARGA DE MODELOS =====================
modelo_flete = joblib.load("modelo_costoflete.sav")
modelo_dias = joblib.load("modelo_dias_pipeline_70.joblib")
label_encoder = joblib.load("label_encoder_dias_70.joblib")

# ===================== SIDEBAR =====================
with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo CSV")
    archivo_csv = st.file_uploader("Archivo con datos (CSV)", type="csv")

# ===================== CARGA DE DATOS =====================
if archivo_csv:
    df = pd.read_csv(archivo_csv)
    df2 = df.copy()

    estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
    with st.sidebar:
        st.subheader("🎛️ Filtro de Estado")
        estado_sel = option_menu("Selecciona un estado", options=estados,
                                 icons=["globe"] + ["geo"] * (len(estados) - 1), default_index=0)

    df_filtrado = df.copy() if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel]

    # Aquí seguiría el desarrollo de las pestañas
    st.success("Datos y modelos cargados correctamente. Dashboard en funcionamiento.")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")

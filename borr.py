# dashboard_app.py

import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import joblib
import os
from PIL import Image
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Cabrito Analytics", layout="wide", initial_sidebar_state="expanded")

# ---------------- STYLE ----------------
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

# ---------------- TÍTULO & LOGO ----------------
with st.sidebar:
    logo = Image.open("danu_logo.png")
    st.image(logo, use_container_width=True)

    st.title("Cabrito Analytics")

tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ---------------- FUNCIONES CACHEADAS ----------------

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

# ---------------- SUBIR ZIP ----------------
st.sidebar.markdown("## 📁 Sube tu archivo ZIP")
zip_file = st.sidebar.file_uploader("Archivo ZIP que contenga DF.csv y DF2.csv", type="zip")

if not zip_file:
    st.warning("📦 Por favor sube un archivo .zip con DF.csv y DF2.csv")
    st.stop()

# ---------------- CARGAR DATOS Y MODELOS ----------------
df, df2 = cargar_datos(zip_file)

@st.cache_resource
def cargar_modelos():
    modelo_flete = joblib.load("modelo_costoflete.sav")
    modelo_dias = joblib.load("modelo_dias_pipeline.joblib")
    label_encoder = joblib.load("label_encoder_dias.joblib")
    return modelo_flete, modelo_dias, label_encoder

modelo_flete, modelo_dias, label_encoder = cargar_modelos()

# ---------------- PESTAÑA 1: DASHBOARD ----------------
with tabs[0]:
    st.header("📊 Panel Logístico")

    # ========== FILTROS EN SIDEBAR ========== #
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

    # ========== FILTRADO Y CONTENIDO SI SE APLICAN FILTROS ========== #
    if aplicar:
        df_filt = df[
            (df['Categoría'].isin(cat_sel)) &
            (df['estado_del_cliente'].isin(est_sel)) &
            (df['año'].isin(año_sel)) &
            (df['mes'].isin(mes_sel))
        ]

        # -------- KPIs -------- #
        st.subheader("📌 Indicadores principales")
        col1, col2, col3 = st.columns(3)
        col1.metric("📦 Total pedidos", f"{len(df_filt):,}")
        pct_flete = (df_filt['costo_de_flete'] / df_filt['precio'] > 0.5).mean() * 100
        col2.metric("🚚 Flete > 50%", f"{pct_flete:.1f}%")
        anticipadas = (df_filt['desviacion_vs_promesa'] < -7).mean() * 100
        col3.metric("⏱ Entregas ≥7 días antes", f"{anticipadas:.1f}%")

        # -------- VISUALIZACIONES -------- #
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



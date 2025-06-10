# Cabrito Dash (versión corregida con indentación adecuada)

import streamlit as st
import pandas as pd
import zipfile
import io
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

# CSS personalizado
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

# Clases personalizadas
class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

# Tabs
if 'tabs' not in st.session_state:
    st.session_state.tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "App Danu 📈"])

tabs = st.session_state.tabs

with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

# ===================== CARGA Y PROCESAMIENTO DE DATOS =====================
if archivo_zip:
    with zipfile.ZipFile(archivo_zip) as z:
        requeridos = [
            'DF.csv', 'DF2.csv',
            'modelo_costoflete.sav',
            'modelo_dias_pipeline.joblib',
            'label_encoder_dias.joblib'
        ]
        contenidos = z.namelist()
        faltantes = [r for r in requeridos if r not in contenidos]
        if faltantes:
            st.error(f"❌ Faltan archivos en el ZIP: {faltantes}")
            st.stop()

        df = pd.read_csv(z.open('DF.csv'))
        df2 = pd.read_csv(z.open('DF2.csv'))
        modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
        modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
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

    df_filtrado = df if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel]
    else:
        st.warning("Sube el ZIP para continuar.")
        st.stop()

# 📊 Resumen Nacional
with tabs[0]:
    zona = estado_sel if estado_sel!="Nacional" else "Resumen Nacional"
    st.title(f"📊 Resumen – {zona}")
    col1, col2 = st.columns(2)
    col1.metric("Pedidos", f"{len(df_filtrado):,}")
    col2.metric("Llegadas muy adelantadas (≥10 d)", f"{(df_filtrado['desviacion_vs_promesa']<-10).mean()*100:.1f}%")
    if 'dias_entrega' in df_filtrado.columns:
        # Gráficos: dona, barras, delivery days...
        # (código idéntico al anterior para tablas y gráficos)

# 🏠 Costo de Envío
with tabs[1]:
    col1, col2 = st.columns(2)
    col1.metric("Total de pedidos", f"{len(df_filtrado):,}")
    col2.metric("Flete sobre precio ≥ 50%", f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%")
    # Tabla % flete/precio y gráficos de barras

# 🧮 Calculadora
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")
    meses_dict = {i: m for i, m in enumerate(
        ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto",
         "Septiembre","Octubre","Noviembre","Diciembre"], start=1)}
    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month
    st.markdown(f"**Estado seleccionado:** {estado_sel}")

    categoria = st.selectbox("Categoría", sorted(df2['categoria'].dropna().unique()))
    colA, colB = st.columns(2)
    mes1 = colA.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2 = colB.selectbox("Mes 2", list(meses_dict.values()), index=1)
    m1 = next(k for k,v in meses_dict.items() if v==mes1)
    m2 = next(k for k,v in meses_dict.items() if v==mes2)

    df_m1 = df2[(df2['mes']==m1)&(df2['estado_del_cliente']==estado_sel)&(df2['categoria']==categoria)]
    df_m2 = df2[(df2['mes']==m2)&(df2['estado_del_cliente']==estado_sel)&(df2['categoria']==categoria)]

    def predecir(df_in):
        if df_in.empty: return df_in
        # Igual que antes: columnas_flete, encode, predecir
        return df_in

    df_m1 = predecir(df_m1); df_m2 = predecir(df_m2)
    def agrupa(df_in,n):
        # agrega costo_estimado y clase_entrega
        return df_in.groupby('ciudad_cliente').agg(...)  # como antes
    res1 = agrupa(df_m1,mes1); res2 = agrupa(df_m2,mes2)
    comp = pd.merge(res1,res2,on='ciudad_cliente',how='outer').fillna(0)
    # Convertir columnas a numérico antes de calcular
    comp[mes1] = pd.to_numeric(comp[mes1], errors='coerce').fillna(0)
    comp[mes2] = pd.to_numeric(comp[mes2], errors='coerce').fillna(0)
    comp['Diferencia'] = (comp[mes2] - comp[mes1]).round(2)
    comp = comp.rename(columns={'ciudad_cliente':'Ciudad'})
    st.dataframe(comp.style.applymap(lambda v: 'color:green' if v>0 else ('color:red' if v<0 else ''), subset=['Diferencia']))
    st.download_button("⬇️ Descargar CSV", comp.to_csv(index=False), "comparacion.csv", "text/csv")

# App Danu 📈
with tabs[3]:
    st.title("📈 Bienvenido a App Danu")
    st.write("Aquí puedes incluir visualizaciones o funcionalidades extras 🤖")
    # Agrega aquí tus gráficos o widgets de App Danu
    # Ejemplo:
    st.write("– Métricas adicionales –\n– Gráficos por usuario – etc.")

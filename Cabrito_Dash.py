# Cabrito Dash 10/06/2025 v2 – Todo integrado

import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu

class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

st.set_page_config(page_title="Cabrito Analytics", layout="wide")
st.markdown("""<style>/* Estilos omitidos por brevedad */</style>""", unsafe_allow_html=True)

def clasificar_zonas(df, estado_sel):
    if estado_sel == "Nacional":
        principales = ['Ciudad de México','Nuevo León','Jalisco']
        return df['estado_del_cliente'].apply(lambda x: x if x in principales else 'Provincia')
    else:
        top = df[df['estado_del_cliente']==estado_sel]['ciudad_cliente'].value_counts().head(3).index.tolist()
        return df['ciudad_cliente'].apply(lambda x: x if x in top else 'Otras')

tabs = st.tabs(["📊 Resumen Nacional","🏠 Costo de Envío","🧮 Calculadora","App Danu 📈"])

with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")
    if archivo_zip:
        with zipfile.ZipFile(archivo_zip) as z:
            required = ['DF.csv','DF2.csv','modelo_costoflete.sav',
                        'modelo_dias_pipeline.joblib','label_encoder_dias.joblib']
            missing = [r for r in required if r not in z.namelist()]
            if missing:
                st.error(f"Faltan archivos: {missing}")
                st.stop()
            df = pd.read_csv(z.open('DF.csv'))
            df2 = pd.read_csv(z.open('DF2.csv'))
            modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
            modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
            label_encoder = joblib.load(z.open('label_encoder_dias.joblib'))
        estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
        estado_sel = option_menu("Filtro de Estado", estados, icons=["globe"]+["geo"]*(len(estados)-1))
        df_filtrado = df if estado_sel=="Nacional" else df[df['estado_del_cliente']==estado_sel]
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

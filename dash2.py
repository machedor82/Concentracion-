# dash2.py

import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------ Clases personalizadas ------------------
class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1
    def fit(self, X, y=None): return self
    def transform(self, X): return X
# ---------------------------------------------------------------------------------------

# Configuración de la app
st.set_page_config(page_title="Cabrito Analytics Profesional", layout="wide")
st.title("📦 Cabrito Analytics App")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# Sidebar: upload + filtros
with st.sidebar:
    st.header("Sube tu archivo ZIP")
    archivo = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if archivo:
    with zipfile.ZipFile(archivo) as z:
        necesarios = ['DF.csv','DF2.csv','modelo_costoflete.sav','modelo_dias_pipeline.joblib','label_encoder_dias.joblib']
        faltantes = [f for f in necesarios if f not in z.namelist()]
        if faltantes:
            st.error(f"❌ Faltan archivos: {faltantes}")
            st.stop()

        df = pd.read_csv(z.open('DF.csv'))
        df2 = pd.read_csv(z.open('DF2.csv'))
        modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
        modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
        encoder = joblib.load(z.open('label_encoder_dias.joblib'))

    # Dashboard
    with tabs[0]:
        st.header("🏠 Dashboard Logístico")
        st.markdown(
    """
**Desfase estimado vs real de entrega**  
Observa cómo varían los tiempos mes a mes y por categoría.
    """
)

        # Filtros interactivos
        st.sidebar.subheader("Filtros")
        estados = sorted(df['estado_del_cliente'].dropna().unique())
        sel_est = st.sidebar.multiselect("Estados", estados, default=estados)
        cats = sorted(df['Categoría'].dropna().unique())
        sel_cat = st.sidebar.multiselect("Categorías", cats, default=cats)

        data = df[df['estado_del_cliente'].isin(sel_est) & df['Categoría'].isin(sel_cat)].copy()
        data['prometido_dias'] = data['dias_entrega'] - data['desviacion_vs_promesa']

        # Métricas resumidas
        est_mean = data['prometido_dias'].mean()
        real_mean = data['dias_entrega'].mean()
        diff_mean = est_mean - real_mean
        c1,c2,c3 = st.columns(3)
        c1.metric("Estimado promedio (d)", f"{est_mean:.1f}")
        c2.metric("Real promedio (d)", f"{real_mean:.1f}")
        c3.metric("Desfase promedio (d)", f"{diff_mean:.1f}")

        # Bar chart por categoría
        cat_agg = data.groupby('Categoría').agg(
            Estimado=('prometido_dias','mean'),
            Real=('dias_entrega','mean')
        ).reset_index()
        fig1 = px.bar(
            cat_agg, x='Categoría', y=['Estimado','Real'], barmode='group',
            color_discrete_map={'Estimado':'#003366','Real':'#6699cc'},
            labels={'value':'Días','variable':'Tipo'},
            title='Tiempos estimado vs real por Categoría'
        )
        real_med = cat_agg['Real'].median()
        fig1.add_hline(y=real_med, line_dash='dash', line_color='#6699cc', annotation_text='Mediana Real', annotation_position='top right')
        st.plotly_chart(fig1, use_container_width=True)

        # Line chart evolución mensual
        ts = data.groupby(['año','mes']).agg(
            Estimado=('prometido_dias','mean'), Real=('dias_entrega','mean')
        ).reset_index().sort_values(['año','mes'])
        ts['Fecha'] = pd.to_datetime(dict(year=ts['año'],month=ts['mes'],day=1))
        fig2 = px.line(
            ts, x='Fecha', y=['Estimado','Real'],
            color_discrete_map={'Estimado':'#003366','Real':'#6699cc'},
            labels={'value':'Días','variable':'Tipo'},
            title='Evolución: estimado vs real'
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Top 10 categorías con mayor desfase
        cat_agg['Desfase'] = cat_agg['Estimado'] - cat_agg['Real']
        top10 = cat_agg.nlargest(10,'Desfase')
        fig3 = px.bar(
            top10, x='Categoría', y='Desfase',
            labels={'Desfase':'Días de desfase'},
            title='Top 10 categorías con mayor desfase',
            color_discrete_sequence=['#003366']
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Calculadora
    with tabs[1]:
        st.header("🧮 Calculadora de Predicción")
        # ... misma lógica que antes ...
        st.info("Calculadora aquí...")

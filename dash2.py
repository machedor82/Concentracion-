# dashboard_main.py
import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import os

# ========================== CONFIGURACIÓN INICIAL ==========================
st.set_page_config(page_title="Cabrito Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #002244;
            color: white;
        }
        .stSlider > div[data-testid="stTickBar"] {
            background-color: #ffffff11;
        }
        .stSlider .css-14pt78w {
            color: white !important;
        }
        .stMultiSelect, .stSlider {
            color: black !important;
        }
        [data-testid="stSidebar"] label {
            color: white;
        }
        body {
            background-color: #f9f9f9;
            color: #1f1f1f;
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

st.title("📊 Panel BI")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ========================== ARCHIVOS LOCALES ==========================
local_path = "/mnt/data/dashboard_final"
zip_file_path = os.path.join(local_path, "DF.csv")
csv_file_path = os.path.join(local_path, "DF2.csv")

# ========================== PESTAÑA 1 - DASHBOARD ==========================
with tabs[0]:
    df = pd.read_csv(zip_file_path)
    st.success("✅ Datos cargados exitosamente")

    with st.sidebar:
        st.header("📂 Filtros")
        categorias = df['Categoría'].dropna().unique()
        estados = df['estado_del_cliente'].dropna().unique()
        años = sorted(df['año'].dropna().unique())
        meses = sorted(df['mes'].dropna().unique())

        categoria_sel = st.multiselect("Categoría de producto", categorias, default=list(categorias))
        estado_sel = st.multiselect("Estado del cliente", estados, default=list(estados))
        año_sel = st.multiselect("Año", años, default=años)
        mes_sel = st.multiselect("Mes", meses, default=meses)

    df_filtrado = df[
        (df['Categoría'].isin(categoria_sel)) &
        (df['estado_del_cliente'].isin(estado_sel)) &
        (df['año'].isin(año_sel)) &
        (df['mes'].isin(mes_sel))
    ]

    st.markdown("## 🧭 Visión General de la Operación")
    st.markdown("### 🔢 Indicadores")
    col1, col2, col3 = st.columns(3)

    col1.metric("📦 Total de pedidos", f"{len(df_filtrado):,}")
    pct_flete_alto = (df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100
    col2.metric("🚚 Flete > 50%", f"{pct_flete_alto:.1f}%")
    pct_anticipadas = (df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100
    col3.metric("⏱️ Entregas ≥7 días antes", f"{pct_anticipadas:.1f}%")

    st.markdown("### 📊 Análisis visual")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌳 Treemap por categoría")
        fig_tree = px.treemap(df_filtrado, path=['Categoría'], values='precio', color='Categoría',
                              color_discrete_sequence=px.colors.sequential.Blues)
        st.plotly_chart(fig_tree, use_container_width=True)

    with col2:
        st.subheader("🗺️ Mapa de entregas de clientes")
        df_mapa = df_filtrado.dropna(subset=['lat_cliente', 'lon_cliente'])
        if not df_mapa.empty:
            st.map(df_mapa.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
        else:
            st.warning("⚠️ No hay ubicaciones disponibles.")

    with col3:
        st.subheader("📈 Entrega vs colchón")
        df_promedios = df_filtrado.groupby('estado_del_cliente')[['dias_entrega', 'colchon_dias']].mean().reset_index()
        fig_bar = px.bar(df_promedios, x='estado_del_cliente', y=['dias_entrega', 'colchon_dias'],
                         barmode='group', color_discrete_sequence=px.colors.sequential.Blues)
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

# ========================== PESTAÑA 2 - CALCULADORA ==========================
with tabs[1]:
    df2 = pd.read_csv(csv_file_path)
    st.success("✅ Archivo CSV cargado exitosamente")

    estados_calc = sorted(df2['estado_del_cliente'].dropna().unique())
    categorias_calc = sorted(df2['Categoría'].dropna().unique())
    estado = st.selectbox("Estado", estados_calc)
    categoria = st.selectbox("Categoría", categorias_calc)

    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    meses_dict = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    col1, col2 = st.columns(2)
    mes1_nombre = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2_nombre = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1 = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
    mes2 = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

    filtro = (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)
    df_mes1 = df2[(df2['mes'] == mes1) & filtro].copy()
    df_mes2 = df2[(df2['mes'] == mes2) & filtro].copy()

    def resumen(df_input, nombre_mes):
        return df_input.groupby('ciudad_cliente')['precio'].mean().reset_index().rename(columns={'precio': nombre_mes})

    r1 = resumen(df_mes1, mes1_nombre)
    r2 = resumen(df_mes2, mes2_nombre)
    comparacion = pd.merge(r1, r2, on='ciudad_cliente', how='outer')
    comparacion['Diferencia'] = comparacion[mes2_nombre] - comparacion[mes1_nombre]

    st.dataframe(comparacion)
    st.download_button("⬇️ Descargar comparación", comparacion.to_csv(index=False), "comparacion.csv", "text/csv")

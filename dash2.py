# dashboard_app.py

import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import os
import joblib
import numpy as np

# ========== CONFIGURACIÓN DE PÁGINA Y ESTILOS ==========
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #002244;
            color: white;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #0099ff;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.image("danu_logo.png", width=200)
st.title("📊 Cabrito Analytics")

# ========== SUBIDA DE ARCHIVO ZIP ==========
zip_file = st.sidebar.file_uploader("📂 Sube archivo ZIP con DF.csv y DF2.csv", type=["zip"])

if zip_file:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        file_names = zip_ref.namelist()
        if 'DF.csv' in file_names and 'DF2.csv' in file_names:
            df = pd.read_csv(zip_ref.open('DF.csv'))
            df2 = pd.read_csv(zip_ref.open('DF2.csv'))
        else:
            st.error("❌ El ZIP debe contener DF.csv y DF2.csv")
            st.stop()
else:
    st.info("🔄 Esperando archivo .zip con DF.csv y DF2.csv...")
    st.stop()

# ========== TABS PRINCIPALES ==========
tab1, tab2 = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ========== DASHBOARD ==========
with tab1:
    st.header("📦 Dashboard General")

    col1, col2, col3 = st.columns(3)
    col1.metric("Pedidos", f"{len(df):,}")
    pct_flete = (df['costo_de_flete'] / df['precio'] > 0.5).mean() * 100
    col2.metric("Flete alto", f"{pct_flete:.1f}%")
    anticipadas = (df['desviacion_vs_promesa'] < -7).mean() * 100
    col3.metric("Entregas muy anticipadas", f"{anticipadas:.1f}%")

    st.subheader("📍 Mapa de Clientes")
    mapa_df = df.dropna(subset=['lat_cliente', 'lon_cliente'])
    if not mapa_df.empty:
        st.map(mapa_df.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
    else:
        st.warning("No hay coordenadas para mostrar.")

    st.subheader("📊 Categorías")
    fig_tree = px.treemap(df, path=['Categoría'], values='precio', color='Categoría')
    st.plotly_chart(fig_tree, use_container_width=True)

# ========== CALCULADORA ==========
with tab2:
    st.header("📈 Calculadora ML por Estado y Categoría")

    estado = st.selectbox("Estado", sorted(df2['estado_del_cliente'].dropna().unique()))
    categoria = st.selectbox("Categoría", sorted(df2['Categoría'].dropna().unique()))

    meses_dict = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    col1, col2 = st.columns(2)
    mes1 = col1.selectbox("Mes 1", list(meses_dict.values()))
    mes2 = col2.selectbox("Mes 2", list(meses_dict.values()))

    num_mes1 = [k for k, v in meses_dict.items() if v == mes1][0]
    num_mes2 = [k for k, v in meses_dict.items() if v == mes2][0]

    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    filtro = (
        (df2['estado_del_cliente'] == estado) &
        (df2['Categoría'] == categoria)
    )

    df_mes1 = df2[(df2['mes'] == num_mes1) & filtro].copy()
    df_mes2 = df2[(df2['mes'] == num_mes2) & filtro].copy()

    # === MODELO ML ===
    try:
        modelo_dias = joblib.load("modelo_dias.pkl")
        label_encoder = joblib.load("label_encoder.pkl")

        def predecir(df_input):
            features = ['colchon_dias', 'precio', 'costo_de_flete']
            X = df_input[features]
            df_input['clase_entrega'] = label_encoder.inverse_transform(modelo_dias.predict(X))
            return df_input

        df_mes1 = predecir(df_mes1)
        df_mes2 = predecir(df_mes2)

        def resumen(df_input, nombre_mes):
            return df_input.groupby('ciudad_cliente').agg({
                'costo_de_flete': 'mean',
                'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
            }).reset_index().rename(columns={
                'costo_de_flete': f"{nombre_mes} ($)",
                'clase_entrega': f"Entrega {nombre_mes}"
            })

        r1 = resumen(df_mes1, mes1)
        r2 = resumen(df_mes2, mes2)
        comparacion = pd.merge(r1, r2, on='ciudad_cliente', how='outer')

        st.dataframe(comparacion)
        st.download_button("⬇️ Descargar comparación", comparacion.to_csv(index=False), "comparacion.csv", "text/csv")

    except Exception as e:
        st.warning("⚠️ No se pudo aplicar el modelo ML. Verifica que los archivos .pkl estén cargados correctamente.")

# ========== NOTA FINAL ==========
from datetime import datetime
st.caption(f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

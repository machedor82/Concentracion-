# dashboard_final.py
import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import joblib
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Cabrito Analytics", layout="wide")

st.title("📊 Cabrito Analytics BI")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

@st.cache_data
def load_csv_from_zip(zip_file, filename):
    with zipfile.ZipFile(zip_file) as z:
        with z.open(filename) as f:
            return pd.read_csv(f)

# ====== CARGA DE ZIP ======
with st.sidebar:
    st.header("📦 Subir ZIP")
    zip_file = st.file_uploader("Carga un archivo ZIP con DF.csv y DF2.csv", type="zip")

if not zip_file:
    st.warning("Por favor, sube un archivo ZIP con DF.csv y DF2.csv.")
    st.stop()

try:
    df = load_csv_from_zip(zip_file, "DF.csv")
    df2 = load_csv_from_zip(zip_file, "DF2.csv")
except Exception as e:
    st.error(f"Error al cargar archivos del ZIP: {e}")
    st.stop()

# ====== DASHBOARD ======
with tabs[0]:
    st.markdown("### 🧭 Visión General")
    st.caption(f"Versión pro optimizada con fondo claro, paleta azul unificada, filtros personalizados, carga de archivo y exportación. Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    with st.sidebar:
        st.subheader("🎛️ Filtros dashboard")
        categoria_sel = st.multiselect("Categoría", df['Categoría'].dropna().unique())
        estado_sel = st.multiselect("Estado cliente", df['estado_del_cliente'].dropna().unique())
        año_sel = st.multiselect("Año", sorted(df['año'].dropna().unique()))
        mes_sel = st.multiselect("Mes", sorted(df['mes'].dropna().unique()))

    df_filtered = df[
        (df['Categoría'].isin(categoria_sel)) &
        (df['estado_del_cliente'].isin(estado_sel)) &
        (df['año'].isin(año_sel)) &
        (df['mes'].isin(mes_sel))
    ]

    st.metric("📦 Total pedidos", len(df_filtered))
    st.metric("🚚 % Flete > 50%", f"{((df_filtered['costo_de_flete'] / df_filtered['precio']) > 0.5).mean()*100:.1f}%")
    st.metric("⏱️ Entregas anticipadas ≥7 días", f"{(df_filtered['desviacion_vs_promesa'] < -7).mean()*100:.1f}%")

    st.subheader("📈 Treemap por categoría")
    fig = px.treemap(df_filtered, path=['Categoría'], values='precio', color='Categoría')
    st.plotly_chart(fig, use_container_width=True)

# ====== CALCULADORA ======
with tabs[1]:
    st.markdown("### 🔍 Calculadora de Flete y Clase de Entrega")

    modelo_flete = joblib.load("modelo_costoflete.sav")
    modelo_dias = joblib.load("modelo_dias_pipeline.joblib")
    label_encoder = joblib.load("label_encoder_dias.joblib")

    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    estados = sorted(df2['estado_del_cliente'].dropna().unique())
    categorias = sorted(df2['Categoría'].dropna().unique())
    estado = st.selectbox("Estado", estados)
    categoria = st.selectbox("Categoría", categorias)

    meses_dict = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    mes1 = st.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2 = st.selectbox("Mes 2", list(meses_dict.values()), index=1)

    m1 = [k for k, v in meses_dict.items() if v == mes1][0]
    m2 = [k for k, v in meses_dict.items() if v == mes2][0]

    filtro = (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)
    df_mes1 = df2[(df2['mes'] == m1) & filtro].copy()
    df_mes2 = df2[(df2['mes'] == m2) & filtro].copy()

    def predecir(df_input):
        if df_input.empty:
            return df_input

        columnas_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                          'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                          'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago']
        df_flete = df_input[columnas_flete].copy()
        df_encoded = pd.get_dummies(df_flete)
        columnas_entrenadas = modelo_flete.get_booster().feature_names
        df_encoded = df_encoded.reindex(columns=columnas_entrenadas, fill_value=0)
        df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
        df_input['costo_de_flete'] = df_input['costo_estimado']

        cols_dias = modelo_dias.feature_names_in_
        faltantes = [col for col in cols_dias if col not in df_input.columns]
        if faltantes:
            st.error(f"⚠️ Faltan columnas para predecir clase de entrega: {faltantes}")
            return df_input

        X_dias = df_input[cols_dias]
        clase_pred = modelo_dias.predict(X_dias)
        df_input['clase_entrega'] = label_encoder.inverse_transform(clase_pred)

        return df_input

    def resumen(df_pred, nombre_mes):
        return df_pred.groupby('ciudad_cliente').agg({
            'costo_estimado': 'mean',
            'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
        }).reset_index().rename(columns={
            'costo_estimado': nombre_mes,
            'clase_entrega': f"Entrega {nombre_mes}"
        })

    df_mes1 = predecir(df_mes1)
    df_mes2 = predecir(df_mes2)
    res1 = resumen(df_mes1, mes1)
    res2 = resumen(df_mes2, mes2)

    comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
    comparacion['Diferencia'] = (comparacion[mes2] - comparacion[mes1]).round(2)

    st.subheader("📊 Comparación")
    st.dataframe(comparacion)

    st.download_button("⬇️ Descargar CSV", data=comparacion.to_csv(index=False), file_name="comparacion.csv")

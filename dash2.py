# cabrito_dashboard.py
import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import os
import datetime

# ========================== CONFIGURACIÓN INICIAL ==========================
st.set_page_config(page_title="Cabrito Analytics", layout="wide", initial_sidebar_state="expanded")

# ========================== ESTILO PERSONALIZADO ==========================
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #002244;
            color: white;
        }
        [data-testid="stSidebar"] label, .stMultiSelect, .stSelectbox {
            color: white !important;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
        }
        .css-1aumxhk, .stFileUploader label {
            color: white !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            padding: 10px;
            border-bottom: 3px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #00b4d8;
            color: #00b4d8;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# ========================== TÍTULO Y LOGO ==========================
st.image("danu_logo.png", width=120)
st.title("Cabrito Analytics App")
tabs = st.tabs(["📊 Dashboard", "🧮 Calculadora"])

# ========================== SUBIR ARCHIVO ZIP ==========================
with st.sidebar:
    st.markdown("### 📂 Sube archivo ZIP con DF.csv y DF2.csv")
    uploaded_zip = st.file_uploader("", type="zip")

# ========================== FUNCIONES DE CARGA ==========================
@st.cache_data
def cargar_datos_zip(uploaded_zip):
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall("temp")
    df = pd.read_csv("temp/DF.csv")
    df2 = pd.read_csv("temp/DF2.csv")
    modelo_flete = joblib.load("temp/modelo_costoflete.sav")
    modelo_dias = joblib.load("temp/modelo_dias_pipeline.joblib")
    label_encoder = joblib.load("temp/label_encoder_dias.joblib")
    return df, df2, modelo_flete, modelo_dias, label_encoder

if uploaded_zip:
    try:
        df, df2, modelo_flete, modelo_dias, label_encoder = cargar_datos_zip(uploaded_zip)
        st.success("✅ Datos cargados correctamente")
    except Exception as e:
        st.error(f"❌ Error al cargar el ZIP: {e}")
        st.stop()
else:
    st.warning("Por favor, sube un archivo ZIP que contenga DF.csv, DF2.csv y los modelos.")
    st.stop()

# ========================== DASHBOARD ==========================
with tabs[0]:
    st.header("📊 Dashboard Logístico")

    with st.sidebar:
        st.subheader("🎚️ Filtros")
        categoria_sel = st.multiselect("Categoría", df['Categoría'].unique(), default=list(df['Categoría'].unique()))
        estado_sel = st.multiselect("Estado", df['estado_del_cliente'].unique(), default=list(df['estado_del_cliente'].unique()))
        anio_sel = st.multiselect("Año", sorted(df['año'].unique()), default=list(df['año'].unique()))
        mes_sel = st.multiselect("Mes", sorted(df['mes'].unique()), default=list(df['mes'].unique()))

    df_filtrado = df[
        (df['Categoría'].isin(categoria_sel)) &
        (df['estado_del_cliente'].isin(estado_sel)) &
        (df['año'].isin(anio_sel)) &
        (df['mes'].isin(mes_sel))
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Pedidos", f"{len(df_filtrado):,}")
    pct_flete = (df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100
    col2.metric("🚛 Flete > 50%", f"{pct_flete:.1f}%")
    anticipadas = (df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100
    col3.metric("⏱️ ≥7 días antes", f"{anticipadas:.1f}%")

    st.subheader("🌳 Treemap")
    fig_tree = px.treemap(df_filtrado, path=['Categoría'], values='precio', color='Categoría')
    st.plotly_chart(fig_tree, use_container_width=True)

    st.subheader("🗺️ Mapa")
    if 'lat_cliente' in df_filtrado.columns and 'lon_cliente' in df_filtrado.columns:
        df_mapa = df_filtrado.dropna(subset=['lat_cliente', 'lon_cliente'])
        st.map(df_mapa.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])

    st.subheader("📈 Entrega vs Colchón")
    if all(col in df_filtrado.columns for col in ['estado_del_cliente', 'dias_entrega', 'colchon_dias']):
        df_avg = df_filtrado.groupby('estado_del_cliente')[['dias_entrega', 'colchon_dias']].mean().reset_index()
        df_avg = df_avg.melt(id_vars='estado_del_cliente', var_name='variable', value_name='value')
        fig_bar = px.bar(df_avg, x='estado_del_cliente', y='value', color='variable', barmode='group')
        st.plotly_chart(fig_bar, use_container_width=True)

# ========================== CALCULADORA ==========================
with tabs[1]:
    st.header("🧮 Calculadora ML por Estado y Categoría")

    estado = st.selectbox("Estado", sorted(df2['estado_del_cliente'].unique()))
    categoria = st.selectbox("Categoría", sorted(df2['Categoría'].unique()))

    meses_dict = {i: m for i, m in enumerate(
        ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"], 1)}
    mes1 = st.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2 = st.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1_n, mes2_n = [k for k, v in meses_dict.items() if v == mes1][0], [k for k, v in meses_dict.items() if v == mes2][0]

    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    def predecir(df_input):
        columnas_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                          'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                          'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago']
        X = df_input[columnas_flete].copy()
        X = pd.get_dummies(X)
        columnas_modelo = modelo_flete.get_booster().feature_names
        X = X.reindex(columns=columnas_modelo, fill_value=0)
        df_input['costo_estimado'] = modelo_flete.predict(X)

        columnas_dias = ['Categoría', 'categoría_peso', '#_deproductos', 'total_peso_g', 'precio', 'costo_de_flete',
                         'distancia_km', 'velocidad_kmh', 'duracion_estimada_min', 'region', 'dc_asignado',
                         'es_feriado', 'es_fin_de_semana', 'dias_promedio_ciudad', 'hora_compra',
                         'nombre_dia', 'mes', 'año', 'temp_origen', 'precip_origen', 'cloudcover_origen',
                         'conditions_origen', 'icon_origen', 'traffic', 'area']
        if not all(col in df_input.columns for col in columnas_dias):
            st.warning("⚠️ Faltan columnas para predicción de clase_entrega")
            return df_input

        X2 = df_input[columnas_dias].copy()
        pred = modelo_dias.predict(X2)
        df_input['clase_entrega'] = label_encoder.inverse_transform(pred)
        return df_input

    df_m1 = df2[(df2['mes'] == mes1_n) & (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)].copy()
    df_m2 = df2[(df2['mes'] == mes2_n) & (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)].copy()

    df_m1 = predecir(df_m1)
    df_m2 = predecir(df_m2)

    def resumen(df_pred, nombre_mes):
        if 'costo_estimado' in df_pred.columns and 'clase_entrega' in df_pred.columns:
            return df_pred.groupby('ciudad_cliente').agg({
                'costo_estimado': 'mean',
                'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
            }).reset_index().rename(columns={
                'costo_estimado': nombre_mes,
                'clase_entrega': f"Entrega {nombre_mes}"
            })
        return pd.DataFrame()

    res1 = resumen(df_m1, mes1)
    res2 = resumen(df_m2, mes2)
    comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
    if not comparacion.empty:
        comparacion['Diferencia'] = (comparacion[mes2] - comparacion[mes1]).round(2)
        st.dataframe(comparacion)
        csv = comparacion.to_csv(index=False).encode('utf-8')
        st.download_button(f"⬇️ Descargar {mes1} vs {mes2}", csv, file_name=f"comparacion_{estado}_{categoria}.csv", mime="text/csv")
    else:
        st.warning("⚠️ No se pudo aplicar el modelo ML. Verifica que los archivos .pkl estén cargados correctamente.")

# ========================== FOOTER ==========================
st.caption(f"Última actualización: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

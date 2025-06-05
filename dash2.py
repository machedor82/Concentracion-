import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import joblib
import numpy as np
from datetime import datetime

# ========================== CONFIGURACIÓN INICIAL ==========================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #002244;
            color: white;
        }
        [data-testid="stSidebar"] label {
            color: white;
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

st.title("📊 Cabrito Analytics - Dashboard & Calculadora")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ========================== CARGA DE ARCHIVO ZIP ==========================
@st.cache_data
def load_dfs_from_zip(uploaded_zip):
    with zipfile.ZipFile(uploaded_zip) as z:
        with z.open("DF.csv") as df_file:
            df1 = pd.read_csv(df_file)
        with z.open("DF2.csv") as df2_file:
            df2 = pd.read_csv(df2_file)
    return df1, df2

uploaded_file = st.sidebar.file_uploader("📦 Sube un ZIP con 'DF.csv' y 'DF2.csv'", type="zip")
if not uploaded_file:
    st.warning("Por favor, sube un archivo ZIP con los CSV requeridos.")
    st.stop()

try:
    df, df2 = load_dfs_from_zip(uploaded_file)
except Exception as e:
    st.error(f"❌ Error al leer el archivo ZIP: {e}")
    st.stop()

# ========================== DASHBOARD ==========================
with tabs[0]:
    st.subheader("📈 Dashboard Operativo")
    st.caption(f"Visión general del desempeño logístico con filtros interactivos y visualizaciones clave. Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    with st.sidebar:
        st.subheader("🎛️ Filtros del Dashboard")
        categoria_sel = st.multiselect("Categoría", df['Categoría'].dropna().unique(), default=list(df['Categoría'].dropna().unique()))
        estado_sel = st.multiselect("Estado del cliente", df['estado_del_cliente'].dropna().unique(), default=list(df['estado_del_cliente'].dropna().unique()))
        año_sel = st.multiselect("Año", sorted(df['año'].dropna().unique()), default=sorted(df['año'].dropna().unique()))
        mes_sel = st.multiselect("Mes", sorted(df['mes'].dropna().unique()), default=sorted(df['mes'].dropna().unique()))

    df_filtrado = df[
        (df['Categoría'].isin(categoria_sel)) &
        (df['estado_del_cliente'].isin(estado_sel)) &
        (df['año'].isin(año_sel)) &
        (df['mes'].isin(mes_sel))
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Pedidos", f"{len(df_filtrado):,}")
    col2.metric("🚚 Flete > 50%", f"{(df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100:.1f}%")
    col3.metric("⏱️ Entregas ≥7 días antes", f"{(df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100:.1f}%")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🌳 Treemap por Categoría")
        fig1 = px.treemap(df_filtrado, path=['Categoría'], values='precio', color='Categoría')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("🗺️ Mapa de Entregas")
        df_map = df_filtrado.dropna(subset=['lat_cliente', 'lon_cliente'])
        if not df_map.empty:
            st.map(df_map.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
        else:
            st.warning("⚠️ No hay coordenadas disponibles.")

    with col3:
        st.subheader("📊 Promedio Entrega vs Colchón")
        df_prom = df_filtrado.groupby('estado_del_cliente')[['dias_entrega', 'colchon_dias']].mean().reset_index()
        fig2 = px.bar(df_prom, x='estado_del_cliente', y=['dias_entrega', 'colchon_dias'], barmode='group')
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

# ========================== CALCULADORA ==========================
with tabs[1]:
    st.subheader("🧮 Calculadora Predictiva por Ciudad")

    estados = sorted(df2['estado_del_cliente'].dropna().unique())
    categorias = sorted(df2['Categoría'].dropna().unique())
    estado = st.selectbox("Estado", estados)
    categoria = st.selectbox("Categoría", categorias)

    meses_dict = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    mes1 = st.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2 = st.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1_n = [k for k, v in meses_dict.items() if v == mes1][0]
    mes2_n = [k for k, v in meses_dict.items() if v == mes2][0]

    filtro = (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)
    df_mes1 = df2[(df2['mes'] == mes1_n) & filtro].copy()
    df_mes2 = df2[(df2['mes'] == mes2_n) & filtro].copy()

    modelo_flete = joblib.load("modelo_costoflete.sav")
    modelo_dias = joblib.load("modelo_dias_pipeline.joblib")
    label_encoder = joblib.load("label_encoder_dias.joblib")

    def predecir(df_input):
        if df_input.empty:
            return df_input

        columnas_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                          'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                          'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago']

        df_encoded = pd.get_dummies(df_input[columnas_flete], drop_first=True)
        expected_cols = modelo_flete.get_booster().feature_names
        df_encoded = df_encoded.reindex(columns=expected_cols, fill_value=0)

        df_input['costo_estimado'] = modelo_flete.predict(df_encoded)

        cols_dias = ['Categoría', 'categoría_peso', '#_deproductos', 'total_peso_g', 'precio', 'costo_estimado',
                     'distancia_km', 'velocidad_kmh', 'duracion_estimada_min', 'region', 'dc_asignado',
                     'es_feriado', 'es_fin_de_semana', 'dias_promedio_ciudad', 'hora_compra',
                     'nombre_dia', 'mes', 'año', 'temp_origen', 'precip_origen', 'cloudcover_origen',
                     'conditions_origen', 'icon_origen', 'traffic', 'area']

        if all(col in df_input.columns for col in cols_dias):
            X_dias = df_input[cols_dias].copy()
            df_input['clase_entrega'] = label_encoder.inverse_transform(modelo_dias.predict(X_dias))
        else:
            df_input['clase_entrega'] = 'N/A'

        return df_input

    def resumen(df_pred, nombre_mes):
        if 'costo_estimado' not in df_pred.columns or 'clase_entrega' not in df_pred.columns:
            return pd.DataFrame(columns=['ciudad_cliente', nombre_mes, f"Entrega {nombre_mes}"])
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
    comparacion['Diferencia'] = comparacion[mes2] - comparacion[mes1]

    st.dataframe(comparacion)
    st.download_button("⬇️ Descargar comparación", comparacion.to_csv(index=False), file_name="comparacion.csv")

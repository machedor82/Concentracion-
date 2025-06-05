import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import joblib
import os
import tempfile
from sklearn.preprocessing import LabelEncoder

# ========================== CONFIGURACIÓN INICIAL ==========================
st.set_page_config(page_title="Cabrito Analytics", layout="wide", initial_sidebar_state="expanded")

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

st.image("danu_logo.png", width=200)
st.title("📊 Cabrito Analytics - Panel BI")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ========================== SUBIR ARCHIVO ZIP ==========================
zip_file = st.sidebar.file_uploader("📂 Sube archivo ZIP con DF.csv y DF2.csv", type="zip")

@st.cache_data(show_spinner="Leyendo archivos...")
def extraer_archivos(zip_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "archivo.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.getvalue())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        df = pd.read_csv(os.path.join(tmpdir, "DF.csv"))
        df2 = pd.read_csv(os.path.join(tmpdir, "DF2.csv"))
        return df, df2

if not zip_file:
    st.warning("🔁 Sube un archivo ZIP para comenzar")
    st.stop()

# ========================== CARGA DE DATOS ==========================
df, df2 = extraer_archivos(zip_file)

# ========================== DASHBOARD ==========================
with tabs[0]:
    with st.sidebar:
        st.header("📌 Filtros")
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

    st.subheader("🪡 Indicadores Generales")
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Total de pedidos", f"{len(df_filtrado):,}")
    col2.metric("🚚 Flete > 50%", f"{(df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100:.1f}%")
    col3.metric("⏱️ Entregas ≥7 días antes", f"{(df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100:.1f}%")

    st.subheader("📊 Visualizaciones")
    col1, col2, col3 = st.columns(3)
    with col1:
        fig_tree = px.treemap(df_filtrado, path=['Categoría'], values='precio', color='Categoría')
        st.plotly_chart(fig_tree, use_container_width=True)
    with col2:
        df_mapa = df_filtrado.dropna(subset=['lat_cliente', 'lon_cliente'])
        if not df_mapa.empty:
            st.map(df_mapa.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
        else:
            st.warning("⚠️ No hay ubicaciones disponibles.")
    with col3:
        df_promedios = df_filtrado.groupby('estado_del_cliente')[['dias_entrega', 'colchon_dias']].mean().reset_index()
        fig_bar = px.bar(df_promedios, x='estado_del_cliente', y=['dias_entrega', 'colchon_dias'], barmode='group')
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

# ========================== CALCULADORA ==========================
with tabs[1]:
    modelo_dias = joblib.load("modelo_dias.joblib")
    label_encoder = joblib.load("label_encoder.joblib")

    estados_calc = sorted(df2['estado_del_cliente'].dropna().unique())
    categorias_calc = sorted(df2['Categoría'].dropna().unique())
    estado = st.selectbox("Estado", estados_calc)
    categoria = st.selectbox("Categoría", categorias_calc)

    meses_dict = {i: mes for i, mes in enumerate(
        ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"], start=1)}

    col1, col2 = st.columns(2)
    mes1_nombre = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2_nombre = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1 = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
    mes2 = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

    filtro = (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)
    df_mes1 = df2[(df2['mes'] == mes1) & filtro].copy()
    df_mes2 = df2[(df2['mes'] == mes2) & filtro].copy()

    def predecir(df_input):
        cols = modelo_dias.feature_names_in_
        X_dias = df_input[cols]
        df_input['costo_estimado'] = modelo_dias.predict(X_dias)
        df_input['clase_entrega'] = label_encoder.inverse_transform(modelo_dias.predict(X_dias))
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

    res1 = resumen(df_mes1, mes1_nombre)
    res2 = resumen(df_mes2, mes2_nombre)
    comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
    comparacion['Diferencia costo'] = comparacion[mes2_nombre] - comparacion[mes1_nombre]

    st.dataframe(comparacion)
    st.download_button("⬇️ Descargar comparación", comparacion.to_csv(index=False), "comparacion.csv", "text/csv")

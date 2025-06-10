# Cabrito Dash 10/06/2025 v1

import streamlit as st
import pandas as pd
import zipfile
import io
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu # Asegúrate de importar esto arriba

# ------------------ Definiciones de clases/funciones personalizadas ------------------

class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

# ---------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import joblib

st.set_page_config(page_title="Cabrito Analytics", layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa !important;
            font-family: 'Segoe UI', sans-serif;
        }
        .main > div {
            color: #1e2022 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a73e8 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #202124 !important;
        }
        [data-testid="stMetricDelta"] {
            font-weight: bold;
            color: #34a853 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e0e0e0;
        }
        [data-testid="stSidebar"] * {
            color: #1a3c5a !important;
        }
        .stExpander > summary {
            font-weight: 600;
            color: #1a3c5a !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 15px;
            padding: 12px;
            border-bottom: 2px solid transparent;
            color: #5f6368;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #1a73e8;
            color: #1a73e8;
            font-weight: 600;
        }
        .stMultiSelect .css-12w0qpk {
            max-height: 0px !important;
            overflow: hidden !important;
        }
        .stMultiSelect {
            height: 38px !important;
        }
        .css-1wa3eu0 {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# ===================== INTERFAZ BÁSICA =====================

def clasificar_zonas(df, estado_sel):
    if estado_sel == "Nacional":
        principales = ['Ciudad de México', 'Nuevo León', 'Jalisco']
        return df['estado_del_cliente'].apply(lambda x: x if x in principales else 'Provincia')
    else:
        top_ciudades = (
            df[df['estado_del_cliente'] == estado_sel]['ciudad_cliente']
            .value_counts()
            .nlargest(3)
            .index
            .tolist()
        )
        return df['ciudad_cliente'].apply(lambda x: x if x in top_ciudades else 'Otras')

tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora","App Danu 📈"])

with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

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

        df_filtrado = df.copy() if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel]

with tabs[2]:
    import joblib
    from sklearn.base import BaseEstimator, TransformerMixin

    st.header("🧮 Calculadora de Predicción")

    meses_dict = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month
    estado = estado_sel
    st.markdown(f"**Estado seleccionado:** {estado}")

    categoria = st.selectbox("Categoría", sorted(df2['Categoría'].dropna().unique()))

    col1, col2 = st.columns(2)
    mes1_nombre = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2_nombre = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1 = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
    mes2 = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

    filtro = (df2['estado_del_cliente'] == estado) & (df2['categoria'] == categoria)
    df_mes1 = df2[(df2['mes'] == mes1) & filtro].copy()
    df_mes2 = df2[(df2['mes'] == mes2) & filtro].copy()

    def predecir(df_input):
        if df_input.empty:
            return df_input

        columnas_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                          'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                          'dias_promedio_ciudad', 'categoria', 'tipo_de_pago']

        df_flete = df_input[columnas_flete].copy()
        df_encoded = pd.get_dummies(df_flete)
        columnas_modelo = modelo_flete.get_booster().feature_names
        df_encoded = df_encoded.reindex(columns=columnas_modelo, fill_value=0)

        df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
        df_input['costo_de_flete'] = df_input['costo_estimado']

        columnas_dias = ['categoria', 'categoria_peso', '#_deproductos', 'total_peso_g', 'precio', 'costo_de_flete',
                         'distancia_km', 'velocidad_kmh', 'duracion_estimada_min', 'region', 'dc_asignado',
                         'es_feriado', 'es_fin_de_semana', 'hora_compra', 'dias_promedio_ciudad', 'nombre_dia',
                         'mes', 'año', 'traffic', 'area']

        if not all(c in df_input.columns for c in columnas_dias):
            return df_input

        X_dias = df_input[columnas_dias]
        pred = modelo_dias.predict(X_dias)
        df_input['clase_entrega'] = label_encoder.inverse_transform(pred)
        return df_input

    def agrupar_resultados(df, nombre_mes):
        if 'costo_estimado' in df.columns and 'clase_entrega' in df.columns:
            return df.groupby('ciudad_cliente').agg({
                'costo_estimado': lambda x: round(x.mean(), 2),
                'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'NA'
            }).rename(columns={
                'costo_estimado': nombre_mes,
                'clase_entrega': f"Entrega {nombre_mes}"
            }).reset_index()
        return pd.DataFrame(columns=['ciudad_cliente', nombre_mes, f"Entrega {nombre_mes}"])

    df_mes1 = predecir(df_mes1)
    df_mes2 = predecir(df_mes2)

    res1 = agrupar_resultados(df_mes1, mes1_nombre)
    res2 = agrupar_resultados(df_mes2, mes2_nombre)
    comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')

    # Conversión numérica segura
    comparacion[mes1_nombre] = pd.to_numeric(comparacion[mes1_nombre], errors='coerce')
    comparacion[mes2_nombre] = pd.to_numeric(comparacion[mes2_nombre], errors='coerce')

    comparacion['Diferencia'] = (comparacion[mes2_nombre] - comparacion[mes1_nombre]).round(2)
    comparacion = comparacion[[
        'ciudad_cliente', mes1_nombre, mes2_nombre, 'Diferencia',
        f"Entrega {mes1_nombre}", f"Entrega {mes2_nombre}"
    ]].rename(columns={'ciudad_cliente': 'Ciudad'})

    def resaltar(val):
        if isinstance(val, (int, float, np.number)):
            if val > 0:
                return 'color: green; font-weight: bold'
            elif val < 0:
                return 'color: red; font-weight: bold'
        return ''

    st.subheader(f"Comparación: {mes1_nombre} vs {mes2_nombre}")
    st.dataframe(
        comparacion.style
        .applymap(resaltar, subset=['Diferencia'])
        .format(precision=2)
    )

    st.download_button(
        "⬇️ Descargar CSV",
        comparacion.to_csv(index=False),
        file_name="comparacion.csv",
        mime="text/csv"
    )

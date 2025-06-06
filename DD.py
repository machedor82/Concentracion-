# dash2.py

import streamlit as st
import pandas as pd
import zipfile
import io
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------ Definiciones de clases/funciones personalizadas ------------------
# Copia aquí las clases o funciones custom que usaste al entrenar 'modelo_dias_pipeline.joblib'.
# Deben llamarse exactamente igual que en tu script original.

class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        # Inicializa tus parámetros
        self.parametro1 = parametro1

    def fit(self, X, y=None):
        # Ajuste si es necesario (o simplemente retorna self)
        return self

    def transform(self, X):
        # Lógica de transformación (ejemplo placeholder)
        return X

# Si tu pipeline usaba más clases o funciones custom, defínelas aquí de la misma forma:
# class OtraClasePersonalizada:
#     def __init__(...): ...
#     def fit(...): ...
#     def transform(...): ...

# ---------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import joblib

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

# ===================== ESTILOS PERSONALIZADOS =====================
st.markdown("""
    <style>
        /* Fondo azul marino en zona principal */
        .main {
            background-color: #002244 !important;
        }

        /* Texto en zona principal */
        .main > div {
            color: white;
        } 
        /* Aumentar tamaño del texto de los títulos de métricas */
[data-testid="stMetricLabel"] {
    font-size: 1.5rem;
    font-weight: 600;
}


        /* Sidebar blanco con texto azul marino */
        [data-testid="stSidebar"] {
            background-color: white !important;
        }

        [data-testid="stSidebar"] * {
            color: #002244 !important;
        }

        .stExpander > summary {
            color: #002244 !important;
        }

        /* Compactar multiselect */
        .css-1wa3eu0 {
            display: none !important;
        }
        .stMultiSelect .css-12w0qpk {
            max-height: 0px !important;
            overflow: hidden !important;
        }
        .stMultiSelect {
            height: 35px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ===================== ENCABEZADO Y CARGA DE ARCHIVO =====================
st.title("📦 Cabrito Analytics App")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

# ===================== PROCESAMIENTO DEL ZIP =====================
if archivo_zip:
    with zipfile.ZipFile(archivo_zip) as z:
        requeridos = [
            'DF.csv',
            'DF2.csv',
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

    # ========================= DASHBOARD =========================
    with tabs[0]:
      

        with st.sidebar:
            st.subheader("🎛️ Filtros")

            with st.expander("📦 Categoría"):
                categorias = sorted(df['Categoría'].dropna().unique())
                categoria_sel = st.multiselect(
                    "Selecciona una o varias categorías:",
                    options=categorias,
                    default=categorias,
                    key="filtro_categoria"
                )

            with st.expander("📍 Estado"):
                estados = sorted(df['estado_del_cliente'].dropna().unique())
                estado_sel = st.multiselect(
                    "Selecciona uno o varios estados:",
                    options=estados,
                    default=estados,
                    key="filtro_estado"
                )

            with st.expander("📅 Año"):
                años = sorted(df['año'].dropna().unique())
                año_sel = st.multiselect(
                    "Selecciona uno o varios años:",
                    options=años,
                    default=años,
                    key="filtro_anio"
                )

            with st.expander("🗓️ Mes"):
                meses = sorted(df['mes'].dropna().unique())
                mes_sel = st.multiselect(
                    "Selecciona uno o varios meses:",
                    options=meses,
                    default=meses,
                    key="filtro_mes"
                )

        # ===================== FILTRADO =====================
        df_filtrado = df[
            (df['Categoría'].isin(categoria_sel)) &
            (df['estado_del_cliente'].isin(estado_sel)) &
            (df['año'].isin(año_sel)) &
            (df['mes'].isin(mes_sel))
        ]

        # ===================== MÉTRICAS =====================
      
        col1, col2, col3 = st.columns(3)
        col1.metric("Pedidos", f"{len(df_filtrado):,}")
        col2.metric(
            "Flete > 50%",
            f"{(df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100:.1f}%"
        )
        col3.metric(
            "≥7 días antes",
            f"{(df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100:.1f}%"
        )

        # ===================== VISUALIZACIONES =====================
        col2, col3 = st.columns(2)


        
        with col2:
            st.subheader("🗺️ Mapa")
            mapa = df_filtrado.dropna(subset=['lat_cliente', 'lon_cliente'])
            if not mapa.empty:
                st.map(mapa.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
            else:
                st.warning("Sin coordenadas válidas.")

        with col3:
            st.subheader("📈 Entrega vs Colchón")
            if {'dias_entrega', 'colchon_dias'}.issubset(df_filtrado.columns):
                medios = df_filtrado.groupby('estado_del_cliente')[['dias_entrega', 'colchon_dias']].mean().reset_index()
                fig2 = px.bar(
                    medios,
                    x='estado_del_cliente',
                    y=['dias_entrega', 'colchon_dias'],
                    barmode='group'
                )
                st.plotly_chart(fig2, use_container_width=True)



    # ========================= CALCULADORA =========================
    with tabs[1]:
        st.header("🧮 Calculadora de Predicción")

        # Convertir timestamp y extraer año/mes
        df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
        df2['año'] = df2['orden_compra_timestamp'].dt.year
        df2['mes'] = df2['orden_compra_timestamp'].dt.month

        estados = df2['estado_del_cliente'].dropna().unique()
        categorias = df2['Categoría'].dropna().unique()

        col1, col2 = st.columns(2)
        estado = col1.selectbox("Estado", sorted(estados))
        categoria = col2.selectbox("Categoría", sorted(categorias))

        meses_dict = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }

        mes1_nombre = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
        mes2_nombre = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)
        mes1 = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
        mes2 = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

        filtro = (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)
        df_mes1 = df2[(df2['mes'] == mes1) & filtro].copy()
        df_mes2 = df2[(df2['mes'] == mes2) & filtro].copy()

        def predecir(df_input):
            if df_input.empty:
                return df_input

            columnas_flete = [
                'total_peso_g', 'precio', '#_deproductos',
                'duracion_estimada_min', 'ciudad_cliente',
                'nombre_dc', 'hora_compra', 'año', 'mes',
                'datetime_origen', 'region',
                'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago'
            ]

            df_flete = df_input[columnas_flete].copy()
            df_encoded = pd.get_dummies(df_flete)
            columnas_modelo = modelo_flete.get_booster().feature_names
            df_encoded = df_encoded.reindex(columns=columnas_modelo, fill_value=0)

            df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
            df_input['costo_de_flete'] = df_input['costo_estimado']

            columnas_dias = [
                'Categoría', 'categoría_peso', '#_deproductos', 'total_peso_g', 'precio',
                'costo_de_flete', 'distancia_km', 'velocidad_kmh', 'duracion_estimada_min',
                'region', 'dc_asignado', 'es_feriado', 'es_fin_de_semana',
                'dias_promedio_ciudad', 'hora_compra', 'nombre_dia', 'mes', 'año',
                'temp_origen', 'precip_origen', 'cloudcover_origen', 'conditions_origen',
                'icon_origen', 'traffic', 'area'
            ]

            if not all(c in df_input.columns for c in columnas_dias):
                return df_input  # evita error si faltan columnas

            X = df_input[columnas_dias]
            pred = modelo_dias.predict(X)
            df_input['clase_entrega'] = label_encoder.inverse_transform(pred)
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
        res1 = resumen(df_mes1, mes1_nombre)
        res2 = resumen(df_mes2, mes2_nombre)

        # Asegurarse de que las columnas de costo sean numéricas
        res1[mes1_nombre] = pd.to_numeric(res1[mes1_nombre], errors='coerce')
        res2[mes2_nombre] = pd.to_numeric(res2[mes2_nombre], errors='coerce')

        comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')

        # Convertir las columnas fusionadas a numérico (por si quedaron object)
        comparacion[mes1_nombre] = pd.to_numeric(comparacion.get(mes1_nombre), errors='coerce')
        comparacion[mes2_nombre] = pd.to_numeric(comparacion.get(mes2_nombre), errors='coerce')

        # Calcular la diferencia convirtiendo primero a numérico y luego redondeando
        diff = (
            pd.to_numeric(comparacion.get(mes2_nombre), errors='coerce')
            - pd.to_numeric(comparacion.get(mes1_nombre), errors='coerce')
        )
        comparacion['Diferencia'] = diff.round(2)

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

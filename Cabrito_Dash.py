# Cabrito Dash 10/06/2025 v3 (Definitivo)

import streamlit as st
import pandas as pd
import zipfile
import io
import joblib
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu

# ------------------ Definiciones de clases/funciones personalizadas ------------------

class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

# (Tu CSS va aquí, se omite por brevedad)
st.markdown("""<style> .main { background-color: #f5f7fa !important; } </style>""", unsafe_allow_html=True)


# ===================== FUNCIÓN DE CLASIFICACIÓN =====================
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

# ===================== INTERFAZ PRINCIPAL =====================
# Se definen las 4 pestañas
tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "App Danu 📈"])

with st.sidebar:
    # st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

# Inicializar DataFrames y variables para evitar errores
df, df2, df_filtrado = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
modelo_flete, modelo_dias, label_encoder = None, None, None
estado_sel = "Nacional"

# ===================== CARGA Y PROCESAMIENTO DE DATOS =====================
if archivo_zip:
    with zipfile.ZipFile(archivo_zip) as z:
        requeridos = ['DF.csv', 'DF2.csv', 'modelo_costoflete.sav', 'modelo_dias_pipeline.joblib', 'label_encoder_dias.joblib']
        if all(f in z.namelist() for f in requeridos):
            df = pd.read_csv(z.open('DF.csv'))
            df2 = pd.read_csv(z.open('DF2.csv'))
            modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
            modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
            label_encoder = joblib.load(z.open('label_encoder_dias.joblib'))
        else:
            st.error(f"❌ Faltan archivos en el ZIP. Se requieren: {requeridos}")
            st.stop()

    with st.sidebar:
        st.subheader("🎛️ Filtro de Estado")
        estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
        estado_sel = option_menu(menu_title="Selecciona un estado", options=estados, icons=["globe"] + ["geo"] * (len(estados) - 1), default_index=0)

    df_filtrado = df.copy() if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel]

# ===================== 📊 RESUMEN NACIONAL (Pestaña 0) =====================
with tabs[0]:
    if not df_filtrado.empty:
        # El contenido de esta pestaña se asume que funciona correctamente
        st.title(f"📊 Resumen Nacional – {estado_sel}")
        st.write("Visualizaciones del Resumen Nacional...")
    else:
        st.info("Carga un archivo ZIP para ver el resumen.")

# ========================= 🏠 COSTO DE ENVÍO (Pestaña 1) =========================
with tabs[1]:
    if not df_filtrado.empty:
        # El contenido de esta pestaña se asume que funciona correctamente
        st.header("Análisis de Costo de Envío")
        st.write("Visualizaciones de Costo de Envío...")
    else:
        st.info("Carga un archivo ZIP para ver el análisis de costos.")

# ========================= 🧮 CALCULADORA (Pestaña 2) =========================
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")

    if not df2.empty and all(m is not None for m in [modelo_flete, modelo_dias, label_encoder]):
        # --- Lógica de la calculadora ---
        meses_dict = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
            7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }

        df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
        df2['año'] = df2['orden_compra_timestamp'].dt.year
        df2['mes'] = df2['orden_compra_timestamp'].dt.month

        st.markdown(f"**Estado seleccionado:** {estado_sel}")
        categoria = st.selectbox("Categoría", sorted(df2['Categoría'].dropna().unique()))

        col1, col2 = st.columns(2)
        mes1_nombre = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
        mes2_nombre = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)
        mes1 = list(meses_dict.keys())[list(meses_dict.values()).index(mes1_nombre)]
        mes2 = list(meses_dict.keys())[list(meses_dict.values()).index(mes2_nombre)]

        def predecir_y_agrupar(df_original, mes_num, cat_sel, est_sel, nom_mes):
            filtro = (df_original['mes'] == mes_num) & (df_original['Categoría'] == cat_sel) & (df_original['estado_del_cliente'] == est_sel)
            df_filtrado = df_original[filtro].copy()

            if df_filtrado.empty:
                return pd.DataFrame(columns=['ciudad_cliente', nom_mes, f"Entrega {nom_mes}"])

            # Predicción de flete
            columnas_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente', 'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region', 'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago']
            df_encoded = pd.get_dummies(df_filtrado[columnas_flete])
            df_encoded = df_encoded.reindex(columns=modelo_flete.get_booster().feature_names, fill_value=0)
            df_filtrado['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
            
            # Predicción de días
            columnas_dias = ['Categoría', 'categoria_peso', '#_deproductos', 'total_peso_g', 'precio', 'costo_de_flete', 'distancia_km', 'velocidad_kmh', 'duracion_estimada_min', 'region', 'dc_asignado', 'es_feriado', 'es_fin_de_semana', 'hora_compra', 'dias_promedio_ciudad', 'nombre_dia', 'mes', 'año', 'traffic', 'area']
            df_filtrado['costo_de_flete'] = df_filtrado['costo_estimado']
            if not all(c in df_filtrado.columns for c in columnas_dias):
                 st.warning(f"Faltan columnas para predecir días en {nom_mes}.")
                 return pd.DataFrame(columns=['ciudad_cliente', nom_mes, f"Entrega {nom_mes}"])

            pred = modelo_dias.predict(df_filtrado[columnas_dias])
            df_filtrado['clase_entrega'] = label_encoder.inverse_transform(pred)
            
            # Agrupación
            agrupado = df_filtrado.groupby('ciudad_cliente').agg(
                costo_promedio=('costo_estimado', 'mean'),
                entrega_moda=('clase_entrega', lambda x: x.mode()[0] if not x.mode().empty else 'N/A')
            ).reset_index()

            agrupado = agrupado.rename(columns={'costo_promedio': nom_mes, 'entrega_moda': f"Entrega {nom_mes}"})
            return agrupado
        
        # --- Bloque principal de ejecución y corrección ---
        res1 = predecir_y_agrupar(df2, mes1, categoria, estado_sel, mes1_nombre)
        res2 = predecir_y_agrupar(df2, mes2, categoria, estado_sel, mes2_nombre)

        # Merge de resultados
        comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')

        # FORZAR CONVERSIÓN NUMÉRICA - ESTA ES LA CORRECCIÓN CLAVE
        # Se asegura de que las columnas existan y sean numéricas antes de restar
        if mes1_nombre in comparacion:
            comparacion[mes1_nombre] = pd.to_numeric(comparacion[mes1_nombre], errors='coerce').fillna(0)
        else:
            comparacion[mes1_nombre] = 0

        if mes2_nombre in comparacion:
            comparacion[mes2_nombre] = pd.to_numeric(comparacion[mes2_nombre], errors='coerce').fillna(0)
        else:
            comparacion[mes2_nombre] = 0

        # Rellenar N/A en columnas de texto
        comparacion.fillna({f"Entrega {mes1_nombre}": 'N/A', f"Entrega {mes2_nombre}": 'N/A'}, inplace=True)

        # Ahora el cálculo es seguro
        comparacion['Diferencia'] = (comparacion[mes2_nombre] - comparacion[mes1_nombre]).round(2)
        
        # --- Presentación de resultados ---
        def resaltar(val):
            if isinstance(val, (int, float)) and val > 0: return 'color: green; font-weight: bold'
            if isinstance(val, (int, float)) and val < 0: return 'color: red; font-weight: bold'
            return ''

        st.subheader(f"Comparación: {mes1_nombre} vs {mes2_nombre}")
        st.dataframe(
            comparacion[['ciudad_cliente', mes1_nombre, mes2_nombre, 'Diferencia', f"Entrega {mes1_nombre}", f"Entrega {mes2_nombre}"]]
            .rename(columns={'ciudad_cliente': 'Ciudad'})
            .style.applymap(resaltar, subset=['Diferencia']).format(precision=2, na_rep='N/A')
        )
        st.download_button("⬇️ Descargar CSV", comparacion.to_csv(index=False).encode('utf-8'), file_name="comparacion.csv", mime="text/csv")

    else:
        st.info("Carga un archivo ZIP válido para usar la calculadora.")


# ========================= App Danu 📈 (Pestaña 3) =========================
with tabs[3]:
    st.header("App Danu 📈")
    st.markdown("### ¡Bienvenido a la sección de Danu!")
    st.success("Esta pestaña ahora se muestra correctamente.")
    
    if not df.empty:
        st.info("Los datos se han cargado y están listos para ser usados en esta sección.")
        st.write("Primeras 5 filas de DF.csv:")
        st.dataframe(df.head())
    else:
        st.warning("Por favor, carga un archivo ZIP para activar las funcionalidades de esta sección.")

# dash2.py

import streamlit as st
import pandas as pd
import zipfile
import io
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu  # Asegúrate de tener instalada esta dependencia
import plotly.graph_objects as go

# ------------------ Definiciones de clases/funciones personalizadas ------------------

class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Tu lógica de transformación
        return X

# (Aquí continúa TODO tu resto de funciones auxiliares, estilos CSS y transformadores
#  exactamente igual que en tu versión original, hasta la definición de tabs.)

# ------------------ Configuración de Streamlit ------------------

st.set_page_config(page_title="Cabrito Analytics", layout="wide")

# ------------------ Definición de pestañas ------------------

tabs = st.tabs([
    "📊 Resumen Nacional",
    "🏠 Costo de Envío",
    "🧮 Calculadora",
    "App Danu 📈"
])

# ==================== PESTAÑA 0: Resumen Nacional ====================
with tabs[0]:
    # ... tu código original para el Resumen Nacional (tab[0])
    pass

# ==================== PESTAÑA 1: Costo de Envío ====================
with tabs[1]:
    # ... tu código original para el Costo de Envío (tab[1])
    pass

# ==================== PESTAÑA 2: Calculadora ====================
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")

    # --- Carga de modelos ---
    modelo_flete = joblib.load('modelo_costoflete.sav')
    modelo_dias = joblib.load('modelo_dias_pipeline_70.joblib')
    label_encoder = joblib.load('label_encoder_dias_70.joblib')

    # Diccionario de meses
    meses_dict = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    # Sidebar para cargar CSV y seleccionar filtros
    st.sidebar.header("Carga tu archivo de datos")
    archivo_subido = st.sidebar.file_uploader("Sube tu CSV con pedidos", type=["csv"])
    if not archivo_subido:
        st.warning("Por favor, carga un archivo CSV para continuar.")
        st.stop()

    df = pd.read_csv(archivo_subido, encoding='utf-8')
    df['orden_compra_timestamp'] = pd.to_datetime(df['orden_compra_timestamp'])
    df['año'] = df['orden_compra_timestamp'].dt.year
    df['mes'] = df['orden_compra_timestamp'].dt.month

    # Filtros de estado y categoría
    estado = st.sidebar.selectbox("Estado", sorted(df['estado_del_cliente'].dropna().unique()))
    categorias = sorted(df['categoria'].dropna().unique())
    categoria = st.sidebar.selectbox("Categoría", categorias)

    # Filtros de mes
    col1, col2 = st.columns(2)
    with col1:
        mes1_nombre = st.selectbox("Mes 1", list(meses_dict.values()), index=0)
    with col2:
        mes2_nombre = st.selectbox("Mes 2", list(meses_dict.values()), index=1)

    mes1_num = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
    mes2_num = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

    # Filtrar datos por estado y categoría
    filtro = (
        (df['estado_del_cliente'] == estado) &
        (df['categoria'] == categoria)
    )
    df_mes1 = df[(df['mes'] == mes1_num) & filtro].copy()
    df_mes2 = df[(df['mes'] == mes2_num) & filtro].copy()

    # Función de predicción
    def predecir(df_input):
        if df_input.empty:
            return df_input

        # Columnas para costo de flete
        columnas_flete = [
            'total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min',
            'ciudad_cliente', 'nombre_dc', 'hora_compra', 'año', 'mes',
            'datetime_origen', 'region', 'dias_promedio_ciudad',
            'categoria', 'tipo_de_pago'
        ]
        df_flete = df_input[columnas_flete].copy()
        df_encoded = pd.get_dummies(df_flete)
        columnas_modelo = modelo_flete.get_booster().feature_names
        df_encoded = df_encoded.reindex(columns=columnas_modelo, fill_value=0)

        df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
        df_input['costo_de_flete'] = df_input['costo_estimado']

        # Columnas para clase de entrega
        cols_dias = [
            'categoria', 'categoria_peso', '#_deproductos', 'total_peso_g', 'precio',
            'costo_de_flete', 'distancia_km', 'velocidad_kmh', 'duracion_estimada_min',
            'region', 'dc_asignado', 'es_feriado', 'es_fin_de_semana',
            'hora_compra', 'dias_promedio_ciudad', 'nombre_dia', 'mes', 'año',
            'traffic', 'area'
        ]
        faltantes = [c for c in cols_dias if c not in df_input.columns]
        if faltantes:
            st.error(f"Faltan columnas para la predicción de clase de entrega: {faltantes}")
            return df_input

        X_dias = df_input[cols_dias].copy()
        pred_clase = modelo_dias.predict(X_dias)
        df_input['clase_entrega'] = label_encoder.inverse_transform(pred_clase)

        return df_input

    # Función para agrupar resultados por ciudad
    def agrupar_resultados(df_pred, nombre_mes):
        if 'costo_estimado' in df_pred.columns and 'clase_entrega' in df_pred.columns:
            return df_pred.groupby('ciudad_cliente').agg({
                'costo_estimado': lambda x: round(x.mean(), 2),
                'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'NA'
            }).rename(columns={
                'costo_estimado': nombre_mes,
                'clase_entrega': f"Entrega {nombre_mes}"
            }).reset_index()
        else:
            st.warning(f"No se pudo calcular para {nombre_mes}.")
            return pd.DataFrame(columns=['ciudad_cliente', nombre_mes, f"Entrega {nombre_mes}"])

    # Ejecutar predicciones
    df_mes1 = predecir(df_mes1)
    df_mes2 = predecir(df_mes2)

    # Crear tabla comparativa
    res1 = agrupar_resultados(df_mes1, mes1_nombre)
    res2 = agrupar_resultados(df_mes2, mes2_nombre)
    comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
    comparacion['Diferencia'] = (comparacion[mes2_nombre] - comparacion[mes1_nombre]).round(2)
    comparacion = comparacion[[
        'ciudad_cliente', mes1_nombre, mes2_nombre, 'Diferencia',
        f"Entrega {mes1_nombre}", f"Entrega {mes2_nombre}"
    ]].rename(columns={'ciudad_cliente': 'Ciudad'})

    # KPIs arriba de la tabla
    costo_prom_mes1 = df_mes1['costo_estimado'].mean() if not df_mes1.empty else np.nan
    costo_prom_mes2 = df_mes2['costo_estimado'].mean() if not df_mes2.empty else np.nan
    cambio_pct = ((costo_prom_mes2 - costo_prom_mes1) / costo_prom_mes1 * 100) if costo_prom_mes1 != 0 else 0

    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"**Costo Promedio {mes1_nombre}**")
    k2.markdown("**% Cambio**")
    k3.markdown(f"**Costo Promedio {mes2_nombre}**")
    k1.markdown(f"<span style='font-size:28px; font-weight:bold'>{costo_prom_mes1:.2f}</span>", unsafe_allow_html=True)
    color = 'green' if cambio_pct > 0 else 'red'
    k2.markdown(f"<span style='color:{color}; font-size:28px; font-weight:bold'>{cambio_pct:.2f}%</span>", unsafe_allow_html=True)
    k3.markdown(f"<span style='font-size:28px; font-weight:bold'>{costo_prom_mes2:.2f}</span>", unsafe_allow_html=True)

    # Función para resaltar diferencia en la tabla
    def resaltar(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: green; font-weight: bold'
            elif val < 0:
                return 'color: red; font-weight: bold'
        return ''

    # Mostrar tabla
    st.subheader(f"Comparación: {mes1_nombre} vs {mes2_nombre}")
    st.dataframe(
        comparacion.style
        .applymap(resaltar, subset=['Diferencia'])
        .format(precision=2),
        use_container_width=True
    )

    # Botón de descarga
    csv = comparacion.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Descargar CSV",
        data=csv,
        file_name=f'comparacion_{estado}_{categoria}_{mes1_nombre}vs{mes2_nombre}.csv',
        mime='text/csv'
    )

# ==================== PESTAÑA 3: App Danu ====================
with tabs[3]:
    # ... tu código original para App Danu (tab[3])
    pass

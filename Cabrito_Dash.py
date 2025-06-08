# Cabrito_Dash.py

import streamlit as st
import pandas as pd
import zipfile
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------ Definiciones de clases/funciones personalizadas ------------------

class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X  # Tu lógica de transformación

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

# ===================== DEFINICIÓN DE PESTAÑAS =====================
tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "App Danu 📈"])

# ==================== PESTAÑA 0: Resumen Nacional ====================
with tabs[0]:
    # ... tu código original para Resumen Nacional ...
    pass

# ==================== PESTAÑA 1: Costo de Envío ====================
with tabs[1]:
    # ... tu código original para Costo de Envío ...
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

    # Sidebar: subir ZIP con el CSV
    st.sidebar.header("Carga tu ZIP con pedidos")
    zip_subido = st.sidebar.file_uploader("ZIP que contenga tu CSV de pedidos", type=["zip"])
    if not zip_subido:
        st.warning("Por favor, sube un archivo .zip para continuar.")
        st.stop()

    with zipfile.ZipFile(zip_subido) as z:
        csv_files = [f for f in z.namelist() if f.lower().endswith('.csv')]
        if not csv_files:
            st.error("No se encontró ningún archivo .csv dentro del ZIP.")
            st.stop()
        nombre_csv = csv_files[0]
        df = pd.read_csv(z.open(nombre_csv), encoding='utf-8')

    # Normalizar nombres de columnas: minúsculas, sin espacios, sin acentos
    import unicodedata
    def normalize(col):
        s = col.strip().lower().replace(' ', '_')
        return unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode('ascii')
    df.columns = [normalize(c) for c in df.columns]

    # Verifica columnas clave
    required = ['orden_compra_timestamp', 'estado_del_cliente', 'categoria']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas obligatorias en el CSV: {missing}. Columnas disponibles: {list(df.columns)}")
        st.stop()

    # Procesamiento de fechas y creación de columnas año/mes
    df['orden_compra_timestamp'] = pd.to_datetime(df['orden_compra_timestamp'])
    df['año'] = df['orden_compra_timestamp'].dt.year
    df['mes'] = df['orden_compra_timestamp'].dt.month

    # Select de Estado y Categoría
    estado = st.sidebar.selectbox("Estado", sorted(df['estado_del_cliente'].dropna().unique()))
    categoria = st.sidebar.selectbox("Categoría", sorted(df['categoria'].dropna().unique()))

    # Selección de Mes 1 y Mes 2
    col1, col2 = st.columns(2)
    with col1:
        mes1_nombre = st.selectbox("Mes 1", list(meses_dict.values()), index=0)
    with col2:
        mes2_nombre = st.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1_num = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
    mes2_num = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

    # Función de predicción
    def predecir(df_input):
        if df_input.empty:
            return df_input

        # Costo de flete
        cols_flete = [
            'total_peso_g', 'precio', 'numero_deproductos', 'duracion_estimada_min',
            'ciudad_cliente', 'nombre_dc', 'hora_compra', 'año', 'mes',
            'datetime_origen', 'region', 'dias_promedio_ciudad',
            'categoria', 'tipo_de_pago'
        ]
        # Ajusta nombres si difieren tras normalizar...
        df_flete = df_input.reindex(columns=cols_flete).copy()
        df_encoded = pd.get_dummies(df_flete)
        feat_names = modelo_flete.get_booster().feature_names
        df_encoded = df_encoded.reindex(columns=feat_names, fill_value=0)
        df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
        df_input['costo_de_flete'] = df_input['costo_estimado']

        # Clase de entrega
        cols_dias = [
            'categoria', 'categoria_peso', 'numero_deproductos', 'total_peso_g', 'precio',
            'costo_de_flete', 'distancia_km', 'velocidad_kmh', 'duracion_estimada_min',
            'region', 'dc_asignado', 'es_feriado', 'es_fin_de_semana',
            'hora_compra', 'dias_promedio_ciudad', 'nombre_dia', 'mes', 'año',
            'traffic', 'area'
        ]
        falt = [c for c in cols_dias if c not in df_input.columns]
        if falt:
            st.error(f"Faltan columnas para predecir clase de entrega: {falt}")
            return df_input

        X_dias = df_input[cols_dias].copy()
        preds = modelo_dias.predict(X_dias)
        df_input['clase_entrega'] = label_encoder.inverse_transform(preds)
        return df_input

    # Filtrado por mes, estado y categoría
    filt = (
        (df['mes'] == mes1_num) &
        (df['estado_del_cliente'] == estado) &
        (df['categoria'] == categoria)
    )
    df_mes1 = predecir(df[filt].copy())

    filt2 = (
        (df['mes'] == mes2_num) &
        (df['estado_del_cliente'] == estado) &
        (df['categoria'] == categoria)
    )
    df_mes2 = predecir(df[filt2].copy())

    # Agrupar resultados
    def agrupar(df_p, nombre):
        if 'costo_estimado' in df_p and 'clase_entrega' in df_p:
            return df_p.groupby('ciudad_cliente').agg({
                'costo_estimado': lambda x: round(x.mean(), 2),
                'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'NA'
            }).rename(columns={
                'costo_estimado': nombre,
                'clase_entrega': f"Entrega {nombre}"
            }).reset_index()
        return pd.DataFrame(columns=['ciudad_cliente', nombre, f"Entrega {nombre}"])

    res1 = agrupar(df_mes1, mes1_nombre)
    res2 = agrupar(df_mes2, mes2_nombre)
    comp = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
    comp['Diferencia'] = (comp[mes2_nombre] - comp[mes1_nombre]).round(2)

    # KPIs
    avg1 = df_mes1['costo_estimado'].mean() if not df_mes1.empty else np.nan
    avg2 = df_mes2['costo_estimado'].mean() if not df_mes2.empty else np.nan
    pct = ((avg2 - avg1) / avg1 * 100) if avg1 else 0

    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"**Costo Promedio {mes1_nombre}:**  {avg1:.2f}")
    color = 'green' if pct > 0 else 'red'
    k2.markdown(f"**% Cambio:**  <span style='color:{color}'>{pct:.2f}%</span>", unsafe_allow_html=True)
    k3.markdown(f"**Costo Promedio {mes2_nombre}:**  {avg2:.2f}")

    # Tabla comparativa
    def style_diff(v):
        if isinstance(v, (int, float)):
            return 'color: green' if v > 0 else ('color: red' if v < 0 else '')
        return ''
    st.subheader(f"{mes1_nombre} vs {mes2_nombre}")
    st.dataframe(
        comp.style.applymap(style_diff, subset=['Diferencia']).format(precision=2),
        use_container_width=True
    )

    # Descarga
    csv = comp.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Descargar CSV", data=csv,
                       file_name=f"comp_{estado}_{categoria}_{mes1_nombre}_vs_{mes2_nombre}.csv")

# ==================== PESTAÑA 3: App Danu ====================
with tabs[3]:
    # ... tu código original para App Danu ...
    pass

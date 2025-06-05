import streamlit as st
import pandas as pd
import zipfile
import io
import plotly.express as px
import joblib
from datetime import datetime

# ====================== CONFIGURACIÓN INICIAL ======================
st.set_page_config(page_title="Cabrito Analytics", layout="wide", initial_sidebar_state="expanded")

st.title("📊 Cabrito Analytics")

tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ====================== CARGA DE ZIP ======================
with st.sidebar:
    st.header("📂 Subir archivo ZIP")
    zip_file = st.file_uploader("Sube un archivo .zip con DF.csv, DF2.csv y modelos", type="zip")

if zip_file:
    try:
        with zipfile.ZipFile(zip_file) as z:
            # Leer archivos desde el ZIP
            with z.open("DF.csv") as f:
                df = pd.read_csv(f)
            with z.open("DF2.csv") as f2:
                df2 = pd.read_csv(f2)
            with z.open("modelo_costoflete.sav") as m1:
                modelo_flete = joblib.load(m1)
            with z.open("modelo_dias_pipeline.joblib") as m2:
                modelo_dias = joblib.load(m2)
            with z.open("label_encoder_dias.joblib") as le:
                label_encoder = joblib.load(le)
    except Exception as e:
        st.error(f"Error al leer archivos del ZIP: {e}")
        st.stop()
else:
    st.info("Por favor sube el archivo .zip")
    st.stop()

# ====================== DASHBOARD ======================
with tabs[0]:
    st.markdown("## 🧭 Visión General de la Operación")
    categorias = df['Categoría'].dropna().unique()
    estados = df['estado_del_cliente'].dropna().unique()
    años = sorted(df['año'].dropna().unique())
    meses = sorted(df['mes'].dropna().unique())

    col1, col2 = st.columns(2)
    categoria_sel = col1.multiselect("Categoría", categorias, default=list(categorias))
    estado_sel = col2.multiselect("Estado", estados, default=list(estados))

    año_sel = st.multiselect("Año", años, default=años)
    mes_sel = st.multiselect("Mes", meses, default=meses)

    df_filtrado = df[
        (df['Categoría'].isin(categoria_sel)) &
        (df['estado_del_cliente'].isin(estado_sel)) &
        (df['año'].isin(año_sel)) &
        (df['mes'].isin(mes_sel))
    ]

    st.metric("Total de pedidos", f"{len(df_filtrado):,}")
    st.plotly_chart(px.treemap(df_filtrado, path=['Categoría'], values='precio'), use_container_width=True)

# ====================== CALCULADORA ======================
with tabs[1]:
    st.markdown("## 🤖 Calculadora ML")

    estados_calc = sorted(df2['estado_del_cliente'].dropna().unique())
    categorias_calc = sorted(df2['Categoría'].dropna().unique())

    estado = st.selectbox("Estado", estados_calc)
    categoria = st.selectbox("Categoría", categorias_calc)

    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    meses_dict = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    col1, col2 = st.columns(2)
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

        columnas_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                          'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                          'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago']
        df_encoded = pd.get_dummies(df_input[columnas_flete])
        df_encoded = df_encoded.reindex(columns=modelo_flete.get_booster().feature_names, fill_value=0)

        df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)

        cols_dias = ['Categoría', 'categoría_peso', '#_deproductos', 'total_peso_g', 'precio', 'costo_estimado',
                     'distancia_km', 'velocidad_kmh', 'duracion_estimada_min', 'region', 'dc_asignado',
                     'es_feriado', 'es_fin_de_semana', 'dias_promedio_ciudad', 'hora_compra',
                     'nombre_dia', 'mes', 'año', 'temp_origen', 'precip_origen', 'cloudcover_origen',
                     'conditions_origen', 'icon_origen', 'traffic', 'area']
        df_input['clase_entrega'] = label_encoder.inverse_transform(modelo_dias.predict(df_input[cols_dias]))

        return df_input

    def resumen(df_input, nombre_mes):
        df_pred = predecir(df_input)
        return df_pred.groupby('ciudad_cliente').agg({
            'costo_estimado': 'mean',
            'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
        }).reset_index().rename(columns={'costo_estimado': nombre_mes, 'clase_entrega': f"Entrega {nombre_mes}"})

    res1 = resumen(df_mes1, mes1_nombre)
    res2 = resumen(df_mes2, mes2_nombre)
    comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
    comparacion['Diferencia'] = (comparacion[mes2_nombre] - comparacion[mes1_nombre]).round(2)

    st.dataframe(comparacion)

    st.download_button("⬇️ Descargar comparación", comparacion.to_csv(index=False), "comparacion.csv", "text/csv")

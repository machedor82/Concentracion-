import streamlit as st
import pandas as pd
import zipfile
import joblib
import os
import plotly.express as px
import numpy as np

# Configuración inicial
st.set_page_config(page_title="Cabrito Analytics App", layout="wide")
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #001f3f;
            color: white;
        }
        .stTabs [aria-selected="true"] {
            color: #004b8d;
            border-bottom: 3px solid #004b8d;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Logo
st.sidebar.image("danu_logo.png", use_column_width=True)
st.sidebar.header("📁 Subida de Archivos")
uploaded_zip = st.sidebar.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

# Variables globales
df = df2 = None
modelo_flete = modelo_dias = label_encoder = None

def cargar_desde_zip(zip_file):
    global modelo_flete, modelo_dias, label_encoder
    with zipfile.ZipFile(zip_file, "r") as z:
        files = z.namelist()
        
        # Carga DF.csv
        if "DF.csv" in files:
            df = pd.read_csv(z.open("DF.csv"))
        else:
            st.error("❌ DF.csv no se encuentra dentro del ZIP.")
            return None, None

        # Carga DF2.csv
        df2 = pd.read_csv(z.open("DF2.csv")) if "DF2.csv" in files else None

        # Carga modelos
        if "modelo_costoflete.sav" in files:
            modelo_flete = joblib.load(z.open("modelo_costoflete.sav"))
        if "modelo_dias_pipeline.joblib" in files:
            modelo_dias = joblib.load(z.open("modelo_dias_pipeline.joblib"))
        if "label_encoder_dias.joblib" in files:
            label_encoder = joblib.load(z.open("label_encoder_dias.joblib"))

        return df, df2

# Procesamiento
if uploaded_zip:
    df, df2 = cargar_desde_zip(uploaded_zip)
    if df is not None:
        tab1, tab2 = st.tabs(["📊 Dashboard", "🧮 Calculadora"])

        # ------------------------ DASHBOARD ------------------------
        with tab1:
            st.header("📦 Dashboard Logístico")

            categorias = df['Categoría'].dropna().unique()
            estados = df['estado_del_cliente'].dropna().unique()
            años = sorted(df['año'].dropna().unique())
            meses = sorted(df['mes'].dropna().unique())

            categoria_sel = st.sidebar.multiselect("Categoría", categorias, default=list(categorias))
            estado_sel = st.sidebar.multiselect("Estado", estados, default=list(estados))
            año_sel = st.sidebar.multiselect("Año", años, default=años)
            mes_sel = st.sidebar.multiselect("Mes", meses, default=meses)

            df_filtrado = df[
                (df['Categoría'].isin(categoria_sel)) &
                (df['estado_del_cliente'].isin(estado_sel)) &
                (df['año'].isin(año_sel)) &
                (df['mes'].isin(mes_sel))
            ]

            st.subheader("📊 Indicadores")
            col1, col2, col3 = st.columns(3)
            col1.metric("Pedidos", f"{len(df_filtrado):,}")
            col2.metric("Flete > 50%", f"{(df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100:.1f}%")
            col3.metric("≥7 días antes", f"{(df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100:.1f}%")

            st.subheader("🌳 Treemap")
            st.plotly_chart(px.treemap(df_filtrado, path=['Categoría'], values='precio', color='Categoría'), use_container_width=True)

            st.subheader("🗺️ Mapa")
            geo_df = df_filtrado.dropna(subset=['lat_cliente', 'lon_cliente'])
            if not geo_df.empty:
                st.map(geo_df.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
            else:
                st.warning("⚠️ No hay datos geográficos.")

            st.subheader("📈 Entrega vs Colchón")
            avg = df_filtrado.groupby('estado_del_cliente')[['dias_entrega', 'colchon_dias']].mean().reset_index()
            st.plotly_chart(px.bar(avg, x='estado_del_cliente', y=['dias_entrega', 'colchon_dias'], barmode='group'), use_container_width=True)

        # ------------------------ CALCULADORA ------------------------
        with tab2:
            st.header("🧮 Predicción de Flete y Clase de Entrega")

            if df2 is None or modelo_flete is None or modelo_dias is None or label_encoder is None:
                st.warning("Sube un ZIP que contenga DF2.csv y los modelos para usar esta sección.")
            else:
                df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
                df2['año'] = df2['orden_compra_timestamp'].dt.year
                df2['mes'] = df2['orden_compra_timestamp'].dt.month

                estado = st.selectbox("Estado", sorted(df2['estado_del_cliente'].dropna().unique()))
                categoria = st.selectbox("Categoría", sorted(df2['Categoría'].dropna().unique()))

                meses_dict = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
                              7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}

                col1, col2 = st.columns(2)
                mes1 = col1.selectbox("Mes 1", list(meses_dict.values()))
                mes2 = col2.selectbox("Mes 2", list(meses_dict.values()))

                m1 = [k for k, v in meses_dict.items() if v == mes1][0]
                m2 = [k for k, v in meses_dict.items() if v == mes2][0]

                filtro = (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)
                df_m1 = df2[(df2['mes'] == m1) & filtro].copy()
                df_m2 = df2[(df2['mes'] == m2) & filtro].copy()

                def predecir(df_input):
                    if df_input.empty:
                        return df_input
                    # FLETE
                    features_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                                      'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                                      'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago']
                    df_encoded = pd.get_dummies(df_input[features_flete])
                    df_encoded = df_encoded.reindex(columns=modelo_flete.get_booster().feature_names, fill_value=0)
                    df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
                    df_input['costo_de_flete'] = df_input['costo_estimado']

                    # CLASE DE ENTREGA
                    features_dias = ['Categoría', 'categoría_peso', '#_deproductos', 'total_peso_g', 'precio', 'costo_de_flete',
                                     'distancia_km', 'velocidad_kmh', 'duracion_estimada_min', 'region', 'dc_asignado',
                                     'es_feriado', 'es_fin_de_semana', 'dias_promedio_ciudad', 'hora_compra',
                                     'nombre_dia', 'mes', 'año', 'temp_origen', 'precip_origen', 'cloudcover_origen',
                                     'conditions_origen', 'icon_origen', 'traffic', 'area']
                    if all(c in df_input.columns for c in features_dias):
                        pred_clase = modelo_dias.predict(df_input[features_dias])
                        df_input['clase_entrega'] = label_encoder.inverse_transform(pred_clase)
                    else:
                        df_input['clase_entrega'] = "N/A"
                    return df_input

                def resumen(df, nombre):
                    if df.empty:
                        return pd.DataFrame()
                    df = predecir(df)
                    return df.groupby('ciudad_cliente').agg({
                        'costo_estimado': 'mean',
                        'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
                    }).reset_index().rename(columns={
                        'costo_estimado': nombre,
                        'clase_entrega': f"Entrega {nombre}"
                    })

                r1 = resumen(df_m1, mes1)
                r2 = resumen(df_m2, mes2)
                comparacion = pd.merge(r1, r2, on='ciudad_cliente', how='outer')
                comparacion['Diferencia'] = (comparacion[mes2] - comparacion[mes1]).round(2)

                st.dataframe(comparacion)
                st.download_button("⬇️ Descargar comparación", comparacion.to_csv(index=False), "comparacion.csv", "text/csv")
else:
    st.info("📦 Sube un archivo ZIP para iniciar.")

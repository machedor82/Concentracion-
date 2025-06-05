
import os
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ========================== CONFIGURACIÓN INICIAL ==========================
st.set_page_config(page_title="Cabrito Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #002244;
            color: white;
        }
        .stSlider > div[data-testid="stTickBar"] {
            background-color: #ffffff11;
        }
        .stSlider .css-14pt78w {
            color: white !important;
        }
        .stMultiSelect, .stSlider {
            color: black !important;
        }
        [data-testid="stSidebar"] label {
            color: white;
        }
        body {
            background-color: #f9f9f9;
            color: #1f1f1f;
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

st.title("📊 Panel BI")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ========================== CARGA DE ARCHIVOS ==========================

with st.sidebar:
    st.header("📂 Subida de Archivos")
    zip_file = st.file_uploader("Sube un archivo ZIP con DF.csv", type="zip", key="zip_loader")
    csv_file = st.file_uploader("Sube DF2.csv para la calculadora", type="csv", key="csv_loader")

# ========================== DASHBOARD ==========================
with tabs[0]:
    if zip_file is not None:
        try:
            with zipfile.ZipFile(zip_file) as z:
                with z.open("DF.csv") as f:
                    df = pd.read_csv(f)

            st.success("✅ DF.csv cargado correctamente")

            # Filtros
            st.sidebar.subheader("🎛️ Filtros")
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

            st.subheader("🔢 Indicadores Clave")
            col1, col2, col3 = st.columns(3)
            col1.metric("Pedidos", f"{len(df_filtrado):,}")
            col2.metric("Flete > 50%", f"{(df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100:.1f}%")
            col3.metric("⏱️ Entregas ≥7 días antes", f"{(df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100:.1f}%")

            st.subheader("📈 Visualizaciones")
            col1, col2, col3 = st.columns(3)

            with col1:
                fig_tree = px.treemap(df_filtrado, path=['Categoría'], values='precio', color='Categoría')
                st.plotly_chart(fig_tree, use_container_width=True)

            with col2:
                mapa = df_filtrado.dropna(subset=['lat_cliente', 'lon_cliente'])
                if not mapa.empty:
                    st.map(mapa.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
                else:
                    st.warning("No hay ubicaciones válidas.")

            with col3:
                prom = df_filtrado.groupby('estado_del_cliente')[['dias_entrega', 'colchon_dias']].mean().reset_index()
                fig_bar = px.bar(prom, x='estado_del_cliente', y=['dias_entrega', 'colchon_dias'], barmode='group')
                st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error cargando el archivo ZIP: {e}")
    else:
        st.warning("🔺 Sube un archivo ZIP con DF.csv para ver el Dashboard.")

# ========================== CALCULADORA ==========================
with tabs[1]:
    if csv_file is not None:
        try:
            df2 = pd.read_csv(csv_file)
            st.success("✅ DF2.csv cargado correctamente")

            # Carga modelos
            modelo_flete = joblib.load("modelo_costoflete.sav")
            modelo_dias = joblib.load("modelo_dias_pipeline.joblib")
            label_encoder = joblib.load("label_encoder_dias.joblib")

            # Filtros
            estado = st.selectbox("Estado", sorted(df2['estado_del_cliente'].dropna().unique()))
            categoria = st.selectbox("Categoría", sorted(df2['Categoría'].dropna().unique()))

            meses_dict = {
                1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
                7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
            }

            col1, col2 = st.columns(2)
            mes1_nombre = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
            mes2_nombre = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)

            mes1 = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
            mes2 = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

            df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
            df2['año'] = df2['orden_compra_timestamp'].dt.year
            df2['mes'] = df2['orden_compra_timestamp'].dt.month

            filtro = (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)
            df_mes1 = df2[(df2['mes'] == mes1) & filtro].copy()
            df_mes2 = df2[(df2['mes'] == mes2) & filtro].copy()

            def predecir(df_input):
                if df_input.empty:
                    return df_input
                columnas_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                                  'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                                  'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago']
                df_flete = df_input[columnas_flete].copy()
                df_encoded = pd.get_dummies(df_flete)
                columnas_entrenadas = modelo_flete.get_booster().feature_names
                df_encoded = df_encoded.reindex(columns=columnas_entrenadas, fill_value=0)
                df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
                return df_input

            def agrupar(df_input, nombre_mes):
                if 'costo_estimado' not in df_input:
                    return pd.DataFrame(columns=['ciudad_cliente', nombre_mes])
                return df_input.groupby('ciudad_cliente')['costo_estimado'].mean().round(2).reset_index().rename(columns={'costo_estimado': nombre_mes})

            df_mes1 = predecir(df_mes1)
            df_mes2 = predecir(df_mes2)

            r1 = agrupar(df_mes1, mes1_nombre)
            r2 = agrupar(df_mes2, mes2_nombre)

            comparacion = pd.merge(r1, r2, on='ciudad_cliente', how='outer')
            comparacion['Diferencia'] = comparacion[mes2_nombre] - comparacion[mes1_nombre]
            comparacion = comparacion.rename(columns={'ciudad_cliente': 'Ciudad'})

            st.subheader(f"📊 Comparación entre {mes1_nombre} y {mes2_nombre}")
            st.dataframe(comparacion)

            st.download_button("⬇️ Descargar CSV", comparacion.to_csv(index=False).encode('utf-8'), "comparacion.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error cargando calculadora: {e}")
    else:
        st.warning("🔺 Sube el archivo DF2.csv para la calculadora.")

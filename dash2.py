import streamlit as st
import pandas as pd
import plotly.express as px
import zipfile
import os
import tempfile
import joblib
import numpy as np
from datetime import datetime

# ================== CONFIGURACIÓN ==================
st.set_page_config(page_title="Cabrito Analytics", layout="wide", initial_sidebar_state="expanded")

# Estilos
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #002244;
            color: white;
        }
        .stFileUploader label, .stTextInput label, .stSelectbox label {
            color: white !important;
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

'''
# ================== CARGA ZIP ==================
st.sidebar.header("📁 Subir archivo ZIP")
zip_file = st.sidebar.file_uploader("Cargar archivo .zip", type="zip")

if zip_file:
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            zip_path = os.path.join(tmpdirname, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdirname)

            # Cargar CSVs y modelos
            df = pd.read_csv(os.path.join(tmpdirname, "DF.csv"))
            df2 = pd.read_csv(os.path.join(tmpdirname, "DF2.csv"))
            modelo_costoflete = joblib.load(os.path.join(tmpdirname, "modelo_costoflete.sav"))
            modelo_dias = joblib.load(os.path.join(tmpdirname, "modelo_dias_pipeline.joblib"))
            label_encoder = joblib.load(os.path.join(tmpdirname, "label_encoder_dias.joblib"))

            # Pestañas
            tabs = st.tabs(["📊 Dashboard", "📈 Calculadora"])

            # ========== PESTAÑA 1: DASHBOARD ==========
            with tabs[0]:
                st.title("📊 Dashboard Logístico")

                st.sidebar.subheader("🎛️ Filtros")
                categorias = df['Categoría'].dropna().unique()
                estados = df['estado_del_cliente'].dropna().unique()
                años = sorted(df['año'].dropna().unique())
                meses = sorted(df['mes'].dropna().unique())

                cat_sel = st.sidebar.multiselect("Categoría", categorias, default=list(categorias))
                est_sel = st.sidebar.multiselect("Estado", estados, default=list(estados))
                año_sel = st.sidebar.multiselect("Año", años, default=list(años))
                mes_sel = st.sidebar.multiselect("Mes", meses, default=list(meses))

                df_filtrado = df[
                    (df['Categoría'].isin(cat_sel)) &
                    (df['estado_del_cliente'].isin(est_sel)) &
                    (df['año'].isin(año_sel)) &
                    (df['mes'].isin(mes_sel))
                ]

                st.markdown("### 📌 Indicadores Clave")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total pedidos", f"{len(df_filtrado):,}")
                pct_flete = (df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100
                c2.metric("Flete > 50%", f"{pct_flete:.1f}%")
                anticipadas = (df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100
                c3.metric("Entregas ≥7 días antes", f"{anticipadas:.1f}%")

                st.markdown("### 📊 Visualizaciones")
                col1, col2, col3 = st.columns(3)

                with col1:
                    fig1 = px.treemap(df_filtrado, path=['Categoría'], values='precio', color='Categoría',
                                      color_discrete_sequence=px.colors.sequential.Blues)
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    mapa = df_filtrado.dropna(subset=['lat_cliente', 'lon_cliente'])
                    if not mapa.empty:
                        st.map(mapa.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
                    else:
                        st.warning("No hay coordenadas para mostrar.")

                with col3:
                    prom = df_filtrado.groupby('estado_del_cliente')[['dias_entrega', 'colchon_dias']].mean().reset_index()
                    fig3 = px.bar(prom, x='estado_del_cliente', y=['dias_entrega', 'colchon_dias'],
                                  barmode='group', color_discrete_sequence=px.colors.sequential.Blues)
                    fig3.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig3, use_container_width=True)

            # ========== PESTAÑA 2: CALCULADORA ==========
            with tabs[1]:
                st.title("📈 Calculadora Predictiva")

                df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
                df2['año'] = df2['orden_compra_timestamp'].dt.year
                df2['mes'] = df2['orden_compra_timestamp'].dt.month

                estado = st.selectbox("Estado", sorted(df2['estado_del_cliente'].unique()))
                categoria = st.selectbox("Categoría", sorted(df2['Categoría'].unique()))

                meses_dict = {
                    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
                    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
                }

                col1, col2 = st.columns(2)
                m1 = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
                m2 = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)
                mes1 = [k for k, v in meses_dict.items() if v == m1][0]
                mes2 = [k for k, v in meses_dict.items() if v == m2][0]

                filtro = (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)
                df1 = df2[(df2['mes'] == mes1) & filtro].copy()
                df2_ = df2[(df2['mes'] == mes2) & filtro].copy()

                def predecir(df_input):
                    columnas_modelo = modelo_dias.feature_names_in_
                    X_dias = df_input[columnas_modelo]
                    X_costos = df_input[modelo_costoflete.feature_names_in_]
                    df_input['costo_estimado'] = modelo_costoflete.predict(X_costos)
                    df_input['clase_entrega'] = label_encoder.inverse_transform(modelo_dias.predict(X_dias))
                    return df_input

                def resumen(df_pred, nombre_mes):
                    if df_pred.empty:
                        return pd.DataFrame(columns=['ciudad_cliente', nombre_mes, f"Entrega {nombre_mes}"])
                    df_pred = predecir(df_pred)
                    return df_pred.groupby('ciudad_cliente').agg({
                        'costo_estimado': 'mean',
                        'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
                    }).reset_index().rename(columns={
                        'costo_estimado': nombre_mes,
                        'clase_entrega': f"Entrega {nombre_mes}"
                    })

                res1 = resumen(df1, m1)
                res2 = resumen(df2_, m2)
                merge = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
                if m1 in merge and m2 in merge:
                    merge['Diferencia'] = merge[m2] - merge[m1]

                st.dataframe(merge)
                st.download_button("⬇️ Descargar comparación", merge.to_csv(index=False), "comparacion.csv", "text/csv")
'''
            # Footer
            st.caption(f"Versión pro optimizada con fondo azul marino, filtros personalizados, carga ZIP, ML y más. Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    except Exception as e:
        st.error(f"❌ Error al cargar el ZIP: {e}")
else:
    st.warning("⬆️ Por favor, sube un archivo .zip que contenga DF.csv, DF2.csv y los modelos.")

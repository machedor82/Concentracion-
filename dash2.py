import streamlit as st
import pandas as pd
import zipfile
import os
import plotly.express as px
import joblib

# ======================== CONFIGURACIÓN ========================
st.set_page_config(page_title="Cabrito Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #001F3F;
            color: white;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .st-bb {
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
        header, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.sidebar.image("danu_logo.png", width=180)
st.sidebar.header("Sube tu archivo ZIP")
zip_file = st.sidebar.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

@st.cache_data
def extraer_archivos(zip_file):
    ruta = "/tmp/archivos_danu"
    os.makedirs(ruta, exist_ok=True)
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(ruta)
    return {
        "df1": pd.read_csv(os.path.join(ruta, "DF.csv")),
        "df2": pd.read_csv(os.path.join(ruta, "DF2.csv")),
        "modelo_flete": joblib.load(os.path.join(ruta, "modelo_costoflete.sav")),
        "modelo_dias": joblib.load(os.path.join(ruta, "modelo_dias_pipeline.joblib")),
        "label_encoder": joblib.load(os.path.join(ruta, "label_encoder_dias.joblib"))
    }

# ======================== DASHBOARD ========================
with tabs[0]:
    if zip_file:
        archivos = extraer_archivos(zip_file)
        df = archivos["df1"]

        st.markdown("## 📦 Dashboard Logístico")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pedidos", f"{len(df):,}")
        col2.metric("Flete > 50%", f"{(df['costo_de_flete'] / df['precio'] > 0.5).mean() * 100:.1f}%")
        col3.metric("≥7 días antes", f"{(df['desviacion_vs_promesa'] < -7).mean() * 100:.1f}%")

        st.subheader("🌳 Treemap")
        st.plotly_chart(px.treemap(df, path=["Categoría"], values="precio", color="Categoría"), use_container_width=True)

        st.subheader("🗺️ Mapa")
        df_mapa = df.dropna(subset=["lat_cliente", "lon_cliente"])
        if not df_mapa.empty:
            st.map(df_mapa.rename(columns={"lat_cliente": "lat", "lon_cliente": "lon"})[["lat", "lon"]])
        else:
            st.warning("No hay coordenadas disponibles")

        st.subheader("📊 Entrega vs Colchón")
        df_prom = df.groupby("estado_del_cliente")[["dias_entrega", "colchon_dias"]].mean().reset_index()
        fig_bar = px.bar(df_prom, x="estado_del_cliente", y=["dias_entrega", "colchon_dias"], barmode="group")
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("Por favor sube el archivo ZIP para ver el dashboard.")

# ======================== CALCULADORA ========================
with tabs[1]:
    if zip_file:
        archivos = extraer_archivos(zip_file)
        df2 = archivos["df2"]
        modelo_flete = archivos["modelo_flete"]
        modelo_dias = archivos["modelo_dias"]
        label_encoder = archivos["label_encoder"]

        df2["orden_compra_timestamp"] = pd.to_datetime(df2["orden_compra_timestamp"])
        df2["año"] = df2["orden_compra_timestamp"].dt.year
        df2["mes"] = df2["orden_compra_timestamp"].dt.month

        estado = st.selectbox("Estado", sorted(df2["estado_del_cliente"].dropna().unique()))
        categoria = st.selectbox("Categoría", sorted(df2["Categoría"].dropna().unique()))
        meses_dict = {1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio",
                      7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"}
        col1, col2 = st.columns(2)
        mes1 = col1.selectbox("Mes 1", list(meses_dict.values()))
        mes2 = col2.selectbox("Mes 2", list(meses_dict.values()))
        m1, m2 = [k for k, v in meses_dict.items() if v == mes1][0], [k for k, v in meses_dict.items() if v == mes2][0]

        filtro = (df2["estado_del_cliente"] == estado) & (df2["Categoría"] == categoria)
        df_mes1 = df2[filtro & (df2["mes"] == m1)].copy()
        df_mes2 = df2[filtro & (df2["mes"] == m2)].copy()

        def predecir(df_input):
            if df_input.empty:
                return df_input
            cols_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                          'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                          'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago']
            X = pd.get_dummies(df_input[cols_flete], dtype=int)
            X = X.reindex(columns=modelo_flete.get_booster().feature_names, fill_value=0)
            df_input["costo_estimado"] = modelo_flete.predict(X).round(2)
            df_input["costo_de_flete"] = df_input["costo_estimado"]

            cols_dias = ['Categoría', 'categoría_peso', '#_deproductos', 'total_peso_g', 'precio', 'costo_de_flete',
                         'distancia_km', 'velocidad_kmh', 'duracion_estimada_min', 'region', 'dc_asignado',
                         'es_feriado', 'es_fin_de_semana', 'dias_promedio_ciudad', 'hora_compra',
                         'nombre_dia', 'mes', 'año', 'temp_origen', 'precip_origen', 'cloudcover_origen',
                         'conditions_origen', 'icon_origen', 'traffic', 'area']
            if not all(col in df_input.columns for col in cols_dias):
                return df_input

            pred_clase = modelo_dias.predict(df_input[cols_dias])
            df_input["clase_entrega"] = label_encoder.inverse_transform(pred_clase)
            return df_input

        def resumen(df_pred, mes_nombre):
            if df_pred.empty:
                return pd.DataFrame()
            return df_pred.groupby("ciudad_cliente").agg({
                "costo_estimado": "mean",
                "clase_entrega": lambda x: x.mode()[0] if not x.mode().empty else "N/A"
            }).reset_index().rename(columns={"costo_estimado": mes_nombre, "clase_entrega": f"Entrega {mes_nombre}"})

        df_mes1 = predecir(df_mes1)
        df_mes2 = predecir(df_mes2)

        res1 = resumen(df_mes1, mes1)
        res2 = resumen(df_mes2, mes2)

        comparacion = pd.merge(res1, res2, on="ciudad_cliente", how="outer")
        if not comparacion.empty:
            comparacion["Diferencia"] = (comparacion[mes2] - comparacion[mes1]).round(2)
            st.dataframe(comparacion)
            st.download_button("⬇️ Descargar resultados", comparacion.to_csv(index=False), "comparacion.csv", "text/csv")
        else:
            st.warning("No hay datos para mostrar. Verifica tus filtros.")
    else:
        st.warning("Primero sube el archivo ZIP para usar la calculadora.")

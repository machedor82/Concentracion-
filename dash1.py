import streamlit as st
import pandas as pd
import zipfile
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

st.title("\ud83d\udcca Panel BI")
tabs = st.tabs(["\ud83c\udfe0 Dashboard", "\ud83e\uddfc Calculadora", "\ud83d\udd27 Por definir"])

# ========================== PESTA\u00d1A 1 ==========================
with tabs[0]:
    st.subheader("\ud83d\udcc2 Cargar base de datos")
    uploaded_file = st.file_uploader("Sube un ZIP que contenga el archivo 'DF.csv'", type="zip")

    @st.cache_data
    def load_zip_csv(upload, internal_name="DF.csv"):
        with zipfile.ZipFile(upload) as z:
            with z.open(internal_name) as f:
                return pd.read_csv(f)

    df = df_filtrado = None

    if uploaded_file:
        try:
            df = load_zip_csv(uploaded_file)
            st.success("\u2705 Datos cargados exitosamente")

            # ========== FILTROS EN SIDEBAR ==========
            with st.sidebar:
                st.header("\ud83c\udf9b\ufe0f Filtros")
                clear_all = st.button("\ud83e\udea9 Quitar selección")

                categorias = df['Categoría'].dropna().unique()
                estados = df['estado_del_cliente'].dropna().unique()
                años = sorted(df['año'].dropna().unique())
                meses = sorted(df['mes'].dropna().unique())

                categoria_sel = st.multiselect("Categoría de producto", categorias, default=[] if clear_all else list(categorias))
                estado_sel = st.multiselect("Estado del cliente", estados, default=[] if clear_all else list(estados))
                año_sel = st.multiselect("Año", años, default=[] if clear_all else años)
                mes_sel = st.multiselect("Mes", meses, default=[] if clear_all else meses)

                df_filtrado = df[
                    (df['Categoría'].isin(categoria_sel)) &
                    (df['estado_del_cliente'].isin(estado_sel)) &
                    (df['año'].isin(año_sel)) &
                    (df['mes'].isin(mes_sel))
                ]

                st.markdown("---")
                st.subheader("\ud83d\udccf Filtros avanzados")
                
                min_flete, max_flete = float(df_filtrado['costo_relativo_envio'].min()), float(df_filtrado['costo_relativo_envio'].max())
                rango_flete = st.slider("Costo relativo de envío (%)", min_value=round(min_flete, 2), max_value=round(max_flete, 2), value=(round(min_flete, 2), round(max_flete, 2)))

                min_peso, max_peso = int(df_filtrado['total_peso_g'].min()), int(df_filtrado['total_peso_g'].max())
                rango_peso = st.slider("Peso total del pedido (g)", min_value=min_peso, max_value=max_peso, value=(min_peso, max_peso))

                df_filtrado = df_filtrado[
                    (df_filtrado['costo_relativo_envio'].between(*rango_flete)) &
                    (df_filtrado['total_peso_g'].between(*rango_peso))
                ]

            # ========== KPIs ==========
            st.markdown("## \ud83e\uddfd Visión General de la Operación")
            with st.container():
                col1, col2, col3 = st.columns(3)
                col1.metric("\ud83d\udce6 Total de pedidos", f"{len(df_filtrado):,}")
                pct_flete_alto = (df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100
                col2.metric("\ud83d\ude9a Flete > 50%", f"{pct_flete_alto:.1f}%")
                pct_anticipadas = (df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100
                col3.metric("\u23f1\ufe0f Entregas ≥7 días antes", f"{pct_anticipadas:.1f}%")

            # ========== GRÁFICAS ==========
            st.markdown("### \ud83d\udcca Análisis visual")

            st.subheader("\ud83c\udfe6 Top 10 centros de distribución")
            top_dc = df_filtrado['dc_asignado'].value_counts().head(10).reset_index()
            top_dc.columns = ['Centro de distribución', 'Cantidad de pedidos']
            st.plotly_chart(px.bar(top_dc, x='Cantidad de pedidos', y='Centro de distribución', orientation='h', color='Cantidad de pedidos', color_continuous_scale='Teal'), use_container_width=True)

            st.subheader("\ud83c\udf0e Pedidos por estado de destino")
            demanda_estado = df_filtrado['estado_del_cliente'].value_counts().reset_index()
            demanda_estado.columns = ['Estado', 'Cantidad de pedidos']
            st.plotly_chart(px.bar(demanda_estado, x='Cantidad de pedidos', y='Estado', orientation='h', color='Cantidad de pedidos', color_continuous_scale='Oranges'), use_container_width=True)

            st.subheader("\ud83c\udf00 Dispersión peso vs costo de flete")
            fig_scatter = px.scatter(df_filtrado, x='total_peso_g', y='costo_de_flete', color='Categoría', opacity=0.6, hover_data=['estado_del_cliente', 'precio'])
            fig_scatter.update_layout(xaxis_title="Peso total del pedido (g)", yaxis_title="Costo de flete ($)")
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("\ud83c\udf33 Treemap por categoría")
            fig_tree = px.treemap(df_filtrado, path=['Categoría'], values='precio', color='Categoría', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_tree, use_container_width=True)

            st.subheader("\ud83d\udcfd\ufe0f Mapa de entregas de clientes")
            df_mapa = df_filtrado.dropna(subset=['lat_cliente', 'lon_cliente'])
            if not df_mapa.empty:
                st.map(df_mapa.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
            else:
                st.warning("\u26a0\ufe0f No hay ubicaciones para mostrar con los filtros actuales.")

            st.download_button("\u2b07\ufe0f Descargar datos filtrados", df_filtrado.to_csv(index=False), "datos_filtrados.csv", "text/csv")

            st.markdown("### \ud83e\udd16 Modelos de predicción")
            col1, col2 = st.columns(2)
            col1.success("Modelo de clasificación de días de entrega: Accuracy ~69%, F1 ~68")
            col2.success("Modelo de regresión del flete: R² ~0.71")
            st.caption("Estos modelos pueden usarse para consolidar entregas, prevenir sobrecostos y predecir el precio antes de la compra.")

        except Exception as e:
            st.error(f"\u26a0\ufe0f Error al cargar los datos: {e}")

# ========================== PESTAÑAS 2 y 3 ==========================
with tabs[1]:
    st.subheader("\ud83e\uddfc Herramienta de C\u00e1lculo")
    st.warning("Aqu\u00ed se incluir\u00e1n funciones interactivas para c\u00e1lculos personalizados.")

with tabs[2]:
    st.subheader("\ud83d\udd27 Contenido en Desarrollo")
    st.success("Esta secci\u00f3n est\u00e1 en construcci\u00f3n. Pronto habr\u00e1 m\u00e1s.")


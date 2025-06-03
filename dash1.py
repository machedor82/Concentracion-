import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import plotly.graph_objects as go

# ========================== CONFIGURACIÓN INICIAL ==========================
st.set_page_config(page_title="Cabrito Analytics", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
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
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora", "🔧 Por definir"])

# ========================== PESTAÑA 1 ==========================
with tabs[0]:
    st.subheader("📂 Cargar base de datos")
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
            st.success("✅ Datos cargados exitosamente")

            # ========== FILTROS ==========
            with st.expander("🎛️ Filtros del dashboard", expanded=False):
                st.markdown("Selecciona los valores que quieres visualizar:")
                clear_all = st.button("🧹 Quitar toda la selección")

                categorias = df['Categoría'].dropna().unique()
                regiones = df['region'].dropna().unique()
                meses = sorted(df['mes'].dropna().unique())

                col1, col2, col3 = st.columns(3)
                with col1:
                    categoria_sel = st.multiselect("Categoría de producto", categorias, default=[] if clear_all else list(categorias))
                with col2:
                    region_sel = st.multiselect("Región", regiones, default=[] if clear_all else list(regiones))
                with col3:
                    mes_sel = st.multiselect("Mes", meses, default=[] if clear_all else meses)

                df_filtrado = df[
                    (df['Categoría'].isin(categoria_sel)) &
                    (df['region'].isin(region_sel)) &
                    (df['mes'].isin(mes_sel))
                ]

            # ========== SLIDERS ==========
            st.markdown("### 📏 Filtros avanzados")
            col1, col2 = st.columns(2)
            with col1:
                min_flete, max_flete = float(df_filtrado['costo_relativo_envio'].min()), float(df_filtrado['costo_relativo_envio'].max())
                rango_flete = st.slider("Costo relativo de envío (%)", min_value=round(min_flete, 2), max_value=round(max_flete, 2), value=(round(min_flete, 2), round(max_flete, 2)))
            with col2:
                min_peso, max_peso = int(df_filtrado['total_peso_g'].min()), int(df_filtrado['total_peso_g'].max())
                rango_peso = st.slider("Peso total del pedido (g)", min_value=min_peso, max_value=max_peso, value=(min_peso, max_peso))

            df_filtrado = df_filtrado[
                (df_filtrado['costo_relativo_envio'].between(*rango_flete)) &
                (df_filtrado['total_peso_g'].between(*rango_peso))
            ]

            # ========== KPIs ==========
            st.markdown("## 🧭 Visión General de la Operación")
            with st.container():
                st.markdown("### 🔢 Indicadores")
                col1, col2, col3 = st.columns(3)

                col1.markdown(f"""<div style='background:linear-gradient(135deg,#2196F3,#64B5F6);padding:20px;border-radius:15px;text-align:center;box-shadow:2px 2px 10px rgba(0,0,0,0.1);color:white;'><div style='font-size:24px;'>📦 Total de pedidos</div><div style='font-size:36px;font-weight:bold;'>{len(df_filtrado):,}</div></div>""", unsafe_allow_html=True)

                pct_flete_alto = (df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100
                col2.markdown(f"""<div style='background:linear-gradient(135deg,#FDD835,#FFF176);padding:20px;border-radius:15px;text-align:center;box-shadow:2px 2px 10px rgba(0,0,0,0.1);color:#333;'><div style='font-size:24px;'>🚚 Flete > 50%</div><div style='font-size:36px;font-weight:bold;'>{pct_flete_alto:.1f}%</div></div>""", unsafe_allow_html=True)

                pct_anticipadas = (df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100
                col3.markdown(f"""<div style='background:linear-gradient(135deg,#66BB6A,#A5D6A7);padding:20px;border-radius:15px;text-align:center;box-shadow:2px 2px 10px rgba(0,0,0,0.1);color:white;'><div style='font-size:24px;'>⏱️ Entregas ≥7 días antes</div><div style='font-size:36px;font-weight:bold;'>{pct_anticipadas:.1f}%</div></div>""", unsafe_allow_html=True)

            # ========== GRÁFICAS ==========
            st.markdown("### 📊 Análisis visual")

            st.subheader("📈 Evolución mensual de entregas a tiempo")
            df_filtrado["mes"] = df_filtrado["mes"].astype(str).str.zfill(2)
            df_filtrado["mes_año_dt"] = pd.to_datetime(df_filtrado["año"].astype(str) + "-" + df_filtrado["mes"])
            entregas_tiempo = df_filtrado.groupby("mes_año_dt")["entrega_a_tiempo"].mean().reset_index()
            entregas_tiempo["entrega_a_tiempo"] *= 100
            fig_line = px.line(entregas_tiempo.sort_values(by="mes_año_dt"), x="mes_año_dt", y="entrega_a_tiempo", markers=True, line_shape="spline", color_discrete_sequence=["#00bfae"])
            st.plotly_chart(fig_line, use_container_width=True)

            st.subheader("📦 Total de pedidos por año")
            pedidos_por_año = df_filtrado['año'].value_counts().sort_index().reset_index()
            pedidos_por_año.columns = ['Año', 'Cantidad de pedidos']
            st.plotly_chart(px.bar(pedidos_por_año, x='Cantidad de pedidos', y='Año', orientation='h', color='Cantidad de pedidos', color_continuous_scale='Blues'), use_container_width=True)

            st.subheader("🏭 Top 10 centros de distribución")
            top_dc = df_filtrado['dc_asignado'].value_counts().head(10).reset_index()
            top_dc.columns = ['Centro de distribución', 'Cantidad de pedidos']
            st.plotly_chart(px.bar(top_dc, x='Cantidad de pedidos', y='Centro de distribución', orientation='h', color='Cantidad de pedidos', color_continuous_scale='Teal'), use_container_width=True)

            st.subheader("🌎 Pedidos por estado de destino")
            demanda_estado = df_filtrado['estado_del_cliente'].value_counts().reset_index()
            demanda_estado.columns = ['Estado', 'Cantidad de pedidos']
            st.plotly_chart(px.bar(demanda_estado, x='Cantidad de pedidos', y='Estado', orientation='h', color='Cantidad de pedidos', color_continuous_scale='Oranges'), use_container_width=True)

           
            st.subheader("🌀 Dispersión peso vs costo de flete")
            fig_scatter = px.scatter(df_filtrado,
                                     x='total_peso_g',
                                     y='costo_de_flete',
                                     color='Categoría',
                                     opacity=0.6,
                                     hover_data=['estado_del_cliente', 'precio'])
            fig_scatter.update_layout(xaxis_title="Peso total del pedido (g)", yaxis_title="Costo de flete ($)")
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("🌳 Treemap por categoría")
            fig_tree = px.treemap(df_filtrado, path=['Categoría'], values='precio', color='Categoría', color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_tree, use_container_width=True)

            st.subheader("🧭 Gauge de entregas a tiempo (último mes)")
            if not entregas_tiempo.empty:
                gauge_value = entregas_tiempo['entrega_a_tiempo'].iloc[-1]
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=gauge_value,
                    title={'text': "📍 % Entregas a Tiempo"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#00bfae"},
                           'steps': [
                               {'range': [0, 70], 'color': "#ffcccc"},
                               {'range': [70, 90], 'color': "#fff6b3"},
                               {'range': [90, 100], 'color': "#ccffcc"}]
                          }))
                st.plotly_chart(fig_gauge, use_container_width=True)

            st.subheader("🗺️ Mapa de entregas de clientes")
            df_mapa = df_filtrado.dropna(subset=['lat_cliente', 'lon_cliente'])
            if not df_mapa.empty:
                st.map(df_mapa.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
            else:
                st.warning("⚠️ No hay ubicaciones para mostrar con los filtros actuales.")

            # ========== DESCARGA ==========
            st.download_button("⬇️ Descargar datos filtrados", df_filtrado.to_csv(index=False), "datos_filtrados.csv", "text/csv")

            # ========== MODELOS ========== 
            st.markdown("### 🤖 Modelos de predicción")
            col1, col2 = st.columns(2)
            col1.success("Modelo de clasificación de días de entrega: Accuracy ~69%, F1 ~68")
            col2.success("Modelo de regresión del flete: R² ~0.71")
            st.caption("Estos modelos pueden usarse para consolidar entregas, prevenir sobrecostos y predecir el precio antes de la compra.")

        except Exception as e:
            st.error(f"⚠️ Error al cargar los datos: {e}")

# ========================== PESTAÑAS 2 y 3 ==========================
with tabs[1]:
    st.subheader("🧮 Herramienta de Cálculo")
    st.warning("Aquí se incluirán funciones interactivas para cálculos personalizados.")

with tabs[2]:
    st.subheader("🔧 Contenido en Desarrollo")
    st.success("Esta sección está en construcción. Pronto habrá más.")


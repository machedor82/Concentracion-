import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px

# ========================== CONFIGURACIÓN INICIAL ==========================
st.set_page_config(page_title="Dashboard Empresarial", layout="wide", initial_sidebar_state="collapsed")

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

st.title("📊 Panel Empresarial")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora", "🔧 Por definir"])

# ========================== PESTAÑA 1: DASHBOARD ==========================
with tabs[0]:
    st.subheader("📂 Cargar base de datos")

    uploaded_file = st.file_uploader("Sube un ZIP que contenga el archivo 'DF.csv'", type="zip")

    @st.cache_data
    def load_zip_csv(upload, internal_name="DF.csv"):
        with zipfile.ZipFile(upload) as z:
            with z.open(internal_name) as f:
                return pd.read_csv(f)

    df = None
    df_filtrado = None

    if uploaded_file:
        try:
            df = load_zip_csv(uploaded_file)
            st.success("✅ Datos cargados exitosamente")
            st.dataframe(df.head())

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

            # ========== KPIs ==========
            st.markdown("## 🧭 Visión General de la Operación")
            with st.container():
                st.markdown("### 🔢 Indicadores clave")
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric(label="Total de pedidos", value=f"{len(df_filtrado):,}")
                with col2:
                    pct_ontime = df_filtrado['entrega_a_tiempo'].mean() * 100
                    st.metric(label="Entregas a tiempo (%)", value=f"{pct_ontime:.1f}%")
                with col3:
                    pct_flete_alto = (df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100
                    st.metric(label="Flete > 50% del producto (%)", value=f"{pct_flete_alto:.1f}%")
                with col4:
                    pct_anticipadas = (df_filtrado['desviacion_vs_promesa'] < -5).mean() * 100
                    st.metric(label="Entregas anticipadas (%)", value=f"{pct_anticipadas:.1f}%")
                with col5:
                    clientes_frecuentes = df_filtrado['cliente_id'].value_counts().gt(5).sum()
                    st.metric(label="Clientes frecuentes", value=clientes_frecuentes)

            # ========== GRÁFICAS ==========
            with st.container():
                st.markdown("### 📊 Análisis visual")
                tab1, tab2, tab3 = st.tabs(["Pedidos por Año", "Centros de Distribución", "Demanda por Estado"])

                # TAB 1: Pedidos por Año
                with tab1:
                    pedidos_por_año = df_filtrado['año'].value_counts().sort_index()
                    pedidos_df = pedidos_por_año.reset_index()
                    pedidos_df.columns = ['Año', 'Cantidad de pedidos']

                    fig1 = px.bar(
                        pedidos_df,
                        x='Cantidad de pedidos',
                        y='Año',
                        orientation='h',
                        color='Cantidad de pedidos',
                        color_continuous_scale='Blues',
                        title="📦 Total de pedidos por año"
                    )
                    fig1.update_layout(
                        xaxis_title="Cantidad de pedidos",
                        yaxis_title="Año",
                        title_x=0.2,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                # TAB 2: Centros de Distribución
                with tab2:
                    top_dc = df_filtrado['dc_asignado'].value_counts().head(10).reset_index()
                    top_dc.columns = ['Centro de distribución', 'Cantidad de pedidos']

                    fig2 = px.bar(
                        top_dc,
                        x='Cantidad de pedidos',
                        y='Centro de distribución',
                        orientation='h',
                        color='Cantidad de pedidos',
                        color_continuous_scale='Teal',
                        title="🏭 Top 10 centros de distribución"
                    )
                    fig2.update_layout(
                        xaxis_title="Cantidad de pedidos",
                        yaxis_title="Centro de distribución",
                        title_x=0.2,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=400
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # TAB 3: Demanda por Estado
                with tab3:
                    demanda_estado = df_filtrado['estado_del_cliente'].value_counts().reset_index()
                    demanda_estado.columns = ['Estado', 'Cantidad de pedidos']

                    fig3 = px.bar(
                        demanda_estado,
                        x='Cantidad de pedidos',
                        y='Estado',
                        orientation='h',
                        color='Cantidad de pedidos',
                        color_continuous_scale='Oranges',
                        title="🌎 Pedidos por estado de destino"
                    )
                    fig3.update_layout(
                        xaxis_title="Cantidad de pedidos",
                        yaxis_title="Estado",
                        title_x=0.2,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=500
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                    
            # ========== MODELOS ==========
            with st.container():
                st.markdown("### 🤖 Modelos de predicción")
                col1, col2 = st.columns(2)
                with col1:
                    st.success("Modelo de clasificación de días de entrega: Accuracy ~69%, F1 ~68")
                with col2:
                    st.success("Modelo de regresión del flete: R² ~0.71")
                st.caption("Estos modelos pueden usarse para consolidar entregas, prevenir sobrecostos y predecir el precio antes de la compra.")

        except Exception as e:
            st.error(f"⚠️ Error al cargar los datos: {e}")

# ========================== PESTAÑA 2: CALCULADORA ==========================
with tabs[1]:
    st.subheader("🧮 Herramienta de Cálculo")
    st.warning("Aquí se incluirán funciones interactivas para cálculos personalizados.")

# ========================== PESTAÑA 3: POR DEFINIR ==========================
with tabs[2]:
    st.subheader("🔧 Contenido en Desarrollo")
    st.success("Esta sección está en construcción. Pronto habrá más.")

      

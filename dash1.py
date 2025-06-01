import streamlit as st
import pandas as pd
import plotly.express as px

# ==== CONFIGURACION AVANZADA ====
st.set_page_config(
    page_title="Cabrito Analytics | Control Total",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==== ESTILO GLOBAL AVANZADO ====
st.markdown("""
    <style>
    body, .main, .block-container {
        background-color: white !important;
        color: #000000;
    }
    header, footer {
        visibility: hidden;
    }
    .metric-label, .metric-value {
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ==== CARGA DE DATOS ROBUSTA ====
@st.cache_data(show_spinner="Cargando datos logísticos...")
def load_data():
    try:
        return pd.read_csv("dfminu.csv.gz", compression="gzip")
    except Exception as e:
        st.exception(f"❌ Error al cargar datos: {e}")
        return pd.DataFrame()

df = load_data()

# ==== VERIFICACIÓN DE DATOS ====
required_cols = ['region', 'cliente', 'entrega_a_tiempo', 'desviacion_entrega',
                 'costo_relativo_envio', 'desviacion_vs_promesa',
                 'lat_origen', 'lon_origen', 'tipo_de_pago']

if not all(col in df.columns for col in required_cols):
    st.error("🚫 El archivo no contiene todas las columnas necesarias para mostrar el dashboard.")
    st.stop()

# ==== SIDEBAR INTERACTIVO ====
st.sidebar.header("🎛️ Filtros dinámicos")
regiones = st.sidebar.multiselect("Región", df['region'].dropna().unique(), default=df['region'].dropna().unique())
tipos_pago = df['tipo_de_pago'].dropna().unique()
pago_sel = st.sidebar.multiselect("Tipo de pago", tipos_pago, default=tipos_pago)

# ==== FILTRO GLOBAL ====
df_filtrado = df[df['region'].isin(regiones) & df['tipo_de_pago'].isin(pago_sel)]

# ==== TÍTULO ====
st.markdown("""
    <h1 style='text-align:center; color:#004C99; margin-bottom:0'>Cabrito Analytics</h1>
    <p style='text-align:center; color:#333; font-size:18px;'>Dashboard ejecutivo conectado • 100% optimizado</p>
""", unsafe_allow_html=True)

# ==== KPIs ====
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("📦 Pedidos", f"{df_filtrado.shape[0]:,}")
kpi2.metric("✅ A tiempo", f"{df_filtrado['entrega_a_tiempo'].mean()*100:.2f}%")
kpi3.metric("📉 Desviación prom.", f"{df_filtrado['desviacion_entrega'].mean():.2f} días")

# ==== GRÁFICAS MULTIFILA ULTRA COMPACTAS ====
col1, col2 = st.columns(2)

with col1:
    fig_region = px.bar(df_filtrado['region'].value_counts().reset_index(),
                        x='index', y='region',
                        title="Distribución por región",
                        color_discrete_sequence=['#004C99'])
    fig_region.update_layout(height=220, margin=dict(t=40, b=30))
    st.plotly_chart(fig_region, use_container_width=True)

with col2:
    fig_treemap = px.treemap(df_filtrado['cliente'].value_counts().reset_index().head(10),
                              path=['index'], values='cliente',
                              title="Top 10 Clientes",
                              color_discrete_sequence=['#AAAAAA'])
    fig_treemap.update_layout(height=220)
    st.plotly_chart(fig_treemap, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    fig_hist = px.histogram(df_filtrado, x='costo_relativo_envio', nbins=30,
                            title="Costo relativo de envío",
                            color_discrete_sequence=['#AAAAAA'])
    fig_hist.update_layout(height=220)
    st.plotly_chart(fig_hist, use_container_width=True)

with col4:
    fig_box = px.box(df_filtrado, x='desviacion_vs_promesa',
                     title="Anticipación vs promesa",
                     color_discrete_sequence=['#004C99'])
    fig_box.update_layout(height=220)
    st.plotly_chart(fig_box, use_container_width=True)

col5, col6 = st.columns(2)
with col5:
    pie_data = df_filtrado['tipo_de_pago'].value_counts().reset_index()
    pie_data.columns = ['Tipo de Pago', 'Cantidad']
    fig_pie = px.pie(pie_data, names='Tipo de Pago', values='Cantidad',
                     title="Distribución de pago",
                     color_discrete_sequence=px.colors.sequential.Blues)
    fig_pie.update_traces(textinfo='percent+label')
    fig_pie.update_layout(height=220)
    st.plotly_chart(fig_pie, use_container_width=True)

with col6:
    geo_df = df_filtrado.dropna(subset=['lat_origen', 'lon_origen'])
    fig_geo = px.scatter_geo(geo_df, lat='lat_origen', lon='lon_origen',
                             title="Origen de pedidos",
                             scope='north america',
                             color_discrete_sequence=['#004C99'],
                             opacity=0.6)
    fig_geo.update_layout(height=220)
    st.plotly_chart(fig_geo, use_container_width=True)

# ==== NOTA FINAL PEQUEÑA ====
st.caption("Versión pro optimizada en Streamlit con layout avanzado y filtros conectados.")

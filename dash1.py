import streamlit as st

st.set_page_config(page_title="Cabrito Analytics | Storytelling Logístico")

import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    try:
        return pd.read_csv("dfminu.csv.gz", compression="gzip")
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return pd.DataFrame()

df = load_data()

# ================== TÍTULO ==================
st.markdown("""
    <h2 style="margin-bottom: 0; color: #0074D9;">Cabrito Analytics</h2>
    <p style="margin-top: 0; font-size: 18px; color: #444;">Eficiencia logística sin inflar costos</p>
""", unsafe_allow_html=True)

# ================== KPIs ==================
col1, col2, col3 = st.columns(3)
col1.metric("Pedidos totales", f"{df.shape[0]:,}")
col2.metric("% Entregas a tiempo", f"{df['entrega_a_tiempo'].mean()*100:.2f}%")
col3.metric("Desviación promedio", f"{df['desviacion_entrega'].mean():.2f} días")

# ================== GRÁFICAS PRINCIPALES ==================
col4, col5 = st.columns(2)

with col4:
    region_counts = df['region'].value_counts().reset_index()
    region_counts.columns = ['Región', 'Pedidos']
    fig_barh = px.bar(region_counts, x='Pedidos', y='Región', orientation='h',
                      color='Región',
                      color_discrete_sequence=['#0074D9', '#AAAAAA'],
                      title="Distribución por región", height=300)
    fig_barh.update_layout(paper_bgcolor="white", plot_bgcolor="white", showlegend=False)
    st.plotly_chart(fig_barh, use_container_width=True)

with col5:
    if 'cliente' in df.columns:
        top_clientes = df['cliente'].value_counts().nlargest(10).reset_index()
        top_clientes.columns = ['Cliente', 'Pedidos']
        fig_treemap = px.treemap(top_clientes, path=['Cliente'], values='Pedidos',
                                 color_discrete_sequence=['#0074D9'],
                                 title="Top 10 clientes")
        fig_treemap.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig_treemap, use_container_width=True)

# ================== HISTOGRAMAS ==================
col6, col7 = st.columns(2)

with col6:
    fig_costo = px.histogram(df, x='costo_relativo_envio', nbins=40,
                             color_discrete_sequence=['#AAAAAA'],
                             title="Costo relativo de envío", height=250)
    fig_costo.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig_costo, use_container_width=True)

with col7:
    fig_anticipacion = px.histogram(df, x='desviacion_vs_promesa', nbins=40,
                                     color_discrete_sequence=['#0074D9'],
                                     title="Días de anticipación", height=250)
    fig_anticipacion.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig_anticipacion, use_container_width=True)

# ================== PIE CHART INTERACTIVO ==================
if 'tipo_de_pago' in df.columns:
    col8, col9 = st.columns([1, 1])
    with col8:
        pie_data = df['tipo_de_pago'].value_counts().reset_index()
        pie_data.columns = ['Tipo de Pago', 'Cantidad']
        fig_pie = px.pie(pie_data, names='Tipo de Pago', values='Cantidad',
                         color_discrete_sequence=px.colors.sequential.Blues,
                         title="Métodos de pago")
        fig_pie.update_traces(textinfo='percent+label')
        fig_pie.update_layout(paper_bgcolor="white", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

# ================== MAPA COMPACTO ==================
col10, _ = st.columns([2, 1])
mapa_df = df.dropna(subset=['lat_origen', 'lon_origen']).copy()
mapa_df = mapa_df[['lat_origen', 'lon_origen']].drop_duplicates().rename(
    columns={'lat_origen': 'lat', 'lon_origen': 'lon'}
)
col10.map(mapa_df, zoom=3)

import streamlit as st
st.set_page_config(page_title="Cabrito Analytics | Storytelling Logístico", layout="wide")
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

# Título compacto
st.markdown("### Cabrito Analytics | Eficiencia Logística sin Inflar Costos")
st.markdown("Camiones medio vacíos. Entregas infladas. Costos invisibles.")

# KPIs compactos
col1, col2, col3 = st.columns(3)
col1.metric("📦 Pedidos", f"{df.shape[0]:,}")
col2.metric("⏱️ A Tiempo", f"{df['entrega_a_tiempo'].mean()*100:.2f}%")
col3.metric("📉 Desviación", f"{df['desviacion_entrega'].mean():.2f} días")

# Región y cliente
col4, col5 = st.columns(2)
with col4:
    region_counts = df['region'].value_counts().reset_index()
    region_counts.columns = ['Región', 'Pedidos']
    fig_region = px.bar(region_counts, x='Región', y='Pedidos', color='Región',
                        color_discrete_sequence=['#0074D9', '#AAAAAA'], height=300,
                        title="Pedidos por región")
    fig_region.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_region, use_container_width=True)

with col5:
    if 'cliente' in df.columns:
        clientes = df['cliente'].value_counts().reset_index()
        clientes.columns = ['Cliente', 'Pedidos']
        fig_clientes = px.treemap(clientes, path=['Cliente'], values='Pedidos',
                                  color_discrete_sequence=['#0074D9'], height=300,
                                  title="Concentración por cliente")
        fig_clientes.update_layout(paper_bgcolor="white")
        st.plotly_chart(fig_clientes, use_container_width=True)

# Costo y anticipación
col6, col7 = st.columns(2)
with col6:
    fig_costo = px.histogram(df, x='costo_relativo_envio', nbins=40,
                             title="Costo relativo de envío",
                             color_discrete_sequence=['#AAAAAA'])
    fig_costo.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig_costo, use_container_width=True)

with col7:
    fig_anticipacion = px.histogram(df, x='desviacion_vs_promesa', nbins=40,
                                     title="Días de anticipación",
                                     color_discrete_sequence=['#0074D9'])
    fig_anticipacion.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    st.plotly_chart(fig_anticipacion, use_container_width=True)

# Mapa
mapa_df = df.dropna(subset=['lat_origen', 'lon_origen'])
st.map(mapa_df[['lat_origen', 'lon_origen']].rename(columns={'lat_origen': 'lat', 'lon_origen': 'lon'}))

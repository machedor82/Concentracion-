import streamlit as st
st.set_page_config(page_title="Cabrito Dashboard", layout="wide")
import pandas as pd
import plotly.express as px


@st.cache_data
def load_data():
    try:
        return pd.read_csv("dfminu.csv.gz", compression="gzip")
    except Exception as e:
        st.error(f"Error al cargar los datos: {e}")
        return pd.DataFrame()

df = load_data()

# === Sidebar: filtros sincronizados ===
st.sidebar.title("🔍 Filtros")
regiones = st.sidebar.multiselect("Región", df['region'].dropna().unique(), default=df['region'].dropna().unique())
pagos = df['tipo_de_pago'].dropna().unique() if 'tipo_de_pago' in df.columns else []
tipo_pago = st.sidebar.multiselect("Tipo de Pago", pagos, default=pagos)

df_filtrado = df[df['region'].isin(regiones)]
if 'tipo_de_pago' in df.columns:
    df_filtrado = df_filtrado[df_filtrado['tipo_de_pago'].isin(tipo_pago)]

# === Colores corporativos ===
azul = "#004C99"
gris = "#AAAAAA"

# === Título superior ===
st.markdown(f"<h3 style='text-align:center; color:{azul}; margin-bottom:0'>CABRITO ANALYTICS</h3>", unsafe_allow_html=True)

# === KPIs en fila 1 ===
k1, k2, k3 = st.columns(3)
k1.metric("📦 Pedidos", f"{df_filtrado.shape[0]:,}")
k2.metric("⏱️ A Tiempo", f"{df_filtrado['entrega_a_tiempo'].mean()*100:.2f}%")
k3.metric("📉 Desviación", f"{df_filtrado['desviacion_entrega'].mean():.2f} días")

# === Fila 2: región y cliente ===
col1, col2 = st.columns(2)
with col1:
    regiones_df = df_filtrado['region'].value_counts().reset_index()
    regiones_df.columns = ['Región', 'Pedidos']
    fig1 = px.bar(regiones_df, x='Pedidos', y='Región', orientation='h',
                  color='Región', height=200,
                  color_discrete_sequence=[azul, gris])
    fig1.update_layout(margin=dict(t=30, b=30), plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    if 'cliente' in df.columns:
        top_clientes = df_filtrado['cliente'].value_counts().head(5).reset_index()
        top_clientes.columns = ['Cliente', 'Pedidos']
        fig2 = px.treemap(top_clientes, path=['Cliente'], values='Pedidos',
                          color_discrete_sequence=[azul], height=200)
        fig2.update_layout(margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig2, use_container_width=True)

# === Fila 3: costos y mapa ===
col3, col4 = st.columns(2)
with col3:
    fig3 = px.histogram(df_filtrado, x='costo_relativo_envio', nbins=30,
                        color_discrete_sequence=[gris], title="Costo relativo", height=200)
    fig3.update_layout(margin=dict(t=30, b=30), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    mapa_df = df_filtrado.dropna(subset=['lat_origen', 'lon_origen'])
    fig4 = px.scatter_geo(mapa_df,
                          lat='lat_origen', lon='lon_origen',
                          scope='north america',
                          height=200, title="Orígenes", opacity=0.6)
    fig4.update_layout(margin=dict(t=30, b=30), geo=dict(bgcolor="white"), paper_bgcolor="white")
    st.plotly_chart(fig4, use_container_width=True)

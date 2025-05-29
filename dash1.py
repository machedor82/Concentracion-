import streamlit as st
st.set_page_config(page_title="Cabrito Analytics | Storytelling Logístico", layout="wide")
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



# Cargar los datos
@st.cache_data
def load_data():
    try:
        return pd.read_csv("dfminu.csv.gz", compression="gzip")
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return pd.DataFrame()

df = load_data()

# Título e introducción
st.markdown("## ¿Y si pudieras entregar igual de rápido… pero gastando menos?")
st.markdown("""
Camiones medio vacíos. Entregas infladas con 10 días de colchón. Costos invisibles.  
Esta historia es sobre cómo pasamos del 96% de entregas a tiempo… al 100% de eficiencia logística.
""")

# --- Fila 1: KPIs y velocímetro ---
col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

col1.metric("📦 Pedidos", f"{df.shape[0]:,}")
col2.metric("⏱️ % A Tiempo", f"{df['entrega_a_tiempo'].mean()*100:.2f}%")
col3.metric("📉 Desviación (días)", f"{df['desviacion_entrega'].mean():.2f}")

with col4:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df['entrega_a_tiempo'].mean()*100,
        title={'text': "Entrega a Tiempo"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 80], 'color': "red"},
                {'range': [80, 95], 'color': "orange"},
                {'range': [95, 100], 'color': "lightgreen"},
            ]
        }
    ))
    fig_gauge.update_layout(height=200, margin=dict(t=20, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- Fila 2: Gráficas de región y treemap de clientes ---
col5, col6 = st.columns(2)

with col5:
    region_counts = df['region'].value_counts().reset_index()
    region_counts.columns = ['Región', 'Pedidos']
    fig_region = px.bar(region_counts, x='Región', y='Pedidos', color='Región',
                        title="Pedidos por región", height=300)
    st.plotly_chart(fig_region, use_container_width=True)

with col6:
    if 'cliente' in df.columns:
        treemap_df = df['cliente'].value_counts().reset_index()
        treemap_df.columns = ['Cliente', 'Pedidos']
        fig_treemap = px.treemap(treemap_df, path=['Cliente'], values='Pedidos',
                                 title="Pedidos por cliente", height=300)
        st.plotly_chart(fig_treemap, use_container_width=True)

# --- Fila 3: Histogramas costo y anticipación ---
col7, col8 = st.columns(2)

with col7:
    fig_costo = px.histogram(df, x='costo_relativo_envio', nbins=50,
                             title="Costo relativo de envío", height=300)
    st.plotly_chart(fig_costo, use_container_width=True)

with col8:
    fig_anticipacion = px.histogram(df, x='desviacion_vs_promesa', nbins=50,
                                     title="Días de anticipación vs promesa", height=300)
    st.plotly_chart(fig_anticipacion, use_container_width=True)

# --- Fila 4: Mapa ---
st.markdown("### Origen de pedidos")
mapa_df = df.dropna(subset=['lat_origen', 'lon_origen'])
st.map(mapa_df[['lat_origen', 'lon_origen']].rename(columns={'lat_origen': 'lat', 'lon_origen': 'lon'}))

# --- Fila 5: Insights y conclusión ---
col9, col10 = st.columns(2)

with col9:
    st.markdown("### 💡 Hallazgos clave")
    st.markdown("""
    - ✅ **83%** llegan más de 5 días antes → rutas optimizables  
    - 🚫 **16%** tienen **flete > 50%** del valor del producto  
    - 📦 **25%** de los días: camiones medio vacíos  
    - 🔁 Solo **10 clientes** han pedido más de 5 veces  
    """)

with col10:
    st.markdown("### 🧠 De la predicción… a la planeación")
    st.markdown("Ya cumplen. Ahora toca optimizar.")
    st.markdown("> No venimos a ofrecer velocidad. Venimos a ofrecer **control**.")
    st.button("Solicitar demo del modelo 📬")

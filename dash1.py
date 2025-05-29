import streamlit as st
st.set_page_config(page_title="Cabrito Analytics | Storytelling Logístico", layout="wide")
import pandas as pd
import plotly.express as px

# Configuración inicial

st.title("¿Y si pudieras entregar igual de rápido… pero gastando menos?")

# Cargar los datos
@st.cache_data
def load_data():
    return pd.read_csv("dfminu.csv.gz", compression="gzip")


df = load_data()

# Introducción
st.markdown("""
Camiones medio vacíos. Entregas infladas con 10 días de colchón. Costos invisibles.
Esta historia es sobre cómo pasamos del 96% de entregas a tiempo… al 100% de eficiencia logística.
""")

# Sección 1: Métricas Generales
st.header("🔎 Evolución de la operación logística")
col1, col2, col3 = st.columns(3)
col1.metric("Pedidos analizados", f"{df.shape[0]:,}")
col2.metric("% Entregas a tiempo", f"{df['entrega_a_tiempo'].mean()*100:.2f}%")
col3.metric("Promedio desviación (días)", f"{df['desviacion_entrega'].mean():.2f}")

# Sección 2: Pedidos por región
st.subheader("📍 Pedidos por región")
region_counts = df['region'].value_counts().reset_index()
region_counts.columns = ['Región', 'Pedidos']
fig_region = px.bar(region_counts, x='Región', y='Pedidos', color='Región', title="Distribución de pedidos por región")
st.plotly_chart(fig_region, use_container_width=True)

# Sección 3: Costo logístico y anticipación
st.subheader("💸 Costo y anticipación logística")
col4, col5 = st.columns(2)
with col4:
    fig_costo = px.histogram(df, x='costo_relativo_envio', nbins=50, title="Distribución del costo relativo de envío")
    st.plotly_chart(fig_costo, use_container_width=True)
with col5:
    fig_anticipacion = px.histogram(df, x='desviacion_vs_promesa', nbins=50, title="Días de anticipación vs promesa")
    st.plotly_chart(fig_anticipacion, use_container_width=True)

# Sección 4: Mapa interactivo
st.subheader("🗺️ Mapa de origen de pedidos")
mapa_df = df.dropna(subset=['lat_origen', 'lon_origen'])
st.map(mapa_df[['lat_origen', 'lon_origen']].rename(columns={'lat_origen': 'lat', 'lon_origen': 'lon'}))

# Sección 5: Insights Clave
st.header("💡 Hallazgos clave")
st.markdown("""
- ✅ **83%** de los pedidos llegan más de 5 días antes → oportunidad de optimizar rutas.
- 🚫 **16%** tienen un **costo de flete > 50% del valor del producto**.
- 📦 **25%** de los días: camiones van medio vacíos.
- 🔁 Solo **10 clientes** han pedido más de 5 veces.
""")

# Sección 6: Conclusión
st.header("🧠 De la predicción… a la planeación")
st.markdown("Ya cumplen. Ahora toca optimizar.")
st.markdown("> No venimos a ofrecer velocidad. Venimos a ofrecer **control**.")

# Botón de contacto
st.button("Solicitar demo del modelo 📬")

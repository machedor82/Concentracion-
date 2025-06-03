import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Dashboard Empresarial", layout="wide", initial_sidebar_state="collapsed")

# Aplicar estilo personalizado con CSS (colores empresariales sobrios)
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

# Título general de la app
st.title("📊 Panel Empresarial")

# Crear las tres pestañas
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora", "🔧 Por definir"])

# Pestaña 1: Dashboard
with tabs[0]:
    st.subheader("Vista General de KPIs y Gráficas")
    st.info("Aquí irá el dashboard con indicadores clave, gráficos y visualizaciones.")

# Pestaña 2: Calculadora
with tabs[1]:
    st.subheader("Herramienta de Cálculo")
    st.warning("Aquí se incluirán funciones interactivas para cálculos personalizados.")

# Pestaña 3: Por definir
with tabs[2]:
    st.subheader("Contenido en Desarrollo")
    st.success("Esta sección está en construcción. Pronto habrá más.")


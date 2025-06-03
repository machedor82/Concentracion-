import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import plotly.graph_objects as go

# ========================== CONFIGURACIÓN INICIAL ==========================
st.set_page_config(page_title="Cabrito Analytics", layout="wide", initial_sidebar_state="collapsed")
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
    </style>
""", unsafe_allow_html=True)

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

            # ========== FILTROS EN SIDEBAR ==========
            with st.sidebar:
                with st.expander("🎛️ Filtros", expanded=True):
                    categorias = df['Categoría'].dropna().unique()
                    estados = df['estado_del_cliente'].dropna().unique()
                    años = sorted(df['año'].dropna().unique())
                    meses = sorted(df['mes'].dropna().unique())
            
                    categoria_sel = st.multiselect("Categoría de producto", categorias, default=list(categorias))
                    estado_sel = st.multiselect("Estado del cliente", estados, default=list(estados))
                    año_sel = st.multiselect("Año", años, default=años)
                    mes_sel = st.multiselect("Mes", meses, default=meses)
            
                with st.expander("📏 Filtros avanzados", expanded=False):
                    min_flete, max_flete = float(df['costo_relativo_envio'].min()), float(df['costo_relativo_envio'].max())
                    rango_flete = st.slider("Costo relativo de envío (%)", min_value=round(min_flete, 2), max_value=round(max_flete, 2), value=(round(min_flete, 2), round(max_flete, 2)))
            
                    min_peso, max_peso = int(df['total_peso_g'].min()), int(df['total_peso_g'].max())
                    rango_peso = st.slider("Peso total del pedido (g)", min_value=min_peso, max_value=max_peso, value=(min_peso, max_peso))
            
            # Aplicar filtros después del sidebar
            df_filtrado = df[
                (df['Categoría'].isin(categoria_sel)) &
                (df['estado_del_cliente'].isin(estado_sel)) &
                (df['año'].isin(año_sel)) &
                (df['mes'].isin(mes_sel)) &
                (df['costo_relativo_envio'].between(*rango_flete)) &
                (df['total_peso_g'].between(*rango_peso))
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
            from streamlit_plotly_events import plotly_events
            
            # Asegúrate de que df_filtrado existe
            st.markdown("### 📊 Análisis visual")
            
            with st.container():
                col1, col2 = st.columns(2)
            
                with col1:
                    st.subheader("🌳 Treemap por categoría")
            
                    # Asegurar que 'precio' esté limpio
                    df_filtrado['precio'] = pd.to_numeric(df_filtrado['precio'], errors='coerce')
            
                    fig_tree = px.treemap(
                        df_filtrado,
                        path=['Categoría'],
                        values='precio',  # usa 'precio' si quieres áreas proporcionales a monto
                        color='Categoría',
                        custom_data=['Categoría'],
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
            
                    selected = plotly_events(
                        fig_tree,
                        click_event=True,
                        hover_event=False,
                        select_event=False,
                        key="treemap_click"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
            
                with col2:
                    st.subheader("🗺️ Mapa de entregas de clientes")
            
                    if selected:
                        categoria_clic = selected[0]["customdata"][0]
                        st.caption(f"🔍 Mostrando entregas para: **{categoria_clic}**")
                        df_mapa = df_filtrado[df_filtrado["Categoría"] == categoria_clic]
                    else:
                        df_mapa = df_filtrado
            
                    df_mapa = df_mapa.dropna(subset=["lat_cliente", "lon_cliente"])
                    if not df_mapa.empty:
                        st.map(df_mapa.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
                    else:
                        st.warning("⚠️ No hay ubicaciones para mostrar con la categoría seleccionada.")
                    st.markdown("</div>", unsafe_allow_html=True)

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


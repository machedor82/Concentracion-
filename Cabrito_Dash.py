import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu  # Asegúrate de instalar streamlit-option-menu

# ------------------ Definiciones de clases/funciones personalizadas ------------------
class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def clasificar_zonas(df, estado_sel):
    if estado_sel == "Nacional":
        principales = ['Ciudad de México', 'Nuevo León', 'Jalisco']
        return df['estado_del_cliente'].apply(lambda x: x if x in principales else 'Provincia')
    else:
        top_ciudades = (
            df[df['estado_del_cliente'] == estado_sel]['ciudad_cliente']
            .value_counts()
            .nlargest(3)
            .index
            .tolist()
        )
        return df['ciudad_cliente'].apply(lambda x: x if x in top_ciudades else 'Otras')

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

st.markdown("""
    <style>
        /* Fondo general sobrio y elegante */
        .main {
            background-color: #f5f7fa !important;
            font-family: 'Segoe UI', sans-serif;
        }
        /* Texto principal en gris oscuro */
        .main > div {
            color: #1e2022 !important;
        }
        /* Estilo de métricas */
        [data-testid="stMetricLabel"] {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a73e8 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #202124 !important;
        }
        [data-testid="stMetricDelta"] {
            font-weight: bold;
            color: #34a853 !important;
        }
        /* Sidebar limpio con acento azul petróleo */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e0e0e0;
        }
        [data-testid="stSidebar"] * {
            color: #1a3c5a !important;
        }
        /* Encabezados de expander elegantes */
        .stExpander > summary {
            font-weight: 600;
            color: #1a3c5a !important;
        }
        /* Tabs refinadas */
        .stTabs [data-baseweb="tab"] {
            font-size: 15px;
            padding: 12px;
            border-bottom: 2px solid transparent;
            color: #5f6368;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #1a73e8;
            color: #1a73e8;
            font-weight: 600;
        }
        /* Ocultar watermark de Streamlit */
        .css-1wa3eu0 {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# ===================== DEFINICIÓN DE PESTAÑAS =====================
tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "App Danu 📈"] )

# ===================== SIDEBAR DE CARGA =====================
with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if not archivo_zip:
    st.warning("Por favor, sube un archivo .zip para continuar.")
    st.stop()

# ===================== CARGA DE DATOS Y MODELOS =====================
with zipfile.ZipFile(archivo_zip) as z:
    requeridos = [
        'DF.csv', 'DF2.csv',
        'modelo_costoflete.sav',
        'modelo_dias_pipeline.joblib',
        'label_encoder_dias.joblib'
    ]
    contenidos = z.namelist()
    falt = [r for r in requeridos if r not in contenidos]
    if falt:
        st.error(f"❌ Faltan archivos en el ZIP: {falt}")
        st.stop()

    df = pd.read_csv(z.open('DF.csv'))
    df2 = pd.read_csv(z.open('DF2.csv'))
    modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
    modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
    label_encoder = joblib.load(z.open('label_encoder_dias.joblib'))

# ===================== FILTRO DE ESTADO =====================
with st.sidebar:
    st.subheader("🎛️ Filtro de Estado")
    estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
    estado_sel = option_menu(
        menu_title="Selecciona un estado",
        options=estados,
        icons=["globe"] + ["geo"] * (len(estados)-1),
        default_index=0
    )

# Datos filtrados para pestañas 0 y 1
df_filtrado = df.copy() if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel].copy()

# ===================== PESTAÑA 0: Resumen Nacional =====================
with tabs[0]:
    zona = estado_sel if estado_sel != "Nacional" else "Resumen Nacional"
    st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {zona}")
    c1, c2 = st.columns(2)
    c1.metric("Pedidos", f"{len(df_filtrado):,}")
    c2.metric("Llegadas muy adelantadas (≥10 días)",
              f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean() * 100:.1f}%")
    # ... (resta de tu lógica original para Resumen Nacional) ...

# ========================= PESTAÑA 1: Costo de Envío =========================
with tabs[1]:
    c1, c2 = st.columns(2)
    c1.metric("📦 Total de Pedidos", f"{len(df_filtrado):,}")
    c2.metric("💰 Flete Alto vs Precio",
              f"{(df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100:.1f}%")
    # ... (resta de tu lógica original para Costo de Envío) ...

# ========================= PESTAÑA 2: Calculadora =========================
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")

    # Normalizar columnas de df2 para evitar KeyErrors
    import unicodedata
    def normalize(col):
        s = col.strip().lower().replace(' ', '_')
        return unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode('ascii')
    df2.columns = [normalize(c) for c in df2.columns]

    # Diccionario de meses
    meses_dict = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    # Procesar fecha y extraer año/mes
    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    # Filtro de Estado y Categoría
    estado = st.selectbox("Estado", sorted(df2['estado_del_cliente'].dropna().unique()))
    categoria = st.selectbox("Categoría", sorted(df2['categoria'].dropna().unique()))

    # Selección de Mes 1 y Mes 2
    c1, c2 = st.columns(2)
    mes1_nombre = c1.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2_nombre = c2.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1_num = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
    mes2_num = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

    # Filtrar datos por filtros
    filtro = (
        (df2['estado_del_cliente'] == estado) &
        (df2['categoria'] == categoria)
    )
    df_mes1 = df2[(df2['mes'] == mes1_num) & filtro].copy()
    df_mes2 = df2[(df2['mes'] == mes2_num) & filtro].copy()

    # Función de predicción
    def predecir(df_input):
        if df_input.empty:
            return df_input
        # Costo de flete
        cols_flete = [
            'total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min',
            'ciudad_cliente', 'nombre_dc', 'hora_compra', 'año', 'mes',
            'datetime_origen', 'region', 'dias_promedio_ciudad',
            'categoria', 'tipo_de_pago'
        ]
        df_f = df_input.reindex(columns=cols_flete).copy()
        dummies = pd.get_dummies(df_f)
        feat_names = modelo_flete.get_booster().feature_names
        dummies = dummies.reindex(columns=feat_names, fill_value=0)
        df_input['costo_estimado'] = modelo_flete.predict(dummies).round(2)
        df_input['costo_de_flete'] = df_input['costo_estimado']
        # Clase de entrega
        cols_dias = [
            'categoria', 'categoria_peso', '#_deproductos', 'total_peso_g', 'precio',
            'costo_de_flete', 'distancia_km', 'velocidad_kmh', 'duracion_estimada_min',
            'region', 'dc_asignado', 'es_feriado', 'es_fin_de_semana',
            'hora_compra', 'dias_promedio_ciudad', 'nombre_dia', 'mes', 'año',
            'traffic', 'area'
        ]
        falt = [c for c in cols_dias if c not in df_input.columns]
        if falt:
            st.error(f"Faltan columnas para predicción de entrega: {falt}")
            return df_input
        Xd = df_input[cols_dias].copy()
        preds = modelo_dias.predict(Xd)
        df_input['clase_entrega'] = label_encoder.inverse_transform(preds)
        return df_input

    # Agrupar resultados
    def agrupar(df_p, nombre):
        if 'costo_estimado' in df_p and 'clase_entrega' in df_p:
            return df_p.groupby('ciudad_cliente').agg({
                'costo_estimado': lambda x: round(x.mean(), 2),
                'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'NA'
            }).rename(columns={
                'costo_estimado': nombre,
                'clase_entrega': f"Entrega {nombre}"
            }).reset_index()
        return pd.DataFrame(columns=['ciudad_cliente', nombre, f"Entrega {nombre}"])

    # Ejecutar predicciones y agrupar
    df_mes1 = predecir(df_mes1)
    df_mes2 = predecir(df_mes2)
    res1 = agrupar(df_mes1, mes1_nombre)
    res2 = agrupar(df_mes2, mes2_nombre)
    comp = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
    comp[mes1_nombre] = pd.to_numeric(comp[mes1_nombre], errors='coerce')
    comp[mes2_nombre] = pd.to_numeric(comp[mes2_nombre], errors='coerce')
    comp['Diferencia'] = (comp[mes2_nombre] - comp[mes1_nombre]).round(2)
    comp = comp[['ciudad_cliente', mes1_nombre, mes2_nombre, 'Diferencia',
                 f"Entrega {mes1_nombre}", f"Entrega {mes2_nombre}" ]]
    comp = comp.rename(columns={'ciudad_cliente': 'Ciudad'})

    # KPIs
    avg1 = df_mes1['costo_estimado'].mean() if not df_mes1.empty else np.nan
    avg2 = df_mes2['costo_estimado'].mean() if not df_mes2.empty else np.nan
    pct = ((avg2 - avg1) / avg1 * 100) if avg1 else 0
    st.markdown("---")
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"**Costo Promedio {mes1_nombre}:**  {avg1:.2f}")
    col = 'green' if pct > 0 else 'red'
    k2.markdown(f"**% Cambio:**  <span style='color:{col}'>{pct:.2f}%</span>", unsafe_allow_html=True)
    k3.markdown(f"**Costo Promedio {mes2_nombre}:**  {avg2:.2f}")

    # Mostrar tabla comparativa
    def style_diff(v):
        if isinstance(v, (int, float)):
            return 'color: green; font-weight: bold' if v > 0 else ('color: red; font-weight: bold' if v < 0 else '')
        return ''
    st.subheader(f"Comparación: {mes1_nombre} vs {mes2_nombre}")
    st.dataframe(
        comp.style.applymap(style_diff, subset=['Diferencia']).format(precision=2),
        use_container_width=True
    )

    # Botón de descarga
    csv = comp.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Descargar CSV",
        data=csv,
        file_name=f"comparacion_{estado}_{categoria}_{mes1_nombre}vs{mes2_nombre}.csv",
        mime="text/csv"
    )

# ===================== PESTAÑA 3: App Danu =====================
with tabs[3]:
    # ... tu código original para App Danu ...
    pass

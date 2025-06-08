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

# Función para normalizar nombres de columnas
import unicodedata

def normalize(col):
    s = col.strip().lower().replace(' ', '_')
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

st.markdown("""
    <style>
        /* Estilos personalizados... */
        .css-1wa3eu0 { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# ===================== DEFINICIÓN DE PESTAÑAS =====================
tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "App Danu 📈"])

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
    requeridos = ['DF.csv', 'DF2.csv', 'modelo_costoflete.sav', 'modelo_dias_pipeline.joblib', 'label_encoder_dias.joblib']
    falt = [r for r in requeridos if r not in z.namelist()]
    if falt:
        st.error(f"❌ Faltan archivos en el ZIP: {falt}")
        st.stop()

    df = pd.read_csv(z.open('DF.csv'))
    df2 = pd.read_csv(z.open('DF2.csv'))
    modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
    modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
    label_encoder = joblib.load(z.open('label_encoder_dias.joblib'))

# Normalizar columnas para df y df2
[df.columns := [normalize(c) for c in df.columns]]
[df2.columns := [normalize(c) for c in df2.columns]]

# ===================== FILTRO DE ESTADO =====================
with st.sidebar:
    st.subheader("🎛️ Filtro de Estado")
    estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
    estado_sel = option_menu(
        menu_title="Selecciona un estado",
        options=estados,
        icons=["globe"] + ["geo"]*(len(estados)-1),
        default_index=0
    )

# Data filtrada para pestañas 0 y 1
df_filtrado = df.copy() if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel].copy()

# ===================== PESTAÑA 0: Resumen Nacional =====================
with tabs[0]:
    zona = estado_sel if estado_sel != "Nacional" else "Resumen Nacional"
    st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {zona}")
    c1, c2 = st.columns(2)
    c1.metric("Pedidos", f"{len(df_filtrado):,}")
    c2.metric("Llegadas muy adelantadas (≥10 días)", f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean()*100:.1f}%")

    if 'dias_entrega' in df_filtrado.columns:
        col1, col2 = st.columns(2)
        # Dona: Pedidos por zona
        with col1:
            df_tmp = df_filtrado.copy()
            df_tmp['zona_entrega'] = classify_zones(df_tmp, estado_sel)
            cnt = df_tmp['zona_entrega'].value_counts().reset_index()
            cnt.columns = ['zona', 'pedidos']
            fig = px.pie(cnt, names='zona', values='pedidos', hole=0.4, title="📍 Pedidos por Zona")
            st.plotly_chart(fig, use_container_width=True)
        # Barras: Entregas a tiempo vs tardías
        with col2:
            df_tmp = df_filtrado.copy()
            df_tmp['zona_entrega'] = classify_zones(df_tmp, estado_sel)
            df_tmp['estatus'] = df_tmp['llego_tarde'].map({0:'A tiempo',1:'Tardío'})
            cz = df_tmp.groupby(['zona_entrega','estatus']).size().reset_index(name='conteo')
            cz['porcentaje'] = cz['conteo']/cz.groupby('zona_entrega')['conteo'].transform('sum')*100
            fig = px.bar(cz, x='zona_entrega', y='porcentaje', color='estatus', barmode='stack', title="🚚 Entregas A Tiempo vs Tardías")
            st.plotly_chart(fig, use_container_width=True)

# ===================== PESTAÑA 1: Costo de Envío =====================
with tabs[1]:
    c1, c2 = st.columns(2)
    c1.metric("📦 Total de Pedidos", f"{len(df_filtrado):,}")
    c2.metric("💰 Flete Alto vs Precio", f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%")
    st.subheader("💸 Relación Envío–Precio")
    df_precio = df_filtrado.copy()
    df_precio['porcentaje_flete'] = df_precio['costo_de_flete']/df_precio['precio']*100
    tabla = df_precio.groupby('categoria')['porcentaje_flete'].mean().reset_index().sort_values('porcentaje_flete',ascending=False)
    tabla['porcentaje'] = tabla['porcentaje_flete'].apply(lambda x: f"🔺 {x:.1f}%" if x>=40 else f"{x:.1f}%")
    st.table(tabla[['categoria','porcentaje']])

# ===================== PESTAÑA 2: Calculadora =====================
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")
    # Procesar df2 fecha
    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month
    est = st.selectbox("Estado", sorted(df2['estado_del_cliente'].unique()))
    cat = st.selectbox("Categoría", sorted(df2['categoria'].unique()))
    c1, c2 = st.columns(2)
    m1_n = c1.selectbox("Mes 1", list(meses_dict.values()))
    m2_n = c2.selectbox("Mes 2", list(meses_dict.values()), index=1)
    m1 = [k for k,v in meses_dict.items() if v==m1_n][0]
    m2 = [k for k,v in meses_dict.items() if v==m2_n][0]
    fil = (df2['estado_del_cliente']==est)&(df2['categoria']==cat)
    d1, d2 = df2[fil & (df2['mes']==m1)].copy(), df2[fil & (df2['mes']==m2)].copy()
    def predecir(d):
        if d.empty: return d
        cols_f = ['total_peso_g','precio','#_deproductos','duracion_estimada_min','ciudad_cliente','nombre_dc','hora_compra','año','mes','datetime_origen','region','dias_promedio_ciudad','categoria','tipo_de_pago']
        df_f = d.reindex(columns=cols_f)
        d_enc = pd.get_dummies(df_f)
        feats = modelo_flete.get_booster().feature_names
        d_enc = d_enc.reindex(columns=feats,fill_value=0)
        d['costo_estimado'] = modelo_flete.predict(d_enc).round(2)
        d['costo_de_flete'] = d['costo_estimado']
        cols_d = ['categoria','categoria_peso','#_deproductos','total_peso_g','precio','costo_de_flete','distancia_km','velocidad_kmh','duracion_estimada_min','region','dc_asignado','es_feriado','es_fin_de_semana','hora_compra','dias_promedio_ciudad','nombre_dia','mes','año','traffic','area']
        if not all(c in d.columns for c in cols_d): return d
        Xd = d[cols_d]
        d['clase_entrega'] = label_encoder.inverse_transform(modelo_dias.predict(Xd))
        return d
    r1, r2 = agrupar(predecir(d1), m1_n), agrupar(predecir(d2), m2_n)
    comp = pd.merge(r1,r2,on='ciudad_cliente',how='outer')
    comp[m1_n] = pd.to_numeric(comp[m1_n],errors='coerce')
    comp[m2_n] = pd.to_numeric(comp[m2_n],errors='coerce')
    comp['Diferencia'] = (comp[m2_n]-comp[m1_n]).round(2)
    st.dataframe(comp)

# ===================== PESTAÑA 3: App Danu =====================
with tabs[3]:
    pass

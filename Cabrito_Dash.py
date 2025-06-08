import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu
import unicodedata

# ------------------ Función para normalizar nombres de columnas ------------------
def normalize(col):
    s = col.strip().lower().replace(' ', '_')
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

st.markdown("""
<style>
  /* Ocultar watermark de Streamlit */
  .css-1wa3eu0 { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ===================== DEFINICIÓN DE PESTAÑAS =====================
tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "App Danu 📈"])

# ===================== SIDEBAR Y CARGA DE ZIP =====================
with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if not archivo_zip:
    st.warning("Por favor, sube un archivo .zip para continuar.")
    st.stop()

with zipfile.ZipFile(archivo_zip) as z:
    files = z.namelist()
    required = ['DF.csv', 'DF2.csv', 'modelo_costoflete.sav', 'modelo_dias_pipeline.joblib', 'label_encoder_dias.joblib']
    missing = [f for f in required if f not in files]
    if missing:
        st.error(f"❌ Faltan archivos en el ZIP: {missing}")
        st.stop()
    df = pd.read_csv(z.open('DF.csv'))
    df2 = pd.read_csv(z.open('DF2.csv'))
    modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
    modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
    label_encoder = joblib.load(z.open('label_encoder_dias.joblib'))

# Normalizar columnas de df y df2
for frame in (df, df2):
    frame.columns = [normalize(c) for c in frame.columns]

# ===================== FILTRO DE ESTADO =====================
with st.sidebar:
    st.subheader("🎛️ Filtro de Estado")
    estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique())
    estado_sel = option_menu("Selecciona un estado", estados, icons=["globe"] + ["geo"]*(len(estados)-1), default_index=0)

# Aplicar filtro a df para pestañas 0 y 1
df_filtrado = df.copy() if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel]

# ------------------ Función zonas para tab0 ------------------
def clasificar_zonas(df_zone, sel):
    if sel == "Nacional":
        top_states = ['Ciudad de México', 'Nuevo León', 'Jalisco']
        return df_zone['estado_del_cliente'].apply(lambda x: x if x in top_states else 'Provincia')
    else:
        top_cities = df_zone[df_zone['estado_del_cliente'] == sel]['ciudad_cliente'].value_counts().nlargest(3).index.tolist()
        return df_zone['ciudad_cliente'].apply(lambda x: x if x in top_cities else 'Otras')

# ===================== PESTAÑA 0: Resumen Nacional =====================
with tabs[0]:
    title_zone = estado_sel if estado_sel != "Nacional" else "Resumen Nacional"
    st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {title_zone}")
    col1, col2 = st.columns(2)
    col1.metric("Pedidos", f"{len(df_filtrado):,}")
    col2.metric("Llegadas muy adelantadas (≥10 días)", f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean()*100:.1f}%")
    if 'dias_entrega' in df_filtrado:
        # Dona: Pedidos por Zona
        c1, c2 = st.columns(2)
        with c1:
            temp = df_filtrado.copy()
            temp['zona'] = clasificar_zonas(temp, estado_sel)
            cnt = temp['zona'].value_counts().reset_index()
            cnt.columns = ['zona', 'pedidos']
            fig = px.pie(cnt, names='zona', values='pedidos', hole=0.4, title="📍 Pedidos por Zona")
            st.plotly_chart(fig, use_container_width=True)
        # Barras: Entregas A tiempo vs Tardías
        with c2:
            temp = df_filtrado.copy()
            temp['zona'] = clasificar_zonas(temp, estado_sel)
            temp['estatus'] = temp['llego_tarde'].map({0: 'A tiempo', 1: 'Tardío'})
            grp = temp.groupby(['zona', 'estatus']).size().reset_index(name='count')
            grp['percent'] = grp['count'] / grp.groupby('zona')['count'].transform('sum') * 100
            fig = px.bar(grp, x='zona', y='percent', color='estatus', barmode='stack', title="🚚 Entregas A Tiempo vs Tardías")
            st.plotly_chart(fig, use_container_width=True)

# ===================== PESTAÑA 1: Costo de Envío =====================
with tabs[1]:
    col1, col2 = st.columns(2)
    col1.metric("📦 Total de Pedidos", f"{len(df_filtrado):,}")
    col2.metric("💰 Flete Alto vs Precio", f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%")
    st.subheader("💸 Relación Envío–Precio: ¿Gasto Justificado?")
    temp = df_filtrado.copy()
    temp['porcentaje_flete'] = temp['costo_de_flete'] / temp['precio'] * 100
    # Asegurar que 'categoria' existe
    if 'categoria' in temp.columns:
        tbl = temp.groupby('categoria')['porcentaje_flete'].mean().reset_index().sort_values('porcentaje_flete', ascending=False)
        tbl['display'] = tbl['porcentaje_flete'].apply(lambda v: f"🔺 {v:.1f}%" if v>=40 else f"{v:.1f}%")
        st.table(tbl[['categoria','display']].rename(columns={'display':'% Flete'}))
    else:
        st.error("No se encontró la columna 'categoria' en los datos.")

# ===================== PESTAÑA 2: Calculadora =====================
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")
    # Procesar fechas en df2
    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'], errors='coerce')
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month
    # Select Estado y Categoría
    est = st.selectbox("Estado", sorted(df2['estado_del_cliente'].dropna().unique()))
    cat = st.selectbox("Categoría", sorted(df2['categoria'].dropna().unique()))
    # Selección de meses
    meses = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    c1, c2 = st.columns(2)
    m1n = c1.selectbox("Mes 1", list(meses.values()), index=0)
    m2n = c2.selectbox("Mes 2", list(meses.values()), index=1)
    m1 = [k for k,v in meses.items() if v==m1n][0]
    m2 = [k for k,v in meses.items() if v==m2n][0]
    # Filtrar
    f = (df2['estado_del_cliente']==est) & (df2['categoria']==cat)
    d1 = df2[f & (df2['mes']==m1)].copy()
    d2 = df2[f & (df2['mes']==m2)].copy()
    # Funciones para predecir y agrupar
    def predecir(d):
        if d.empty: return d
        cf = ['total_peso_g','precio','#_deproductos','duracion_estimada_min','ciudad_cliente','nombre_dc','hora_compra','año','mes','datetime_origen','region','dias_promedio_ciudad','categoria','tipo_de_pago']
        df_f = d.reindex(columns=cf).copy()
        df_enc = pd.get_dummies(df_f)
        feats = modelo_flete.get_booster().feature_names
        df_enc = df_enc.reindex(columns=feats, fill_value=0)
        d['costo_estimado'] = modelo_flete.predict(df_enc).round(2)
        d['costo_de_flete'] = d['costo_estimado']
        cd = ['categoria','categoria_peso','#_deproductos','total_peso_g','precio','costo_de_flete','distancia_km','velocidad_kmh','duracion_estimada_min','region','dc_asignado','es_feriado','es_fin_de_semana','hora_compra','dias_promedio_ciudad','nombre_dia','mes','año','traffic','area']
        if not all(col in d.columns for col in cd): return d
        Xd = d[cd]
        d['clase_entrega'] = label_encoder.inverse_transform(modelo_dias.predict(Xd))
        return d
    def agrupar(d, name):
        if 'costo_estimado' not in d or 'clase_entrega' not in d:
            return pd.DataFrame(columns=['ciudad_cliente', name, f"Entrega {name}"])
        g = d.groupby('ciudad_cliente').agg({
            'costo_estimado': lambda x: round(x.mean(),2),
            'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'NA'
        }).reset_index()
        g = g.rename(columns={'costo_estimado': name, 'clase_entrega': f"Entrega {name}"})
        return g
    r1 = agrupar(predecir(d1), m1n)
    r2 = agrupar(predecir(d2), m2n)
    comp = pd.merge(r1, r2, on='ciudad_cliente', how='outer')
    # Asegurar numérico y diferencia
    for col in (m1n, m2n):
        comp[col] = pd.to_numeric(comp[col], errors='coerce')
    comp['Diferencia'] = (comp[m2n] - comp[m1n]).round(2)
    comp = comp.rename(columns={'ciudad_cliente':'Ciudad'})
    # Mostrar tabla
    st.subheader(f"Comparación: {m1n} vs {m2n}")
    st.dataframe(comp, use_container_width=True)
    # Descargar
    st.download_button("⬇️ Descargar CSV", data=comp.to_csv(index=False).encode('utf-8'), file_name='comparacion.csv', mime='text/csv')

# ===================== PESTAÑA 3: App Danu =====================
with tabs[3]:
    pass

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu
import unicodedata

# Función para normalizar nombres de columnas

def normalize(col):
    s = col.strip().lower().replace(' ', '_')
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

# Configuración de página
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

st.markdown("""
<style>
  /* Ocultar watermark de Streamlit */
  .css-1wa3eu0 { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Definición de pestañas
tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "App Danu 📈"])

# Sidebar: subir CSV de datos
st.sidebar.header("Sube tu CSV de pedidos")
csv_file = st.sidebar.file_uploader("Selecciona un archivo .csv", type=["csv"] )
if not csv_file:
    st.warning("Por favor, sube un archivo .csv para continuar.")
    st.stop()

# Cargar datos y modelos
df = pd.read_csv(csv_file)
df.columns = [normalize(c) for c in df.columns]

modelo_flete = joblib.load('modelo_costoflete.sav')
modelo_dias = joblib.load('modelo_dias_pipeline_70.joblib')
label_encoder = joblib.load('label_encoder_dias_70.joblib')

# Filtro de estado
with st.sidebar:
    st.subheader("🎛️ Filtro de Estado")
    estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique())
    estado_sel = option_menu("Selecciona un estado", estados,
                            icons=["globe"] + ["geo"]*(len(estados)-1),
                            default_index=0)
# Filtrar df para pestañas 0 y 1
df_filtrado = df.copy() if estado_sel == "Nacional" else df[df['estado_del_cliente']==estado_sel].copy()

# Función para clasificar zonas en pestaña 0
def clasificar_zonas(df_zone, sel):
    if sel == "Nacional":
        top_states = ['ciudad_de_mexico','nuevo_leon','jalisco']
        return df_zone['estado_del_cliente'].apply(lambda x: x if normalize(x) in top_states else 'Provincia')
    else:
        top_cities = df_zone[df_zone['estado_del_cliente']==sel]['ciudad_cliente'].value_counts().nlargest(3).index.tolist()
        return df_zone['ciudad_cliente'].apply(lambda x: x if x in top_cities else 'Otras')

# Pestaña 0: Resumen Nacional
with tabs[0]:
    title_zone = estado_sel if estado_sel != "Nacional" else "Resumen Nacional"
    st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {title_zone}")
    c1, c2 = st.columns(2)
    c1.metric("Pedidos", f"{len(df_filtrado):,}")
    if 'desviacion_vs_promesa' in df_filtrado:
        c2.metric("Llegadas muy adelantadas (≥10 días)", f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean()*100:.1f}%")
    if 'dias_entrega' in df_filtrado:
        col1, col2 = st.columns(2)
        with col1:
            tmp = df_filtrado.copy()
            tmp['zona'] = clasificar_zonas(tmp, estado_sel)
            cnt = tmp['zona'].value_counts().reset_index()
            cnt.columns = ['zona','pedidos']
            fig = px.pie(cnt, names='zona', values='pedidos', hole=0.4, title="📍 Pedidos por Zona")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            tmp = df_filtrado.copy()
            tmp['zona'] = clasificar_zonas(tmp, estado_sel)
            if 'llego_tarde' in tmp:
                tmp['estatus'] = tmp['llego_tarde'].map({0:'A tiempo',1:'Tardío'})
                grp = tmp.groupby(['zona','estatus']).size().reset_index(name='count')
                grp['percent'] = grp['count']/grp.groupby('zona')['count'].transform('sum')*100
                fig = px.bar(grp, x='zona', y='percent', color='estatus', barmode='stack', title="🚚 Entregas A Tiempo vs Tardías")
                st.plotly_chart(fig, use_container_width=True)

# Pestaña 1: Costo de Envío
with tabs[1]:
    c1, c2 = st.columns(2)
    c1.metric("📦 Total de Pedidos", f"{len(df_filtrado):,}")
    if all(col in df_filtrado.columns for col in ['costo_de_flete','precio']):
        c2.metric("💰 Flete Alto vs Precio", f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%")
        st.subheader("💸 Relación Envío–Precio: ¿Gasto Justificado?")
        tmp = df_filtrado.copy()
        tmp['porcentaje_flete'] = tmp['costo_de_flete']/tmp['precio']*100
        if 'categoria' in tmp:
            tbl = tmp.groupby('categoria')['porcentaje_flete'].mean().reset_index().sort_values('porcentaje_flete',ascending=False)
            tbl['display'] = tbl['porcentaje_flete'].apply(lambda v: f"🔺 {v:.1f}%" if v>=40 else f"{v:.1f}%")
            st.table(tbl[['categoria','display']].rename(columns={'display':'% Flete'}))
        else:
            st.error("No se encontró 'categoria' en los datos.")
    else:
        st.error("Columnas 'costo_de_flete' o 'precio' faltantes.")

# Pestaña 2: Calculadora de Predicción
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")
    # Preparar filtro
    if 'orden_compra_timestamp' in df:
        df['orden_compra_timestamp'] = pd.to_datetime(df['orden_compra_timestamp'], errors='coerce')
        df['año'] = df['orden_compra_timestamp'].dt.year
        df['mes'] = df['orden_compra_timestamp'].dt.month
    est = st.selectbox("Estado", sorted(df['estado_del_cliente'].dropna().unique()))
    cat = st.selectbox("Categoría", sorted(df['categoria'].dropna().unique()))
    meses = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    m1n = st.selectbox("Mes 1", list(meses.values()))
    m2n = st.selectbox("Mes 2", list(meses.values()), index=1)
    m1 = [k for k,v in meses.items() if v==m1n][0]
    m2 = [k for k,v in meses.items() if v==m2n][0]
    df1 = df[(df['mes']==m1) & (df['estado_del_cliente']==est) & (df['categoria']==cat)].copy()
    df2 = df[(df['mes']==m2) & (df['estado_del_cliente']==est) & (df['categoria']==cat)].copy()

    def predecir(d):
        if d.empty: return d
        cols = ['total_peso_g','precio','#_deproductos','duracion_estimada_min','ciudad_cliente','nombre_dc','hora_compra','año','mes','datetime_origen','region','dias_promedio_ciudad','categoria','tipo_de_pago']
        d_f = d.reindex(columns=cols).copy()
        enc = pd.get_dummies(d_f)
        feats = modelo_flete.get_booster().feature_names
        enc = enc.reindex(columns=feats, fill_value=0)
        d['costo_estimado'] = modelo_flete.predict(enc).round(2)
        d['costo_de_flete'] = d['costo_estimado']
        return d

    def agrupar(d, name):
        if 'costo_estimado' not in d:
            return pd.DataFrame(columns=['ciudad_cliente', name])
        g = d.groupby('ciudad_cliente')['costo_estimado'].mean().round(2).reset_index().rename(columns={'costo_estimado':name})
        return g

    r1 = agrupar(predecir(df1), m1n)
    r2 = agrupar(predecir(df2), m2n)
    comp = pd.merge(r1, r2, on='ciudad_cliente', how='outer')
    comp[m1n] = pd.to_numeric(comp[m1n], errors='coerce')
    comp[m2n] = pd.to_numeric(comp[m2n], errors='coerce')
    comp['Diferencia'] = (comp[m2n] - comp[m1n]).round(2)
    comp = comp.rename(columns={'ciudad_cliente':'Ciudad'})
    st.subheader(f"Comparación: {m1n} vs {m2n}")
    st.dataframe(comp, use_container_width=True)
    st.download_button("⬇️ Descargar CSV", data=comp.to_csv(index=False).encode('utf-8'), file_name='comparacion.csv', mime='text/csv')

# Pestaña 3: App Danu
with tabs[3]:
    pass

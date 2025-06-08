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
# Filtrar df
if estado_sel == "Nacional":
    df_filtrado = df.copy()
else:
    df_filtrado = df[df['estado_del_cliente']==estado_sel].copy()

# Función para clasificar zonas en pestaña 0
def clasificar_zonas(df_zone, sel):
    if sel == "Nacional":
        principales = ['ciudad_de_mexico','nuevo_leon','jalisco']
        return df_zone['estado_del_cliente'].apply(lambda x: x if normalize(x) in principales else 'Provincia')
    else:
        top_ciudades = df_zone[df_zone['estado_del_cliente']==sel]['ciudad_cliente'].value_counts().nlargest(3).index.tolist()
        return df_zone['ciudad_cliente'].apply(lambda x: x if x in top_ciudades else 'Otras')

# ================= PESTAÑA 0: Resumen Nacional =================
with tabs[0]:
    title = estado_sel if estado_sel != "Nacional" else "Resumen Nacional"
    st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {title}")

    # Métricas principales
    m1, m2 = st.columns(2)
    m1.metric("Pedidos", f"{len(df_filtrado):,}")
    if 'desviacion_vs_promesa' in df_filtrado:
        m2.metric("Llegadas muy adelantadas (≥10 días)", f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean()*100:.1f}%")

    # Fila 1: Gráfica de dona y barras apiladas
    col1, col2 = st.columns(2)
    with col1:
        tmp = df_filtrado.copy()
        tmp['zona'] = clasificar_zonas(tmp, estado_sel)
        cnt = tmp['zona'].value_counts().reset_index()
        cnt.columns = ['zona', 'pedidos']
        fig1 = px.pie(cnt, names='zona', values='pedidos', hole=0.4, title="📍 Pedidos por Zona")
        fig1.update_traces(textinfo='percent+label', hovertemplate="<b>%{label}</b><br>Pedidos: %{value}<br>Porcentaje: %{percent}")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        if 'llego_tarde' in df_filtrado:
            tmp2 = df_filtrado.copy()
            tmp2['zona'] = clasific_zonas = clasificar_zonas(tmp2, estado_sel)  # remap
            tmp2['estatus'] = tmp2['llego_tarde'].map({0:'A tiempo',1:'Tardío'})
            grp = tmp2.groupby(['zona','estatus']).size().reset_index(name='count')
            grp['percent'] = grp['count']/grp.groupby('zona')['count'].transform('sum')*100
            fig2 = px.bar(grp, x='zona', y='percent', color='estatus', barmode='stack', title="🚚 Entregas A Tiempo vs Tardías")
            fig2.update_layout(xaxis_title='Zona', yaxis_title='Porcentaje (%)', legend_title='Estatus')
            st.plotly_chart(fig2, use_container_width=True)

    # Fila 2: Distribución de días y colchón por zona
    if 'dias_entrega' in df_filtrado and 'colchon_dias' in df_filtrado:
        col3, col4 = st.columns(2)
        with col3:
            tmp3 = df_filtrado.copy()
            tmp3['grupo_dias'] = pd.cut(tmp3['dias_entrega'], bins=[0,5,10,float('inf')], labels=["1-5","6-10",">10"])
            tmp3['zona'] = clasificar_zonas(tmp3, estado_sel)
            grp2 = tmp3.groupby(['zona','grupo_dias']).size().reset_index(name='count')
            grp2['percent'] = grp2['count']/grp2.groupby('zona')['count'].transform('sum')*100
            fig3 = px.bar(grp2, x='zona', y='percent', color='grupo_dias', barmode='stack', title="📦 Días de Entrega por Zona")
            fig3.update_layout(xaxis_title='Zona', yaxis_title='Porcentaje (%)', legend_title='Rango Días')
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            tmp4 = df_filtrado.copy()
            tmp4['zona'] = clasificar_zonas(tmp4, estado_sel)
            medios = tmp4.groupby('zona')[['dias_entrega','colchon_dias']].mean().reset_index()
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(y=medios['zona'], x=medios['dias_entrega'], name='Días Entrega', orientation='h'))
            fig4.add_trace(go.Bar(y=medios['zona'], x=medios['colchon_dias'], name='Colchón Días', orientation='h'))
            fig4.update_layout(barmode='group', title='📦 Días vs Colchón por Zona', xaxis_title='Promedio Días', yaxis_title='Zona')
            st.plotly_chart(fig4, use_container_width=True)

# ================= PESTAÑA 1: Costo de Envío =================
with tabs[1]:
    c1, c2 = st.columns(2)
    c1.metric("📦 Total de Pedidos", f"{len(df_filtrado):,}")
    if 'costo_de_flete' in df_filtrado and 'precio' in df_filtrado:
        c2.metric("💰 Flete Alto vs Precio", f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%")
        st.subheader("💸 Relación Envío–Precio")
        tmp = df_filtrado.copy()
        tmp['porcentaje_flete'] = tmp['costo_de_flete']/tmp['precio']*100
        if 'categoria' in tmp:
            tbl = tmp.groupby('categoria')['porcentaje_flete'].mean().reset_index().sort_values('porcentaje_flete', ascending=False)
            tbl['display'] = tbl['porcentaje_flete'].apply(lambda v: f"🔺 {v:.1f}%" if v>=40 else f"{v:.1f}%")
            st.table(tbl[['categoria','display']].rename(columns={'display':'% Flete'}))

# ================ PESTAÑA 2: Calculadora ================
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")
    if 'orden_compra_timestamp' in df:
        df['orden_compra_timestamp'] = pd.to_datetime(df['orden_compra_timestamp'], errors='coerce')
        df['año'] = df['orden_compra_timestamp'].dt.year
        df['mes'] = df['orden_compra_timestamp'].dt.month
    est2 = st.selectbox("Estado", sorted(df['estado_del_cliente'].dropna().unique()))
    cat2 = st.selectbox("Categoría", sorted(df['categoria'].dropna().unique()))
    meses = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    m1n2 = st.selectbox("Mes 1", list(meses.values()))
    m2n2 = st.selectbox("Mes 2", list(meses.values()), index=1)
    m1_ = [k for k,v in meses.items() if v==m1n2][0]
    m2_ = [k for k,v in meses.items() if v==m2n2][0]
    df1 = df[(df['mes']==m1_) & (df['estado_del_cliente']==est2) & (df['categoria']==cat2)].copy()
    df2b = df[(df['mes']==m2_) & (df['estado_del_cliente']==est2) & (df['categoria']==cat2)].copy()
    def predecir(df_pred):
        if df_pred.empty: return df_pred
        cols = ['total_peso_g','precio','#_deproductos','duracion_estimada_min','ciudad_cliente','nombre_dc','hora_compra','año','mes','datetime_origen','region','dias_promedio_ciudad','categoria','tipo_de_pago']
        dfp = df_pred.reindex(columns=cols).copy()
        enc = pd.get_dummies(dfp)
        feats = modelo_flete.get_booster().feature_names
        enc = enc.reindex(columns=feats, fill_value=0)
        df_pred['costo_estimado'] = modelo_flete.predict(enc).round(2)
        return df_pred
    def agr(df_pred, name):
        if 'costo_estimado' not in df_pred:
            return pd.DataFrame(columns=['ciudad_cliente', name])
        agg = df_pred.groupby('ciudad_cliente')['costo_estimado'].mean().round(2).reset_index().rename(columns={'costo_estimado':name})
        return agg
    res1 = agr(predecir(df1), m1n2)
    res2 = agr(predecir(df2b), m2n2)
    comp2 = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
    comp2[m1n2] = pd.to_numeric(comp2[m1n2], errors='coerce')
    comp2[m2n2] = pd.to_numeric(comp2[m2n2], errors='coerce')
    comp2['Diferencia'] = (comp2[m2n2] - comp2[m1n2]).round(2)
    comp2 = comp2.rename(columns={'ciudad_cliente':'Ciudad'})
    st.subheader(f"Comparación: {m1n2} vs {m2n2}")
    st.dataframe(comp2, use_container_width=True)
    st.download_button("⬇️ Descargar CSV", data=comp2.to_csv(index=False).encode('utf-8'),file_name='comparacion.csv',mime='text/csv')

# Pestaña 3: App Danu
with tabs[3]:
    pass

import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------------------
# 1) Debe ser la primera llamada Streamlit
st.set_page_config(page_title="Cabrito Analytics Profesional", layout="wide")

# ---------------------------------------------------------------------------------------
# Inyectar CSS corporativo
st.markdown(
    """
    <style>
    :root {
      --primary-blue: #003366;
      --secondary-blue: #6699cc;
      --theme-primary-color: #003366;
    }
    .stButton > button {
      background-color: var(--primary-blue) !important;
      color: white !important;
      border: 1px solid var(--primary-blue) !important;
    }
    .stButton > button:hover {
      background-color: var(--secondary-blue) !important;
    }
    [data-baseweb="tag"] {
      background-color: var(--primary-blue) !important;
      color: white !important;
    }
    button[role="tab"][aria-selected="true"] {
      color: var(--primary-blue) !important;
      border-bottom: 3px solid var(--primary-blue) !important;
      font-weight: bold;
    }
    button[role="tab"]:hover {
      color: var(--primary-blue) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📦 Cabrito Analytics App")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ------------------ Sidebar: ZIP + filtros ------------------
with st.sidebar:
    st.header("Sube tu archivo ZIP")
    archivo = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if not archivo:
    st.stop()

# Carga datos y modelos desde ZIP
with zipfile.ZipFile(archivo) as z:
    df = pd.read_csv(z.open('DF.csv'))
    df2 = pd.read_csv(z.open('DF2.csv'))
    modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
    modelo_dias   = joblib.load(z.open('modelo_dias_pipeline.joblib'))
    encoder       = joblib.load(z.open('label_encoder_dias.joblib'))

# --------- Dashboard ---------
with tabs[0]:
    st.header("🏠 Dashboard Logístico")
    st.markdown("**Desfase estimado vs real de entrega**  \nFiltra y analiza sin scroll.")

    # Filtros
    estados    = sorted(df['estado_del_cliente'].dropna().unique())
    categorias = sorted(df['Categoría'].dropna().unique())
    if 'sel_est' not in st.session_state: st.session_state.sel_est = estados.copy()
    if 'sel_cat' not in st.session_state: st.session_state.sel_cat = categorias.copy()
    if st.sidebar.button("Seleccionar todo (Estados)", key="e"): st.session_state.sel_est = estados.copy()
    sel_est = st.sidebar.multiselect("Estados", estados, default=st.session_state.sel_est, key='sel_est')
    if st.sidebar.button("Seleccionar todo (Categorías)", key="c"): st.session_state.sel_cat = categorias.copy()
    sel_cat = st.sidebar.multiselect("Categorías", categorias, default=st.session_state.sel_cat, key='sel_cat')

    d = df[df['estado_del_cliente'].isin(sel_est) & df['Categoría'].isin(sel_cat)].copy()
    d['prometido_dias'] = d['dias_entrega'] - d['desviacion_vs_promesa']

    # KPIs
    a1, a2, a3 = st.columns(3)
    a1.metric("Estimado promedio (d)", f"{d['prometido_dias'].mean():.1f}")
    a2.metric("Real promedio (d)",    f"{d['dias_entrega'].mean():.1f}")
    a3.metric("Desfase promedio (d)", f"{(d['prometido_dias'].mean()-d['dias_entrega'].mean()):.1f}")

    # Gráficas en fila
    cat_agg = d.groupby('Categoría').agg(Estimado=('prometido_dias','mean'),Real=('dias_entrega','mean')).reset_index()
    cat_agg['Desfase'] = cat_agg['Estimado'] - cat_agg['Real']
    time_agg = d.groupby(['año','mes']).agg(Estimado=('prometido_dias','mean'),Real=('dias_entrega','mean')).reset_index()
    time_agg['Fecha'] = pd.to_datetime(dict(year=time_agg['año'],month=time_agg['mes'],day=1))

    c1, c2, c3 = st.columns(3, gap='medium')
    fig1 = px.bar(cat_agg, x='Categoría', y=['Estimado','Real'], barmode='group', color_discrete_map={'Estimado':'#003366','Real':'#6699cc'},template='plotly_white')
    fig1.update_layout(margin=dict(t=30,b=10,l=10,r=10),height=300)
    c1.plotly_chart(fig1, use_container_width=True)
    fig2 = px.line(time_agg, x='Fecha', y=['Estimado','Real'], color_discrete_map={'Estimado':'#003366','Real':'#6699cc'},template='plotly_white')
    fig2.update_layout(margin=dict(t=30,b=10,l=10,r=10),height=300)
    c2.plotly_chart(fig2, use_container_width=True)
    top10 = cat_agg.nlargest(10,'Desfase')
    fig3 = px.bar(top10, x='Categoría', y='Desfase', color_discrete_sequence=['#003366'],template='plotly_white')
    fig3.update_layout(margin=dict(t=30,b=10,l=10,r=10),height=300)
    c3.plotly_chart(fig3, use_container_width=True)

# ------- Calculadora (reemplazada) -------
with tabs[1]:
    st.header("🧮 Calculadora de Predicción")

    # Fechas
    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'], df2['mes'] = df2['orden_compra_timestamp'].dt.year, df2['orden_compra_timestamp'].dt.month

    # Selecciones
    estados2    = sorted(df2['estado_del_cliente'].dropna().unique())
    categorias2 = sorted(df2['categoria'].dropna().unique())
    e_sel = st.selectbox("Estado", estados2)
    c_sel = st.selectbox("Categoría", categorias2)
    meses = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    m1 = st.selectbox("Mes 1", list(meses.values()), index=0)
    m2 = st.selectbox("Mes 2", list(meses.values()), index=1)
    m1n = [k for k,v in meses.items() if v==m1][0]
    m2n = [k for k,v in meses.items() if v==m2][0]

    # Filtrar meses
    dfm1 = df2[(df2['mes']==m1n)&(df2['estado_del_cliente']==e_sel)&(df2['categoria']==c_sel)].copy()
    dfm2 = df2[(df2['mes']==m2n)&(df2['estado_del_cliente']==e_sel)&(df2['categoria']==c_sel)].copy()

    # Función predecir
    def predecir(df_in):
        if df_in.empty: return df_in
        cols_f = ['total_peso_g','precio','#_deproductos','duracion_estimada_min','ciudad_cliente','nombre_dc','hora_compra','año','mes','datetime_origen','region','dias_promedio_ciudad','categoria','tipo_de_pago']
        Xf = pd.get_dummies(df_in[cols_f])
        feats = modelo_flete.get_booster().feature_names
        Xf = Xf.reindex(columns=feats, fill_value=0)
        df_in['costo_estimado'] = modelo_flete.predict(Xf).round(2)
        clase = modelo_dias.predict(df_in[[c for c in df_in.columns if c in feats]])
        df_in['clase_entrega'] = encoder.inverse_transform(clase)
        return df_in

    # Agrupar
    def agrupar(df_p, name):
        if 'costo_estimado' not in df_p: return pd.DataFrame(columns=['Ciudad',name])
        out = df_p.groupby('ciudad_cliente').agg({
            'costo_estimado': lambda x: round(x.mean(),2),
            'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'NA'
        }).rename(columns={'costo_estimado':name,'clase_entrega':f"Entrega {name}"}).reset_index().rename(columns={'ciudad_cliente':'Ciudad'})
        return out

    dfp1 = predecir(dfm1)
    dfp2 = predecir(dfm2)
    r1 = agrupar(dfp1, m1)
    r2 = agrupar(dfp2, m2)
    comp = pd.merge(r1, r2, on='Ciudad', how='outer')
    comp[[m1,m2]] = comp[[m1,m2]].apply(pd.to_numeric, errors='coerce')
    comp['Diferencia'] = (comp[m2]-comp[m1]).round(2)

    # KPIs superiores
    kx1, kx2, kx3 = st.columns(3)
    avg1 = dfp1['costo_estimado'].mean() if not dfp1.empty else np.nan
    avg2 = dfp2['costo_estimado'].mean() if not dfp2.empty else np.nan
    pct = ((avg2-avg1)/avg1*100) if avg1 else 0
    kx1.metric(f"Avg {m1}", f"{avg1:.2f}")
    kx2.metric("% Cambio", f"{pct:.1f}%")
    kx3.metric(f"Avg {m2}", f"{avg2:.2f}")

    # Tabla
    def highlight(v): return 'color: green' if v>0 else ('color: red' if v<0 else '')
    st.dataframe(comp.style.applymap(highlight, subset=['Diferencia']).format(precision=2))
    st.download_button("Descargar CSV", comp.to_csv(index=False), file_name="comparacion.csv")

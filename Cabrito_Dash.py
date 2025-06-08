import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu
import plotly.graph_objects as go

# --------------------------------
# Configuración inicial y tema
st.set_page_config(page_title="Cabrito Analytics", layout="wide")
st.markdown(
    """
    <style>
        :root { --primary-blue: #003366; --secondary-blue: #6699cc; }
        .stButton > button { background-color: var(--primary-blue)!important; color:white!important; border:1px solid var(--primary-blue)!important; }
        .stButton > button:hover { background-color: var(--secondary-blue)!important; }
        [data-baseweb="tag"] { background-color:var(--primary-blue)!important; color:white!important; }
        button[role="tab"][aria-selected="true"] { color:var(--primary-blue)!important; border-bottom:3px solid var(--primary-blue)!important; font-weight:bold; }
        button[role="tab"]:hover { color:var(--primary-blue)!important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Título y pestañas
st.title("📦 Cabrito Analytics App")
tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "App Danu 📈"])

# Sidebar: carga de ZIP
tab_sidebar = st.sidebar
with tab_sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

# Detener si no hay archivo
if not archivo_zip:
    st.stop()

# Carga datos y modelos del ZIP
with zipfile.ZipFile(archivo_zip) as z:
    requeridos = ['DF.csv','DF2.csv','modelo_costoflete.sav','modelo_dias_pipeline.joblib','label_encoder_dias.joblib']
    faltantes = [f for f in requeridos if f not in z.namelist()]
    if faltantes:
        st.error(f"❌ Faltan archivos en el ZIP: {faltantes}")
        st.stop()
    df  = pd.read_csv(z.open('DF.csv'))
    df2 = pd.read_csv(z.open('DF2.csv'))
    modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
    modelo_dias   = joblib.load(z.open('modelo_dias_pipeline.joblib'))
    label_encoder = joblib.load(z.open('label_encoder_dias.joblib'))

# Sidebar: filtro de estado
with tab_sidebar:
    st.subheader("🎛️ Filtro de Estado")
    estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
    estado_sel = option_menu(
        menu_title=None,
        options=estados,
        icons=["globe"] + ["geo"] * (len(estados)-1),
        menu_icon="caret-down-fill",
        default_index=0,
        orientation="vertical"
    )

# Filtrado global
df_filtrado = df.copy() if estado_sel == "Nacional" else df[df['estado_del_cliente']==estado_sel]

# ========================= Resumen Nacional =========================
with tabs[0]:
    st.title(f"📊 Entrega vs Promesa – {estado_sel}")
    # Métricas principales
    c1, c2 = st.columns(2)
    c1.metric("Pedidos", f"{len(df_filtrado):,}")
    c2.metric("Llegadas ≥10 días antes", f"{(df_filtrado['desviacion_vs_promesa']<-10).mean()*100:.1f}%")
    # Gráficos de resumen (omitir para brevedad)

# ========================= Costo de Envío =========================
with tabs[1]:
    st.header("📦 Costo de Envío vs Precio")
    c1, c2 = st.columns(2)
    c1.metric("Total pedidos", f"{len(df_filtrado):,}")
    c2.metric("Flete >50% precio", f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%")
    # Gráficos de costo/precio (omitir para brevedad)

# ========================= Calculadora =========================
with tabs[2]:
    st.header("🧮 Calculadora de Predicción de Flete y Entrega")

    # Preprocesar timestamps
    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    # Selección de estado y categoría
    col1, col2 = st.columns(2)
    estado2    = col1.selectbox("Estado", sorted(df2['estado_del_cliente'].dropna().unique()))
    categoria2 = col2.selectbox("Categoría", sorted(df2['Categoría'].dropna().unique()))

    # Selección de meses
    meses_dict = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
                  7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    mcol1, mcol2 = st.columns(2)
    mes1_nombre = mcol1.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2_nombre = mcol2.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1 = [k for k,v in meses_dict.items() if v==mes1_nombre][0]
    mes2 = [k for k,v in meses_dict.items() if v==mes2_nombre][0]

    # Filtrar por selección
    dfm1 = df2[(df2['mes']==mes1)&(df2['estado_del_cliente']==estado2)&(df2['Categoría']==categoria2)].copy()
    dfm2 = df2[(df2['mes']==mes2)&(df2['estado_del_cliente']==estado2)&(df2['Categoría']==categoria2)].copy()

    # Función de predicción
    def predecir(df_in):
        if df_in.empty: return df_in
        cols_f = ['total_peso_g','precio','#_deproductos','duracion_estimada_min',
                  'ciudad_cliente','nombre_dc','hora_compra','año','mes','datetime_origen',
                  'region','dias_promedio_ciudad','Categoría','tipo_de_pago']
        Xf = pd.get_dummies(df_in[cols_f])
        feats = modelo_flete.get_booster().feature_names
        Xf = Xf.reindex(columns=feats, fill_value=0)
        df_in['costo_estimado'] = modelo_flete.predict(Xf).round(2)
        df_in['costo_de_flete'] = df_in['costo_estimado']
        # Días de entrega
        dias_feat = [c for c in df_in.columns if c in modelo_dias.feature_names_in_]
        if dias_feat:
            preds = modelo_dias.predict(df_in[dias_feat])
            df_in['clase_entrega'] = label_encoder.inverse_transform(preds)
        return df_in

    # Función de agrupación
    def agrupar(df_p, name):
        if 'costo_estimado' not in df_p.columns:
            return pd.DataFrame(columns=['Ciudad', name])
        out = df_p.groupby('ciudad_cliente').agg(
            **{name:('costo_estimado','mean')},
            **{f"Entrega {name}":('clase_entrega', lambda x: x.mode()[0] if not x.mode().empty else 'NA')}
        ).reset_index().rename(columns={'ciudad_cliente':'Ciudad'})
        return out

    # Ejecutar predicción y agrupación
    dfp1 = predecir(dfm1)
    dfp2 = predecir(dfm2)
    r1 = agrupar(dfp1, mes1_nombre)
    r2 = agrupar(dfp2, mes2_nombre)
    comp = pd.merge(r1, r2, on='Ciudad', how='outer')

    # Calcular diferencia
    for col in [mes1_nombre, mes2_nombre]:
        comp[col] = pd.to_numeric(comp.get(col), errors='coerce')
    comp['Diferencia'] = (comp[mes2_nombre] - comp[mes1_nombre]).round(2)

    # KPIs comparativos
    k1, k2, k3 = st.columns(3)
    avg1 = dfp1['costo_estimado'].mean() if not dfp1.empty else np.nan
    avg2 = dfp2['costo_estimado'].mean() if not dfp2.empty else np.nan
    pct  = ((avg2-avg1)/avg1*100) if avg1 else 0
    k1.metric(f"Avg {mes1_nombre}", f"{avg1:.2f}")
    k2.metric("% Cambio", f"{pct:.1f}%")
    k3.metric(f"Avg {mes2_nombre}", f"{avg2:.2f}")

    # Mostrar tabla
    def style_diff(val):
        return 'color: green' if val>0 else ('color: red' if val<0 else '')
    st.subheader(f"Comparación {mes1_nombre} vs {mes2_nombre} – {estado2}/{categoria2}")
    st.dataframe(
        comp.style.applymap(style_diff, subset=['Diferencia']).format(precision=2)
    )
    st.download_button("⬇️ Descargar CSV", comp.to_csv(index=False), file_name="calculadora_comparacion.csv")

# ========================= App Danu =========================
with tabs[3]:
    st.header("App Danu – Insights 📈")
    st.write("Aquí va contenido adicional de App Danu...")

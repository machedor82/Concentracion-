# dash2.py

import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------ Clases personalizadas ------------------
class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None): self.parametro1 = parametro1
    def fit(self, X, y=None): return self
    def transform(self, X): return X
# ---------------------------------------------------------------------------------------

# Configuración de la app y tema corporativo
st.set_page_config(page_title="Cabrito Analytics Profesional", layout="wide")
st.markdown(
    "<style>"
    ":root { --primary-blue: #003366; --secondary-blue: #6699cc; }"
    "</style>",
    unsafe_allow_html=True
)

st.title("📦 Cabrito Analytics App")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("Sube tu archivo ZIP")
    archivo = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if archivo:
    # Cargar datos y modelos
    with zipfile.ZipFile(archivo) as z:
        requeridos = ['DF.csv','DF2.csv','modelo_costoflete.sav','modelo_dias_pipeline.joblib','label_encoder_dias.joblib']
        falt = [f for f in requeridos if f not in z.namelist()]
        if falt:
            st.error(f"❌ Faltan archivos: {falt}")
            st.stop()
        df = pd.read_csv(z.open('DF.csv'))
        df2 = pd.read_csv(z.open('DF2.csv'))
        modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
        modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
        encoder = joblib.load(z.open('label_encoder_dias.joblib'))

    # Filtros
    st.sidebar.subheader("Filtros de Dashboard")
    estados = sorted(df['estado_del_cliente'].dropna().unique())
    categorias = sorted(df['Categoría'].dropna().unique())
    if 'sel_est' not in st.session_state: st.session_state.sel_est = estados.copy()
    if 'sel_cat' not in st.session_state: st.session_state.sel_cat = categorias.copy()

    if st.sidebar.button("Seleccionar todo Estados"):
        st.session_state.sel_est = estados.copy()
    sel_est = st.sidebar.multiselect(
        "Estados", estados, default=st.session_state.sel_est, key='sel_est'
    )

    if st.sidebar.button("Seleccionar todo Categorías"):
        st.session_state.sel_cat = categorias.copy()
    sel_cat = st.sidebar.multiselect(
        "Categorías", categorias, default=st.session_state.sel_cat, key='sel_cat'
    )

    # ------------------ Dashboard ------------------
    with tabs[0]:
        st.header("🏠 Dashboard Logístico")
        st.markdown(
            "**Desfase estimado vs real de entrega**  
            Analiza las métricas clave y la evolución temporal sin desplazamiento."
        )

        # Filtrado
        df_sel = df[df['estado_del_cliente'].isin(sel_est) & df['Categoría'].isin(sel_cat)].copy()
        df_sel['prometido_dias'] = df_sel['dias_entrega'] - df_sel['desviacion_vs_promesa']

        # Métricas KPI
        col1, col2, col3 = st.columns(3)
        col1.metric("Estimado promedio (d)", f"{df_sel['prometido_dias'].mean():.1f}")
        col2.metric("Real promedio (d)", f"{df_sel['dias_entrega'].mean():.1f}")
        col3.metric("Desfase promedio (d)", f"{(df_sel['prometido_dias'].mean()-df_sel['dias_entrega'].mean()):.1f}")

        # Prepara gráficas juntas
        fig_kwargs = dict(use_container_width=True)
        color_map = {'Estimado':'var(--primary-blue)','Real':'var(--secondary-blue)'}
        template = 'plotly_white'

        # Agregar agregados
        agg_cat = df_sel.groupby('Categoría').agg(
            Estimado=('prometido_dias','mean'),
            Real=('dias_entrega','mean')
        ).reset_index()
        agg_time = df_sel.groupby(['año','mes']).agg(
            Estimado=('prometido_dias','mean'),
            Real=('dias_entrega','mean')
        ).reset_index().sort_values(['año','mes'])
        agg_time['Fecha'] = pd.to_datetime(dict(year=agg_time['año'], month=agg_time['mes'], day=1))
        agg_cat['Desfase'] = agg_cat['Estimado'] - agg_cat['Real']
        top10 = agg_cat.nlargest(10,'Desfase')

        # Mostrar 3 plots en una sola fila
        p1, p2, p3 = st.columns(3)

        # Bar chart categorías
        fig1 = px.bar(
            agg_cat, x='Categoría', y=['Estimado','Real'], barmode='group',
            color_discrete_map=color_map, labels={'value':'Días','variable':'Tipo'},
            title='Estimado vs Real por Categoría', template=template
        )
        fig1.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=350)
        p1.plotly_chart(fig1, **fig_kwargs)

        # Line chart evolución mensual
        fig2 = px.line(
            agg_time, x='Fecha', y=['Estimado','Real'],
            color_discrete_map=color_map, labels={'value':'Días','variable':'Tipo'},
            title='Evolución mensual', template=template
        )
        fig2.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=350)
        p2.plotly_chart(fig2, **fig_kwargs)

        # Top10 categorías desfase
        fig3 = px.bar(
            top10, x='Categoría', y='Desfase',
            color_discrete_sequence=['var(--primary-blue)'],
            labels={'Desfase':'Días de desfase'},
            title='Top 10 desfase', template=template
        )
        fig3.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=350)
        p3.plotly_chart(fig3, **fig_kwargs)

    # ------------------ Calculadora ------------------
    with tabs[1]:
        st.header("🧮 Calculadora de Predicción")
        # Convertir timestamp
        df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
        df2['año'] = df2['orden_compra_timestamp'].dt.year
        df2['mes'] = df2['orden_compra_timestamp'].dt.month
        
        # Inputs usuario
        e2 = sorted(df2['estado_del_cliente'].dropna().unique())
        c2 = sorted(df2['Categoría'].dropna().unique())
        col4, col5 = st.columns(2)
        sel_e2 = col4.selectbox("Estado", e2)
        sel_c2 = col5.selectbox("Categoría", c2)

        meses = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
                 7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
        m1 = col4.selectbox("Mes 1", list(meses.values()), index=0)
        m2 = col5.selectbox("Mes 2", list(meses.values()), index=1)
        m1_k = [k for k,v in meses.items() if v==m1][0]
        m2_k = [k for k,v in meses.items() if v==m2][0]

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
            return df_in

        def resumen(df_pred, mes_nom):
            if 'costo_estimado' not in df_pred.columns:
                return pd.DataFrame()
            out = df_pred.groupby('ciudad_cliente').agg(
                **{mes_nom:('costo_estimado','mean')}
            ).reset_index()
            return out

        d1 = predecir(df2[(df2['estado_del_cliente']==sel_e2)&(df2['Categoría']==sel_c2)&(df2['mes']==m1_k)].copy())
        d2 = predecir(df2[(df2['estado_del_cliente']==sel_e2)&(df2['Categoría']==sel_c2)&(df2['mes']==m2_k)].copy())
        r1 = resumen(d1,m1)
        r2 = resumen(d2,m2)
        cmp = r1.merge(r2, on='ciudad_cliente')
        cmp['Diferencia'] = (cmp[m2] - cmp[m1]).round(2)
        cmp.rename(columns={'ciudad_cliente':'Ciudad'}, inplace=True)

        st.dataframe(cmp.style.format(precision=2))
        st.download_button("⬇️ Descargar CSV", cmp.to_csv(index=False), file_name="calculadora.csv")

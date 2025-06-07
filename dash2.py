# dash2.py

import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------ Inyectar CSS para branding ------------------
st.markdown(
    """
    <style>
    /* Variable de color */
    :root {
      --primary-blue: #003366;
    }

    /* Botones en el sidebar */
    .stSidebar .stButton>button {
      background-color: var(--primary-blue) !important;
      color: white !important;
      border-color: var(--primary-blue) !important;
    }
    .stSidebar .stButton>button:hover {
      background-color: #002244 !important;
    }

    /* Etiquetas de multiselect */
    .stMultiSelect .css-1gk9p0s,  /* etiqueta */
    .stMultiSelect .css-19bb58m {  /* botón de cerrar */
      background-color: var(--primary-blue) !important;
      color: white !important;
    }

    /* Pestañas: etiqueta activa */
    [data-baseweb="tab"] {
      color: #7a7a7a;
    }
    [data-baseweb="tab"][aria-selected="true"] {
      color: var(--primary-blue) !important;
      border-bottom: 3px solid var(--primary-blue) !important;
      font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Configuración de la app
st.set_page_config(page_title="Cabrito Analytics Profesional", layout="wide")
st.title("📦 Cabrito Analytics App")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ------------------ Sidebar: carga y filtros ------------------
with st.sidebar:
    st.header("Sube tu archivo ZIP")
    archivo = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if archivo:
    # Leer ZIP
    with zipfile.ZipFile(archivo) as z:
        reqs = [
            'DF.csv', 'DF2.csv',
            'modelo_costoflete.sav',
            'modelo_dias_pipeline.joblib',
            'label_encoder_dias.joblib'
        ]
        falt = [f for f in reqs if f not in z.namelist()]
        if falt:
            st.error(f"❌ Faltan archivos: {falt}")
            st.stop()
        df = pd.read_csv(z.open('DF.csv'))
        df2 = pd.read_csv(z.open('DF2.csv'))
        modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
        modelo_dias   = joblib.load(z.open('modelo_dias_pipeline.joblib'))
        encoder       = joblib.load(z.open('label_encoder_dias.joblib'))

    # Preparar filtros
    estados     = sorted(df['estado_del_cliente'].dropna().unique())
    categorias  = sorted(df['Categoría'].dropna().unique())
    if 'sel_est' not in st.session_state: st.session_state.sel_est = estados.copy()
    if 'sel_cat' not in st.session_state: st.session_state.sel_cat = categorias.copy()

    with st.sidebar:
        st.subheader("Filtros de Dashboard")
        # Botón y multiselect Estados
        if st.button("Seleccionar todo (Estados)", key="btn_est"):
            st.session_state.sel_est = estados.copy()
        sel_est = st.multiselect(
            "Estados", estados,
            default=st.session_state.sel_est,
            key="sel_est"
        )
        # Botón y multiselect Categorías
        if st.button("Seleccionar todo (Categorías)", key="btn_cat"):
            st.session_state.sel_cat = categorias.copy()
        sel_cat = st.multiselect(
            "Categorías", categorias,
            default=st.session_state.sel_cat,
            key="sel_cat"
        )

    # ========================= DASHBOARD =========================
    with tabs[0]:
        st.header("🏠 Dashboard Logístico")
        st.markdown(
            """
**Desfase estimado vs real de entrega**  
Analiza las métricas clave y la evolución temporal de forma clara.
            """
        )

        # Filtrar y calcular
        df_sel = df[
            df['estado_del_cliente'].isin(sel_est) &
            df['Categoría'].isin(sel_cat)
        ].copy()
        df_sel['prometido_dias'] = df_sel['dias_entrega'] - df_sel['desviacion_vs_promesa']

        # KPI cards
        est_mean  = df_sel['prometido_dias'].mean()
        real_mean = df_sel['dias_entrega'].mean()
        diff_mean = est_mean - real_mean
        k1, k2, k3 = st.columns(3)
        k1.metric("Estimado promedio (d)", f"{est_mean:.1f}")
        k2.metric("Real promedio (d)",    f"{real_mean:.1f}")
        k3.metric("Desfase promedio (d)", f"{diff_mean:.1f}")

        # Preparar datos agregados
        agg_cat  = df_sel.groupby('Categoría').agg(
            Estimado=('prometido_dias','mean'),
            Real=('dias_entrega','mean')
        ).reset_index()
        agg_cat['Desfase'] = agg_cat['Estimado'] - agg_cat['Real']

        agg_time = df_sel.groupby(['año','mes']).agg(
            Estimado=('prometido_dias','mean'),
            Real=('dias_entrega','mean')
        ).reset_index().sort_values(['año','mes'])
        agg_time['Fecha'] = pd.to_datetime(dict(year=agg_time['año'], month=agg_time['mes'], day=1))

        # Mostrar gráficas en una sola fila
        col1, col2, col3 = st.columns(3)

        # 1) Barras por categoría
        fig1 = px.bar(
            agg_cat, x='Categoría', y=['Estimado','Real'], barmode='group',
            color_discrete_map={'Estimado':'var(--primary-blue)','Real':'var(--secondary-blue)'},
            labels={'value':'Días','variable':'Tipo'},
            title='Estimado vs Real por Categoría',
            template='plotly_white'
        )
        med_real = agg_cat['Real'].median()
        fig1.add_hline(y=med_real, line_dash='dash', line_color='var(--secondary-blue)',
                       annotation_text='Mediana Real', annotation_position='top right')
        fig1.update_layout(margin={'l':10,'r':10,'t':35,'b':10}, height=350)
        col1.plotly_chart(fig1, use_container_width=True)

        # 2) Línea evolución mensual
        fig2 = px.line(
            agg_time, x='Fecha', y=['Estimado','Real'],
            color_discrete_map={'Estimado':'var(--primary-blue)','Real':'var(--secondary-blue)'},
            labels={'value':'Días','variable':'Tipo'},
            title='Evolución Mensual',
            template='plotly_white'
        )
        fig2.update_layout(margin={'l':10,'r':10,'t':35,'b':10}, height=350)
        col2.plotly_chart(fig2, use_container_width=True)

        # 3) Top10 desfase
        top10 = agg_cat.nlargest(10,'Desfase')
        fig3 = px.bar(
            top10, x='Categoría', y='Desfase',
            labels={'Desfase':'Días de desfase'},
            title='Top 10 Categorías con Mayor Desfase',
            color_discrete_sequence=['var(--primary-blue)'],
            template='plotly_white'
        )
        fig3.update_layout(margin={'l':10,'r':10,'t':35,'b':10}, height=350)
        col3.plotly_chart(fig3, use_container_width=True)

    # ========================= CALCULADORA =========================
    with tabs[1]:
        st.header("🧮 Calculadora de Predicción")

        # Preprocesar timestamps
        df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
        df2['año'] = df2['orden_compra_timestamp'].dt.year
        df2['mes'] = df2['orden_compra_timestamp'].dt.month

        # Inputs usuario
        estados2    = sorted(df2['estado_del_cliente'].dropna().unique())
        categorias2 = sorted(df2['Categoría'].dropna().unique())
        c1, c2 = st.columns(2)
        estado2    = c1.selectbox("Estado",    estados2)
        categoria2 = c2.selectbox("Categoría", categorias2)

        meses = {
            1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",
            5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",
            9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
        }
        mes1_nombre = c1.selectbox("Mes 1", list(meses.values()), index=0)
        mes2_nombre = c2.selectbox("Mes 2", list(meses.values()), index=1)
        mes1 = [k for k,v in meses.items() if v==mes1_nombre][0]
        mes2 = [k for k,v in meses.items() if v==mes2_nombre][0]

        # Filtrar
        filt2 = (df2['estado_del_cliente']==estado2)&(df2['Categoría']==categoria2)
        df_m1 = df2[(df2['mes']==mes1)&filt2].copy()
        df_m2 = df2[(df2['mes']==mes2)&filt2].copy()

        # Predicción
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

        def resumen(df_pred, nombre_mes):
            if 'costo_estimado' not in df_pred.columns:
                return pd.DataFrame()
            out = df_pred.groupby('ciudad_cliente').agg(**{nombre_mes:('costo_estimado','mean')}).reset_index()
            return out

        d1 = predecir(df_m1)
        d2 = predecir(df_m2)
        r1 = resumen(d1, mes1_nombre)
        r2 = resumen(d2, mes2_nombre)

        comp = r1.merge(r2, on='ciudad_cliente', how='outer')
        comp['Diferencia'] = (comp[mes2_nombre] - comp[mes1_nombre]).round(2)
        comp.rename(columns={'ciudad_cliente':'Ciudad'}, inplace=True)

        # Mostrar resultados
        st.dataframe(comp.style.format(precision=2))
        st.download_button("⬇️ Descargar CSV", comp.to_csv(index=False), file_name="comparacion.csv")


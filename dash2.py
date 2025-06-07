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
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1
    def fit(self, X, y=None): return self
    def transform(self, X): return X
# ---------------------------------------------------------------------------------------

# Configuración de la app
st.set_page_config(page_title="Cabrito Analytics Profesional", layout="wide")
st.title("📦 Cabrito Analytics App")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# Sidebar: carga de ZIP + filtros
with st.sidebar:
    st.header("Sube tu archivo ZIP")
    archivo = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if archivo:
    # Leer modelos y datos
    with zipfile.ZipFile(archivo) as z:
        necesarios = [
            'DF.csv', 'DF2.csv',
            'modelo_costoflete.sav',
            'modelo_dias_pipeline.joblib',
            'label_encoder_dias.joblib'
        ]
        faltantes = [f for f in necesarios if f not in z.namelist()]
        if faltantes:
            st.error(f"❌ Faltan archivos: {faltantes}")
            st.stop()
        df = pd.read_csv(z.open('DF.csv'))
        df2 = pd.read_csv(z.open('DF2.csv'))
        modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
        modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
        encoder = joblib.load(z.open('label_encoder_dias.joblib'))

    # Sidebar filtros dinámicos
    estados = sorted(df['estado_del_cliente'].dropna().unique())
    categorias = sorted(df['Categoría'].dropna().unique())
    if 'sel_est' not in st.session_state:
        st.session_state.sel_est = estados.copy()
    if 'sel_cat' not in st.session_state:
        st.session_state.sel_cat = categorias.copy()

    with st.sidebar:
        st.subheader("Filtros de Dashboard")

        # Estados
        if st.button("Seleccionar todo", key="btn_est"):
            st.session_state.sel_est = estados.copy()
        sel_est = st.multiselect(
            "Estados", estados,
            default=st.session_state.sel_est,
            key="sel_est"
        )

        # Categorías
        if st.button("Seleccionar todo", key="btn_cat"):
            st.session_state.sel_cat = categorias.copy()
        sel_cat = st.multiselect(
            "Categorías", categorias,
            default=st.session_state.sel_cat,
            key="sel_cat"
        )

    # ================ DASHBOARD ================
    with tabs[0]:
        st.header("🏠 Dashboard Logístico")
        st.markdown(
            """
**Desfase estimado vs real de entrega**  
Observa cómo varían los tiempos mes a mes y por categoría.
            """
        )

        # Filtrar datos
        data = df[
            df['estado_del_cliente'].isin(sel_est) &
            df['Categoría'].isin(sel_cat)
        ].copy()
        data['prometido_dias'] = data['dias_entrega'] - data['desviacion_vs_promesa']

        # Métricas clave
        est_mean  = data['prometido_dias'].mean()
        real_mean = data['dias_entrega'].mean()
        diff_mean = est_mean - real_mean
        c1, c2, c3 = st.columns(3)
        c1.metric("Estimado promedio (d)", f"{est_mean:.1f}")
        c2.metric("Real promedio (d)",    f"{real_mean:.1f}")
        c3.metric("Desfase promedio (d)", f"{diff_mean:.1f}")

        # Gráfica de barras por categoría
        cat_agg = data.groupby('Categoría').agg(
            Estimado=('prometido_dias','mean'),
            Real=('dias_entrega','mean')
        ).reset_index()
        fig1 = px.bar(
            cat_agg, x='Categoría', y=['Estimado','Real'], barmode='group',
            color_discrete_map={'Estimado':'#003366','Real':'#6699cc'},
            labels={'value':'Días','variable':'Tipo'},
            title='Tiempos estimado vs real por Categoría'
        )
        med_real = cat_agg['Real'].median()
        fig1.add_hline(
            y=med_real, line_dash='dash', line_color='#6699cc',
            annotation_text='Mediana Real', annotation_position='top right'
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Gráfica de líneas: evolución mensual
        ts = data.groupby(['año','mes']).agg(
            Estimado=('prometido_dias','mean'),
            Real=('dias_entrega','mean')
        ).reset_index().sort_values(['año','mes'])
        ts['Fecha'] = pd.to_datetime(dict(year=ts['año'], month=ts['mes'], day=1))
        fig2 = px.line(
            ts, x='Fecha', y=['Estimado','Real'],
            color_discrete_map={'Estimado':'#003366','Real':'#6699cc'},
            labels={'value':'Días','variable':'Tipo'},
            title='Evolución mensual: estimado vs real'
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Top 10 categorías con mayor desfase
        cat_agg['Desfase'] = cat_agg['Estimado'] - cat_agg['Real']
        top10 = cat_agg.nlargest(10,'Desfase')
        fig3 = px.bar(
            top10, x='Categoría', y='Desfase',
            labels={'Desfase':'Días de desfase'},
            title='Top 10 categorías con mayor desfase',
            color_discrete_sequence=['#003366']
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ================ CALCULADORA ================
    with tabs[1]:
        st.header("🧮 Calculadora de Predicción")

        # Preprocesamiento
        df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
        df2['año'] = df2['orden_compra_timestamp'].dt.year
        df2['mes'] = df2['orden_compra_timestamp'].dt.month

        # Selección de estado y categoría
        estados2    = sorted(df2['estado_del_cliente'].dropna().unique())
        categorias2 = sorted(df2['Categoría'].dropna().unique())
        col1, col2  = st.columns(2)
        estado2     = col1.selectbox("Estado",    estados2)
        categoria2  = col2.selectbox("Categoría", categorias2)

        # Selección de meses
        meses_dict = {
            1:"Enero",   2:"Febrero",  3:"Marzo",     4:"Abril",
            5:"Mayo",    6:"Junio",    7:"Julio",     8:"Agosto",
            9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
        }
        mes1_nombre = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
        mes2_nombre = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)
        mes1 = [k for k,v in meses_dict.items() if v==mes1_nombre][0]
        mes2 = [k for k,v in meses_dict.items() if v==mes2_nombre][0]

        # Filtrar y predecir
        filtro2 = (df2['estado_del_cliente']==estado2)&(df2['Categoría']==categoria2)
        df_mes1 = df2[(df2['mes']==mes1)&filtro2].copy()
        df_mes2 = df2[(df2['mes']==mes2)&filtro2].copy()

        def predecir(df_input):
            if df_input.empty: return df_input
            cols_flete = [
                'total_peso_g','precio','#_deproductos','duracion_estimada_min',
                'ciudad_cliente','nombre_dc','hora_compra','año','mes',
                'datetime_origen','region','dias_promedio_ciudad',
                'Categoría','tipo_de_pago'
            ]
            df_f  = df_input[cols_flete]
            df_enc = pd.get_dummies(df_f)
            feats  = modelo_flete.get_booster().feature_names
            df_enc = df_enc.reindex(columns=feats, fill_value=0)
            df_input['costo_estimado']   = modelo_flete.predict(df_enc).round(2)
            df_input['costo_de_flete']   = df_input['costo_estimado']

            cols_dias = [
                'Categoría','categoría_peso','#_deproductos','total_peso_g','precio',
                'costo_de_flete','distancia_km','velocidad_kmh',
                'duracion_estimada_min','region','dc_asignado','es_feriado',
                'es_fin_de_semana','dias_promedio_ciudad','hora_compra',
                'nombre_dia','mes','año','temp_origen','precip_origen',
                'cloudcover_origen','conditions_origen','icon_origen','traffic','area'
            ]
            if not all(c in df_input.columns for c in cols_dias):
                return df_input
            X     = df_input[cols_dias]
            preds = modelo_dias.predict(X)
            df_input['clase_entrega'] = encoder.inverse_transform(preds)
            return df_input

        def resumen(df_pred, nombre_mes):
            if 'costo_estimado' not in df_pred.columns or 'clase_entrega' not in df_pred.columns:
                return pd.DataFrame(columns=['ciudad_cliente', nombre_mes, f"Entrega {nombre_mes}"])
            return df_pred.groupby('ciudad_cliente').agg({
                'costo_estimado':'mean',
                'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
            }).reset_index().rename(columns={
                'costo_estimado': nombre_mes,
                'clase_entrega': f"Entrega {nombre_mes}"
            })

        df1 = predecir(df_mes1)
        df2_ = predecir(df_mes2)
        res1  = resumen(df1, mes1_nombre)
        res2  = resumen(df2_, mes2_nombre)

        # Asegurar numérico y calcular diferencia
        res1[mes1_nombre] = pd.to_numeric(res1[mes1_nombre], errors='coerce')
        res2[mes2_nombre] = pd.to_numeric(res2[mes2_nombre], errors='coerce')
        comp = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
        comp[mes1_nombre] = pd.to_numeric(comp[mes1_nombre], errors='coerce')
        comp[mes2_nombre] = pd.to_numeric(comp[mes2_nombre], errors='coerce')
        comp['Diferencia'] = (comp[mes2_nombre] - comp[mes1_nombre]).round(2)
        comp = comp[[
            'ciudad_cliente', mes1_nombre, mes2_nombre, 'Diferencia',
            f"Entrega {mes1_nombre}", f"Entrega {mes2_nombre}"
        ]].rename(columns={'ciudad_cliente': 'Ciudad'})

        # Resaltar diferencias
        def resaltar(val):
            if isinstance(val, (int, float, np.number)):
                if val > 0: return 'color: green; font-weight: bold'
                elif val < 0: return 'color: red; font-weight: bold'
            return ''

        st.subheader(f"Comparación: {mes1_nombre} vs {mes2_nombre}")
        st.dataframe(
            comp.style
                .applymap(resaltar, subset=['Diferencia'])
                .format(precision=2)
        )
        st.download_button(
            "⬇️ Descargar CSV",
            comp.to_csv(index=False),
            file_name="comparacion.csv",
            mime="text/csv"
        )

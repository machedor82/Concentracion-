# dash2.py

import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------ Definiciones de clases/funciones personalizadas ------------------
# Copia aquí las clases o funciones custom que usaste al entrenar 'modelo_dias_pipeline.joblib'.
# Deben llamarse exactamente igual que en tu script original.

class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

# ---------------------------------------------------------------------------------------

# Configuración principal
st.set_page_config(page_title="Cabrito Analytics Profesional", layout="wide")
st.title("📦 Cabrito Analytics App")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# Sidebar upload
with st.sidebar:
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if archivo_zip:
    with zipfile.ZipFile(archivo_zip) as z:
        requeridos = [
            'DF.csv', 'DF2.csv',
            'modelo_costoflete.sav',
            'modelo_dias_pipeline.joblib',
            'label_encoder_dias.joblib'
        ]
        faltantes = [r for r in requeridos if r not in z.namelist()]
        if faltantes:
            st.error(f"❌ Faltan archivos en el ZIP: {faltantes}")
            st.stop()

        df = pd.read_csv(z.open('DF.csv'))
        df2 = pd.read_csv(z.open('DF2.csv'))
        modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
        modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
        label_encoder = joblib.load(z.open('label_encoder_dias.joblib'))

    # ========================= DASHBOARD PROFESIONAL =========================
    with tabs[0]:
        st.header("🏠 Dashboard Logístico")
        st.markdown(
            "**Nuestros datos muestran un desfase importante entre lo estimado y lo real.**" \
            "Por ejemplo, se prometen 21 días cuando la entrega real es de 4 días en promedio."
        )

        # Slider de desviación
        st.sidebar.subheader("🎛️ Filtrar por desviación")
        dev_min, dev_max = int(df['desviacion_vs_promesa'].min()), int(df['desviacion_vs_promesa'].max())
        slider_min, slider_max = st.sidebar.slider(
            "Desviación (Estimado - Real) días", dev_min, dev_max, (dev_min, dev_max)
        )
        df_filtered = df[(df['desviacion_vs_promesa'] >= slider_min) & (df['desviacion_vs_promesa'] <= slider_max)].copy()

        # Métricas clave
        df_filtered['prometido_dias'] = df_filtered['dias_entrega'] - df_filtered['desviacion_vs_promesa']
        avg_prom = df_filtered['prometido_dias'].mean()
        avg_act = df_filtered['dias_entrega'].mean()
        avg_diff = avg_prom - avg_act

        col1, col2, col3 = st.columns(3)
        col1.metric("Estimado (días)", f"{avg_prom:.1f}")
        col2.metric("Real (días)", f"{avg_act:.1f}")
        col3.metric("Diferencia media", f"{avg_diff:.1f}")

        # Comparativa estimado vs real por categoría con dos tonos de azul
        medios_cat = (
            df_filtered.groupby('Categoría')
              .agg(Estimado=('prometido_dias', 'mean'), Real=('dias_entrega', 'mean'))
              .reset_index()
        )
        fig_cat = px.bar(
            medios_cat,
            x='Categoría',
            y=['Estimado', 'Real'],
            barmode='group',
            labels={'value': 'Días', 'variable': 'Tipo'},
            title='Tiempos estimados vs reales por Categoría',
            color_discrete_map={
                'Estimado': '#003366',  # azul marino
                'Real': '#6699cc'       # azul claro
            }
        )
        # Añadir líneas de mediana
        med_est = medios_cat['Estimado'].median()
        med_real = medios_cat['Real'].median()
        fig_cat.add_hline(y=med_est, line_dash='dash', line_color='#003366', annotation_text='Mediana Estimado', annotation_position='bottom right')
        fig_cat.add_hline(y=med_real, line_dash='dash', line_color='#6699cc', annotation_text='Mediana Real', annotation_position='top right')
        st.plotly_chart(fig_cat, use_container_width=True)

        # Violín de desviaciones por categoría en azul claro
        fig_violin = px.violin(
            df_filtered,
            x='Categoría',
            y='desviacion_vs_promesa',
            box=True,
            points='all',
            labels={'desviacion_vs_promesa': 'Estimado - Real (días)'},
            title='Distribución de desviaciones por Categoría',
            color_discrete_sequence=['#6699cc']
        )
        st.plotly_chart(fig_violin, use_container_width=True)

        # Top 10 categorías con mayor desfase medio
        medios_cat['Desviación_media'] = medios_cat['Estimado'] - medios_cat['Real']
        top_dev = medios_cat.sort_values('Desviación_media', ascending=False).head(10)
        fig_top = px.bar(
            top_dev,
            x='Categoría',
            y='Desviación_media',
            labels={'Desviación_media': 'Diferencia media (días)'},
            title='Top 10 categorías con mayor desfase',
            color_discrete_sequence=['#003366']
        )
        st.plotly_chart(fig_top, use_container_width=True)

    # ========================= CALCULADORA =========================
    with tabs[1]:
        st.header("🧮 Calculadora de Predicción")
        df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
        df2['año'] = df2['orden_compra_timestamp'].dt.year
        df2['mes'] = df2['orden_compra_timestamp'].dt.month

        estados = df2['estado_del_cliente'].dropna().unique()
        categorias = df2['Categoría'].dropna().unique()
        c1, c2 = st.columns(2)
        estado = c1.selectbox("Estado", sorted(estados))
        categoria = c2.selectbox("Categoría", sorted(categorias))

        meses_dict = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
                      7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
        mes1_nombre = c1.selectbox("Mes 1", list(meses_dict.values()), index=0)
        mes2_nombre = c2.selectbox("Mes 2", list(meses_dict.values()), index=1)
        mes1 = [k for k,v in meses_dict.items() if v==mes1_nombre][0]
        mes2 = [k for k,v in meses_dict.items() if v==mes2_nombre][0]

        filtro = (df2['estado_del_cliente']==estado)&(df2['Categoría']==categoria)
        df_mes1 = df2[(df2['mes']==mes1)&filtro].copy()
        df_mes2 = df2[(df2['mes']==mes2)&filtro].copy()

        def predecir(df_input):
            if df_input.empty:
                return df_input
            cols_flete = ['total_peso_g','precio','#_deproductos','duracion_estimada_min',
                          'ciudad_cliente','nombre_dc','hora_compra','año','mes','datetime_origen',
                          'region','dias_promedio_ciudad','Categoría','tipo_de_pago']
            df_f = df_input[cols_flete]
            df_enc = pd.get_dummies(df_f)
            feats = modelo_flete.get_booster().feature_names
            df_enc = df_enc.reindex(columns=feats, fill_value=0)
            df_input['costo_estimado'] = modelo_flete.predict(df_enc).round(2)
            df_input['costo_de_flete'] = df_input['costo_estimado']

            cols_dias = ['Categoría','categoría_peso','#_deproductos','total_peso_g','precio',
                         'costo_de_flete','distancia_km','velocidad_kmh','duracion_estimada_min',
                         'region','dc_asignado','es_feriado','es_fin_de_semana','dias_promedio_ciudad',
                         'hora_compra','nombre_dia','mes','año','temp_origen','precip_origen',
                         'cloudcover_origen','conditions_origen','icon_origen','traffic','area']
            if not all(c in df_input.columns for c in cols_dias):
                return df_input
            X = df_input[cols_dias]
            preds = modelo_dias.predict(X)
            df_input['clase_entrega'] = label_encoder.inverse_transform(preds)
            return df_input

        def resumen(df_pred, nombre_mes):
            if 'costo_estimado' not in df_pred.columns or 'clase_entrega' not in df_pred.columns:
                return pd.DataFrame(columns=['ciudad_cliente', nombre_mes, f"Entrega {nombre_mes}"])
            return df_pred.groupby('ciudad_cliente').agg({
                'costo_estimado':'mean',
                'clase_entrega':lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
            }).reset_index().rename(columns={'costo_estimado':nombre_mes,'clase_entrega':f"Entrega {nombre_mes}"})

        df_m1, df_m2 = predecir(df_mes1), predecir(df_mes2)
        r1, r2 = resumen(df_m1, mes1_nombre), resumen(df_m2, mes2_nombre)
        r1[mes1_nombre] = pd.to_numeric(r1[mes1_nombre], errors='coerce')
        r2[mes2_nombre] = pd.to_numeric(r2[mes2_nombre], errors='coerce')
        comp = pd.merge(r1, r2, on='ciudad_cliente', how='outer')
        comp[mes1_nombre] = pd.to_numeric(comp.get(mes1_nombre), errors='coerce')
        comp[mes2_nombre] = pd.to_numeric(comp.get(mes2_nombre), errors='coerce')
        diffv = pd.to_numeric(comp.get(mes2_nombre), errors='coerce') - pd.to_numeric(comp.get(mes1_nombre), errors='coerce')
        comp['Diferencia'] = diffv.round(2)
        comp = comp[['ciudad_cliente', mes1_nombre, mes2_nombre, 'Diferencia', f"Entrega {mes1_nombre}", f"Entrega {mes2_nombre}" ]]
        comp.rename(columns={'ciudad_cliente':'Ciudad'}, inplace=True)

        def resaltar(v):
            if isinstance(v, (int,float,np.number)):
                return 'color: green; font-weight: bold' if v>0 else ('color: red; font-weight: bold' if v<0 else '')
            return ''

        st.subheader(f"Comparación: {mes1_nombre} vs {mes2_nombre}")
        st.dataframe(comp.style.applymap(resaltar, subset=['Diferencia']).format(precision=2))
        st.download_button("⬇️ Descargar CSV", comp.to_csv(index=False), file_name="comparacion.csv", mime="text/csv")

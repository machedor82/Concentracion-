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
            "**Nuestro análisis muestra que los tiempos de entrega estimados son significativamente mayores que los reales.**\n"
            "Por ejemplo, se prometen 21 días cuando la entrega real es de 4 días en promedio."
        )

        # Filtro de desviación
        st.sidebar.subheader("🎛️ Rango de desviación")
        dev_min = int(df['desviacion_vs_promesa'].min())
        dev_max = int(df['desviacion_vs_promesa'].max())
        dev_range = st.sidebar.slider(
            "Desviación (Estimado - Real) días", dev_min, dev_max, (dev_min, dev_max)
        )
        df_filtered = df[
            (df['desviacion_vs_promesa'] >= dev_range[0]) &
            (df['desviacion_vs_promesa'] <= dev_range[1])
        ].copy()

        # Métricas clave
        df_filtered['prometido_dias'] = df_filtered['dias_entrega'] - df_filtered['desviacion_vs_promesa']
        avg_prom = df_filtered['prometido_dias'].mean()
        avg_act = df_filtered['dias_entrega'].mean()
        avg_diff = avg_prom - avg_act

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimado (días)", f"{avg_prom:.1f}")
        m2.metric("Real (días)", f"{avg_act:.1f}")
        m3.metric("Diferencia media", f"{avg_diff:.1f}")

        # Comparativa estimado vs real por categoría
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
            color_discrete_sequence=['#003366']
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        # Boxplot de desviaciones por categoría
        fig_box = px.box(
            df_filtered,
            x='Categoría',
            y='desviacion_vs_promesa',
            points='all',
            labels={'desviacion_vs_promesa': 'Estimado - Real (días)'},
            title='Distribución de desviaciones por Categoría',
            color_discrete_sequence=['#003366']
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Top 10 categorías con mayor desviación media
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
        col1, col2 = st.columns(2)
        estado = col1.selectbox("Estado", sorted(estados))
        categoria = col2.selectbox("Categoría", sorted(categorias))

        meses_dict = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }
        mes1_nombre = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
        mes2_nombre = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)
        mes1 = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
        mes2 = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

        filtro = (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)
        df_mes1 = df2[(df2['mes'] == mes1) & filtro].copy()
        df_mes2 = df2[(df2['mes'] == mes2) & filtro].copy()

        def predecir(df_input):
            if df_input.empty:
                return df_input

            columnas_flete = [
                'total_peso_g', 'precio', '#_deproductos',
                'duracion_estimada_min', 'ciudad_cliente', 'nombre_dc',
                'hora_compra', 'año', 'mes', 'datetime_origen',
                'region','dias_promedio_ciudad', 'Categoría','tipo_de_pago'
            ]
            df_flete = df_input[columnas_flete].copy()
            df_encoded = pd.get_dummies(df_flete)
            cols_model = modelo_flete.get_booster().feature_names
            df_encoded = df_encoded.reindex(columns=cols_model, fill_value=0)

            df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
            df_input['costo_de_flete'] = df_input['costo_estimado']

            columnas_dias = [
                'Categoría','categoría_peso','#_deproductos','total_peso_g',
                'precio','costo_de_flete','distancia_km','velocidad_kmh',
                'duracion_estimada_min','region','dc_asignado','es_feriado',
                'es_fin_de_semana','dias_promedio_ciudad','hora_compra',
                'nombre_dia','mes','año','temp_origen','precip_origen',
                'cloudcomentarios_removed'
            ]
            if not all(c in df_input.columns for c in columnas_dias):
                return df_input

            X = df_input[columnas_dias]
            pred = modelo_dias.predict(X)
            df_input['clase_entrega'] = label_encoder.inverse_transform(pred)
            return df_input

        def resumen(df_pred, nombre_mes):
            if 'costo_estimado' not in df_pred.columns or 'clase_entrega' not in df_pred.columns:
                return pd.DataFrame(columns=['ciudad_cliente', nombre_mes, f"Entrega {nombre_mes}"])
            return df_pred.groupby('ciudad_cliente').agg({
                'costo_estimado': 'mean',
                'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
            }).reset_index().rename(columns={
                'costo_estimado': nombre_mes,
                'clase_entrega': f"Entrega {nombre_mes}"
            })

        df_mes1 = predecir(df_mes1)
        df_mes2 = predecir(df_mes2)
        res1 = resumen(df_mes1, mes1_nombre)
        res2 = resumen(df_mes2, mes2_nombre)

        res1[mes1_nombre] = pd.to_numeric(res1[mes1_nombre], errors='coerce')
        res2[mes2_nombre] = pd.to_numeric(res2[mes2_nombre], errors='coerce')
        comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
        comparacion[mes1_nombre] = pd.to_numeric(comparacion.get(mes1_nombre), errors='coerce')
        comparacion[mes2_nombre] = pd.to_numeric(comparacion.get(mes2_nombre), errors='coerce')
        diff = (
            pd.to_numeric(comparacion.get(mes2_nombre), errors='coerce')
            - pd.to_numeric(comparacion.get(mes1_nombre), errors='coerce')
        )
        comparacion['Diferencia'] = diff.round(2)
        comparacion = comparacion[[
            'ciudad_cliente', mes1_nombre, mes2_nombre, 'Diferencia',
            f"Entrega {mes1_nombre}", f"Entrega {mes2_nombre}"
        ]].rename(columns={'ciudad_cliente': 'Ciudad'})

        def resaltar(val):
            if isinstance(val, (int, float, np.number)):
                if val > 0:
                    return 'color: green; font-weight: bold'
                elif val < 0:
                    return 'color: red; font-weight: bold'
            return ''

        st.subheader(f"Comparación: {mes1_nombre} vs {mes2_nombre}")
        st.dataframe(
            comparacion.style
                      .applymap(resaltar, subset=['Diferencia'])
                      .format(precision=2)
        )
        st.download_button(
            "⬇️ Descargar CSV",
            comparacion.to_csv(index=False),
            file_name="comparacion.csv",
            mime="text/csv"
        )

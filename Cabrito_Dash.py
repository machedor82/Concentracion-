import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Cabrito Analytics", layout="wide")

# ------------------ Subida de archivo ------------------
with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("📂 Sube tu archivo CSV")
    archivo_csv = st.file_uploader("Archivo .csv con los datos completos", type="csv")

# ------------------ Función auxiliar ------------------
def clasificar_zonas(df, estado_sel):
    if estado_sel == "Nacional":
        principales = ['Ciudad de México', 'Nuevo León', 'Jalisco']
        return df['estado_del_cliente'].apply(lambda x: x if x in principales else 'Provincia')
    else:
        top_ciudades = (
            df[df['estado_del_cliente'] == estado_sel]['ciudad_cliente']
            .value_counts()
            .nlargest(3)
            .index
            .tolist()
        )
        return df['ciudad_cliente'].apply(lambda x: x if x in top_ciudades else 'Otras')

# ------------------ Si se sube el CSV ------------------
if archivo_csv:
    df = pd.read_csv(archivo_csv)
    df2 = df.copy()

    # Calcular desviacion_vs_promesa si no existe
    if 'desviacion_vs_promesa' not in df.columns and {'fecha_entrega_al_cliente', 'fecha_de_entrega_estimada'}.issubset(df.columns):
        df['desviacion_vs_promesa'] = (
            pd.to_datetime(df['fecha_entrega_al_cliente'], errors='coerce') -
            pd.to_datetime(df['fecha_de_entrega_estimada'], errors='coerce')
        ).dt.days

    # Cargar modelos
    modelo_flete = joblib.load("modelo_costoflete.sav")
    modelo_dias = joblib.load("modelo_dias_pipeline_70.joblib")
    label_encoder = joblib.load("label_encoder_dias_70.joblib")

    with st.sidebar:
        st.subheader("🎛️ Filtro de Estado")
        estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
        estado_sel = option_menu(
            menu_title="Selecciona un estado",
            options=estados,
            icons=["globe"] + ["geo"] * (len(estados) - 1),
            default_index=0
        )

    df_filtrado = df if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel]

    st.success("✅ Datos y modelos cargados correctamente. Dashboard en funcionamiento.")

    tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "🗃️ Data"])

    # ========== TAB 0: Resumen Nacional ==========
    with tabs[0]:
        st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {estado_sel}")
        col1, col2 = st.columns(2)
        col1.metric("Pedidos", f"{len(df_filtrado):,}")
        col2.metric(
            "Llegadas muy adelantadas (≥10 días)",
            f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean() * 100:.1f}%"
        )

        col1, col2 = st.columns(2)
        with col1:
            df_tmp = df_filtrado.copy()
            df_tmp['zona_entrega'] = clasificar_zonas(df_tmp, estado_sel)
            conteo = df_tmp['zona_entrega'].value_counts().reset_index()
            conteo.columns = ['zona', 'cantidad']  # ← renombrar columnas
            fig = px.pie(conteo, names='zona', values='cantidad', hole=0.4,
                 title="📍 Pedidos por Zona")
            st.plotly_chart(fig, use_container_width=True)


        with col2:
            df_tmp['estatus_entrega'] = df_tmp['llego_tarde'].apply(lambda x: 'A tiempo' if x == 0 else 'Tardío')
            conteo = df_tmp.groupby(['zona_entrega', 'estatus_entrega']).size().reset_index(name='conteo')
            fig = px.bar(conteo, x='zona_entrega', y='conteo', color='estatus_entrega', barmode='stack')
            st.plotly_chart(fig, use_container_width=True)

    # ========== TAB 1: Costo de Envío ==========
    with tabs[1]:
        st.header("💰 Costo de Envío")
        col1, col2 = st.columns(2)
        col1.metric("Pedidos", f"{len(df_filtrado):,}")
        col2.metric(
            "% Flete > 50% del precio",
            f"{(df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100:.1f}%"
        )

        df_filtrado['porcentaje_flete'] = (df_filtrado['costo_de_flete'] / df_filtrado['precio']) * 100
        resumen = df_filtrado[['categoria', 'porcentaje_flete']].groupby('categoria').mean().round(2).sort_values(by='porcentaje_flete', ascending=False)
        st.dataframe(resumen, use_container_width=True)

    # ========== TAB 2: Calculadora ==========
    with tabs[2]:
        st.header("🧮 Calculadora de Predicción")

        meses_dict = {1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
                      7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"}

        df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'], errors='coerce')
        df2['año'] = df2['orden_compra_timestamp'].dt.year
        df2['mes'] = df2['orden_compra_timestamp'].dt.month

        categoria = st.selectbox("Categoría", sorted(df2['categoria'].dropna().unique()))
        col1, col2 = st.columns(2)
        mes1_nombre = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
        mes2_nombre = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)
        mes1 = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
        mes2 = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

        filtro = (df2['estado_del_cliente'] == estado_sel) & (df2['categoria'] == categoria)
        df_mes1 = df2[(df2['mes'] == mes1) & filtro].copy()
        df_mes2 = df2[(df2['mes'] == mes2) & filtro].copy()

        def predecir(df_input):
            if df_input.empty:
                return df_input
            columnas_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                              'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                              'dias_promedio_ciudad', 'categoria', 'tipo_de_pago']
            df_flete = df_input[columnas_flete].copy()
            df_encoded = pd.get_dummies(df_flete)
            columnas_modelo = modelo_flete.get_booster().feature_names
            df_encoded = df_encoded.reindex(columns=columnas_modelo, fill_value=0)
            df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
            df_input['costo_de_flete'] = df_input['costo_estimado']

            columnas_dias = ['categoria', 'categoria_peso', '#_deproductos', 'total_peso_g', 'precio', 'costo_de_flete',
                             'distancia_km', 'velocidad_kmh', 'duracion_estimada_min', 'region', 'dc_asignado',
                             'es_feriado', 'es_fin_de_semana', 'hora_compra', 'dias_promedio_ciudad', 'nombre_dia',
                             'mes', 'año', 'traffic', 'area']
            if not all(c in df_input.columns for c in columnas_dias):
                return df_input
            X_dias = df_input[columnas_dias]
            pred = modelo_dias.predict(X_dias)
            df_input['clase_entrega'] = label_encoder.inverse_transform(pred)
            return df_input

        def agrupar(df, mes_nombre):
            if 'costo_estimado' in df.columns and 'clase_entrega' in df.columns:
                return df.groupby('ciudad_cliente').agg({
                    'costo_estimado': 'mean',
                    'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'NA'
                }).rename(columns={'costo_estimado': mes_nombre, 'clase_entrega': f'Entrega {mes_nombre}'}).reset_index()
            return pd.DataFrame(columns=['ciudad_cliente', mes_nombre, f'Entrega {mes_nombre}'])

        df_mes1 = predecir(df_mes1)
        df_mes2 = predecir(df_mes2)

        res1 = agrupar(df_mes1, mes1_nombre)
        res2 = agrupar(df_mes2, mes2_nombre)
        comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
        comparacion['Diferencia'] = (comparacion[mes2_nombre] - comparacion[mes1_nombre]).round(2)
        comparacion = comparacion.rename(columns={'ciudad_cliente': 'Ciudad'})

        st.dataframe(comparacion.style.format(precision=2))
        st.download_button("⬇️ Descargar Comparación CSV", comparacion.to_csv(index=False), file_name="comparacion.csv")

    # ========== TAB 3: Vista de Datos ==========
    with tabs[3]:
        st.subheader("🔍 Vista Previa del Dataset")
        st.dataframe(df.head(100), use_container_width=True)

else:
    st.warning("⚠️ Sube un archivo CSV para activar el dashboard.")

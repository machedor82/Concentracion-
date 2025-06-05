import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
#
# ========================== CONFIGURACIÓN INICIAL ==========================
st.set_page_config(page_title="Cabrito Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #002244;
            color: white;
        }
        .stSlider > div[data-testid="stTickBar"] {
            background-color: #ffffff11;
        }
        .stSlider .css-14pt78w {
            color: white !important;
        }
        .stMultiSelect, .stSlider {
            color: black !important;
        }
        [data-testid="stSidebar"] label {
            color: white;
        }
        body {
            background-color: #f9f9f9;
            color: #1f1f1f;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            padding: 10px;
            border-bottom: 3px solid transparent;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #004b8d;
            color: #004b8d;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Panel BI")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora", "🔧 Por definir"])

# ========================== LÓGICA DE CARGA EN SIDEBAR ==========================
with st.sidebar:
    st.header("📂 Carga de datos")

    # ZIP para Dashboard
    uploaded_file = st.file_uploader("📦 Sube un ZIP con el archivo 'DF.csv'", type="zip", key="zip_uploader")

    # CSV para Calculadora
    archivo_subido = st.file_uploader("📑 Sube tu archivo CSV para cálculo", type="csv", key="csv_uploader")

    with st.expander("🎛️ Filtros"):
        st.markdown("Los filtros se activan cuando cargas datos en la pestaña correspondiente.")


# ========================== PESTAÑA 1 ==========================
with tabs[0]:
    st.session_state.active_tab = "🏠 Dashboard"

    @st.cache_data
    def load_zip_csv(upload, internal_name="DF.csv"):
        with zipfile.ZipFile(upload) as z:
            with z.open(internal_name) as f:
                return pd.read_csv(f)

    df = df_filtrado = None

    if uploaded_file:
        try:
            df = load_zip_csv(uploaded_file)
            st.success("✅ Datos cargados exitosamente")

            # ========================== SIDEBAR: FILTROS PARA DASHBOARD ==========================
            if df is not None:
                with st.sidebar:
                    with st.expander("🎛️ Filtros", expanded=True):
                        categorias = df['Categoría'].dropna().unique()
                        estados = df['estado_del_cliente'].dropna().unique()
                        años = sorted(df['año'].dropna().unique())
                        meses = sorted(df['mes'].dropna().unique())

                        categoria_sel = st.multiselect("Categoría de producto", categorias, default=list(categorias))
                        estado_sel = st.multiselect("Estado del cliente", estados, default=list(estados))
                        año_sel = st.multiselect("Año", años, default=años)
                        mes_sel = st.multiselect("Mes", meses, default=meses)

                    with st.expander("📏 Filtros avanzados", expanded=False):
                        min_flete, max_flete = float(df['costo_relativo_envio'].min()), float(df['costo_relativo_envio'].max())
                        rango_flete = st.slider("Costo relativo de envío (%)", min_value=round(min_flete, 2), max_value=round(max_flete, 2), value=(round(min_flete, 2), round(max_flete, 2)))

                        min_peso, max_peso = int(df['total_peso_g'].min()), int(df['total_peso_g'].max())
                        rango_peso = st.slider("Peso total del pedido (g)", min_value=min_peso, max_value=max_peso, value=(min_peso, max_peso))

                # Aplicar filtros
                df_filtrado = df[
                    (df['Categoría'].isin(categoria_sel)) &
                    (df['estado_del_cliente'].isin(estado_sel)) &
                    (df['año'].isin(año_sel)) &
                    (df['mes'].isin(mes_sel)) &
                    (df['costo_relativo_envio'].between(*rango_flete)) &
                    (df['total_peso_g'].between(*rango_peso))
                ]

                if not df_filtrado.empty:
                    st.markdown("## 🧭 Visión General de la Operación")
                    st.markdown("### 🔢 Indicadores")
                    col1, col2, col3 = st.columns(3)

                    col1.metric("📦 Total de pedidos", f"{len(df_filtrado):,}")

                    pct_flete_alto = (df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100
                    col2.metric("🚚 Flete > 50%", f"{pct_flete_alto:.1f}%")

                    pct_anticipadas = (df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100
                    col3.metric("⏱️ Entregas ≥7 días antes", f"{pct_anticipadas:.1f}%")

                    st.markdown("### 📊 Análisis visual")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader("🌳 Treemap por categoría")
                        try:
                            fig_tree = px.treemap(
                                df_filtrado,
                                path=['Categoría'],
                                values='precio',
                                color='Categoría',
                                color_discrete_sequence=px.colors.qualitative.Pastel
                            )
                            st.plotly_chart(fig_tree, use_container_width=True)
                        except Exception as e:
                            st.error(f"❌ Error al generar el Treemap: {e}")

                    with col2:
                        st.subheader("🗺️ Mapa de entregas de clientes")
                        df_mapa = df_filtrado.dropna(subset=['lat_cliente', 'lon_cliente'])
                        if not df_mapa.empty:
                            st.map(df_mapa.rename(columns={'lat_cliente': 'lat', 'lon_cliente': 'lon'})[['lat', 'lon']])
                        else:
                            st.warning("⚠️ No hay ubicaciones disponibles con los filtros actuales.")

                    with col3:
                        st.subheader("📈 Promedio de entrega vs colchón por estado")
                        if all(col in df_filtrado.columns for col in ['estado_del_cliente', 'dias_entrega', 'colchon_dias']):
                            df_promedios = df_filtrado.groupby('estado_del_cliente')[['dias_entrega', 'colchon_dias']].mean().reset_index()
                            df_promedios = df_promedios.round(2)

                            fig_bar = px.bar(
                                df_promedios,
                                x='estado_del_cliente',
                                y=['dias_entrega', 'colchon_dias'],
                                barmode='group',
                                labels={'value': 'Días promedio', 'estado_del_cliente': 'Estado'},
                                title='Comparación de días de entrega vs colchón por estado',
                                text_auto='.2s'
                            )
                            fig_bar.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.warning("⚠️ Faltan columnas necesarias: 'estado_del_cliente', 'dias_entrega' o 'colchon_dias'.")
        except Exception as e:
            st.error(f"❌ Error al cargar el ZIP: {e}")


# ========================== PESTAÑA 2 ==========================
with tabs[1]:
    st.session_state.active_tab = "🧮 Calculadora"

    if archivo_subido is None:
        st.warning("Por favor, carga un archivo CSV para continuar.")
        st.stop()

    df2 = pd.read_csv(archivo_subido)
    st.success("✅ Archivo CSV cargado exitosamente")

    with st.sidebar:
        st.markdown("### 🎯 Filtros para predicción")
        estados_calc = sorted(df2['estado_del_cliente'].dropna().unique())
        categorias_calc = sorted(df2['Categoría'].dropna().unique())
        estado = st.selectbox("Estado", estados_calc)
        categoria = st.selectbox("Categoría", categorias_calc)

    @st.cache_resource
    def cargar_modelos():
        modelo_flete = joblib.load('modelo_costoflete.sav')
        modelo_dias = joblib.load('modelo_dias_pipeline.joblib')
        label_encoder = joblib.load('label_encoder_dias.joblib')
        return modelo_flete, modelo_dias, label_encoder

    modelo_flete, modelo_dias, label_encoder = cargar_modelos()

    meses_dict = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    st.title("Predicción de Costo de Flete y Clase de Entrega por Ciudad y Categoría")

    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    col1, col2 = st.columns(2)
    with col1:
        mes1_nombre = st.selectbox("Mes 1", list(meses_dict.values()), index=0)
    with col2:
        mes2_nombre = st.selectbox("Mes 2", list(meses_dict.values()), index=1)

    mes1_num = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
    mes2_num = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

    filtro = (df2['estado_del_cliente'] == estado) & (df2['Categoría'] == categoria)
    df_mes1 = df2[(df2['mes'] == mes1_num) & filtro].copy()
    df_mes2 = df2[(df2['mes'] == mes2_num) & filtro].copy()

    def predecir(df_input):
        if df_input.empty:
            return df_input

        columnas_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                          'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                          'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago']

        df_flete = df_input[columnas_flete].copy()
        df_encoded = pd.get_dummies(df_flete)
        columnas_entrenadas = modelo_flete.get_booster().feature_names
        df_encoded = df_encoded.reindex(columns=columnas_entrenadas, fill_value=0)

        df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
        df_input['costo_de_flete'] = df_input['costo_estimado']

        cols_dias = ['Categoría', 'categoría_peso', '#_deproductos', 'total_peso_g', 'precio', 'costo_de_flete',
                     'distancia_km', 'velocidad_kmh', 'duracion_estimada_min', 'region', 'dc_asignado',
                     'es_feriado', 'es_fin_de_semana', 'dias_promedio_ciudad', 'hora_compra',
                     'nombre_dia', 'mes', 'año', 'temp_origen', 'precip_origen', 'cloudcover_origen',
                     'conditions_origen', 'icon_origen', 'traffic', 'area']

        faltantes = [col for col in cols_dias if col not in df_input.columns]
        if faltantes:
            st.error(f"Faltan columnas para la predicción de clase de entrega: {faltantes}")
            return df_input

        X_dias = df_input[cols_dias].copy()
        if X_dias.empty:
            return df_input

        clase_pred = modelo_dias.predict(X_dias)
        df_input['clase_entrega'] = label_encoder.inverse_transform(clase_pred)

        return df_input

    def agrupar_resultados(df_input, nombre_mes):
        if 'costo_estimado' in df_input.columns and 'clase_entrega' in df_input.columns:
            return df_input.groupby('ciudad_cliente').agg({
                'costo_estimado': lambda x: round(x.mean(), 2),
                'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'NA'
            }).rename(columns={
                'costo_estimado': nombre_mes,
                'clase_entrega': f"Entrega {nombre_mes}"
            }).reset_index()
        else:
            st.warning(f"No se pudo calcular el costo de flete o la clase de entrega para {nombre_mes}.")
            return pd.DataFrame(columns=['ciudad_cliente', nombre_mes, f"Entrega {nombre_mes}"])

    df_mes1 = predecir(df_mes1)
    df_mes2 = predecir(df_mes2)

    res1 = agrupar_resultados(df_mes1, mes1_nombre)
    res2 = agrupar_resultados(df_mes2, mes2_nombre)

    comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
    comparacion['Diferencia'] = (comparacion[mes2_nombre] - comparacion[mes1_nombre]).round(2)

    comparacion = comparacion[[
        'ciudad_cliente',
        mes1_nombre,
        mes2_nombre,
        'Diferencia',
        f"Entrega {mes1_nombre}",
        f"Entrega {mes2_nombre}"
    ]].rename(columns={'ciudad_cliente': 'Ciudad'})

    def resaltar_diferencia(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: green; font-weight: bold'
            elif val < 0:
                return 'color: red; font-weight: bold'
        return ''

    styled_df = (
        comparacion.style
        .applymap(resaltar_diferencia, subset=['Diferencia'])
        .format(precision=2)
    )

    st.subheader(f"Comparación entre {mes1_nombre} y {mes2_nombre} en {estado} para categoría {categoria}")
    st.write(styled_df)

    csv = comparacion.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"⬇️ Descargar comparación: {mes1_nombre} vs {mes2_nombre}",
        data=csv,
        file_name=f'comparacion_{estado}_{categoria}_{mes1_nombre}_vs_{mes2_nombre}.csv',
        mime='text/csv'
    )

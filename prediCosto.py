import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Cargar modelos
modelo_flete = joblib.load('modelo_costoflete.sav')
modelo_dias = joblib.load('modelo_dias_pipeline_70.joblib')
label_encoder = joblib.load('label_encoder_dias_70.joblib')

# Diccionario de meses
meses_dict = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
}

st.set_page_config(page_title="Predicción de Flete", layout="wide")
st.title("Predicción de Costo de Flete y Clase de Entrega por Ciudad y Categoría")

# Subida de archivo
st.sidebar.header("Carga tu archivo de datos")
archivo_subido = st.sidebar.file_uploader("Sube tu CSV con pedidos", type=["csv"])

if archivo_subido:
    df = pd.read_csv(archivo_subido, encoding='utf-8')
else:
    st.warning("Por favor, carga un archivo CSV para continuar.")
    st.stop()

# Procesar fecha
df['orden_compra_timestamp'] = pd.to_datetime(df['orden_compra_timestamp'])
df['año'] = df['orden_compra_timestamp'].dt.year
df['mes'] = df['orden_compra_timestamp'].dt.month

# Filtro de estado y categoría
estado = st.sidebar.selectbox("Estado", sorted(df['estado_del_cliente'].dropna().unique()))
categorias = sorted(df['categoria'].dropna().unique())
categoria = st.sidebar.selectbox("Categoría", categorias)

# Filtros de mes
col1, col2 = st.columns(2)
with col1:
    mes1_nombre = st.selectbox("Mes 1", list(meses_dict.values()), index=0)
with col2:
    mes2_nombre = st.selectbox("Mes 2", list(meses_dict.values()), index=1)

mes1_num = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
mes2_num = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

# Filtrar datos
filtro = (df['estado_del_cliente'] == estado) & (df['categoria'] == categoria)
df_mes1 = df[(df['mes'] == mes1_num) & filtro].copy()
df_mes2 = df[(df['mes'] == mes2_num) & filtro].copy()

# Función de predicción
def predecir(df_input):
    if df_input.empty:
        return df_input

    columnas_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                      'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                      'dias_promedio_ciudad', 'categoria', 'tipo_de_pago']

    df_flete = df_input[columnas_flete].copy()
    df_encoded = pd.get_dummies(df_flete)
    columnas_entrenadas = modelo_flete.get_booster().feature_names
    df_encoded = df_encoded.reindex(columns=columnas_entrenadas, fill_value=0)

    df_input['costo_estimado'] = modelo_flete.predict(df_encoded)
    df_input['costo_estimado'] = df_input['costo_estimado'].round(2)
    df_input['costo_de_flete'] = df_input['costo_estimado']

    cols_dias = ['categoria', 'categoria_peso', '#_deproductos', 'total_peso_g', 'precio', 'costo_de_flete',
                 'distancia_km', 'velocidad_kmh', 'duracion_estimada_min', 'region', 'dc_asignado',
                 'es_feriado', 'es_fin_de_semana', 'hora_compra', 'dias_promedio_ciudad',
                 'nombre_dia', 'mes', 'año', 'traffic', 'area']

    faltantes = [col for col in cols_dias if col not in df_input.columns]
    if faltantes:
        st.error(f"Faltan columnas para la predicción de clase de entrega: {faltantes}")
        return df_input

    X_dias = df_input[cols_dias].copy()
    clase_pred = modelo_dias.predict(X_dias)
    df_input['clase_entrega'] = label_encoder.inverse_transform(clase_pred)

    return df_input

# Agrupar resultados
def agrupar_resultados(df, nombre_mes):
    if 'costo_estimado' in df.columns and 'clase_entrega' in df.columns:
        return df.groupby('ciudad_cliente').agg({
            'costo_estimado': lambda x: round(x.mean(), 2),
            'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'NA'
        }).rename(columns={
            'costo_estimado': nombre_mes,
            'clase_entrega': f"Entrega {nombre_mes}"
        }).reset_index()
    else:
        st.warning(f"No se pudo calcular el costo de flete o la clase de entrega para {nombre_mes}.")
        return pd.DataFrame(columns=['ciudad_cliente', nombre_mes, f"Entrega {nombre_mes}"])

# Predecir
df_mes1 = predecir(df_mes1)
df_mes2 = predecir(df_mes2)

# Comparación por ciudad
res1 = agrupar_resultados(df_mes1, mes1_nombre)
res2 = agrupar_resultados(df_mes2, mes2_nombre)
comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
comparacion['Diferencia'] = (comparacion[mes2_nombre] - comparacion[mes1_nombre]).round(2)
comparacion = comparacion[[
    'ciudad_cliente', mes1_nombre, mes2_nombre, 'Diferencia',
    f"Entrega {mes1_nombre}", f"Entrega {mes2_nombre}"
]].rename(columns={'ciudad_cliente': 'Ciudad'})

# KPIs de costo promedio (🔼 arriba de tabla)
costo_prom_mes1 = df_mes1['costo_estimado'].mean() if not df_mes1.empty else np.nan
costo_prom_mes2 = df_mes2['costo_estimado'].mean() if not df_mes2.empty else np.nan
cambio_pct = ((costo_prom_mes2 - costo_prom_mes1) / costo_prom_mes1 * 100) if costo_prom_mes1 != 0 else 0

st.markdown("---")
cols_kpi_arriba = st.columns(3)
cols_kpi_arriba[0].markdown(f"**Costo de Flete Promedio {mes1_nombre}**")
cols_kpi_arriba[1].markdown("**% Cambio**")
cols_kpi_arriba[2].markdown(f"**Costo de Flete Promedio {mes2_nombre}**")

cols_kpi_arriba[0].markdown(
    f"<span style='font-size:28px; font-weight:bold'>{costo_prom_mes1:.2f}</span>", unsafe_allow_html=True)
color_cambio = 'green' if cambio_pct > 0 else 'red'
cols_kpi_arriba[1].markdown(
    f"<span style='color:{color_cambio}; font-size:28px; font-weight:bold'>{cambio_pct:.2f}%</span>",
    unsafe_allow_html=True)
cols_kpi_arriba[2].markdown(
    f"<span style='font-size:28px; font-weight:bold'>{costo_prom_mes2:.2f}</span>", unsafe_allow_html=True)

# Estilo condicional
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

# Tabla grande
st.markdown(
    """
    <style>
    .big-table-container {
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.subheader(f"Comparación entre {mes1_nombre} y {mes2_nombre} en {estado} para categoría {categoria}")
st.markdown("<div class='big-table-container'>", unsafe_allow_html=True)
st.write(styled_df)
st.markdown("</div>", unsafe_allow_html=True)

# Descarga CSV
csv = comparacion.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Descargar CSV de Comparación",
    data=csv,
    file_name=f'comparacion_{estado}_{categoria}_{mes1_nombre}vs{mes2_nombre}.csv',
    mime='text/csv',
)

# KPIs de entrega: ABAJO
#st.markdown("---")
#st.subheader("Indicadores de desempeño de entrega")

#porcentaje_entregas_anticipadas = (
#    (df['diferencia_entrega_estimada'] < -10).sum() / len(df) * 100
#    if 'diferencia_entrega_estimada' in df.columns else np.nan
#)
#porcentaje_entregas_retrasadas = (
#    (df['diferencia_entrega_estimada'] > 0).sum() / len(df) * 100
#    if 'diferencia_entrega_estimada' in df.columns else np.nan
#)

#col1, col2 = st.columns(2)
#col1.metric("% Entregas >10 días antes (antes)", f"{porcentaje_entregas_anticipadas:.2f}%")
#col2.metric("% Entregas con retraso (antes)", f"{porcentaje_entregas_retrasadas:.2f}%")

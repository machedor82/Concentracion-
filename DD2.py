# dash2.py

import streamlit as st
import pandas as pd
import zipfile
import io
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu # Asegúrate de importar esto arriba

# ------------------ Definiciones de clases/funciones personalizadas ------------------
# Copia aquí las clases o funciones custom que usaste al entrenar 'modelo_dias_pipeline.joblib'.
# Deben llamarse exactamente igual que en tu script original.

class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        # Inicializa tus parámetros
        self.parametro1 = parametro1

    def fit(self, X, y=None):
        # Ajuste si es necesario (o simplemente retorna self)
        return self

    def transform(self, X):
        # Lógica de transformación (ejemplo placeholder)
        return X

# Si tu pipeline usaba más clases o funciones custom, defínelas aquí de la misma forma:
# class OtraClasePersonalizada:
#     def __init__(...): ...
#     def fit(...): ...
#     def transform(...): ...

# ---------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import joblib

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

# ===================== ESTILOS PERSONALIZADOS =====================
st.markdown("""
    <style>
        /* Fondo azul marino en zona principal */
        .main {
            background-color: #002244 !important;
        }

        /* Texto en zona principal */
        .main > div {
            color: white;
        } 
        /* Aumentar tamaño del texto de los títulos de métricas */
[data-testid="stMetricLabel"] {
    font-size: 1.5rem;
    font-weight: 600;
}


        /* Sidebar blanco con texto azul marino */
        [data-testid="stSidebar"] {
            background-color: white !important;
        }

        [data-testid="stSidebar"] * {
            color: #002244 !important;
        }

        .stExpander > summary {
            color: #002244 !important;
        }

        /* Compactar multiselect */
        .css-1wa3eu0 {
            display: none !important;
        }
        .stMultiSelect .css-12w0qpk {
            max-height: 0px !important;
            overflow: hidden !important;
        }
        .stMultiSelect {
            height: 35px !important;
        }
    </style>
""", unsafe_allow_html=True)
# ===================== ENCABEZADO Y CARGA DE ARCHIVO =====================
tabs = st.tabs(["📊 Resumen Nacional", "🏠 Dashboard", "🧮 Calculadora"])

with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

# ===================== PROCESAMIENTO DEL ZIP =====================
if archivo_zip:
    with zipfile.ZipFile(archivo_zip) as z:
        requeridos = [
            'DF.csv',
            'DF2.csv',
            'modelo_costoflete.sav',
            'modelo_dias_pipeline.joblib',
            'label_encoder_dias.joblib'
        ]
        contenidos = z.namelist()
        faltantes = [r for r in requeridos if r not in contenidos]
        if faltantes:
            st.error(f"❌ Faltan archivos en el ZIP: {faltantes}")
            st.stop()

        # Cargar datos y modelos
        df = pd.read_csv(z.open('DF.csv'))
        df2 = pd.read_csv(z.open('DF2.csv'))
        modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
        modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
        label_encoder = joblib.load(z.open('label_encoder_dias.joblib'))


# ========== 📊 RESUMEN NACIONAL ==========
with tabs[0]:
    st.title("📊 Resumen Nacional")
    
    st.info("Esta sección aún está en construcción. Pronto podrás ver un resumen agregado de la operación a nivel país.")

    # 👇 Usa df sin filtrar
    if 'dias_entrega' in df.columns:

        # --------- HISTOGRAMA DE ENTREGA ---------
        st.subheader("⏱️ Distribución de Tiempos de Entrega")
        fig_hist = px.histogram(
            df,
            x='dias_entrega',
            nbins=30,
            title="Distribución de pedidos por días de entrega",
            labels={'dias_entrega': 'Días entre orden y entrega'},
        )
        fig_hist.update_layout(
            xaxis_title="Días de entrega",
            yaxis_title="Número de pedidos",
            bargap=0.1,
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # --------- GRÁFICO DE BARRAS APILADAS POR ESTADO ---------
        st.subheader("📦 Proporción de Tiempos de Entrega por Estado")

        df_tmp = df.copy()
        df_tmp['rango_entrega'] = pd.cut(
            df_tmp['dias_entrega'],
            bins=[0, 5, 10, float('inf')],
            labels=["1-5 días", "6-10 días", "Más de 10 días"],
            right=True
        )

        conteo = df_tmp.groupby(['estado_del_cliente', 'rango_entrega']).size().reset_index(name='conteo')
        total_por_estado = conteo.groupby('estado_del_cliente')['conteo'].transform('sum')
        conteo['porcentaje'] = conteo['conteo'] / total_por_estado * 100

        fig = px.bar(
            conteo,
            x='estado_del_cliente',
            y='porcentaje',
            color='rango_entrega',
            labels={
                'estado_del_cliente': 'Estado',
                'porcentaje': 'Porcentaje',
                'rango_entrega': 'Rango'
            },
            title='⏱️ Tiempo de Entrega por Estado (Distribución %)',
            text_auto='.1f'
        )

        fig.update_layout(
            barmode='stack',
            xaxis_title=None,
            yaxis_title='Porcentaje (%)',
            legend_title='Rango de Entrega',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)



    # ========================= PESTAÑA 1: DASHBOARD =========================
    with tabs[1]:

        # --------- SIDEBAR FILTRO ---------
        with st.sidebar:
            st.subheader("🎛️ Filtro de Estado")
            estados = sorted(df['estado_del_cliente'].dropna().unique())
            estado_sel = option_menu(
                menu_title="Selecciona un estado",
                options=estados,
                icons=["geo"] * len(estados),
                default_index=0
            )

        # --------- FILTRADO DE DATOS ---------
        df_filtrado = df[df['estado_del_cliente'] == estado_sel]

        # --------- MÉTRICAS PRINCIPALES ---------
        col1, col2, col3 = st.columns(3)
        col1.metric("Pedidos", f"{len(df_filtrado):,}")
        col2.metric(
            "Transporte costoso para su valor",
            f"{(df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100:.1f}%"
        )
        col3.metric(
            "Llegadas muy adelantadas (1 semana antes o más)",
            f"{(df_filtrado['desviacion_vs_promesa'] < -7).mean() * 100:.1f}%"
        )

        # --------- TABLA HORIZONTAL: % Flete sobre Precio por Categoría ---------
        st.subheader("💸 % del Flete sobre el Precio")

        df_precio = df_filtrado.copy()
        df_precio['porcentaje_flete'] = (df_precio['costo_de_flete'] / df_precio['precio']) * 100
        tabla = df_precio.groupby('Categoría')['porcentaje_flete'].mean().reset_index()
        tabla = tabla.sort_values(by='porcentaje_flete', ascending=False)
        max_val = tabla['porcentaje_flete'].max()
        tabla['porcentaje_flete'] = tabla['porcentaje_flete'].apply(
            lambda x: f"🔺 {x:.1f}%" if x == max_val else f"{x:.1f}%"
        )
        tabla_h = tabla.set_index('Categoría').T

        def highlight_emoji_red(s):
            return ['color: red; font-weight: bold' if '🔺' in str(v) else '' for v in s]

        st.dataframe(
            tabla_h.style.apply(highlight_emoji_red, axis=1),
            use_container_width=True,
            height=100,
            hide_index=True
        )

        # --------- LAYOUT SUPERIOR: BARRAS + HEATMAP ---------
        col1, col2 = st.columns([1, 1])

        with col1:
            totales = df_filtrado.groupby('Categoría')[['precio', 'costo_de_flete']].sum().reset_index()
            totales = totales.sort_values(by='precio', ascending=False)

            fig_totales = px.bar(
                totales,
                x='Categoría',
                y=['precio', 'costo_de_flete'],
                barmode='group',
                labels={'value': 'Monto ($)', 'variable': 'Concepto'}
            )
            fig_totales.update_layout(
                height=320,
                xaxis_title=None,
                yaxis_title=None,
                margin=dict(t=40, b=40, l=10, r=10),
                legend_title="",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_totales, use_container_width=True)

        with col2:
            df_heat = df_filtrado.copy()
            df_heat['dia_semana'] = pd.to_datetime(df_heat['orden_compra_timestamp']).dt.day_name()
            heat_data = df_heat.groupby(['Categoría', 'dia_semana'])['costo_de_flete'].mean().reset_index()

            fig = px.density_heatmap(
                heat_data,
                x="dia_semana",
                y="Categoría",
                z="costo_de_flete",
                color_continuous_scale="Blues",
                title="🔥 Costo Promedio de Flete por Día y Categoría"
            )
            st.plotly_chart(fig, use_container_width=True)

        # --------- LAYOUT INFERIOR: BARRAS HORIZONTALES ---------
        if {'dias_entrega', 'colchon_dias'}.issubset(df_filtrado.columns):
            import plotly.graph_objects as go

            medios = df_filtrado.groupby('Categoría')[['dias_entrega', 'colchon_dias']].mean().reset_index()
            medios = medios.sort_values(by='dias_entrega', ascending=False)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=medios['Categoría'],
                x=medios['dias_entrega'],
                name='Días Entrega',
                orientation='h'
            ))
            fig.add_trace(go.Bar(
                y=medios['Categoría'],
                x=medios['colchon_dias'],
                name='Colchón Días',
                orientation='h'
            ))

            promedio_entrega = medios['dias_entrega'].mean()
            fig.add_shape(
                type="line",
                x0=promedio_entrega,
                x1=promedio_entrega,
                y0=-0.5,
                y1=len(medios) - 0.5,
                line=dict(color="blue", dash="dash")
            )

            fig.update_layout(
                barmode='group',
                height=350,
                xaxis_title=None,
                yaxis_title=None,
                margin=dict(t=40, b=40, l=10, r=10),
                legend_title="Métrica"
            )

            st.plotly_chart(fig, use_container_width=True)

        


    # ========================= CALCULADORA =========================
    with tabs[2]:
        st.header("🧮 Calculadora de Predicción")

        # Convertir timestamp y extraer año/mes
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
                'duracion_estimada_min', 'ciudad_cliente',
                'nombre_dc', 'hora_compra', 'año', 'mes',
                'datetime_origen', 'region',
                'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago'
            ]

            df_flete = df_input[columnas_flete].copy()
            df_encoded = pd.get_dummies(df_flete)
            columnas_modelo = modelo_flete.get_booster().feature_names
            df_encoded = df_encoded.reindex(columns=columnas_modelo, fill_value=0)

            df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
            df_input['costo_de_flete'] = df_input['costo_estimado']

            columnas_dias = [
                'Categoría', 'categoría_peso', '#_deproductos', 'total_peso_g', 'precio',
                'costo_de_flete', 'distancia_km', 'velocidad_kmh', 'duracion_estimada_min',
                'region', 'dc_asignado', 'es_feriado', 'es_fin_de_semana',
                'dias_promedio_ciudad', 'hora_compra', 'nombre_dia', 'mes', 'año',
                'temp_origen', 'precip_origen', 'cloudcover_origen', 'conditions_origen',
                'icon_origen', 'traffic', 'area'
            ]

            if not all(c in df_input.columns for c in columnas_dias):
                return df_input  # evita error si faltan columnas

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

        # Asegurarse de que las columnas de costo sean numéricas
        res1[mes1_nombre] = pd.to_numeric(res1[mes1_nombre], errors='coerce')
        res2[mes2_nombre] = pd.to_numeric(res2[mes2_nombre], errors='coerce')

        comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')

        # Convertir las columnas fusionadas a numérico (por si quedaron object)
        comparacion[mes1_nombre] = pd.to_numeric(comparacion.get(mes1_nombre), errors='coerce')
        comparacion[mes2_nombre] = pd.to_numeric(comparacion.get(mes2_nombre), errors='coerce')

        # Calcular la diferencia convirtiendo primero a numérico y luego redondeando
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

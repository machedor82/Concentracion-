# Cabrito Dash 11/06/2025 12:17 pm

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



# ---------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import zipfile
import plotly.express as px
import joblib

# ===================== CONFIGURACIÓN DE PÁGINA =====================
st.set_page_config(page_title="Cabrito Analytics", layout="wide")

st.markdown("""
    <style>
        /* Fondo general sobrio y elegante */
        .main {
            background-color: #f5f7fa !important;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Texto principal en gris oscuro */
        .main > div {
            color: #1e2022 !important;
        }

        /* Estilo de métricas */
        [data-testid="stMetricLabel"] {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1a73e8 !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #202124 !important;
        }

        [data-testid="stMetricDelta"] {
            font-weight: bold;
            color: #34a853 !important;
        }

        /* Sidebar limpio con acento azul petróleo */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            border-right: 1px solid #e0e0e0;
        }

        [data-testid="stSidebar"] * {
            color: #1a3c5a !important;
        }

        /* Encabezados de expander elegantes */
        .stExpander > summary {
            font-weight: 600;
            color: #1a3c5a !important;
        }

        /* Tabs refinadas */
        .stTabs [data-baseweb="tab"] {
            font-size: 15px;
            padding: 12px;
            border-bottom: 2px solid transparent;
            color: #5f6368;
        }

        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #1a73e8;
            color: #1a73e8;
            font-weight: 600;
        }

        /* Selector multiple compacto */
        .stMultiSelect .css-12w0qpk {
            max-height: 0px !important;
            overflow: hidden !important;
        }

        .stMultiSelect {
            height: 38px !important;
        }

        /* Ocultar watermark de Streamlit */
        .css-1wa3eu0 {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)
# ===================== INTERFAZ BÁSICA =====================

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


tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora","App Danu 📈"])

with st.sidebar: 
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo CSV")
    archivo_csv = st.file_uploader("Archivo CSV", type="csv")
# ===================== CARGA Y PROCESAMIENTO DE DATOS =====================
if archivo_csv:
    df = pd.read_csv(archivo_csv)

    # Calcular columna 'desviacion_vs_promesa'
    df['fecha_entrega_al_cliente'] = pd.to_datetime(df['fecha_entrega_al_cliente'])
    df['fecha_de_entrega_estimada'] = pd.to_datetime(df['fecha_de_entrega_estimada'])
    df['desviacion_vs_promesa'] = (df['fecha_entrega_al_cliente'] - df['fecha_de_entrega_estimada']).dt.days

    # Copia auxiliar para la calculadora
    df2 = df.copy()

    # Cargar modelos locales
    modelo_flete = joblib.load('modelo_costoflete.sav')
    modelo_dias = joblib.load('modelo_dias_pipeline_70.joblib')
    label_encoder = joblib.load('label_encoder_dias_70.joblib')

    # --------- SIDEBAR: Filtro de Estado ---------
    with st.sidebar:
        st.subheader("🎛️ Filtro de Estado")
        estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
        estado_sel = option_menu(
            menu_title="Selecciona un estado",
            options=estados,
            icons=["globe"] + ["geo"] * (len(estados) - 1),
            default_index=0
        )

    # --------- FILTRADO DE DATOS ---------
    df_filtrado = df.copy() if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel]

# ===================== 📊 RESUMEN NACIONAL =====================
with tabs[0]:
    if archivo_csv:
        zona_display = estado_sel if estado_sel != "Nacional" else "Resumen Nacional"
        st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {zona_display}")

        # --------- MÉTRICAS PRINCIPALES ---------
        col1, col2 = st.columns(2)
        col1.metric("Pedidos", f"{len(df_filtrado):,}")
        col2.metric(
            "Llegadas muy adelantadas (≥10 días)",
            f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean() * 100:.1f}%"
        )

        if 'dias_entrega' in df_filtrado.columns:

            # ========== FILA 1 ==========
            col1, col2 = st.columns(2)

            # --------- Barras: Entregas a tiempo vs tardías ---------
            with col1:
                df_tmp = df_filtrado.copy()
                df_tmp['zona_entrega'] = clasificar_zonas(df_tmp, estado_sel)
                df_tmp['estatus_entrega'] = df_tmp['llego_tarde'].apply(lambda x: 'A tiempo' if x == 0 else 'Tardío')

                conteo_zona = df_tmp.groupby(['zona_entrega', 'estatus_entrega']).size().reset_index(name='conteo')
                conteo_zona['porcentaje'] = conteo_zona['conteo'] / conteo_zona.groupby('zona_entrega')['conteo'].transform('sum') * 100
                orden_zonas = df_tmp['zona_entrega'].value_counts().index.tolist()

                fig = px.bar(
                    conteo_zona,
                    x='zona_entrega',
                    y='porcentaje',
                    color='estatus_entrega',
                    category_orders={'zona_entrega': orden_zonas},
                    color_discrete_map={'A tiempo': '#A7D3F4', 'Tardío': '#B0B0B0'},
                    title="🚚 Si somos puntuales, ¿cuál es el problema?",
                    labels={'zona_entrega': 'Zona', 'porcentaje': 'Porcentaje', 'estatus_entrega': 'Tipo de Entrega'},
                    text_auto='.1f'
                )

                fig.update_traces(hovertemplate="<b>%{x}</b><br>%{color}: %{y:.1f}%")
                fig.update_layout(barmode='stack', xaxis_title=None, yaxis_title='Porcentaje (%)',
                                  legend_title='Tipo de Entrega', height=500)

                st.plotly_chart(fig, use_container_width=True)

            # --------- Gráfico horizontal: Días vs colchón ---------
            with col2:
                label_plural = "Ciudades" if estado_sel != "Nacional" else "Estados"
                st.subheader(f"📦 {label_plural} con mayor colchón de entrega")

                if (
                    {'dias_entrega', 'colchon_dias'}.issubset(df_filtrado.columns)
                    and not df_filtrado[['dias_entrega', 'colchon_dias']].dropna().empty
                ):
                    import plotly.graph_objects as go

                    df_tmp = df_filtrado.copy()
                    df_tmp['zona_entrega'] = clasificar_zonas(df_tmp, estado_sel)

                    medios = df_tmp.groupby('zona_entrega')[['dias_entrega', 'colchon_dias']].mean().reset_index()
                    medios = medios.sort_values(by='dias_entrega', ascending=False)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=medios['zona_entrega'],
                        x=medios['dias_entrega'],
                        name='Días Entrega',
                        orientation='h',
                        marker_color='#4FA0D9'
                    ))
                    fig.add_trace(go.Bar(
                        y=medios['zona_entrega'],
                        x=medios['colchon_dias'],
                        name='Colchón Días',
                        orientation='h',
                        marker_color='#B0B0B0'
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
                        height=500,
                        xaxis_title='Días Promedio',
                        yaxis_title=label_plural,
                        margin=dict(t=40, b=40, l=80, r=10),
                        legend_title="Métrica"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay datos suficientes para mostrar el gráfico.")

            # ========== FILA 2 ==========
            col3, col4 = st.columns(2)

            # --------- Barras: Días de entrega por zona dinámica ---------
            with col3:
                df_tmp = df_filtrado[df_filtrado['dias_entrega'].notna()].copy()
                df_tmp['grupo_dias'] = pd.cut(
                    df_tmp['dias_entrega'],
                    bins=[0, 5, 10, float('inf')],
                    labels=["1-5", "6-10", "Más de 10"]
                )
                df_tmp['zona_entrega'] = clasificar_zonas(df_tmp, estado_sel)

                conteo = df_tmp.groupby(['zona_entrega', 'grupo_dias']).size().reset_index(name='conteo')
                conteo['porcentaje'] = conteo['conteo'] / conteo.groupby('zona_entrega')['conteo'].transform('sum') * 100
                orden_zonas = df_tmp['zona_entrega'].value_counts().index.tolist()

                colores_dias = {
                    "1-5": "#A7D3F4",
                    "6-10": "#4FA0D9",
                    "Más de 10": "#FF6B6B"
                }

                fig_barras = px.bar(
                    conteo,
                    x='zona_entrega',
                    y='porcentaje',
                    color='grupo_dias',
                    category_orders={'zona_entrega': orden_zonas},
                    color_discrete_map=colores_dias,
                    title="📦 ¿Éxito logístico o maquillaje de tiempos?",
                    labels={
                        'zona_entrega': 'Zona',
                        'porcentaje': 'Porcentaje',
                        'grupo_dias': 'Días de Entrega'
                    },
                    text_auto='.1f'
                )

                fig_barras.update_layout(
                    barmode='stack',
                    xaxis_title=None,
                    yaxis_title='Porcentaje (%)',
                    legend_title='Días de Entrega',
                    height=500
                )

                st.plotly_chart(fig_barras, use_container_width=True)

            # --------- Dona: Pedidos por zona dinámica ---------
            with col4:
                df_tmp = df_filtrado.copy()
                df_tmp['zona_entrega'] = clasificar_zonas(df_tmp, estado_sel)

                conteo_zona = df_tmp['zona_entrega'].value_counts().reset_index()
                conteo_zona.columns = ['Zona', 'Pedidos']
                zonas = conteo_zona['Zona'].tolist()

                tonos_azules = [
                    '#08306b', '#08519c', '#2171b5', '#4292c6',
                    '#6baed6', '#9ecae1', '#c6dbef', '#deebf7'
                ]

                colores_discretos = {
                    zona: '#B0B0B0' if zona == 'Provincia' else tonos_azules[i % len(tonos_azules)]
                    for i, zona in enumerate(zonas)
                }

                fig_pie = px.pie(
                    conteo_zona,
                    names='Zona',
                    values='Pedidos',
                    hole=0.4,
                    title="📍 Pedidos por Zona",
                    color='Zona',
                    color_discrete_map=colores_discretos
                )

                fig_pie.update_traces(
                    textinfo='label+percent',
                    textfont_color='white',
                    hovertemplate="<b>%{label}</b><br>Porcentaje: %{percent}"
                )

                st.plotly_chart(fig_pie, use_container_width=True)


# ========================= PESTAÑA 1: Costo de Envío =========================
with tabs[1]:
    if archivo_csv: 
        # ==================== MÉTRICAS PRINCIPALES ====================
        col1, col2 = st.columns(2)
        col1.metric("📦 Total de Pedidos", f"{len(df_filtrado):,}")
        col2.metric(
            "💰 Flete Alto vs Precio",
            f"{(df_filtrado['costo_de_flete'] / df_filtrado['precio'] > 0.5).mean() * 100:.1f}%"
        )

        # ==================== TABLA DE % FLETE SOBRE PRECIO ====================
        st.subheader("💸 Relación Envío–Precio: ¿Gasto Justificado?")
        
        df_precio = df_filtrado.copy()
        df_precio['porcentaje_flete'] = (df_precio['costo_de_flete'] / df_precio['precio']) * 100
        
        tabla = df_precio.groupby('categoria')['porcentaje_flete'].mean().reset_index()
        tabla = tabla.sort_values(by='porcentaje_flete', ascending=False)
        
        # Aplicar el emoji y formatear
        tabla['porcentaje_flete_raw'] = tabla['porcentaje_flete']  # guardamos valor real para condicional
        tabla['porcentaje_flete'] = tabla['porcentaje_flete'].apply(
            lambda x: f"🔺 {x:.1f}%" if x >= 40 else f"{x:.1f}%"
        )
        
        tabla_h = tabla.set_index('categoria')[['porcentaje_flete']].T
        
        # Estilo condicional: rojo si ≥ 40%
        def highlight_if_high(s):
            return [
                'color: red; font-weight: bold' if '🔺' in str(v) else ''
                for v in s
            ]
        
        st.dataframe(
            tabla_h.style.apply(highlight_if_high, axis=1),
            use_container_width=True,
            height=100,
            hide_index=True
        )

        # ==================== GRÁFICAS COMPARATIVAS ====================
        col1, col2 = st.columns(2)

        # --------- BARRA: Precio vs Flete por Categoría ---------
        with col1:
            totales = df_filtrado.groupby('categoria')[['precio', 'costo_de_flete']].sum().reset_index()
            totales = totales.rename(columns={
                'precio': 'Precio',
                'costo_de_flete': 'Costo de Flete'
            })
            totales = totales.sort_values(by='Precio', ascending=False)

            fig_totales = px.bar(
                totales,
                x='categoria',
                y=['Precio', 'Costo de Flete'],
                barmode='group',
                title="📊 Total Precio vs Costo de Envío",
                labels={'value': 'Monto ($)', 'variable': 'Concepto'},
                color_discrete_map={
                    'Precio': '#005BAC',
                    'Costo de Flete': '#4FA0D9'
                }
            )

            fig_totales.update_layout(
                height=360,
                margin=dict(t=40, b=60, l=10, r=10),
                legend_title="",
                legend=dict(
                    orientation="v",          # vertical en lugar de horizontal
                    yanchor="top",
                    y=1,
                    xanchor="right",
                    x=1
                )
            )

            fig_totales.update_traces(
                hovertemplate="<b>%{x}</b><br>%{legendgroup}: $%{y:,.0f}<extra></extra>"
            )
            fig_totales.update_xaxes(tickangle=-40)

            st.plotly_chart(fig_totales, use_container_width=True)

        # --------- BOXPLOT: Variabilidad % Flete / Precio ---------
        with col2:
            # Agrupar por mes y calcular el promedio general (sin distinguir por año)
            df_promedio_mensual = df_filtrado.groupby('mes')['costo_de_flete'].mean().reset_index()
            
            # Convertir número de mes a nombre
            meses_texto = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                        'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            df_promedio_mensual['mes_nombre'] = df_promedio_mensual['mes'].apply(lambda x: meses_texto[x - 1])
            
            # Ordenar por calendario
            df_promedio_mensual = df_promedio_mensual.sort_values('mes')
            
            # Crear gráfica de línea
            fig = px.line(
                df_promedio_mensual,
                x='mes_nombre',
                y='costo_de_flete',
                markers=True,
                title="📈 Costo Promedio de Flete por Mes (Promedio General 2016–2018)",
                labels={'mes_nombre': 'Mes', 'costo_de_flete': 'Costo Promedio de Flete ($)'}
            )
            
            fig.update_layout(
                height=420,
                xaxis=dict(categoryorder='array', categoryarray=meses_texto),
                yaxis_title="Costo Promedio ($)",
                margin=dict(t=50, b=50, l=40, r=10)
            )
            
            fig.update_traces(line=dict(width=3, color='#2c7be5'), marker=dict(size=7, color='#2c7be5'))
            
            st.plotly_chart(fig, use_container_width=True)
        

# ========================= CALCULADORA =========================
with tabs[2]:
    if archivo_csv:
        import joblib
        from sklearn.base import BaseEstimator, TransformerMixin

        st.header("🧮 Calculadora de Predicción")

        meses_dict = {
            1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
            5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
            9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
        }

        df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
        df2['año'] = df2['orden_compra_timestamp'].dt.year
        df2['mes'] = df2['orden_compra_timestamp'].dt.month
        estado = estado_sel
        st.markdown(f"**Estado seleccionado:** {estado}")

        categoria = st.selectbox("Categoría", sorted(df2['categoria'].dropna().unique()))

        col1, col2 = st.columns(2)
        mes1_nombre = col1.selectbox("Mes 1", list(meses_dict.values()), index=0)
        mes2_nombre = col2.selectbox("Mes 2", list(meses_dict.values()), index=1)
        mes1 = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
        mes2 = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

        filtro = (df2['estado_del_cliente'] == estado) & (df2['categoria'] == categoria)
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

        def agrupar_resultados(df, nombre_mes):
            if 'costo_estimado' in df.columns and 'clase_entrega' in df.columns:
                return df.groupby('ciudad_cliente').agg({
                    'costo_estimado': lambda x: round(x.mean(), 2),
                    'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'NA'
                }).rename(columns={
                    'costo_estimado': nombre_mes,
                    'clase_entrega': f"Entrega {nombre_mes}"
                }).reset_index()
            return pd.DataFrame(columns=['ciudad_cliente', nombre_mes, f"Entrega {nombre_mes}"])

        df_mes1 = predecir(df_mes1)
        df_mes2 = predecir(df_mes2)

        res1 = agrupar_resultados(df_mes1, mes1_nombre)
        res2 = agrupar_resultados(df_mes2, mes2_nombre)
        comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')

        comparacion[f"Flete {mes1_nombre}"] = pd.to_numeric(comparacion[mes1_nombre], errors='coerce')
        comparacion[f"Flete {mes2_nombre}"] = pd.to_numeric(comparacion[mes2_nombre], errors='coerce')
        comparacion['Diferencia Flete'] = (
            comparacion[f"Flete {mes2_nombre}"] - comparacion[f"Flete {mes1_nombre}"]
        ).round(2)

        comparacion.drop(columns=[col for col in [mes1_nombre, mes2_nombre] if col in comparacion.columns], inplace=True)

        comparacion = comparacion[[ 
            'ciudad_cliente',
            f"Flete {mes1_nombre}",
            f"Flete {mes2_nombre}",
            'Diferencia Flete',
            f"Entrega {mes1_nombre}",
            f"Entrega {mes2_nombre}"
        ]].rename(columns={'ciudad_cliente': 'Ciudad'})

        def resaltar(val):
            if isinstance(val, (int, float, np.number)):
                if val > 0:
                    return 'color: green; font-weight: bold'
                elif val < 0:
                    return 'color: red; font-weight: bold'
            return ''

        st.markdown("---")
        cols_kpi_arriba = st.columns(3)
        cols_kpi_arriba[0].markdown(f"**Costo de Flete Promedio {mes1_nombre}**")
        cols_kpi_arriba[1].markdown("**% Cambio**")
        cols_kpi_arriba[2].markdown(f"**Costo de Flete Promedio {mes2_nombre}**")

        costo_prom_mes1 = comparacion[f"Flete {mes1_nombre}"].mean(skipna=True)
        costo_prom_mes2 = comparacion[f"Flete {mes2_nombre}"].mean(skipna=True)
        if costo_prom_mes1 and not np.isnan(costo_prom_mes1):
            cambio_pct = ((costo_prom_mes2 - costo_prom_mes1) / costo_prom_mes1) * 100
        else:
            cambio_pct = 0

        cols_kpi_arriba[0].markdown(
            f"<span style='font-size:28px; font-weight:bold'>${costo_prom_mes1:,.2f}</span>", unsafe_allow_html=True)
        color_cambio = 'green' if cambio_pct > 0 else 'red'
        cols_kpi_arriba[1].markdown(
            f"<span style='color:{color_cambio}; font-size:28px; font-weight:bold'>{cambio_pct:.2f}%</span>",
            unsafe_allow_html=True)
        cols_kpi_arriba[2].markdown(
            f"<span style='font-size:28px; font-weight:bold'>${costo_prom_mes2:,.2f}</span>", unsafe_allow_html=True)

        st.subheader(f"Comparación: {mes1_nombre} vs {mes2_nombre}")
        st.table(
            comparacion.style
            .applymap(resaltar, subset=['Diferencia Flete'])
            .format({
                f"Flete {mes1_nombre}": "${:,.2f}",
                f"Flete {mes2_nombre}": "${:,.2f}",
                "Diferencia Flete": "${:,.2f}"
            })
        )

        st.download_button(
            "⬇️ Descargar CSV",
            comparacion.to_csv(index=False),
            file_name="comparacion.csv",
            mime="text/csv"
        )
    else:
        st.info("Por favor sube un archivo CSV para activar esta sección.")

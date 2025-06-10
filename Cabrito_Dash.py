import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu
import unicodedata

# Función para normalizar nombres de columnas
def normalize(col):
    s = col.strip().lower().replace(' ', '_')
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

# Configuración de página
st.set_page_config(page_title="Cabrito Analytics", layout="wide")
st.markdown("""
<style>
  /* Ocultar watermark de Streamlit */
  .css-1wa3eu0 { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Definición de pestañas
tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "App Danu 📈"])

# Sidebar: subir CSV de datos
st.sidebar.header("Sube tu CSV de pedidos")
csv_file = st.sidebar.file_uploader("Selecciona un archivo .csv", type=["csv"])
if not csv_file:
    st.sidebar.warning("Por favor, sube un archivo .csv para continuar.")
    st.stop()

# Cargar datos y modelos
df = pd.read_csv(csv_file)
df.columns = [normalize(c) for c in df.columns]
modelo_flete   = joblib.load('modelo_costoflete.sav')
modelo_dias    = joblib.load('modelo_dias_pipeline_70.joblib')
label_encoder  = joblib.load('label_encoder_dias_70.joblib')

# Filtro de estado
with st.sidebar:
    st.subheader("🎛️ Filtro de Estado")
    estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique())
    estado_sel = option_menu(
        "Selecciona un estado", estados,
        icons=["globe"] + ["geo"]*(len(estados)-1), default_index=0
    )

# Filtrar df
df_filtrado = df.copy() if estado_sel == "Nacional" else df[df['estado_del_cliente']==estado_sel].copy()

# Función zonas
def clasificar_zonas(df_zone, sel):
    if sel == "Nacional":
        principales = ['ciudad_de_mexico','nuevo_leon','jalisco']
        return df_zone['estado_del_cliente'].apply(
            lambda x: x if normalize(x) in principales else 'Provincia'
        )
    top_cities = (
        df_zone[df_zone['estado_del_cliente']==sel]['ciudad_cliente']
        .value_counts().nlargest(3).index.tolist()
    )
    return df_zone['ciudad_cliente'].apply(lambda x: x if x in top_cities else 'Otras')

# Colores azules fuertes (reversa)
blue_seq = px.colors.sequential.Blues_r

# Pestaña 0: Resumen Nacional
with tabs[0]:
    title = estado_sel if estado_sel != "Nacional" else "Resumen Nacional"
    st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {title}")
    m1, m2 = st.columns(2)
    m1.metric("Pedidos", f"{len(df_filtrado):,}")
    if 'desviacion_vs_promesa' in df_filtrado:
        m2.metric(
            "Llegadas muy adelantadas (≥10 días)",
            f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean()*100:.1f}%"
        )
    c1, c2 = st.columns(2)
    with c1:
        tmp = df_filtrado.copy()
        tmp['zona'] = clasificar_zonas(tmp, estado_sel)
        cnt = tmp['zona'].value_counts().reset_index()
        cnt.columns = ['zona','pedidos']
        fig1 = px.pie(
            cnt, names='zona', values='pedidos', hole=0.4,
            title="📍 Pedidos por Zona",
            color_discrete_sequence=blue_seq
        )
        fig1.update_traces(
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Pedidos: %{value}<br>Porcentaje: %{percent}"
        )
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        if 'llego_tarde' in df_filtrado:
            tmp2 = df_filtrado.copy()
            tmp2['zona'] = clasificar_zonas(tmp2, estado_sel)
            tmp2['estatus'] = tmp2['llego_tarde'].map({0:'A tiempo',1:'Tardío'})
            grp = tmp2.groupby(['zona','estatus']).size().reset_index(name='count')
            grp['percent'] = grp['count']/grp.groupby('zona')['count'].transform('sum')*100
            fig2 = px.bar(
                grp, x='zona', y='percent', color='estatus', barmode='stack',
                title="🚚 Entregas A Tiempo vs Tardías",
                color_discrete_sequence=blue_seq
            )
            fig2.update_layout(xaxis_title='Zona', yaxis_title='Porcentaje (%)', legend_title='Estatus')
            st.plotly_chart(fig2, use_container_width=True)
    if all(col in df_filtrado.columns for col in ['dias_entrega','colchon_dias']):
        c3, c4 = st.columns(2)
        with c3:
            tmp3 = df_filtrado.copy()
            tmp3['grupo_dias'] = pd.cut(
                tmp3['dias_entrega'], bins=[0,5,10,float('inf')],
                labels=["1-5","6-10",">10"]
            )
            tmp3['zona'] = clasificar_zonas(tmp3, estado_sel)
            grp2 = tmp3.groupby(['zona','grupo_dias']).size().reset_index(name='count')
            grp2['percent'] = grp2['count']/grp2.groupby('zona')['count'].transform('sum')*100
            fig3 = px.bar(
                grp2, x='zona', y='percent', color='grupo_dias', barmode='stack',
                title="📦 Días de Entrega por Zona",
                color_discrete_map={"1-5":"#A7D3F4","6-10":"#4FA0D9",">10":"#FF6B6B"}
            )
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            tmp4 = df_filtrado.copy()
            tmp4['zona'] = clasificar_zonas(tmp4, estado_sel)
            medios = tmp4.groupby('zona')[['dias_entrega','colchon_dias']].mean().reset_index()
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                y=medios['zona'], x=medios['dias_entrega'],
                name='Días Entrega', orientation='h', marker_color=blue_seq[3]
            ))
            fig4.add_trace(go.Bar(
                y=medios['zona'], x=medios['colchon_dias'],
                name='Colchón Días', orientation='h', marker_color=blue_seq[1]
            ))
            fig4.update_layout(
                barmode='group', title='📦 Días vs Colchón por Zona',
                xaxis_title='Promedio Días', yaxis_title='Zona'
            )
            st.plotly_chart(fig4, use_container_width=True)

# Pestaña 1: Costo de Envío
with tabs[1]:
    c1, c2 = st.columns(2)
    c1.metric("📦 Total de Pedidos", f"{len(df_filtrado):,}")
    if all(col in df_filtrado.columns for col in ['costo_de_flete','precio']):
        c2.metric(
            "💰 Flete Alto vs Precio",
            f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%"
        )

        # → Aquí comienza la tabla horizontal con nombres abreviados
        st.subheader("💸 Relación Envío–Precio")
        tmp = df_filtrado.copy()
        tmp['porcentaje_flete'] = tmp['costo_de_flete'] / tmp['precio'] * 100

        if 'categoria' in tmp:
            # calculamos el promedio por categoría
            tbl = (
                tmp.groupby('categoria')['porcentaje_flete']
                   .mean()
                   .reset_index()
                   .sort_values('porcentaje_flete', ascending=False)
            )
            # formateamos el display
            tbl['display'] = tbl['porcentaje_flete'] \
                                 .apply(lambda v: f"🔺 {v:.1f}%" if v >= 40 else f"{v:.1f}%")

            # diccionario de abreviaturas para encabezados
            name_map = {
                'Festividades':             'Festividades',
                'Florería':                 'Florería',
                'Electronica y Tecnología': 'Electrónica/Tec.',
                'Alimentos y Bebidas':      'Alimentos/Bebidas',
                'Automotriz':               'Auto',
                'Industria y Comercio':     'Ind./Comercio',
                'Libros y Papelería':       'Libros/Papel',
                'Hogar y Muebles':          'Hogar',
                'Moda y Accesorios':        'Moda/Acc.',
                'Mascotas':                 'Mascotas',
                'Deportes':                 'Deportes',
                'Belleza y Salud':          'Belleza',
                'Bebés y Niños':            'Bebés',
                'Servicios y Otros':        'Servicios',
                'Arte y Manualidades':      'Arte/Manual.'
            }
            # aplicar abreviaturas
            tbl['categoria'] = tbl['categoria'].map(name_map).fillna(tbl['categoria'])

            # pivoteamos para que quede una sola fila
            tbl_horiz = (
                tbl[['categoria','display']]
                   .rename(columns={'display':'% Flete'})
                   .set_index('categoria')
                   .T
            )
            st.table(tbl_horiz)
        # ← Aquí termina la tabla horizontal

        # resto de gráficas...
        tot = df_filtrado.groupby('categoria')[['precio','costo_de_flete']].sum().reset_index()
        fig_tot = px.bar(
            tot, x='categoria', y=['precio','costo_de_flete'], barmode='group',
            title="📊 Total Precio vs Costo de Envío",
            labels={'value':'Monto ($)','variable':'Concepto'},
            color_discrete_sequence=blue_seq
        )
        st.plotly_chart(fig_tot, use_container_width=True)

        df_month = (
            df_filtrado.groupby('mes')['costo_de_flete']
                        .mean()
                        .reset_index()
        )
        meses_txt = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        df_month['mes_nombre'] = df_month['mes'].apply(lambda x: meses_txt[x-1])
        fig_line = px.line(
            df_month, x='mes_nombre', y='costo_de_flete', markers=True,
            title="📈 Costo Promedio de Flete por Mes",
            labels={'mes_nombre':'Mes','costo_de_flete':'Costo Promedio ($)'},
            color_discrete_sequence=blue_seq
        )
        st.plotly_chart(fig_line, use_container_width=True)

# Pestaña 2: Calculadora de Predicción
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")
    if 'orden_compra_timestamp' in df:
        df['orden_compra_timestamp'] = pd.to_datetime(df['orden_compra_timestamp'], errors='coerce')
        df['año']  = df['orden_compra_timestamp'].dt.year
        df['mes']  = df['orden_compra_timestamp'].dt.month

    est2 = st.selectbox("Estado", sorted(df['estado_del_cliente'].dropna().unique()))
    cat2 = st.selectbox("Categoría", sorted(df['categoria'].dropna().unique()))

    meses = {
        1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
        7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",
        11:"Noviembre

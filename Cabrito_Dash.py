import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu
import unicodedata

# — Función para normalizar nombres de columnas —
def normalize(col):
    s = col.strip().lower().replace(' ', '_')
    return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

# — Configuración de página —
st.set_page_config(page_title="Cabrito Analytics", layout="wide")
st.markdown("""
<style>
  /* Ocultar watermark de Streamlit */
  .css-1wa3eu0 { display: none !important; }
</style>
""", unsafe_allow_html=True)

# — Definición de pestañas —
tabs = st.tabs([
    "📊 Resumen Nacional",
    "🏠 Costo de Envío",
    "🧮 Calculadora",
    "App Danu 📈"
])

# — Sidebar: subir CSV de datos —
st.sidebar.header("Sube tu CSV de pedidos")
csv_file = st.sidebar.file_uploader("Selecciona un archivo .csv", type=["csv"])
if not csv_file:
    st.sidebar.warning("Por favor, sube un archivo .csv para continuar.")
    st.stop()

# — Cargar datos y modelos —
df = pd.read_csv(csv_file)
df.columns = [normalize(c) for c in df.columns]

modelo_flete = joblib.load('modelo_costoflete.sav')
modelo_dias = joblib.load('modelo_dias_pipeline_70.joblib')
label_encoder = joblib.load('label_encoder_dias_70.joblib')

# — Filtro de estado —
with st.sidebar:
    st.subheader("🎛️ Filtro de Estado")
    estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique())
    estado_sel = option_menu(
        "Selecciona un estado",
        estados,
        icons=["globe"] + ["geo"] * (len(estados)-1),
        default_index=0
    )

# — Filtrar df para pestañas 0 y 1 —
if estado_sel == "Nacional":
    df_filtrado = df.copy()
else:
    df_filtrado = df[df['estado_del_cliente'] == estado_sel].copy()

# — Función para clasificar zonas en pestaña 0 —
def clasificar_zonas(df_zone, sel):
    if sel == "Nacional":
        top_states = ['ciudad_de_mexico', 'nuevo_leon', 'jalisco']
        return df_zone['estado_del_cliente'].apply(
            lambda x: x if normalize(x) in top_states else 'Provincia'
        )
    else:
        top_cities = df_zone[df_zone['estado_del_cliente']==sel]['ciudad_cliente'] \
            .value_counts().nlargest(3).index.tolist()
        return df_zone['ciudad_cliente'].apply(
            lambda x: x if x in top_cities else 'Otras'
        )

# — Pestaña 0: Resumen Nacional —
with tabs[0]:
    title = estado_sel if estado_sel != "Nacional" else "Resumen Nacional"
    st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {title}")

    # Métricas principales
    m1, m2 = st.columns(2)
    m1.metric("Pedidos", f"{len(df_filtrado):,}")
    if 'desviacion_vs_promesa' in df_filtrado:
        m2.metric(
            "Llegadas muy adelantadas (≥10 días)",
            f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean()*100:.1f}%"
        )

    # Fila 1: Dona y barras apiladas (azul)
    c1, c2 = st.columns(2)
    with c1:
        tmp = df_filtrado.copy()
        tmp['zona'] = clasificar_zonas(tmp, estado_sel)
        cnt = tmp['zona'].value_counts().reset_index()
        cnt.columns = ['zona','pedidos']
        fig1 = px.pie(
            cnt, names='zona', values='pedidos', hole=0.4,
            title="📍 Pedidos por Zona",
            color_discrete_sequence=px.colors.sequential.Blues
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
            grp['percent'] = grp['count'] / grp.groupby('zona')['count'].transform('sum') * 100
            fig2 = px.bar(
                grp, x='zona', y='percent', color='estatus', barmode='stack',
                title="🚚 Entregas A Tiempo vs Tardías",
                color_discrete_sequence=px.colors.sequential.Blues
            )
            fig2.update_layout(xaxis_title='Zona', yaxis_title='Porcentaje (%)', legend_title='Estatus')
            st.plotly_chart(fig2, use_container_width=True)

    # Fila 2: Días y colchón (mantiene escala original)
    if all(col in df_filtrado.columns for col in ['dias_entrega','colchon_dias']):
        c3, c4 = st.columns(2)
        with c3:
            tmp3 = df_filtrado.copy()
            tmp3['grupo_dias'] = pd.cut(
                tmp3['dias_entrega'],
                bins=[0,5,10,float('inf')],
                labels=["1-5","6-10",">10"]
            )
            tmp3['zona'] = clasificar_zonas(tmp3, estado_sel)
            grp2 = tmp3.groupby(['zona','grupo_dias']).size().reset_index(name='count')
            grp2['percent'] = grp2['count'] / grp2.groupby('zona')['count'].transform('sum') * 100
            fig3 = px.bar(
                grp2, x='zona', y='percent', color='grupo_dias', barmode='stack',
                title="📦 Días de Entrega por Zona",
                color_discrete_map={"1-5":"#A7D3F4","6-10":"#4FA0D9",">10":"#FF6B6B"}
            )
            fig3.update_layout(xaxis_title='Zona', yaxis_title='Porcentaje (%)', legend_title='Rango Días')
            st.plotly_chart(fig3, use_container_width=True)
        with c4:
            tmp4 = df_filtrado.copy()
            tmp4['zona'] = clasificar_zonas(tmp4, estado_sel)
            medios = tmp4.groupby('zona')[['dias_entrega','colchon_dias']].mean().reset_index()
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                y=medios['zona'], x=medios['dias_entrega'],
                name='Días Entrega', orientation='h',
                marker_color=px.colors.sequential.Blues[3]
            ))
            fig4.add_trace(go.Bar(
                y=medios['zona'], x=medios['colchon_dias'],
                name='Colchón Días', orientation='h',
                marker_color=px.colors.sequential.Blues[1]
            ))
            fig4.update_layout(
                barmode='group',
                title='📦 Días vs Colchón por Zona',
                xaxis_title='Promedio Días',
                yaxis_title='Zona'
            )
            st.plotly_chart(fig4, use_container_width=True)

# — Pestaña 1: Costo de Envío —
with tabs[1]:
    c1, c2 = st.columns(2)
    c1.metric("📦 Total de Pedidos", f"{len(df_filtrado):,}")
    if all(col in df_filtrado.columns for col in ['costo_de_flete','precio']):
        c2.metric(
            "💰 Flete Alto vs Precio",
            f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%"
        )
        st.subheader("💸 Relación Envío–Precio")
        tmp = df_filtrado.copy()
        tmp['porcentaje_flete'] = tmp['costo_de_flete']/tmp['precio']*100
        if 'categoria' in tmp:
            tbl = tmp.groupby('categoria')['porcentaje_flete'] \
                     .mean().reset_index() \
                     .sort_values('porcentaje_flete', ascending=False)
            tbl['display'] = tbl['porcentaje_flete'] \
                             .apply(lambda v: f"🔺 {v:.1f}%" if v>=40 else f"{v:.1f}%")
            st.table(tbl[['categoria','display']].rename(columns={'display':'% Flete'}))
        # Total Precio vs Costo de Envío en azul
        tot = df_filtrado.groupby('categoria')[['precio','costo_de_flete']] \
                         .sum().reset_index()
        fig_tot = px.bar(
            tot,
            x='categoria', y=['precio','costo_de_flete'],
            barmode='group',
            title="📊 Total Precio vs Costo de Envío",
            labels={'value':'Monto ($)','variable':'Concepto'},
            color_discrete_sequence=px.colors.sequential.Blues
        )
        st.plotly_chart(fig_tot, use_container_width=True)
        # Línea de costo promedio por mes en azul
        df_month = df_filtrado.groupby('mes')['costo_de_flete'] \
                              .mean().reset_index()
        meses_txt = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        df_month['mes_nombre'] = df_month['mes'].apply(lambda x: meses_txt[x-1])
        fig_line = px.line(
            df_month,
            x='mes_nombre', y='costo_de_flete', markers=True,
            title="📈 Costo Promedio de Flete por Mes",
            labels={'mes_nombre':'Mes','costo_de_flete':'Costo Promedio ($)'},
            color_discrete_sequence=px.colors.sequential.Blues
        )
        st.plotly_chart(fig_line, use_container_width=True)

# — Pestaña 2: Calculadora de Predicción —
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")
    # Prepara dataframe con año/mes
    if 'orden_compra_timestamp' in df:
        df['orden_compra_timestamp'] = pd.to_datetime(df['orden_compra_timestamp'], errors='coerce')
        df['año'] = df['orden_compra_timestamp'].dt.year
        df['mes'] = df['orden_compra_timestamp'].dt.month

    est2 = st.selectbox("Estado", sorted(df['estado_del_cliente'].dropna().unique()))
    cat2 = st.selectbox("Categoría", sorted(df['categoria'].dropna().unique()))
    meses = {
        1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",
        5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",
        9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
    }
    m1n2 = st.selectbox("Mes 1", list(meses.values()), index=0)
    m2n2 = st.selectbox("Mes 2", list(meses.values()), index=1)
    m1_ = [k for k,v in meses.items() if v==m1n2][0]
    m2_ = [k for k,v in meses.items() if v==m2n2][0]

    df1 = df[(df['mes']==m1_) & (df['estado_del_cliente']==est2) & (df['categoria']==cat2)].copy()
    df2b = df[(df['mes']==m2_) & (df['estado_del_cliente']==est2) & (df['categoria']==cat2)].copy()

    def predecir(d):
        if d.empty:
            return d
        cols = [
            'total_peso_g','precio','#_deproductos','duracion_estimada_min',
            'ciudad_cliente','nombre_dc','hora_compra','año','mes',
            'datetime_origen','region','dias_promedio_ciudad','categoria','tipo_de_pago'
        ]
        d_f = d.reindex(columns=cols).copy()
        enc = pd.get_dummies(d_f)
        feats = modelo_flete.get_booster().feature_names
        enc = enc.reindex(columns=feats, fill_value=0)
        d['costo_estimado'] = modelo_flete.predict(enc).round(2)
        return d

    def agrupar(d, name):
        if 'costo_estimado' not in d:
            return pd.DataFrame(columns=['ciudad_cliente', name])
        agg = d.groupby('ciudad_cliente')['costo_estimado'] \
               .mean().round(2).reset_index() \
               .rename(columns={'costo_estimado': name})
        return agg

    r1 = agrupar(predecir(df1), m1n2)
    r2 = agrupar(predecir(df2b), m2n2)
    comp2 = pd.merge(r1, r2, on='ciudad_cliente', how='outer')
    comp2[m1n2] = pd.to_numeric(comp2[m1n2], errors='coerce')
    comp2[m2n2] = pd.to_numeric(comp2[m2n2], errors='coerce')
    comp2['Diferencia'] = (comp2[m2n2] - comp2[m1n2]).round(2)
    comp2 = comp2.rename(columns={'ciudad_cliente': 'Ciudad'})

    st.subheader(f"Comparación: {m1n2} vs {m2n2}")
    st.dataframe(comp2, use_container_width=True)
    st.download_button(
        "⬇️ Descargar CSV",
        data=comp2.to_csv(index=False).encode('utf-8'),
        file_name='comparacion.csv',
        mime='text/csv'
    )

# — Pestaña 3: App Danu —
with tabs[3]:
    pass

import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import numpy as np
from streamlit_option_menu import option_menu

# ----------------------------
# Configuración inicial y tema
st.set_page_config(page_title="Cabrito Analytics", layout="wide")
st.markdown(
    """
    <style>
      :root { --primary-blue: #003366; --secondary-blue: #6699cc; }
      .stButton > button {
        background-color: var(--primary-blue)!important;
        color: white!important;
        border: 1px solid var(--primary-blue)!important;
      }
      .stButton > button:hover {
        background-color: var(--secondary-blue)!important;
      }
      [data-baseweb="tag"] {
        background-color: var(--primary-blue)!important;
        color: white!important;
      }
      button[role="tab"][aria-selected="true"] {
        color: var(--primary-blue)!important;
        border-bottom: 3px solid var(--primary-blue)!important;
        font-weight: bold;
      }
      button[role="tab"]:hover {
        color: var(--primary-blue)!important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
st.title("📦 Cabrito Analytics App")
tabs = st.tabs([
    "📊 Resumen Nacional",
    "🏠 Costo de Envío",
    "🧮 Calculadora",
    "App Danu 📈"
])

# ----------------------------
# Sidebar: carga de ZIP
with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if not archivo_zip:
    st.stop()

# ----------------------------
# Carga de datos y modelos desde ZIP
with zipfile.ZipFile(archivo_zip) as z:
    requeridos = [
        'DF.csv',
        'DF2.csv',
        'modelo_costoflete.sav',
        'modelo_dias_pipeline.joblib',
        'label_encoder_dias.joblib'
    ]
    faltantes = [f for f in requeridos if f not in z.namelist()]
    if faltantes:
        st.error(f"❌ Faltan archivos en el ZIP: {faltantes}")
        st.stop()

    df  = pd.read_csv(z.open('DF.csv'))
    df2 = pd.read_csv(z.open('DF2.csv'))
    modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
    modelo_dias   = joblib.load(z.open('modelo_dias_pipeline.joblib'))
    label_encoder = joblib.load(z.open('label_encoder_dias.joblib'))

# ----------------------------
# Sidebar: filtro de estado
with st.sidebar:
    st.subheader("🎛️ Filtro de Estado")
    estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
    estado_sel = option_menu(
        menu_title=None,
        options=estados,
        icons=['globe'] + ['geo'] * (len(estados)-1),
        default_index=0,
        orientation='vertical'
    )

# Filtrado global
df_filtrado = df if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel]

# ==================== Resumen Nacional ====================
with tabs[0]:
    st.header(f"📊 Resumen Nacional de Entrega – {estado_sel}")
    # KPIs
    k1, k2 = st.columns(2)
    k1.metric("Total Pedidos", f"{len(df_filtrado):,}")
    k2.metric("Llegadas ≥10 días antes", f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean()*100:.1f}%")

    # Gráfica 1: desviación promedio por categoría
    agg_cat = df_filtrado.groupby('Categoría').agg(
        Promedio_Desvio=('desviacion_vs_promesa', 'mean')
    ).reset_index()
    fig_cat = px.bar(
        agg_cat,
        x='Categoría',
        y='Promedio_Desvio',
        color='Promedio_Desvio',
        color_continuous_scale=['#6699cc', '#003366'],
        labels={'Promedio_Desvio':'Días'},
        title='Desviación promedio vs promesa por Categoría',
        template='plotly_white'
    )
    fig_cat.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=350)
    st.plotly_chart(fig_cat, use_container_width=True)

    # Gráfica 2: evolución mensual de desviación promedio
    agg_time = df_filtrado.groupby(['año', 'mes']).agg(
        Promedio_Desvio=('desviacion_vs_promesa', 'mean')
    ).reset_index()
    agg_time['Fecha'] = pd.to_datetime(
        dict(year=agg_time['año'], month=agg_time['mes'], day=1)
    )
    fig_time = px.line(
        agg_time,
        x='Fecha',
        y='Promedio_Desvio',
        labels={'Promedio_Desvio':'Días'},
        title='Evolución mensual de desviación promedio',
        template='plotly_white'
    )
    fig_time.update_traces(line_color='#003366')
    fig_time.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=350)
    st.plotly_chart(fig_time, use_container_width=True)

    # Gráfica 3: Top 10 categorías con mayor desviación
    top10 = agg_cat.nsmallest(10, 'Promedio_Desvio')
    fig_top = px.bar(
        top10,
        x='Categoría',
        y='Promedio_Desvio',
        color_discrete_sequence=['#003366'],
        labels={'Promedio_Desvio':'Días'},
        title='Top 10 categorías con mayor desviación',
        template='plotly_white'
    )
    fig_top.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=350)
    st.plotly_chart(fig_top, use_container_width=True)

# ==================== Costo de Envío ====================
with tabs[1]:
    st.header(f"🏠 Costo de Envío vs Precio – {estado_sel}")
    c1, c2 = st.columns(2)
    c1.metric("Total Pedidos", f"{len(df_filtrado):,}")
    c2.metric("% Flete >50% precio", f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%")

    # Gráfica 1: porcentaje promedio de flete por categoría
    df_filtrado['Pct_Flete'] = df_filtrado['costo_de_flete'] / df_filtrado['precio']
    agg_flete = df_filtrado.groupby('Categoría').agg(
        Avg_Pct=('Pct_Flete','mean')
    ).reset_index()
    fig_f = px.bar(
        agg_flete,
        x='Categoría',
        y='Avg_Pct',
        color='Avg_Pct',
        color_continuous_scale=['#6699cc','#003366'],
        labels={'Avg_Pct':'% Flete'},
        title='Porcentaje promedio de flete por Categoría',
        template='plotly_white'
    )
    fig_f.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=350)
    st.plotly_chart(fig_f, use_container_width=True)

    # Gráfica 2: scatter flete vs precio
    fig_scatter = px.scatter(
        df_filtrado,
        x='precio',
        y='costo_de_flete',
        color='Categoría',
        labels={'precio':'Precio','costo_de_flete':'Costo de Flete'},
        title='Costo de flete vs Precio',
        template='plotly_white'
    )
    fig_scatter.update_traces(marker=dict(size=7))
    fig_scatter.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=350)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Gráfica 3: histograma de porcentaje de flete
    fig_hist = px.histogram(
        df_filtrado,
        x='Pct_Flete',
        nbins=20,
        labels={'Pct_Flete':'% Flete'},
        title='Distribución de porcentaje de flete',
        template='plotly_white'
    )
    fig_hist.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

# ==================== Calculadora ====================
with tabs[2]:
    st.header("🧮 Calculadora de Predicción de Flete y Entrega")

    # Preprocesamiento de fechas
    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    # Selección de estado y categoría
    col1, col2 = st.columns(2)
    estado2    = col1.selectbox("Estado", sorted(df2['estado_del_cliente'].dropna().unique()))
    categoria2 = col2.selectbox("Categoría", sorted(df2['Categoría'].dropna().unique()))

    # Selección de meses
    meses_dict = {
        1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril",
        5:"Mayo", 6:"Junio", 7:"Julio", 8:"Agosto",
        9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
    }
    mcol1, mcol2 = st.columns(2)
    mes1_nombre = mcol1.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2_nombre = mcol2.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1 = [k for k,v in meses_dict.items() if v == mes1_nombre][0]
    mes2 = [k for k,v in meses_dict.items() if v == mes2_nombre][0]

    # Filtrar los dataframes por estado, categoría y mes
    dfm1 = df2[
        (df2['estado_del_cliente'] == estado2) &
        (df2['Categoría'] == categoria2) &
        (df2['mes'] == mes1)
    ].copy()
    dfm2 = df2[
        (df2['estado_del_cliente'] == estado2) &
        (df2['Categoría'] == categoria2) &
        (df2['mes'] == mes2)
    ].copy()

    # Función de predicción
    def predecir(df_in):
        if df_in.empty:
            return df_in
        cols_f = [
            'total_peso_g','precio','#_deproductos','duracion_estimada_min',
            'ciudad_cliente','nombre_dc','hora_compra','año','mes',
            'datetime_origen','region','dias_promedio_ciudad',
            'Categoría','tipo_de_pago'
        ]
        Xf = pd.get_dummies(df_in[cols_f])
        feats = modelo_flete.get_booster().feature_names
        Xf = Xf.reindex(columns=feats, fill_value=0)
        df_in['costo_estimado'] = modelo_flete.predict(Xf).round(2)
        df_in['costo_de_flete'] = df_in['costo_estimado']
        dias_feats = [c for c in df_in.columns if c in modelo_dias.feature_names_in_]
        if dias_feats:
            preds = modelo_dias.predict(df_in[dias_feats])
            df_in['clase_entrega'] = label_encoder.inverse_transform(preds)
        return df_in

    # Función de agrupación por ciudad
    def agrupar(df_p, nombre):
        if 'costo_estimado' not in df_p.columns:
            return pd.DataFrame(columns=['Ciudad', nombre])
        out = df_p.groupby('ciudad_cliente').agg(
            **{nombre:('costo_estimado','mean')},
            **{f"Entrega {nombre}":('clase_entrega',
                                    lambda x: x.mode()[0] if not x.mode().empty else 'N/A')}
        ).reset_index().rename(columns={'ciudad_cliente':'Ciudad'})
        return out

    # Ejecutar predicción y agrupación
    dfp1 = predecir(dfm1)
    dfp2 = predecir(dfm2)
    r1 = agrupar(dfp1, mes1_nombre)
    r2 = agrupar(dfp2, mes2_nombre)
    comp = pd.merge(r1, r2, on='Ciudad', how='outer')

    # Calcular diferencias
    for c in [mes1_nombre, mes2_nombre]:
        comp[c] = pd.to_numeric(comp.get(c), errors='coerce')
    comp['Diferencia'] = (comp[mes2_nombre] - comp[mes1_nombre]).round(2)

    # KPIs comparativos
    kk1, kk2, kk3 = st.columns(3)
    avg1 = dfp1['costo_estimado'].mean() if not dfp1.empty else np.nan
    avg2 = dfp2['costo_estimado'].mean() if not dfp2.empty else np.nan
    pct  = ((avg2 - avg1) / avg1 * 100) if avg1 else 0
    kk1.metric(f"Avg {mes1_nombre}", f"{avg1:.2f}")
    kk2.metric("% Cambio", f"{pct:.1f}%")
    kk3.metric(f"Avg {mes2_nombre}", f"{avg2:.2f}")

    # Tabla de comparación
    def style_diff(val):
        return 'color: green' if val > 0 else ('color: red' if val < 0 else '')
    st.subheader(f"Comparación {mes1_nombre} vs {mes2_nombre} — {estado2}/{categoria2}")
    st.dataframe(
        comp.style
            .applymap(style_diff, subset=['Diferencia'])
            .format(precision=2)
    )
    st.download_button(
        "⬇️ Descargar CSV",
        comp.to_csv(index=False),
        file_name="calculadora_comparacion.csv"
    )

# ==================== App Danu ====================
with tabs[3]:
    st.header("App Danu – Insights 📈")
    st.write("Aquí va contenido adicional de App Danu...")

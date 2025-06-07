# dash2.py

import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------------------
# 1) Esta llamada DEBE IR PRIMERO
st.set_page_config(page_title="Cabrito Analytics Profesional", layout="wide")

# 2) Sobrescribimos la variable de tema para que el subrayado activo ya no salga rojo, sino nuestro azul
st.markdown(
    """
    <style>
    :root {
      --theme-primary-color: #003366 !important;
      --primary-blue: #003366;
      --secondary-blue: #6699cc;
    }
    /* Botones en general */
    .stButton > button {
      background-color: var(--primary-blue) !important;
      color: white !important;
      border: 1px solid var(--primary-blue) !important;
    }
    .stButton > button:hover {
      background-color: var(--secondary-blue) !important;
    }
    /* Etiquetas de multiselect */
    [data-baseweb="tag"] {
      background-color: var(--primary-blue) !important;
      color: white !important;
    }
    /* Pestaña activa (underline) */
    [data-baseweb="tab"][aria-selected="true"] {
      color: var(--primary-blue) !important;
      border-bottom: 3px solid var(--primary-blue) !important;
      font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------------------
st.title("📦 Cabrito Analytics App")
tabs = st.tabs(["🏠 Dashboard", "🧮 Calculadora"])

# ------------------ Sidebar: carga + filtros ------------------
with st.sidebar:
    st.header("Sube tu archivo ZIP")
    archivo = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if archivo:
    # cargar datos y modelos
    with zipfile.ZipFile(archivo) as z:
        necesarios = ["DF.csv", "DF2.csv", "modelo_costoflete.sav",
                      "modelo_dias_pipeline.joblib", "label_encoder_dias.joblib"]
        falt = [f for f in necesarios if f not in z.namelist()]
        if falt:
            st.error(f"❌ Faltan archivos: {falt}")
            st.stop()
        df = pd.read_csv(z.open("DF.csv"))
        df2 = pd.read_csv(z.open("DF2.csv"))
        modelo_flete = joblib.load(z.open("modelo_costoflete.sav"))
        modelo_dias   = joblib.load(z.open("modelo_dias_pipeline.joblib"))
        encoder       = joblib.load(z.open("label_encoder_dias.joblib"))

    # filtros dinámicos
    estados    = sorted(df["estado_del_cliente"].dropna().unique())
    categorias = sorted(df["Categoría"].dropna().unique())
    if "sel_est" not in st.session_state:
        st.session_state.sel_est = estados.copy()
    if "sel_cat" not in st.session_state:
        st.session_state.sel_cat = categorias.copy()

    with st.sidebar:
        st.subheader("Filtros de Dashboard")
        if st.button("Seleccionar todo (Estados)", key="btn_est"):
            st.session_state.sel_est = estados.copy()
        sel_est = st.multiselect("Estados", estados,
                                 default=st.session_state.sel_est,
                                 key="sel_est")
        if st.button("Seleccionar todo (Categorías)", key="btn_cat"):
            st.session_state.sel_cat = categorias.copy()
        sel_cat = st.multiselect("Categorías", categorias,
                                 default=st.session_state.sel_cat,
                                 key="sel_cat")

    # ========================= DASHBOARD =========================
    with tabs[0]:
        st.header("🏠 Dashboard Logístico")
        st.markdown(
            """
**Desfase estimado vs real de entrega**  
Analiza las métricas clave y la evolución temporal sin desplazamientos.
            """
        )
        df_sel = df[df["estado_del_cliente"].isin(sel_est) &
                    df["Categoría"].isin(sel_cat)].copy()
        df_sel["prometido_dias"] = (df_sel["dias_entrega"] -
                                    df_sel["desviacion_vs_promesa"])
        # KPI
        e_mean = df_sel["prometido_dias"].mean()
        r_mean = df_sel["dias_entrega"].mean()
        d_mean = e_mean - r_mean
        k1, k2, k3 = st.columns(3)
        k1.metric("Estimado promedio (d)", f"{e_mean:.1f}")
        k2.metric("Real promedio (d)",    f"{r_mean:.1f}")
        k3.metric("Desfase promedio (d)", f"{d_mean:.1f}")

        # Datos agregados
        agg_cat = df_sel.groupby("Categoría").agg(
            Estimado=("prometido_dias", "mean"),
            Real=("dias_entrega", "mean")
        ).reset_index()
        agg_cat["Desfase"] = agg_cat["Estimado"] - agg_cat["Real"]
        agg_time = (df_sel.groupby(["año", "mes"])
                    .agg(Estimado=("prometido_dias", "mean"),
                         Real=("dias_entrega", "mean"))
                    .reset_index()
                    .sort_values(["año", "mes"]))
        agg_time["Fecha"] = pd.to_datetime(
            dict(year=agg_time["año"], month=agg_time["mes"], day=1)
        )

        # Tres gráficas en fila
        colA, colB, colC = st.columns(3, gap="medium")

        # 1) Barras por categoría
        fig1 = px.bar(agg_cat, x="Categoría", y=["Estimado", "Real"],
                      barmode="group",
                      color_discrete_map={
                        "Estimado": "#003366",
                        "Real":      "#6699cc"},
                      labels={"value": "Días", "variable": "Tipo"},
                      title="Estimado vs Real por Categoría",
                      template="plotly_white")
        fig1.add_hline(y=agg_cat["Real"].median(),
                       line_dash="dash", line_color="#6699cc",
                       annotation_text="Mediana Real",
                       annotation_position="top right")
        fig1.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=300)
        colA.plotly_chart(fig1, use_container_width=True)

        # 2) Línea evolución mensual
        fig2 = px.line(agg_time, x="Fecha", y=["Estimado", "Real"],
                       color_discrete_map={
                         "Estimado": "#003366",
                         "Real":      "#6699cc"},
                       labels={"value": "Días", "variable": "Tipo"},
                       title="Evolución Mensual",
                       template="plotly_white")
        fig2.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=300)
        colB.plotly_chart(fig2, use_container_width=True)

        # 3) Top10 desfase
        top10 = agg_cat.nlargest(10, "Desfase")
        fig3 = px.bar(top10, x="Categoría", y="Desfase",
                      labels={"Desfase": "Días de desfase"},
                      title="Top 10 Categorías con Mayor Desfase",
                      color_discrete_sequence=["#003366"],
                      template="plotly_white")
        fig3.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=300)
        colC.plotly_chart(fig3, use_container_width=True)

    # ========================= CALCULADORA =========================
    with tabs[1]:
        st.header("🧮 Calculadora de Predicción")

        df2["orden_compra_timestamp"] = pd.to_datetime(
            df2["orden_compra_timestamp"])
        df2["año"] = df2["orden_compra_timestamp"].dt.year
        df2["mes"] = df2["orden_compra_timestamp"].dt.month

        estados2    = sorted(df2["estado_del_cliente"].dropna().unique())
        categorias2 = sorted(df2["Categoría"].dropna().unique())
        d1, d2 = st.columns(2)
        sel_e2 = d1.selectbox("Estado",    estados2)
        sel_c2 = d2.selectbox("Categoría", categorias2)

        meses_map = {
            1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",
            5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",
            9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
        }
        m1_name = d1.selectbox("Mes 1", list(meses_map.values()), index=0)
        m2_name = d2.selectbox("Mes 2", list(meses_map.values()), index=1)
        m1 = [k for k,v in meses_map.items() if v==m1_name][0]
        m2 = [k for k,v in meses_map.items() if v==m2_name][0]

        def predecir(df_i):
            if df_i.empty: return df_i
            cols_f = ["total_peso_g","precio","#_deproductos",
                      "duracion_estimada_min","ciudad_cliente","nombre_dc",
                      "hora_compra","año","mes","datetime_origen","region",
                      "dias_promedio_ciudad","Categoría","tipo_de_pago"]
            Xi = pd.get_dummies(df_i[cols_f])
            feats = modelo_flete.get_booster().feature_names
            Xi = Xi.reindex(columns=feats, fill_value=0)
            df_i["costo_estimado"] = modelo_flete.predict(Xi).round(2)
            df_i["costo_de_flete"] = df_i["costo_estimado"]
            return df_i

        def resumen(df_p, name):
            if "costo_estimado" not in df_p.columns:
                return pd.DataFrame(columns=["ciudad_cliente", name])
            return (df_p.groupby("ciudad_cliente")["costo_estimado"]
                        .mean().reset_index().rename(columns={"costo_estimado": name}))

        r1 = resumen(predecir(df2[(df2["mes"]==m1)&
                                  (df2["estado_del_cliente"]==sel_e2)&
                                  (df2["Categoría"]==sel_c2)]), m1_name)
        r2 = resumen(predecir(df2[(df2["mes"]==m2)&
                                  (df2["estado_del_cliente"]==sel_e2)&
                                  (df2["Categoría"]==sel_c2)]), m2_name)

        comp = pd.merge(r1, r2, on="ciudad_cliente", how="outer")
        for col in [m1_name, m2_name]:
            comp[col] = pd.to_numeric(comp[col], errors="coerce")
        comp["Diferencia"] = (comp[m2_name] - comp[m1_name]).round(2)
        comp.rename(columns={"ciudad_cliente":"Ciudad"}, inplace=True)

        st.dataframe(comp.style.format(precision=2))
        st.download_button("⬇️ Descargar CSV", comp.to_csv(index=False),
                           file_name="comparacion.csv")

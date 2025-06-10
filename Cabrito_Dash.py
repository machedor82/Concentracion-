import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from streamlit_option_menu import option_menu
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Cabrito Analytics", layout="wide")

# Sidebar para carga del CSV
with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("📂 Sube tu archivo CSV")
    archivo_csv = st.file_uploader("Archivo .csv con los datos completos", type="csv")

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

if archivo_csv:
    df = pd.read_csv(archivo_csv)
    df2 = df.copy()

    if 'desviacion_vs_promesa' not in df.columns and {'fecha_entrega_al_cliente','fecha_de_entrega_estimada'}.issubset(df.columns):
        df['desviacion_vs_promesa'] = (
            pd.to_datetime(df['fecha_entrega_al_cliente'], errors='coerce') -
            pd.to_datetime(df['fecha_de_entrega_estimada'], errors='coerce')
        ).dt.days

    modelo_flete = joblib.load("modelo_costoflete.sav")
    modelo_dias = joblib.load("modelo_dias_pipeline_70.joblib")
    label_encoder = joblib.load("label_encoder_dias_70.joblib")

    with st.sidebar:
        st.subheader("🎛️ Filtro de Estado")
        estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
        estado_sel = option_menu(
            menu_title="Selecciona un estado",
            options=estados,
            icons=["globe"] + ["geo"] * (len(estados)-1),
            default_index=0
        )

    df_filtrado = df if estado_sel == "Nacional" else df[df['estado_del_cliente'] == estado_sel]
    st.success("✅ Datos y modelos cargados correctamente.")
    tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "🗃️ Data"])
    with tabs[0]:
        st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {estado_sel}")
        col1, col2 = st.columns(2)
        col1.metric("Pedidos", f"{len(df_filtrado):,}")
        col2.metric("Llegadas muy adelantadas (≥10 días)",
                    f"{(df_filtrado['desviacion_vs_promesa'] < -10).mean()*100:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            df_tmp = df_filtrado.copy()
            df_tmp['zona_entrega'] = clasificar_zonas(df_tmp, estado_sel)
            conteo = df_tmp['zona_entrega'].value_counts().reset_index()
            conteo.columns = ['zona', 'cantidad']

            tonos_azules = ['#005BAC', '#4FA0D9', '#A7D3F4']
            color_map = {
                z: '#B0B0B0' if z == 'Provincia' else tonos_azules[min(i, len(tonos_azules)-1)]
                for i, z in enumerate(conteo['zona'])
            }

            fig = px.pie(
                conteo,
                names='zona',
                values='cantidad',
                hole=0.4,
                color='zona',
                color_discrete_map=color_map,
                title="📍 Pedidos por Zona"
            )

            fig.update_traces(
                textinfo='percent+label+value',
                hovertemplate="<b>%{label}</b><br>Pedidos: %{value}<br>Porcentaje: %{percent}"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df_tmp = df_filtrado.copy()
            df_tmp['zona_entrega'] = clasificar_zonas(df_tmp, estado_sel)
            df_tmp['estatus_entrega'] = df_tmp['llego_tarde'].apply(lambda x: 'A tiempo' if x==0 else 'Tardío')
            conteo2 = df_tmp.groupby(['zona_entrega','estatus_entrega']).size().reset_index(name='conteo')
            orden = df_tmp['zona_entrega'].value_counts().index.tolist()
            fig2 = px.bar(conteo2, x='zona_entrega', y='conteo', color='estatus_entrega',
                          category_orders={'zona_entrega': orden},
                          color_discrete_map={'A tiempo':'#A7D3F4','Tardío':'#B0B0B0'},
                          title="🚚 Entregas a tiempo vs tardías",
                          labels={'zona_entrega':'Zona','conteo':'Cantidad','estatus_entrega':'Estado'})
            fig2.update_traces(hovertemplate="<b>%{x}</b><br>%{color}: %{y}")
            fig2.update_layout(barmode='stack',height=500)
            st.plotly_chart(fig2, use_container_width=True)
                col3, col4 = st.columns(2)

        with col3:
            tot = df_filtrado.groupby('categoria')[['precio', 'costo_de_flete']].sum().reset_index()
            tot = tot.sort_values(by='precio', ascending=False)

            # Convertimos a formato largo
            tot_long = tot.melt(id_vars='categoria', value_vars=['precio', 'costo_de_flete'],
                                var_name='Concepto', value_name='Monto')

            # Orden forzado
            orden_categorias = tot['categoria'].tolist()

            fig5 = px.bar(
                tot_long,
                x='categoria',
                y='Monto',
                color='Concepto',
                barmode='group',
                title="📊 Total Precio vs Costo de Envío",
                color_discrete_map={
                    'precio': '#005BAC',
                    'costo_de_flete': '#4FA0D9'
                },
                category_orders={'categoria': orden_categorias}
            )

            fig5.update_layout(
                height=360,
                xaxis_title='Categoría',
                yaxis_title='Monto ($)',
                legend_title_text='',
                margin=dict(t=40, b=60, l=10, r=10)
            )
            fig5.update_xaxes(tickangle=-40)
            st.plotly_chart(fig5, use_container_width=True)


        with col4:
            label = "Ciudad" if estado_sel!="Nacional" else "Estado"
            st.subheader(f"📦 {label}s con mayor colchón de entrega")
            df_tmp = df_filtrado.copy()
            df_tmp['zona_entrega'] = clasificar_zonas(df_tmp, estado_sel)
            medios = df_tmp.groupby('zona_entrega')[['dias_entrega','colchon_dias']].mean().reset_index().sort_values('dias_entrega',ascending=False)
            avg = medios['dias_entrega'].mean()
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(y=medios['zona_entrega'], x=medios['dias_entrega'], name='Días Entrega', orientation='h', marker_color='#4FA0D9'))
            fig4.add_trace(go.Bar(y=medios['zona_entrega'], x=medios['colchon_dias'], name='Colchón Días', orientation='h', marker_color='#B0B0B0'))
            fig4.add_shape(type='line', x0=avg, x1=avg, y0=-0.5, y1=len(medios)-0.5, line=dict(color='blue', dash='dash'))
            fig4.update_layout(barmode='group',height=500,xaxis_title='Días Promedio', yaxis_title=label)
            st.plotly_chart(fig4, use_container_width=True)
# Tab 1: Costo de Envío
with tabs[1]:
    col1, col2 = st.columns(2)
    col1.metric("📦 Total Pedidos", f"{len(df_filtrado):,}")
    col2.metric("💰 % Flete > 50% del precio",
                f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%")

    st.subheader("💸 Relación Envío vs Precio por Categoría")
    df_rel = df_filtrado.copy()
    df_rel['porcentaje_flete'] = df_rel['costo_de_flete']/df_rel['precio']*100
    tabla = df_rel.groupby('categoria')['porcentaje_flete'].mean().reset_index().sort_values('porcentaje_flete',ascending=False)
    tabla['porcentaje_form'] = tabla['porcentaje_flete'].apply(lambda x: f"🔺 {x:.1f}%" if x>=40 else f"{x:.1f}%")
    tabla_display = tabla.set_index('categoria')['porcentaje_form'].to_frame().T
    st.dataframe(tabla_display.style.applymap(lambda v: 'color:red;font-weight:bold' if '🔺' in v else ''), use_container_width=True)

            col3, col4 = st.columns(2)

        with col3:
            tot = df_filtrado.groupby('categoria')[['precio', 'costo_de_flete']].sum().reset_index()
            tot = tot.sort_values(by='precio', ascending=False)

            # Convertimos a formato largo
            tot_long = tot.melt(id_vars='categoria', value_vars=['precio', 'costo_de_flete'],
                                var_name='Concepto', value_name='Monto')

            # Orden forzado
            orden_categorias = tot['categoria'].tolist()

            fig5 = px.bar(
                tot_long,
                x='categoria',
                y='Monto',
                color='Concepto',
                barmode='group',
                title="📊 Total Precio vs Costo de Envío",
                color_discrete_map={
                    'precio': '#005BAC',
                    'costo_de_flete': '#4FA0D9'
                },
                category_orders={'categoria': orden_categorias}
            )

            fig5.update_layout(
                height=360,
                xaxis_title='Categoría',
                yaxis_title='Monto ($)',
                legend_title_text='',
                margin=dict(t=40, b=60, l=10, r=10)
            )
            fig5.update_xaxes(tickangle=-40)
            st.plotly_chart(fig5, use_container_width=True)


    with col4:
        df_m = df_filtrado.groupby('mes')['costo_de_flete'].mean().reset_index()
        meses_txt = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        df_m['mes_nombre'] = df_m['mes'].apply(lambda x: meses_txt[x-1])
        fig6 = px.line(df_m, x='mes_nombre', y='costo_de_flete', markers=True,
                      title="📈 Costo Promedio Mensual de Flete",
                      labels={'mes_nombre':'Mes','costo_de_flete':'Costo Promedio'})
        fig6.update_traces(line=dict(width=3, color='#2c7be5'), marker=dict(size=7, color='#2c7be5'))
        fig6.update_layout(height=420)
        st.plotly_chart(fig6, use_container_width=True)

# Tab 2: Calculadora
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")
    meses_dict = {
        1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
        7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"
    }
    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'], errors='coerce')
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month

    categoria = st.selectbox("Categoría", sorted(df2['categoria'].dropna().unique()))
    c1, c2 = st.columns(2)
    mes1_nombre = c1.selectbox("Mes 1", list(meses_dict.values()), index=0)
    mes2_nombre = c2.selectbox("Mes 2", list(meses_dict.values()), index=1)
    mes1 = [k for k, v in meses_dict.items() if v == mes1_nombre][0]
    mes2 = [k for k, v in meses_dict.items() if v == mes2_nombre][0]

    filtro = (df2['estado_del_cliente'] == estado_sel) & (df2['categoria'] == categoria)
    df_mes1 = df2[(df2['mes'] == mes1) & filtro].copy()
    df_mes2 = df2[(df2['mes'] == mes2) & filtro].copy()

    def predecir(df_input):
        if df_input.empty: return df_input
        cols_f = ['total_peso_g','precio','#_deproductos','duracion_estimada_min',
                  'ciudad_cliente','nombre_dc','hora_compra','año','mes','datetime_origen',
                  'region','dias_promedio_ciudad','categoria','tipo_de_pago']
        df_f = df_input[cols_f].copy()
        df_enc = pd.get_dummies(df_f)
        cols_modelo = modelo_flete.get_booster().feature_names
        df_enc = df_enc.reindex(columns=cols_modelo, fill_value=0)
        df_input['costo_estimado'] = modelo_flete.predict(df_enc).round(2)
        df_input['costo_de_flete'] = df_input['costo_estimado']

        cols_d = ['categoria','categoria_peso','#_deproductos','total_peso_g','precio',
                  'costo_de_flete','distancia_km','velocidad_kmh','duracion_estimada_min',
                  'region','dc_asignado','es_feriado','es_fin_de_semana','hora_compra',
                  'dias_promedio_ciudad','nombre_dia','mes','año','traffic','area']
        if not all(c in df_input.columns for c in cols_d): return df_input
        Xd = df_input[cols_d]
        pred = modelo_dias.predict(Xd)
        df_input['clase_entrega'] = label_encoder.inverse_transform(pred)
        return df_input

    def agrupar(df_input, nombre):
        if 'costo_estimado' in df_input and 'clase_entrega' in df_input:
            return df_input.groupby('ciudad_cliente').agg({
                'costo_estimado':'mean',
                'clase_entrega': lambda x: x.mode()[0] if not x.mode().empty else 'NA'
            }).rename(columns={
                'costo_estimado': nombre,
                'clase_entrega': f'Entrega {nombre}'
            }).reset_index()
        return pd.DataFrame(columns=['ciudad_cliente', nombre, f'Entrega {nombre}'])

    df_mes1 = predecir(df_mes1)
    df_mes2 = predecir(df_mes2)
    res1 = agrupar(df_mes1, mes1_nombre)
    res2 = agrupar(df_mes2, mes2_nombre)
    comparacion = pd.merge(res1, res2, on='ciudad_cliente', how='outer')
    comparacion[mes1_nombre] = pd.to_numeric(comparacion[mes1_nombre], errors='coerce')
    comparacion[mes2_nombre] = pd.to_numeric(comparacion[mes2_nombre], errors='coerce')
    comparacion['Diferencia'] = (comparacion[mes2_nombre] - comparacion[mes1_nombre]).round(2)
    comparacion = comparacion.rename(columns={'ciudad_cliente': 'Ciudad'})
    st.dataframe(comparacion.style.format(precision=2))
    st.download_button("⬇️ Descargar CSV", comparacion.to_csv(index=False), file_name="comparacion.csv")

# Tab 3: Vista de Datos
with tabs[3]:
    st.subheader("🔍 Vista Previa del Dataset")
    st.dataframe(df.head(100), use_container_width=True)

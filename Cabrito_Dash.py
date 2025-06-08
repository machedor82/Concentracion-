0
import streamlit as st
import pandas as pd
import zipfile
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_option_menu import option_menu  # Asegúrate de instalar streamlit-option-menu

# ------------------ Definiciones de clases/funciones personalizadas ------------------
class MiTransformadorEspecial(BaseEstimator, TransformerMixin):
    def __init__(self, parametro1=None):
        self.parametro1 = parametro1

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


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
        /* Ocultar watermark de Streamlit */
        .css-1wa3eu0 {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# ===================== DEFINICIÓN DE PESTAÑAS =====================
tabs = st.tabs(["📊 Resumen Nacional", "🏠 Costo de Envío", "🧮 Calculadora", "App Danu 📈"])

# ===================== SIDEBAR DE CARGA =====================
with st.sidebar:
    st.image("danu_logo.png", use_container_width=True)
    st.header("Sube tu archivo ZIP")
    archivo_zip = st.file_uploader("ZIP con DF.csv, DF2.csv y modelos", type="zip")

if not archivo_zip:
    st.warning("Por favor, sube un archivo .zip para continuar.")
    st.stop()

# ===================== CARGA DE DATOS Y MODELOS =====================
with zipfile.ZipFile(archivo_zip) as z:
    requeridos = ['DF.csv', 'DF2.csv', 'modelo_costoflete.sav', 'modelo_dias_pipeline.joblib', 'label_encoder_dias.joblib']
    falt = [r for r in requeridos if r not in z.namelist()]
    if falt:
        st.error(f"❌ Faltan archivos en el ZIP: {falt}")
        st.stop()

    df = pd.read_csv(z.open('DF.csv'))
    df2 = pd.read_csv(z.open('DF2.csv'))
    modelo_flete = joblib.load(z.open('modelo_costoflete.sav'))
    modelo_dias = joblib.load(z.open('modelo_dias_pipeline.joblib'))
    label_encoder = joblib.load(z.open('label_encoder_dias.joblib'))

# ===================== FILTRO DE ESTADO =====================
with st.sidebar:
    st.subheader("🎛️ Filtro de Estado")
    estados = ["Nacional"] + sorted(df['estado_del_cliente'].dropna().unique().tolist())
    estado_sel = option_menu(
        menu_title="Selecciona un estado",
        options=estados,
        icons=["globe"] + ["geo"]*(len(estados)-1),
        default_index=0
    )

# Data filtrada
if estado_sel == "Nacional":
    df_filtrado = df.copy()
else:
    df_filtrado = df[df['estado_del_cliente']==estado_sel].copy()

# ===================== PESTAÑA 0: Resumen Nacional =====================
with tabs[0]:
    zona = estado_sel if estado_sel!="Nacional" else "Resumen Nacional"
    st.title(f"📊 ¿Entrega Rápida o Margen Inflado? – {zona}")
    c1, c2 = st.columns(2)
    c1.metric("Pedidos", f"{len(df_filtrado):,}")
    c2.metric("Llegadas muy adelantadas (≥10 días)", f"{(df_filtrado['desviacion_vs_promesa']< -10).mean()*100:.1f}%")

    if 'dias_entrega' in df_filtrado.columns:
        col1, col2 = st.columns(2)
        # Dona: Pedidos por zona
        with col1:
            df_tmp = df_filtrado.copy()
            df_tmp['zona_entrega'] = clasificar_zonas(df_tmp, estado_sel)
            conteo_zona = df_tmp['zona_entrega'].value_counts().reset_index()
            conteo_zona.columns=['Zona','Pedidos']
            zonas = conteo_zona['Zona'].tolist()
            tonos_azules=['#005BAC','#4FA0D9','#A7D3F4']
            cd={}
            for i,z in enumerate(zonas): cd[z]=('#B0B0B0' if z=='Provincia' else tonos_azules[min(i,2)])
            fig=px.pie(conteo_zona,names='Zona',values='Pedidos',hole=0.4,title="📍 Pedidos por Zona",
                       color='Zona',color_discrete_map=cd)
            fig.update_traces(textinfo='percent+label+value',hovertemplate="<b>%{label}</b><br>Pedidos: %{value}<br>Porcentaje: %{percent}")
            st.plotly_chart(fig,use_container_width=True)
        # Barras: Entregas a tiempo vs tardías
        with col2:
            df_tmp = df_filtrado.copy()
            df_tmp['zona_entrega']=clasificar_zonas(df_tmp,estado_sel)
            df_tmp['estatus_entrega']=df_tmp['llego_tarde'].apply(lambda x:'A tiempo' if x==0 else 'Tardío')
            cz=df_tmp.groupby(['zona_entrega','estatus_entrega']).size().reset_index(name='conteo')
            cz['porcentaje']=cz['conteo']/cz.groupby('zona_entrega')['conteo'].transform('sum')*100
            order=df_tmp['zona_entrega'].value_counts().index.tolist()
            fig=px.bar(cz,x='zona_entrega',y='porcentaje',color='estatus_entrega',category_orders={'zona_entrega':order},
                       color_discrete_map={'A tiempo':'#A7D3F4','Tardío':'#B0B0B0'},title="🚚 Si somos puntuales, ¿cuál es el problema?",
                       labels={'zona_entrega':'Zona','porcentaje':'Porcentaje','estatus_entrega':'Tipo de Entrega'},text_auto='.1f')
            fig.update_traces(hovertemplate="<b>%{x}</b><br>%{color}: %{y:.1f}%")
            fig.update_layout(barmode='stack',xaxis_title=None,yaxis_title='Porcentaje (%)',legend_title='Tipo de Entrega',height=500)
            st.plotly_chart(fig,use_container_width=True)
        # Fila 2: barras y horizontal
        col3,col4=st.columns(2)
        with col3:
            df_tmp=df_filtrado[df_filtrado['dias_entrega'].notna()].copy()
            df_tmp['grupo_dias']=pd.cut(df_tmp['dias_entrega'],bins=[0,5,10,float('inf')],labels=["1-5","6-10","Más de 10"])
            df_tmp['zona_entrega']=clasificar_zonas(df_tmp,estado_sel)
            cnt=df_tmp.groupby(['zona_entrega','grupo_dias']).size().reset_index(name='conteo')
            cnt['porcentaje']=cnt['conteo']/cnt.groupby('zona_entrega')['conteo'].transform('sum')*100
            order=df_tmp['zona_entrega'].value_counts().index.tolist()
            colors={'1-5':'#A7D3F4','6-10':'#4FA0D9','Más de 10':'#FF6B6B'}
            fig=px.bar(cnt,x='zona_entrega',y='porcentaje',color='grupo_dias',category_orders={'zona_entrega':order},
                       color_discrete_map=colors,title="📦 ¿Éxito logístico o maquillaje de tiempos?",
                       labels={'zona_entrega':'Zona','porcentaje':'Porcentaje','grupo_dias':'Días de Entrega'},text_auto='.1f')
            fig.update_layout(barmode='stack',xaxis_title=None,yaxis_title='Porcentaje (%)',legend_title='Días de Entrega',height=500)
            st.plotly_chart(fig,use_container_width=True)
        with col4:
            label="Ciudad" if estado_sel!="Nacional" else "Estado"
            st.subheader(f"📦 {label}s con mayor colchón de entrega")
            if {'dias_entrega','colchon_dias'}.issubset(df_filtrado.columns):
                df_tmp=df_filtrado.copy();df_tmp['zona_entrega']=clasificar_zonas(df_tmp,estado_sel)
                medios=df_tmp.groupby('zona_entrega')[['dias_entrega','colchon_dias']].mean().reset_index().sort_values('dias_entrega',ascending=False)
                fig=go.Figure();fig.add_trace(go.Bar(y=medios['zona_entrega'],x=medios['dias_entrega'],name='Días Entrega',orientation='h',marker_color='#4FA0D9'))
                fig.add_trace(go.Bar(y=medios['zona_entrega'],x=medios['colchon_dias'],name='Colchón Días',orientation='h',marker_color='#B0B0B0'))
                m=medios['dias_entrega'].mean()
                fig.add_shape(type="line",x0=m,x1=m,y0=-0.5,y1=len(medios)-0.5,line=dict(color="blue",dash="dash"))
                fig.update_layout(barmode='group',height=500,xaxis_title='Días Promedio',yaxis_title=label,margin=dict(t=40,b=40,l=80,r=10),legend_title="Métrica")
                st.plotly_chart(fig,use_container_width=True)

# ========================= PESTAÑA 1: Costo de Envío =========================
with tabs[1]:
    c1,c2=st.columns(2)
    c1.metric("📦 Total de Pedidos",f"{len(df_filtrado):,}")
    c2.metric("💰 Flete Alto vs Precio",f"{(df_filtrado['costo_de_flete']/df_filtrado['precio']>0.5).mean()*100:.1f}%")
    st.subheader("💸 Relación Envío–Precio: ¿Gasto Justificado?")
    df_precio=df_filtrado.copy();df_precio['porcentaje_flete']=df_precio['costo_de_flete']/df_precio['precio']*100
    tabla=df_precio.groupby('categoria')['porcentaje_flete'].mean().reset_index().sort_values('porcentaje_flete',ascending=False)
    tabla['raw']=tabla['porcentaje_flete'];tabla['porcentaje_flete']=tabla['porcentaje_flete'].apply(lambda x:(f"🔺 {x:.1f}%" if x>=40 else f"{x:.1f}%"))
    tabla_h=tabla.set_index('categoria')[['porcentaje_flete']].T
    def hig(s):return ['color:red;font-weight:bold' if '🔺' in str(v) else '' for v in s]
    st.dataframe(tabla_h.style.apply(hig,axis=1),use_container_width=True,height=100,hide_index=True)
    col1,col2=st.columns(2)
    with col1:
        tot=df_filtrado.groupby('categoria')[['precio','costo_de_flete']].sum().reset_index().sort_values('precio',ascending=False)
        fig=px.bar(tot,x='categoria',y=['precio','costo_de_flete'],barmode='group',title="📊 Total Precio vs Costo de Envío",labels={'value':'Monto ($)','variable':'Concepto'},color_discrete_map={'precio':'#005BAC','costo_de_flete':'#4FA0D9'})
        fig.update_layout(height=360,margin=dict(t=40,b=60,l=10,r=10),legend=dict(orientation='h',yanchor='bottom',y=-0.3,xanchor='center',x=0.5),legend_title=None)
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{legendgroup}: %{y:,.0f} $<extra></extra>")
        fig.update_xaxes(tickangle=-40)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        dpm=df_filtrado.groupby('mes')['costo_de_flete'].mean().reset_index()
        meses_txt=['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        dpm['mes_nombre']=dpm['mes'].apply(lambda x:meses_txt[x-1])
        dpm=dpm.sort_values('mes')
        fig=px.line(dpm,x='mes_nombre',y='costo_de_flete',markers=True,title="📈 Costo Promedio de Flete por Mes",labels={'mes_nombre':'Mes','costo_de_flete':'Costo Promedio ($)'})
        fig.update_layout(height=420,xaxis=dict(categoryorder='array',categoryarray=meses_txt),yaxis_title='Costo Promedio ($)',margin=dict(t=50,b=50,l=40,r=10))
        fig.update_traces(line=dict(width=3),marker=dict(size=7))
        st.plotly_chart(fig,use_container_width=True)

# ========================= PESTAÑA 2: Calculadora =========================
with tabs[2]:
    st.header("🧮 Calculadora de Predicción")
    # Normalizar columnas de df2
    import unicodedata
    def normalize(col): s=col.strip().lower().replace(' ','_');return unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode('ascii')
    df2.columns=[normalize(c) for c in df2.columns]
    meses_dict={1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}
    df2['orden_compra_timestamp']=pd.to_datetime(df2['orden_compra_timestamp']);df2['año']=df2['orden_compra_timestamp'].dt.year;df2['mes']=df2['orden_compra_timestamp'].dt.month
    est=st.selectbox("Estado",sorted(df2['estado_del_cliente'].dropna().unique()));cat=st.selectbox("Categoría",sorted(df2['categoria'].dropna().unique()))
    c1,c2=st.columns(2)
    mes1_n=c1.selectbox("Mes 1",list(meses_dict.values()),index=0);mes2_n=c2.selectbox("Mes 2",list(meses_dict.values()),index=1)
    m1=[k for k,v in meses_dict.items() if v==mes1_n][0];m2=[k for k,v in meses_dict.items() if v==mes2_n][0]
    filt=(df2['estado_del_cliente']==est)&(df2['categoria']==cat)
    d1=df2[(df2['mes']==m1)&filt].copy();d2=df2[(df2['mes']==m2)&filt].copy()
    def predecir(df_in):
        if df_in.empty: return df_in
        cols_f=['total_peso_g','precio','#_deproductos','duracion_estimada_min','ciudad_cliente','nombre_dc','hora_compra','año','mes','datetime_origen','region','dias_promedio_ciudad','categoria','tipo_de_pago']
        dff=df_in.reindex(columns=cols_f).copy();dum=pd.get_dummies(dff);feat=modelo_flete.get_booster().feature_names;dum=dum.reindex(columns=feat,fill_value=0)
        df_in['costo_estimado']=modelo_flete.predict(dum).round(2);df_in['costo_de_flete']=df_in['costo_estimado']
        cols_d=['categoria','categoria_peso','#_deproductos','total_peso_g','precio','costo_de_flete','distancia_km','velocidad_kmh','duracion_estimada_min','region','dc_asignado','es_feriado','es_fin_de_semana','hora_compra','dias_promedio_ciudad','nombre_dia','mes','año','traffic','area']
        if not all(c in df_in.columns for c in cols_d): return df_in
        Xd=df_in[cols_d].copy();p=modelo_dias.predict(Xd);df_in['clase_entrega']=label_encoder.inverse_transform(p);return df_in
    def agr(df_p,nm):
        if 'costo_estimado' in df_p and 'clase_entrega' in df_p:
            return df_p.groupby('ciudad_cliente').agg({'costo_estimado':lambda x:round(x.mean(),2),'clase_entrega':lambda x:x.mode()[0] if not x.mode().empty else 'NA'}).rename(columns={'costo_estimado':nm,'clase_entrega':f"Entrega {nm}"}).reset_index()
        return pd.DataFrame(columns=['ciudad_cliente',nm,f"Entrega {nm}"])
    r1=agr(predecir(d1),mes1_n);r2=agr(predecir(d2),mes2_n)
    comp=pd.merge(r1,r2,on='ciudad_cliente',how='outer');comp[mes1_n]=pd.to_numeric(comp[mes1_n],errors='coerce');comp[mes2_n]=pd.to_numeric(comp[mes2_n],errors='coerce');comp['Diferencia']=(comp[mes2_n]-comp[mes1_n]).round(2)
    comp=comp[['ciudad_cliente',mes1_n,mes2_n,'Diferencia',f"Entrega {mes1_n}",f"Entrega {mes2_n}"]].rename(columns={'ciudad_cliente':'Ciudad'})
    a1=d1['costo_estimado'].mean() if not d1.empty else np.nan; a2=d2['costo_estimado'].mean() if not d2.empty else np.nan; pct=((a2-a1)/a1*100) if a1 else 0
    st.markdown("---");k1,k2,k3=st.columns(3);k1.markdown(f"**Costo Promedio {mes1_n}:** {a1:.2f}");colr='green' if pct>0 else 'red';k2.markdown(f"**% Cambio:** <span style='color:{colr}'>{pct:.2f}%</span>",unsafe_allow_html=True);k3.markdown(f"**Costo Promedio {mes2_n}:** {a2:.2f}")
    st.subheader(f"Comparación: {mes1_n} vs {mes2_n}");st.dataframe(comp.style.applymap(lambda v:'color:green;font-weight:bold' if isinstance(v,(int,float)) and v>0 else ('color:red;font-weight:bold' if isinstance(v,(int,float)) and v<0 else ''),subset=['Diferencia']).format(precision=2),use_container_width=True)
    st.download_button("⬇️ Descargar CSV",data=comp.to_csv(index=False).encode('utf-8'),file_name=f"comparacion_{est}_{cat}_{mes1_n}vs{mes2_n}.csv",mime='text/csv')

# ===================== PESTAÑA 3: App Danu =====================
with tabs[3]:
    # ... tu código original para App Danu ...
    pass

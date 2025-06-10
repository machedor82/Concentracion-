# ========================= CALCULADORA =========================
with tabs[2]:
    import joblib
    from sklearn.base import BaseEstimator, TransformerMixin

    st.header("🧮 Calculadora de Predicción")

    # Diccionario de meses
    meses_dict = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }

    df2['orden_compra_timestamp'] = pd.to_datetime(df2['orden_compra_timestamp'])
    df2['año'] = df2['orden_compra_timestamp'].dt.year
    df2['mes'] = df2['orden_compra_timestamp'].dt.month
    estado = estado_sel  # Usamos la selección del sidebar
    st.markdown(f"**Estado seleccionado:** {estado}")

    categoria = st.selectbox("Categoría", sorted(df2['Categoría'].dropna().unique()))

    col1, col2 = st.columns(2)
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

        columnas_flete = ['total_peso_g', 'precio', '#_deproductos', 'duracion_estimada_min', 'ciudad_cliente',
                          'nombre_dc', 'hora_compra', 'año', 'mes', 'datetime_origen', 'region',
                          'dias_promedio_ciudad', 'Categoría', 'tipo_de_pago']

        df_flete = df_input[columnas_flete].copy()
        df_encoded = pd.get_dummies(df_flete)
        columnas_modelo = modelo_flete.get_booster().feature_names
        df_encoded = df_encoded.reindex(columns=columnas_modelo, fill_value=0)

        df_input['costo_estimado'] = modelo_flete.predict(df_encoded).round(2)
        df_input['costo_de_flete'] = df_input['costo_estimado']

        columnas_dias = ['Categoría', 'categoria_peso', '#_deproductos', 'total_peso_g', 'precio', 'costo_de_flete',
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

    # Asegurar que las columnas sean numéricas
    comparacion[mes1_nombre] = pd.to_numeric(comparacion[mes1_nombre], errors='coerce')
    comparacion[mes2_nombre] = pd.to_numeric(comparacion[mes2_nombre], errors='coerce')
    comparacion['Diferencia'] = (comparacion[mes2_nombre] - comparacion[mes1_nombre]).round(2)

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

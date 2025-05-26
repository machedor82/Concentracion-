import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Tablero Iris Local", layout="wide")
st.title("🌸 Dashboard de Iris")

# Cargar dataset Iris desde seaborn
df = sns.load_dataset("iris")

# Mostrar los datos
st.subheader("Vista previa de los datos")
st.dataframe(df)

# Histograma interactivo
st.subheader("Histograma por columna")
numeric_cols = df.select_dtypes(include='number').columns.tolist()
col = st.selectbox("Selecciona una columna", numeric_cols)

fig, ax = plt.subplots()
ax.hist(df[col], bins=20, color="skyblue", edgecolor="black")
ax.set_title(f"Histograma de {col}")
st.pyplot(fig)

# Gráfico de dispersión
st.subheader("Gráfico de dispersión por especie")
x_col = st.selectbox("Eje X", numeric_cols, index=0)
y_col = st.selectbox("Eje Y", numeric_cols, index=1)

fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x=x_col, y=y_col, hue="species", ax=ax2)
ax2.set_title(f"{y_col} vs {x_col} por especie")
st.pyplot(fig2)

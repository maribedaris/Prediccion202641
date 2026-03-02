import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración inicial de la página
st.set_page_config(page_title="Predictor de Modelo Neuronal", layout="centered")

# 1. Cargar todos los artefactos de Machine Learning en caché para que la app sea rápida
@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load('modelo_red_neuronal.joblib')
    scaler = joblib.load('scaler.joblib')
    ohe = joblib.load('one_hot_encoder.joblib')
    label_encoders = joblib.load('label_encoders_binarios.joblib')
    col_escalar = joblib.load('columnas_escalar.joblib')
    col_cat = joblib.load('columnas_categoricas.joblib')
    features_finales = joblib.load('feature_columns.joblib')
    
    return modelo, scaler, ohe, label_encoders, col_escalar, col_cat, features_finales

modelo, scaler, ohe, label_encoders, col_escalar, col_cat, features_finales = cargar_artefactos()

st.title("Clasificador de Datos - Interfaz de Predicción")
st.markdown("Ingresa los datos manualmente en los siguientes campos para generar una predicción.")

# 2. Crear el formulario de entrada
with st.form("formulario_prediccion"):
    st.subheader("Datos Numéricos")
    input_data = {}
    
    # Crear dos columnas para que el formulario se vea ordenado
    col1, col2 = st.columns(2)
    
    # Campos Numéricos (cuadros editables)
    with col1:
        for col in col_escalar:
            # Se asume valor por defecto 0.0, puedes ajustarlo
            input_data[col] = st.number_input(f"Ingrese: {col}", value=0.0, step=1.0)
            
    # Campos Binarios (menús desplegables extraídos del LabelEncoder)
    st.subheader("Datos Binarios")
    col3, col4 = st.columns(2)
    for i, (col, le) in enumerate(label_encoders.items()):
        # Alternar entre columnas para la interfaz
        column = col3 if i % 2 == 0 else col4
        with column:
            opciones = le.classes_
            input_data[col] = st.selectbox(f"Seleccione: {col}", opciones)
            
    # Campos Categóricos (menús desplegables extraídos del OneHotEncoder)
    st.subheader("Datos Categóricos")
    col5, col6 = st.columns(2)
    for i, col in enumerate(col_cat):
        column = col5 if i % 2 == 0 else col6
        with column:
            opciones = ohe.categories_[i]
            input_data[col] = st.selectbox(f"Seleccione: {col}", opciones)

    # Botón de predicción
    submit_button = st.form_submit_button(label="Generar Predicción")

# 3. Lógica de procesamiento y predicción al presionar el botón
if submit_button:
    # Convertir el diccionario de entradas en un DataFrame de una sola fila
    df_input = pd.DataFrame([input_data])
    
    try:
        # A. Transformar columnas numéricas (Escalado)
        df_input[col_escalar] = scaler.transform(df_input[col_escalar])
        
        # B. Transformar columnas binarias (Label Encoding)
        for col, le in label_encoders.items():
            df_input[col] = le.transform(df_input[col])
            
        # C. Transformar columnas categóricas (One-Hot Encoding)
        datos_categoricos_codificados = ohe.transform(df_input[col_cat])
        
        # El OneHotEncoder puede devolver una matriz rala o densa. La convertimos a array denso si es necesario.
        if hasattr(datos_categoricos_codificados, "toarray"):
            datos_categoricos_codificados = datos_categoricos_codificados.toarray()
            
        nombres_columnas_ohe = ohe.get_feature_names_out(col_cat)
        df_categorico = pd.DataFrame(datos_categoricos_codificados, columns=nombres_columnas_ohe)
        
        # D. Unir todos los datos preprocesados
        df_procesado = df_input.drop(columns=col_cat)
        df_final = pd.concat([df_procesado.reset_index(drop=True), df_categorico.reset_index(drop=True)], axis=1)
        
        # E. Alinear las columnas con las que espera el modelo de red neuronal
        # Garantiza el mismo orden exacto y rellena con 0 si por alguna razón falta alguna dummy variable
        df_final = df_final.reindex(columns=features_finales, fill_value=0)
        
        # Generar Predicción
        prediccion = modelo.predict(df_final)[0]
        
        # Generar las probabilidades si el modelo lo permite
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(df_final)[0]
            prob_max = np.max(probabilidades) * 100
            st.success(f"### Predicción del Modelo: {prediccion}")
            st.info(f"Confianza de la predicción: {prob_max:.2f}%")
        else:
            st.success(f"### Predicción del Modelo: {prediccion}")
            
    except Exception as e:
        st.error(f"Ocurrió un error al procesar los datos: {e}")

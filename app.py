
import streamlit as st
import pandas as pd
import joblib

# Configuración inicial
st.set_page_config(page_title="Predictor Red Neuronal", layout="wide", page_icon="🧠")
st.title("Aplicación de Predicción con Red Neuronal")
st.markdown("Ingresa los datos en los formularios a continuación para obtener una predicción del modelo.")

# ==========================================================
# 1. Cargar modelo y preprocesadores (.pkl)
# ==========================================================

@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    le_binarios = joblib.load('label_encoders_binarios.pkl')
    ohe_categoricas = joblib.load('one_hot_encoder_categoricas.pkl')
    modelo_nn = joblib.load('modelo_red_neuronal_entrenado.pkl')
    return scaler, le_binarios, ohe_categoricas, modelo_nn

scaler, le_binarios, ohe_categoricas, modelo_nn = load_models()

# Cargar los componentes
scaler, le_binarios, ohe_categoricas, modelo_nn = load_models()

# 2. Extraer automáticamente los nombres de las variables de los preprocesadores
num_features = scaler.feature_names_in_
bin_features = list(le_binarios.keys())
cat_features = ohe_categoricas.feature_names_in_

# 3. Crear la Interfaz Gráfica con el formulario
with st.form("prediction_form"):
    st.header("Datos de Entrada")
    
    # Dividir la pantalla en 3 columnas para organizar mejor los campos
    col1, col2, col3 = st.columns(3)
    input_data = {}
    
    # Columna 1: Variables Numéricas (Cuadros de entrada numérica)
    with col1:
        st.subheader("Variables Numéricas")
        for col in num_features:
            # Puedes ajustar el paso (step) o los valores min/max si los conoces
            input_data[col] = st.number_input(f"{col}", value=0.0)
            
    # Columna 2: Variables Binarias (Menús desplegables)
    with col2:
        st.subheader("Variables Binarias")
        for col in bin_features:
            opciones_bin = le_binarios[col].classes_
            input_data[col] = st.selectbox(f"{col}", opciones_bin)
            
    # Columna 3: Variables Categóricas OHE (Menús desplegables múltiples)
    with col3:
        st.subheader("Variables Categóricas")
        for i, col in enumerate(cat_features):
            opciones_cat = ohe_categoricas.categories_[i]
            input_data[col] = st.selectbox(f"{col}", opciones_cat)
            
    st.markdown("---")
    # Botón de predicción
    submit_button = st.form_submit_button("Realizar Predicción 🚀")

# 4. Lógica que se ejecuta al presionar el botón
if submit_button:
    # Convertir el diccionario de entradas en un DataFrame de 1 fila
    df_input = pd.DataFrame([input_data])
    
    try:
        # A. Preprocesar variables binarias
        for col in bin_features:
            df_input[col] = le_binarios[col].transform(df_input[col])
            
        # B. Preprocesar variables categóricas (One-Hot Encoding)
        cat_encoded = ohe_categoricas.transform(df_input[cat_features])
        
        # Verificar si la matriz es "sparse" (dispersa) y convertirla a un array denso si es necesario
        if hasattr(cat_encoded, "toarray"):
            cat_encoded = cat_encoded.toarray()
            
        # Crear un DataFrame con las características OHE
        cat_encoded_df = pd.DataFrame(
            cat_encoded, 
            columns=ohe_categoricas.get_feature_names_out(cat_features)
        )
        
        # C. Preprocesar variables numéricas (Escalado)
        df_input[num_features] = scaler.transform(df_input[num_features])
        
        # D. Separar los dataframes procesados para concatenarlos correctamente
        num_scaled_df = df_input[num_features].reset_index(drop=True)
        bin_encoded_df = df_input[bin_features].reset_index(drop=True)
        
        # E. Concatenar todo en el conjunto de variables final (X)
        X_procesado = pd.concat([num_scaled_df, bin_encoded_df, cat_encoded_df], axis=1)
        
        # Asegurar que el orden de las columnas coincida exactamente con el que vio la Red Neuronal en el entrenamiento
        if hasattr(modelo_nn, "feature_names_in_"):
            # Rellenar con ceros las columnas que la red espera pero no se generaron (por categorías ausentes en el nuevo input)
            columnas_faltantes = set(modelo_nn.feature_names_in_) - set(X_procesado.columns)
            for c in columnas_faltantes:
                X_procesado[c] = 0
                
            # Ordenar las columnas
            X_procesado = X_procesado[modelo_nn.feature_names_in_]
            
        # F. Realizar la predicción
        prediccion = modelo_nn.predict(X_procesado)
        probabilidad = modelo_nn.predict_proba(X_procesado)[0]
        
        # G. Mostrar los resultados en la interfaz
        st.success("¡Análisis completado con éxito!")
        
        # Usamos métricas de Streamlit para destacar el resultado
        st.metric(label="Resultado de Predicción", value=str(prediccion[0]))
        
        st.markdown("### Probabilidades por clase:")
        # Mostrar las probabilidades formateadas
        probs_dict = {f"Clase {modelo_nn.classes_[i]}": f"{prob*100:.2f}%" for i, prob in enumerate(probabilidad)}
        st.json(probs_dict)
        
    except Exception as e:
        st.error(f"Se produjo un error al procesar los datos o hacer la predicción: {e}")

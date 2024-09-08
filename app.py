import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="🧠Easy Train AI", page_icon="🤖", layout="wide")

# Título y descripción de la aplicación
st.title("🤖 Easy Train AI: Entrenamiento Fácil de Redes Neuronales")
st.markdown("""
    Bienvenido a **Easy Train AI**, una aplicación diseñada para hacer que el entrenamiento de redes neuronales sea accesible y simple, 
    sin la necesidad de programar. Solo necesitas subir tus datos, configurar el modelo a través de opciones intuitivas, entrenarlo y 
    obtener predicciones en tiempo real. Es ideal para tareas como predicción de cualquier tipo de datos, desde sensores hasta finanzas 
    o análisis de ventas.

    **Características principales:**
    - Soporte para archivos en formato CSV, TXT y XLSX
    - Configuración personalizada de variables de entrada (X) y salida (y)
    - Arquitectura flexible de la red neuronal: ajusta las capas, neuronas y funciones de activación
    - Gráficos interactivos que muestran el rendimiento del modelo durante el entrenamiento
    - Predicciones en tiempo real utilizando datos nuevos cargados por el usuario
    - Carga y uso de modelos previamente entrenados
""")

# Cargar el archivo de datos
uploaded_file = st.file_uploader("📤 Sube tu archivo de datos", type=["csv", "txt", "xlsx"])

if uploaded_file is not None:
    # Crear columnas para organizar el layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Selección del delimitador si es un archivo CSV o TXT
        delimiter = st.selectbox("Selecciona el delimitador", [",", ";", "\t", " "], index=0)
        
        # Cargar el archivo según su formato
        if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".txt"):
            data_mult = pd.read_csv(uploaded_file, delimiter=delimiter, header=None)
        elif uploaded_file.name.endswith(".xlsx"):
            data_mult = pd.read_excel(uploaded_file, header=None)
        
        st.write("Vista previa de los datos:")
        st.dataframe(data_mult.head(), use_container_width=True)
    
    with col2:
        # Añadir nombres de columnas si el usuario lo solicita
        if st.checkbox("Añadir nombres de columnas"):
            columns = st.text_area("Especifica los nombres de las columnas separados por comas:", 
                                   value="ID,Temperature,Humidity,UV,Voltage,Current,Illuminance,ClientIP,SensorID,DateTime")
            data_mult.columns = [col.strip() for col in columns.split(",")]
            st.write("Nombres de columnas actualizados:")
            st.dataframe(data_mult.head(), use_container_width=True)
    
    # Seleccionar las variables independientes y dependientes
    st.subheader("🎯 Selección de Variables")
    input_vars = st.multiselect("Variables independientes (X)", data_mult.columns.tolist(), default=data_mult.columns.tolist()[:-1])
    target_var = st.selectbox("Variable dependiente (y)", data_mult.columns.tolist(), index=len(data_mult.columns.tolist()) - 1)
    
    X = data_mult[input_vars]
    y = data_mult[target_var]
    
    # Opción de estandarizar los datos
    if st.checkbox("Estandarizar las variables independientes"):
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        st.write("Datos estandarizados:")
        st.dataframe(pd.DataFrame(X, columns=input_vars).head(), use_container_width=True)
    
    # Dividir los datos en entrenamiento y prueba
    test_size = st.slider("Tamaño del conjunto de prueba (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    
    # Configurar la red neuronal
    st.subheader("🛠️ Configuración de la Red Neuronal")
    num_layers = st.slider("Número de capas ocultas", 1, 10, 3)
    layers = []
    for i in range(num_layers):
        col1, col2 = st.columns(2)
        with col1:
            neurons = st.number_input(f"Neurones en la capa {i+1}", min_value=1, value=64)
        with col2:
            activation = st.selectbox(f"Activación capa {i+1}", ["relu", "sigmoid", "tanh"], index=0)
        layers.append((neurons, activation))
    
    epochs = st.number_input("Número de épocas", min_value=10, value=100)
    
    if st.button("🚀 Entrenar el modelo"):
        with st.spinner("Entrenando el modelo..."):
            # Crear el modelo
            model = Sequential()
            model.add(Dense(layers[0][0], input_dim=X_train.shape[1], activation=layers[0][1]))
            for neurons, activation in layers[1:]:
                model.add(Dense(neurons, activation=activation))
            model.add(Dense(1))  # Capa de salida para regresión

            # Compilar el modelo
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Entrenar el modelo
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), verbose=0)

            # Hacer predicciones
            y_pred = model.predict(X_test)

            # Calcular R^2 y RMSE
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{r2:.4f}")
            with col2:
                st.metric("RMSE", f"{rmse:.4f}")

            # Gráficos de la pérdida durante el entrenamiento y la validación
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history.history['loss'], label='Entrenamiento')
            ax1.plot(history.history['val_loss'], label='Validación')
            ax1.set_title('Pérdida durante el entrenamiento')
            ax1.set_xlabel('Época')
            ax1.set_ylabel('Pérdida')
            ax1.legend()

            ax2.scatter(y_test, y_pred, alpha=0.5)
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax2.set_title('Predicciones vs Valores Reales')
            ax2.set_xlabel('Valores Reales')
            ax2.set_ylabel('Predicciones')

            st.pyplot(fig)

            st.success("Entrenamiento completado.")
        
        # Guardar el modelo entrenado
        model.save('trained_model.h5')
        if 'scaler' in locals():
            joblib.dump(scaler, 'scaler.pkl')

        st.download_button('📥 Descargar modelo', data=open('trained_model.h5', 'rb'), file_name='trained_model.h5')

# Sección para cargar un modelo previamente entrenado
st.subheader("📁 Cargar Modelo Entrenado")
uploaded_model = st.file_uploader("Sube un modelo entrenado (.h5)", type=["h5"])
uploaded_scaler = st.file_uploader("Sube el scaler (opcional, .pkl)", type=["pkl"])

if uploaded_model is not None:
    # Guardar el modelo cargado
    with open("loaded_model.h5", "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    # Cargar el modelo
    model = load_model("loaded_model.h5")
    st.success("Modelo cargado correctamente.")

    # Cargar el scaler si se proporcionó
    if uploaded_scaler is not None:
        with open("loaded_scaler.pkl", "wb") as f:
            f.write(uploaded_scaler.getbuffer())
        scaler = joblib.load("loaded_scaler.pkl")
        st.success("Scaler cargado correctamente.")
    
    # Obtener la estructura del modelo
    input_shape = model.input_shape[1]
    
    # Probar el modelo con nuevos datos
    st.subheader("🔮 Predicción en Tiempo Real")
    st.write("Ingresa los valores para las variables de entrada:")
    
    # Crear un diccionario para almacenar los valores de entrada
    user_input = {}
    
    # Crear dos columnas para organizar los inputs
    col1, col2 = st.columns(2)
    
    # Distribuir los inputs en las columnas
    for i in range(input_shape):
        if i % 2 == 0:
            with col1:
                user_input[f"Variable {i+1}"] = st.number_input(f"Valor para Variable {i+1}", value=0.0, format="%.4f")
        else:
            with col2:
                user_input[f"Variable {i+1}"] = st.number_input(f"Valor para Variable {i+1}", value=0.0, format="%.4f")
    
    if st.button("Realizar Predicción"):
        # Convertir los inputs del usuario a un array numpy
        input_data = np.array(list(user_input.values())).reshape(1, -1)
        
        # Aplicar el scaler si está disponible
        if 'scaler' in locals():
            input_data = scaler.transform(input_data)
        
        # Realizar la predicción
        prediction = model.predict(input_data)
        
        # Mostrar la predicción
        st.metric("Predicción del modelo", f"{prediction[0][0]:.4f}")
        
        # Visualizar la predicción en un gráfico
        fig, ax = plt.subplots()
        ax.bar(["Predicción"], prediction[0])
        ax.set_ylabel("Valor Predicho")
        ax.set_title("Resultado de la Predicción")
        st.pyplot(fig)
else:
    st.info("Por favor, carga un modelo entrenado para realizar predicciones.")
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Detección de Números", layout="centered")

# Título
st.title("🧠 Big Data - Detección de Números (MNIST)")

# Cargar modelo
@st.cache_resource
def load_cnn_model():
    return load_model("model_Mnist_LeNet.h5")

model = load_cnn_model()

# Subir imagen
uploaded_file = st.file_uploader("📤 Sube una imagen (fondo blanco)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Escala de grises

    # Mostrar imagen
    st.image(image, caption="🖼 Imagen cargada", use_container_width=True)

    # Preprocesar
    image_resized = ImageOps.invert(image).resize((28, 28))
    img_array = np.array(image_resized).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predecir
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.success(f"✅ Número detectado: **{predicted_class}**")

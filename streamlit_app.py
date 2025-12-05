import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

st.title("Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit and let the model predict it.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

def preprocess_image(img):
    img = img.convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize (change if your model uses a different size)
    img_array = np.array(img)

    # Normalize if model expects [0,1]
    img_array = img_array / 255.0

    # Reshape to match model input (batch, height, width, channels)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=False)

    with st.spinner("Processing..."):
        processed = preprocess_image(image)
        prediction = model.predict(processed)
        predicted_digit = np.argmax(prediction)

    st.success(f"Predicted Digit: **{predicted_digit}**")
    st.bar_chart(prediction[0])

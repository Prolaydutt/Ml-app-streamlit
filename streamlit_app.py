import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Try to import cv2, fall back to PIL if unavailable
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV not available, using PIL for image processing")

# Configure page
st.set_page_config(
    page_title="Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Error handling for model loading
@st.cache_resource
def load_model():
    try:
        # Replace 'your_model.h5' with your actual model filename
        model = keras.models.load_model('hog_cnn_model.h5')
        return model, None
    except FileNotFoundError:
        return None, "Model file not found. Please ensure 'your_model.h5' is in the same directory."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def preprocess_image(image):
    """Preprocess uploaded image for model prediction"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            if CV2_AVAILABLE:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                # Use PIL for grayscale conversion
                image_gray = image.convert('L')
                img_array = np.array(image_gray)
        
        # Resize to 28x28
        if CV2_AVAILABLE:
            img_resized = cv2.resize(img_array, (28, 28))
        else:
            # Use PIL for resizing
            img_pil = Image.fromarray(img_array)
            img_pil_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
            img_resized = np.array(img_pil_resized)
        
        # Invert colors if needed (MNIST has white digits on black background)
        # Comment out the next line if your model expects black digits on white
        img_resized = 255 - img_resized
        
        # Normalize pixel values to 0-1
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Reshape for model input (batch_size, height, width, channels)
        img_final = img_normalized.reshape(1, 28, 28, 1)
        
        return img_final, None
    except Exception as e:
        return None, f"Error preprocessing image: {str(e)}"

# Main app
def main():
    st.title("🔢 Handwritten Digit Recognition")
    st.write("Upload an image of a handwritten digit (0-9) and let the model predict it!")
    
    # Load model
    model, error = load_model()
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Make sure your model file is named 'your_model.h5' and placed in the same directory as this script.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a single handwritten digit"
    )
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True)
            
            # Preprocess image
            processed_image, preprocess_error = preprocess_image(image)
            
            if preprocess_error:
                st.error(f"❌ {preprocess_error}")
                return
            
            # Make prediction
            with st.spinner("Analyzing..."):
                try:
                    prediction = model.predict(processed_image, verbose=0)
                    predicted_digit = np.argmax(prediction[0])
                    confidence = np.max(prediction[0]) * 100
                    
                    with col2:
                        st.subheader("Prediction")
                        st.markdown(f"### Predicted Digit: **{predicted_digit}**")
                        st.markdown(f"Confidence: **{confidence:.2f}%**")
                        
                        # Show all probabilities
                        st.write("#### All Probabilities:")
                        for digit in range(10):
                            prob = prediction[0][digit] * 100
                            st.write(f"Digit {digit}: {prob:.2f}%")
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("This might be due to model-image compatibility issues. Check your model's expected input shape.")
        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        st.write("""
        1. Upload a clear image of a handwritten digit (0-9)
        2. The model will automatically analyze it
        3. View the prediction and confidence scores
        
        **Tips for best results:**
        - Use images with clear, centered digits
        - Prefer white or light-colored digits on dark backgrounds
        - Avoid cluttered backgrounds
        """)
    
    # Technical info
    with st.expander("🔧 Technical Details"):
        st.write("""
        **Model Configuration:**
        - Expected input: 28×28 grayscale images
        - Output: 10 classes (digits 0-9)
        - Framework: TensorFlow/Keras
        
        **Preprocessing Steps:**
        1. Convert to grayscale
        2. Resize to 28×28 pixels
        3. Invert colors (if needed)
        4. Normalize pixel values (0-1)
        5. Reshape for model input
        """)

if __name__ == "__main__":
    main()

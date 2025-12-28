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

# Configure page with modern styling
st.set_page_config(
    page_title="AI Digit Recognition",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Card styling */
    .stApp {
        background: transparent;
    }
    
    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Upload area */
    div[data-testid="stFileUploader"] > div {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"] > div:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        transform: translateY(-2px);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Result cards */
    .prediction-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .prediction-digit {
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
    }
    
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
    }
    
    /* Progress bars */
    .prob-bar {
        background: #f0f0f0;
        border-radius: 10px;
        height: 30px;
        margin: 0.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .prob-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 1rem;
        color: white;
        font-weight: 600;
        transition: width 0.8s ease;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Image containers */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Error handling for model loading
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('hog_cnn_model.h5')
        input_shape = model.input_shape
        expected_height = input_shape[1] if input_shape[1] else 28
        expected_width = input_shape[2] if input_shape[2] else 28
        return model, None, (expected_height, expected_width)
    except FileNotFoundError:
        return None, "Model file not found. Please ensure 'hog_cnn_model.h5' is in the same directory.", (28, 28)
    except Exception as e:
        return None, f"Error loading model: {str(e)}", (28, 28)

def preprocess_image(image, target_size=(28, 28)):
    """Preprocess uploaded image for model prediction"""
    try:
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            if CV2_AVAILABLE:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image.convert('L')
                img_array = np.array(image_gray)
        
        if CV2_AVAILABLE:
            img_resized = cv2.resize(img_array, target_size)
        else:
            img_pil = Image.fromarray(img_array)
            img_pil_resized = img_pil.resize(target_size, Image.Resampling.LANCZOS)
            img_resized = np.array(img_pil_resized)
        
        img_resized = 255 - img_resized
        img_normalized = img_resized.astype('float32') / 255.0
        img_final = img_normalized.reshape(1, target_size[0], target_size[1], 1)
        
        return img_final, None
    except Exception as e:
        return None, f"Error preprocessing image: {str(e)}"

# Main app
def main():
    # Sidebar
    with st.sidebar:
        st.title("Digit Recognition")
        st.markdown("---")
        st.markdown("### About")
        st.write("Advanced neural network for recognizing handwritten digits with high accuracy.")
        st.markdown("---")
        st.markdown("### Quick Tips")
        st.write("✨ Use clear, centered images")
        st.write("🎨 Light digits on dark background work best")
        st.write("📸 Avoid cluttered backgrounds")
        st.markdown("---")
        st.markdown("### Model Info")
        model, error, input_size = load_model()
        if not error:
            st.write(f"📐 Input: {input_size[0]}×{input_size[1]}")
            st.write(f"🎯 Classes: 0-9")
            st.write(f"🧠 Framework: TensorFlow")
    
    # Main content
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0;'>🎯 Handwritten Digit Recognition</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem; margin-top: 0;'>Powered by Deep Learning & Computer Vision</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load model
    model, error, input_size = load_model()
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Make sure your model file is named 'hog_cnn_model.h5' and placed in the same directory.")
        return
    
    # File uploader in center
    col_space1, col_upload, col_space2 = st.columns([1, 3, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Drop your image here or click to browse",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a single handwritten digit (0-9)"
        )
    
    if uploaded_file is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        
        try:
            image = Image.open(uploaded_file)
            
            # Create two columns for image and results
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color: #667eea; margin-top: 0;'>📸 Your Image</h3>", unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Preprocess image
            processed_image, preprocess_error = preprocess_image(image, target_size=input_size)
            
            if preprocess_error:
                st.error(f"❌ {preprocess_error}")
                return
            
            # Make prediction
            with st.spinner("🔮 Analyzing your digit..."):
                try:
                    prediction = model.predict(processed_image, verbose=0)
                    predicted_digit = np.argmax(prediction[0])
                    confidence = np.max(prediction[0]) * 100
                    
                    with col2:
                        st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                        st.markdown("<h3 style='color: #667eea; margin-top: 0;'>🎯 Prediction Result</h3>", unsafe_allow_html=True)
                        st.markdown(f"<div class='prediction-digit'>{predicted_digit}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='confidence-badge'>{confidence:.1f}% Confident</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            
                    # Probability distribution
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color: #667eea; margin-top: 0;'>📊 Confidence Distribution</h3>", unsafe_allow_html=True)
                    
                    for digit in range(10):
                        prob = prediction[0][digit] * 100
                        st.markdown(f"""
                        <div class='prob-bar'>
                            <div class='prob-fill' style='width: {prob}%;'>
                                <span>Digit {digit}</span>
                                <span>{prob:.1f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("This might be due to model-image compatibility issues.")
        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
    else:
        # Show empty state with example
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class='prediction-card' style='padding: 3rem;'>
                <h2 style='color: #667eea;'>👆 Upload an image to get started</h2>
                <p style='color: #666; font-size: 1.1rem;'>The AI will instantly recognize the handwritten digit</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

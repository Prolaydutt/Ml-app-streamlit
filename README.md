# 🔢 Handwritten Digit Recognition App

A Streamlit web application for recognizing handwritten digits (0-9) using a custom-trained CNN model.

## 📋 Overview

This application allows users to upload images of handwritten digits and get real-time predictions using a trained deep learning model. The app automatically detects the model's expected input size and handles image preprocessing, making it easy to deploy any custom CNN digit classifier.

## ✨ Features

- **Automatic Input Detection**: Automatically detects your model's expected image dimensions
- **Real-time Predictions**: Instant digit recognition with confidence scores
- **Probability Distribution**: View prediction probabilities for all digits (0-9)
- **Robust Error Handling**: Comprehensive error handling for model loading, image processing, and predictions
- **User-friendly Interface**: Clean, intuitive UI built with Streamlit
- **Flexible Image Processing**: Works with or without OpenCV (falls back to PIL)

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Your trained model file (`.h5`  format)

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained model file in the project directory and rename it to `your_model.h5` (or update the filename in line 21 of `streamlit_app.py`)

### Running Locally

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Dependencies

```
streamlit
tensorflow
pillow
numpy
```

## 🎯 Usage

1. **Launch the app** using the command above
2. **Upload an image** of a handwritten digit (PNG, JPG, or JPEG)
3. **View the prediction** along with confidence scores
4. **Check probability distribution** for all digits

### Tips for Best Results

- Use clear, centered images of single digits
- Prefer white or light-colored digits on dark backgrounds
- Avoid cluttered backgrounds or multiple digits
- Ensure the digit is clearly visible and not too small

## 🏗️ Project Structure

```
.
├── streamlit_app.py       # Main application file
├── your_model.h5          # Your trained model (you provide this)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Configuration

### Changing the Model File

Edit line 21 in `streamlit_app.py`:
```python
model = keras.models.load_model('your_model_name.h5')
```

### Adjusting Color Inversion

If your model expects **black digits on white background** instead of white on black, comment out line 49:
```python
# img_resized = 255 - img_resized  # Comment this line
```

### Custom Input Size

The app automatically detects your model's input size, but if you need to manually set it, modify the `preprocess_image` function's `target_size` parameter.

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub (including your model file if it's under 100MB)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" and select your repository
5. Set `streamlit_app.py` as the main file
6. Click "Deploy"

### Deploy to Other Platforms

This app can also be deployed to:
- **Heroku**: Use the included `requirements.txt`
- **Google Cloud Run**: Containerize with Docker
- **AWS EC2**: Run directly on a virtual machine
- **Render**: Deploy as a web service

## 🛠️ Error Handling

The app includes comprehensive error handling for:

- **Model Loading Errors**: Missing files, corrupted models, incompatible formats
- **Image Upload Errors**: Invalid file formats, corrupted images
- **Preprocessing Errors**: Dimension mismatches, color space issues
- **Prediction Errors**: Model-input incompatibilities, runtime errors
- **Library Dependencies**: Graceful fallback when OpenCV is unavailable

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

Your Name - [Your GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [TensorFlow](https://www.tensorflow.org/)
- Image processing with [Pillow](https://python-pillow.org/)

## 📞 Support

If you encounter any issues or have questions, please [open an issue](https://github.com/yourusername/yourrepo/issues) on GitHub.

---

**Made with ❤️ for digit recognition**

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import plotly.graph_objects as go
import cv2
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        background-color: #f5f7fa;
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
        color: #1f2937;
    }
    .prediction-box {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
        text-align: center;
        margin-top: 20px;
    }
    .confidence-bar {
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define class labels
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_DISPLAY_NAMES = {
    'glioma': '🧬 Glioma',
    'meningioma': '🧬 Meningioma',
    'pituitary': '🧬 Pituitary Tumor',
    'notumor': '✅ No Tumor'
}
CLASS_COLORS = {
    'glioma': '#FF6B6B',
    'meningioma': '#4ECDC4',
    'pituitary': '#FFD93D',
    'notumor': '#6BCB77'
}
CLASS_DESCRIPTIONS = {
    'glioma': "A tumor that arises from glial cells in the brain or spine.",
    'meningioma': "A tumor that forms on the membranes covering the brain and spinal cord.",
    'pituitary': "A tumor that forms in the pituitary gland at the base of the brain.",
    'notumor': "No abnormal growth detected in the brain scan."
}

# Model loading with caching
@st.cache_resource
def load_trained_model():
    """Load the trained brain tumor detection model."""
    try:
        # Try to load from the saved model file
        model = load_model('brain_tumor_model.h5')
        return model
    except:
        # If model not found, show an error
        st.error("⚠️ Model file not found. Please ensure 'brain_tumor_model.h5' is in the app directory.")
        return None

def load_and_preprocess_image(img_file, target_size=(224, 224)):
    """Load and preprocess image for model prediction."""
    img = Image.open(img_file)
    img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array, img

def get_gradcam_heatmap(model, img_array, last_conv_layer_name='Conv_1'):
    """Generate Grad-CAM heatmap for model prediction."""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]
    
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img, alpha=0.4):
    """Overlay heatmap on the original image."""
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (original_img.size[0], original_img.size[1]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert original image to array
    original_img_np = np.array(original_img)
    
    # Overlay heatmap
    superimposed_img = cv2.addWeighted(original_img_np, 1 - alpha, heatmap, alpha, 0)
    
    return Image.fromarray(superimposed_img)

def plot_confidence_chart(probabilities, class_names):
    """Create a Plotly bar chart for confidence scores."""
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities[0] * 100,
            y=class_names,
            orientation='h',
            marker_color=[CLASS_COLORS.get(name.split()[1] if len(name.split()) > 1 else name, '#808080') 
                          for name in class_names],
            text=[f"{p*100:.1f}%" for p in probabilities[0]],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Confidence Scores by Class",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 20, 'family': 'Arial Black'}
        },
        xaxis_title="Confidence (%)",
        yaxis_title="",
        xaxis=dict(range=[0, 100], gridcolor='#e0e0e0'),
        yaxis=dict(gridcolor='#e0e0e0'),
        height=350,
        margin=dict(l=10, r=10, t=60, b=10),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family="Arial", size=14),
    )
    
    # Add a vertical line at 50% for reference
    fig.add_vline(x=50, line_width=1, line_dash="dash", line_color="gray")
    
    return fig

def main():
    """Main function to run the Streamlit app."""
    st.title("🧠 Brain Tumor Detection")
    st.markdown("### Upload a brain MRI scan for instant classification")
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses a deep learning model (MobileNetV2) trained on brain MRI scans 
        to detect and classify tumors into four categories:
        - **Glioma**
        - **Meningioma**
        - **Pituitary Tumor**
        - **No Tumor**
        
        **How it works:**
        1. Upload a brain MRI image
        2. The model analyzes the scan
        3. Get instant results with confidence scores
        """)
        
        st.divider()
        st.header("📊 Model Performance")
        st.metric("Accuracy", "95.8%")
        st.metric("Precision", "95.6%")
        st.metric("Recall", "95.4%")
        st.metric("F1-Score", "95.5%")
        
        st.divider()
        st.caption("Built with TensorFlow and Streamlit")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a brain MRI scan image (JPG, JPEG, PNG)"
        )
    
    # Initialize session state
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'last_probabilities' not in st.session_state:
        st.session_state.last_probabilities = None
    if 'last_image' not in st.session_state:
        st.session_state.last_image = None
    
    if uploaded_file is not None:
        with col1:
            # Display uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded MRI Scan", use_container_width=True)
        
        with col2:
            with st.spinner("🧠 Analyzing the brain scan..."):
                # Load model
                model = load_trained_model()
                
                if model is not None:
                    # Preprocess image
                    img_array, original_img = load_and_preprocess_image(uploaded_file)
                    
                    # Make prediction
                    predictions = model.predict(img_array, verbose=0)
                    predicted_class_idx = np.argmax(predictions[0])
                    predicted_class = CLASS_NAMES[predicted_class_idx]
                    confidence = predictions[0][predicted_class_idx] * 100
                    
                    # Store in session state
                    st.session_state.prediction_made = True
                    st.session_state.last_prediction = predicted_class
                    st.session_state.last_probabilities = predictions
                    st.session_state.last_image = original_img
                    
                    # Display prediction result
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="color: #1f2937; margin-bottom: 10px;">🧠 Prediction Result</h2>
                        <div style="font-size: 48px; margin: 10px 0;">
                            {CLASS_DISPLAY_NAMES[predicted_class]}
                        </div>
                        <div style="font-size: 20px; color: #4B5563; margin: 5px 0;">
                            Confidence: <strong style="color: {CLASS_COLORS[predicted_class]};">{confidence:.1f}%</strong>
                        </div>
                        <div style="font-size: 16px; color: #6B7280; margin: 15px 0 5px 0;">
                            {CLASS_DESCRIPTIONS[predicted_class]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Generate Grad-CAM heatmap
                    try:
                        heatmap = get_gradcam_heatmap(model, img_array)
                        heatmap_img = overlay_heatmap(heatmap, original_img)
                        st.session_state.heatmap_img = heatmap_img
                    except Exception as e:
                        st.warning("Grad-CAM visualization not available for this image.")
                        st.session_state.heatmap_img = None
                else:
                    st.error("Model could not be loaded. Please check the model file.")
    
    # Show results if prediction was made
    if st.session_state.prediction_made and st.session_state.last_prediction is not None:
        st.divider()
        st.markdown("### 📊 Detailed Analysis")
        
        # Create two columns for confidence chart and heatmap
        col3, col4 = st.columns([3, 2])
        
        with col3:
            # Display confidence chart
            class_display_names = [CLASS_DISPLAY_NAMES[c] for c in CLASS_NAMES]
            fig = plot_confidence_chart(st.session_state.last_probabilities, class_display_names)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.markdown("### 🔍 Activation Map")
            st.markdown("*The heatmap shows which areas of the image influenced the model's decision.*")
            
            if st.session_state.get('heatmap_img') is not None:
                st.image(st.session_state.heatmap_img, use_container_width=True)
            else:
                st.info("Heatmap visualization not available for this image.")
        
        # Add a reset button
        if st.button("🔄 Analyze Another Image", use_container_width=True):
            st.session_state.prediction_made = False
            st.session_state.last_prediction = None
            st.session_state.last_probabilities = None
            st.session_state.last_image = None
            st.session_state.heatmap_img = None
            st.rerun()
    
    # If no image uploaded, show instructions
    else:
        with col2:
            st.markdown("""
            <div style="background-color: #f8fafc; border-radius: 15px; padding: 40px 20px; text-align: center; border: 2px dashed #cbd5e1;">
                <div style="font-size: 60px; margin-bottom: 20px;">🧠</div>
                <h3 style="color: #1f2937;">Upload a Brain MRI Scan</h3>
                <p style="color: #6B7280;">Upload an image to get started with brain tumor detection.</p>
                <p style="color: #6B7280; font-size: 14px; margin-top: 10px;">
                    Supported formats: JPG, JPEG, PNG
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

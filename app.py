import streamlit as st
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

from preprocessing.preprocess import preprocess_image
from preprocessing.explanation_text import explanation_text
from utils.gradcam import generate_gradcam
from utils.lime_explainer import explain_with_lime

# Cache the model loading so it doesn't reload every run
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Sefat33/retinal",
        filename="model.keras.keras",
        repo_type="model"
    )
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Find last conv layer for Grad-CAM
last_conv_layer = [l.name for l in reversed(model.layers) if 'conv' in l.name or 'mhsa' in l.name][0]
class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

st.set_page_config(page_title="Retinal Disease Diagnosis", layout="wide")
st.title("👁️ Retinal Disease Diagnosis with Explainability")

uploaded_file = st.sidebar.file_uploader("📤 Upload Retinal Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load uploaded image as RGB
    img = Image.open(uploaded_file).convert("RGB")
    temp_path = "temp.jpg"
    img.save(temp_path)

    if st.sidebar.button("🔎 Analyze"):
        # Preprocess image: returns (1, H, W, 3) tensor and display image numpy array
        preprocessed, display_img = preprocess_image(temp_path)

        # Predict
        pred = model.predict(preprocessed)[0]
        pred_idx = np.argmax(pred)
        pred_class = class_names[pred_idx]
        confidence = pred[pred_idx] * 100

        st.image(display_img, caption=f"Prediction: {pred_class} ({confidence:.2f}%)", use_column_width=True)
        st.success(explanation_text.get(pred_class, "No explanation available."))

        # Grad-CAM visualization
        st.subheader("🔥 Grad-CAM")
        gradcam_img = generate_gradcam(model, preprocessed, pred_idx, last_conv_layer, display_img.shape[:2])
        st.image(gradcam_img, use_column_width=True)

        # LIME explanation visualization
        st.subheader("🧠 LIME Explanation")
        lime_img = explain_with_lime(model, display_img, pred_idx)
        st.image(lime_img, use_column_width=True)

        # Show class probabilities bar chart
        st.subheader("📊 Class Probabilities")
        probs_dict = {class_names[i]: float(pred[i]) for i in range(len(class_names))}
        st.bar_chart(probs_dict)
else:
    st.info("Please upload a retinal image from the sidebar to start diagnosis.")

import streamlit as st
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

from preprocessing.preprocess import preprocess_image
from preprocessing.explanation_text import explanation_text
from utils.gradcam import generate_gradcam
from utils.lime_explainer import explain_with_lime
from keras_cv_attention_models.coatnet import CoAtNet0

# Import to ensure custom layers are registered
_ = CoAtNet0

# Load model from Hugging Face
with st.spinner("🔄 Downloading model from Hugging Face..."):
    model_path = hf_hub_download(
        repo_id="Sefat33/retinal",
        filename="model.keras.keras",  # Your model's actual file name
        repo_type="model"
    )
    model = tf.keras.models.load_model(model_path)

last_conv_layer = [l.name for l in reversed(model.layers) if 'conv' in l.name or 'mhsa' in l.name][0]
class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

st.set_page_config(page_title="Retinal Disease Diagnosis", layout="wide")
st.title("👁️ Retinal Disease Diagnosis with Explainability")

uploaded_file = st.sidebar.file_uploader("📤 Upload Retinal Image", type=["jpg", "jpeg", "png"])

if st.sidebar.button("🔎 Analyze") and uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    preprocessed, display_img = preprocess_image("temp.jpg")
    pred = model.predict(preprocessed)[0]
    pred_idx = np.argmax(pred)
    pred_class = class_names[pred_idx]
    conf = pred[pred_idx] * 100

    st.image(display_img, caption=f"Prediction: {pred_class} ({conf:.2f}%)", use_column_width=True)
    st.success(explanation_text[pred_class])

    st.subheader("🔥 Grad-CAM")
    gradcam = generate_gradcam(model, preprocessed, pred_idx, last_conv_layer, display_img.shape[:2])
    st.image(gradcam, use_column_width=True)

    st.subheader("🧠 LIME Explanation")
    lime_img = explain_with_lime(model, display_img, pred_idx)
    st.image(lime_img, use_column_width=True)

    st.subheader("📊 Class Probabilities")
    st.bar_chart({class_names[i]: float(pred[i]) for i in range(len(class_names))})

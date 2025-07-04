import os
import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from utils.gradcam import generate_gradcam
from utils.lime_explainer import explain_with_lime
from preprocessing.preprocess import preprocess_image
from preprocessing.explanation_text import explanation_text

model = tf.keras.models.load_model("model.keras")
last_conv_layer = [l.name for l in reversed(model.layers) if 'conv' in l.name or 'mhsa' in l.name][0]
class_names = ['Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Others']

st.set_page_config(page_title="Retinal Disease Diagnosis", layout="wide")
st.title("👁️ Retinal Disease Diagnosis with Explainability")

st.sidebar.header("🔍 Select Input Image")
image_folder = "data/2850-aug"
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
selected_image = st.sidebar.selectbox("Choose an image", image_files)

if st.sidebar.button("🔎 Analyze"):
    img_path = os.path.join(image_folder, selected_image)
    preprocessed, display_img = preprocess_image(img_path)
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
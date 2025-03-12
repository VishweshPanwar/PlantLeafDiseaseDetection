import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

st.set_page_config(page_title="Plant Disease Classifier", layout="wide")

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"

model = tf.keras.models.load_model(model_path)

class_indices = json.load(open(f"{working_dir}/class_indices.json"))

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.  
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: bold;
            text-align: center;
            color: #2E8B57;
        }
        .uploaded-img {
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            margin-top: 10px;
        }
        .prediction-text {
            font-size: 1.5rem;
            font-weight: bold;
            color: #d9534f;
            text-align: center;
        }
        .button-style {
            background-color: #2E8B57;
            color: white;
            font-size: 1.2rem;
            font-weight: bold;
            padding: 10px;
            border-radius: 10px;
            width: 100%;
        }
        .button-style:hover {
            background-color: #228B22;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🌿 Plant Disease Classifier</div>', unsafe_allow_html=True)

uploaded_image = st.file_uploader("📤 Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        image = Image.open(uploaded_image)
        resized_img = image.resize((250, 250))
        st.image(resized_img, caption="Uploaded Image", use_column_width=True, output_format="JPEG")

    with col2:
        st.write("\n")
        if st.button("🔍 Predict", key="predict_button"):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.markdown(f'<p class="prediction-text">🟢 Prediction: {str(prediction)}</p>', unsafe_allow_html=True)

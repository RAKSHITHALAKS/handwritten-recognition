import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt

# -------------------------------
# Config
# -------------------------------
MODEL_FILE = "emnist_cnn.keras"

# EMNIST labels (62 classes: 0–9, A–Z, a–z)
EMNIST_LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

# -------------------------------
# Load model
# -------------------------------
if os.path.exists(MODEL_FILE):
    model = tf.keras.models.load_model(MODEL_FILE)
else:
    st.error(f"❌ Model file {MODEL_FILE} not found! Please train using train_emnist.py first.")
    st.stop()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Handwritten Character Recognition", page_icon="✍️")
st.title("Handwritten Character Recognition")
st.markdown("Draw a **digit (0–9)** or **letter (A–Z, a–z)** below and click **Predict**!")

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data).astype("uint8")).convert("L")
    img = img.resize((28, 28))
    img_arr = np.array(img)
    img_arr = img_arr.reshape(1, 28, 28, 1) / 255.0

    col1, col2 = st.columns([1,1])

    with col1:
        if st.button("Predict"):
            preds = model.predict(img_arr)
            pred_class = np.argmax(preds)
            st.success(f" Prediction: {EMNIST_LABELS[pred_class]}")

            # Confidence bar chart
            st.subheader("Prediction Confidence")
            fig, ax = plt.subplots(figsize=(12,4))
            ax.bar(range(len(EMNIST_LABELS)), preds[0])
            ax.set_xticks(range(len(EMNIST_LABELS)))
            ax.set_xticklabels(EMNIST_LABELS, rotation=90)
            ax.set_ylabel("Probability")
            st.pyplot(fig)

    with col2:
        if st.button("Clear Canvas"):
            st.experimental_rerun()


Add app.py


# Handwritten Character Recognition (0–9, A–Z)

An interactive Python-based web application to recognize handwritten digits (0–9) and letters (A–Z, a–z) using a Convolutional Neural Network (CNN) trained on the EMNIST dataset. The app provides real-time predictions along with confidence visualization, and is deployable via Streamlit.

---

## Features
- Recognizes **digits 0–9 and letters A–Z / a–z.
- Interactive **drawing canvas** using Streamlit.
- **Confidence bar chart** showing prediction probabilities.
- Extensible to OCR systems, form digitization, and handwritten text recognition.

---

## Project Structure

├─ app.py # Streamlit app for EMNIST
├─ train_emnist.py # Script to train CNN on EMNIST
├─ emnist_cnn.keras # Trained CNN model (optional)
├─ requirements.txt # Required Python packages
├─ images/ # Screenshots & GIFs
│ ├─ canvas_example.png
│ ├─ prediction_chart.png
│ └─ app_demo.gif
└─ README.md

Dependencies
Python 3.10+
TensorFlow
Keras
TensorFlow Datasets
Streamlit
Streamlit Drawable Canvas
Matplotlib
NumPy
Pillow

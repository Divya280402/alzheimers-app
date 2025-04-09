import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Path to the trained model
MODEL_PATH = "alzheimer_cnn_model.h5"

# Load the model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model Loaded Successfully!")
else:
    st.error("❌ Model file not found! Please upload `alzheimer_cnn_model.h5`.")

# Define class labels and descriptions
CLASS_NAMES = ["Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"]
CLASS_DESCRIPTIONS = {
    "Non Demented": "No signs of dementia.",
    "Very Mild Demented": "Minimal cognitive decline, early-stage dementia.",
    "Mild Demented": "Moderate memory loss and cognitive issues.",
    "Moderate Demented": "Significant memory loss, requires assistance with daily activities."
}

# Function to find last conv layer
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

# Grad-CAM heatmap function
def grad_cam(model, img_array, layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        predicted_class = np.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs.numpy()[0]

    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap, predicted_class

# App Title and Description
st.title("Alzheimer's Detection with Grad-CAM")
st.write("Upload one or more MRI scans to classify dementia stage and visualize model focus.")

# Multi-file Upload
uploaded_files = st.file_uploader("📂 Upload MRI Scan(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader(f"Processing: {uploaded_file.name}")

        file_details = {
            "Filename": uploaded_file.name,
            "File Size (KB)": round(len(uploaded_file.getvalue()) / 1024, 2),
            "Format": uploaded_file.type
        }
        st.sidebar.write("**File Details:**")
        st.sidebar.json(file_details)

        # Load and preprocess image
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Grad-CAM
        last_conv_layer = find_last_conv_layer(model)
        if last_conv_layer is None:
            st.error("❌ No convolutional layer found!")
            continue

        heatmap, predicted_class = grad_cam(model, img_array, last_conv_layer)

        # Create overlay
        img_cv = np.array(img)
        heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        alpha = st.sidebar.slider(f"Grad-CAM Intensity for {uploaded_file.name}", 0.1, 1.0, 0.3, 0.05)
        overlay = cv2.addWeighted(img_cv, 1 - alpha, heatmap_color, alpha, 0)

        # Side-by-side view
        st.write("### 🔍 Visual Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

        # Prediction
        st.subheader("🩺 Prediction Results")
        st.write(f"**Prediction:** {CLASS_NAMES[predicted_class]}")
        st.info(CLASS_DESCRIPTIONS[CLASS_NAMES[predicted_class]])

        # Download heatmap
        heatmap_path = f"grad_cam_result_{uploaded_file.name.replace(' ', '_')}.jpg"
        cv2.imwrite(heatmap_path, overlay)
        st.download_button("📥 Download Heatmap", data=open(heatmap_path, "rb"), file_name=heatmap_path)

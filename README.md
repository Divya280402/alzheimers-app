# alzheimers-app
# Alzheimer's Detection Using Grad-CAM 

An interactive Streamlit application that uses a Convolutional Neural Network (CNN) to classify Alzheimer's disease stages from MRI brain scans, with **Grad-CAM** (Gradient-weighted Class Activation Mapping) used to visualize and interpret which regions of the scan influenced the model's prediction.


## 🔍 Overview

Early detection of Alzheimer's disease is critical for timely intervention. This project combines deep learning-based classification with explainable AI (Grad-CAM) so predictions aren't just accurate — they're interpretable, showing *which parts of the brain scan* the model focused on to reach its conclusion.

## ✨ Features

- Upload one or multiple MRI scans (JPG/PNG) for batch analysis
- CNN-based classification into 4 stages:
  - Non Demented
  - Very Mild Demented
  - Mild Demented
  - Moderate Demented
- Grad-CAM heatmap overlay to visualize model attention/interpretability
- Adjustable heatmap intensity slider for clearer visualization
- Side-by-side comparison of original scan vs. Grad-CAM overlay
- Downloadable heatmap results

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras** — CNN model training and inference
- **OpenCV** — heatmap generation and image processing
- **Streamlit** — interactive web app interface
- **Grad-CAM** — model interpretability technique

## 🚀 How It Works

1. User uploads an MRI scan via the Streamlit interface
2. Image is preprocessed and resized for the model
3. The trained CNN model predicts the dementia stage
4. Grad-CAM computes a heatmap highlighting the regions most influential to the prediction, using gradients from the last convolutional layer
5. The heatmap is overlaid on the original scan for visual interpretation

## 📦 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Divya280402/alzheimers-app.git
cd alzheimers-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 Project Background

This project was developed as part of my M.Sc in Data Analytics coursework, applying deep learning and explainable AI techniques to a real-world healthcare use case. It reflects my interest in building not just accurate models, but interpretable ones — an increasingly important consideration in applied data science and analytics work.

🏆 **Recognition:** Secured **Third Place at Research Day 2025**, Sri Ramachandra Faculty of Engineering and Technology, for this project.

## 👤 Author

**Divya Lakshmi R**
Data Analyst | SQL · Python · Tableau · Power BI | M.Sc Data Analytics
[LinkedIn](https://www.linkedin.com/in/divya-lakshmi-ravi-kumar-b471441ba)

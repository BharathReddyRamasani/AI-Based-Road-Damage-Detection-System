import os
import io
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import model_utils

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "road_damage_cnn_model.keras"))
CACHE_PATH_FILE = os.path.normpath(os.path.join(BASE_DIR, "dataset_path_cache.txt"))

# Page configuration
st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Premium Obsidian Custom CSS Stylesheet
st.markdown("""
<style>
/* Background and text styling */
.stApp {
    background: linear-gradient(135deg, #0a0c10 0%, #151922 100%);
    color: #f0f2f6;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* Glassmorphism Containers */
.glass-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 25px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}
.glass-card:hover {
    background: rgba(255, 255, 255, 0.035);
    border-color: rgba(200, 200, 200, 0.12);
    transform: translateY(-2px);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.45);
}

/* Glowing Typography */
.glowing-title {
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 38px;
    margin-bottom: 2px;
    text-align: center;
}
.glowing-subtitle {
    font-size: 17px;
    color: #a0aec0;
    font-weight: 500;
    margin-bottom: 25px;
    text-align: center;
    letter-spacing: 0.5px;
}

/* Status styles */
.severity-high {
    color: #ff4757;
    text-shadow: 0 0 10px rgba(255, 71, 87, 0.4);
    font-weight: 700;
}
.severity-medium {
    color: #ff9f43;
    text-shadow: 0 0 10px rgba(255, 159, 67, 0.4);
    font-weight: 700;
}
.severity-low {
    color: #2ed573;
    text-shadow: 0 0 10px rgba(46, 213, 115, 0.4);
    font-weight: 700;
}

/* Clean recommendations panels */
.rec-box {
    border-left: 5px solid;
    padding: 15px;
    border-radius: 4px;
    background: rgba(255,255,255,0.01);
}
.rec-high {
    border-left-color: #ff4757;
    background: rgba(255, 71, 87, 0.05);
}
.rec-medium {
    border-left-color: #ff9f43;
    background: rgba(255, 159, 67, 0.05);
}
.rec-low {
    border-left-color: #2ed573;
    background: rgba(46, 213, 115, 0.05);
}

/* File Upload drag area */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.01) !important;
    border: 2px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

/* Hide Streamlit default branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Helper function to get cached dataset path
def get_cached_dataset_path():
    if os.path.exists(CACHE_PATH_FILE):
        with open(CACHE_PATH_FILE, "r") as f:
            path = f.read().strip()
            if os.path.exists(path):
                return path
    return None

def set_cached_dataset_path(path):
    with open(CACHE_PATH_FILE, "w") as f:
        f.write(path)

# Initialize Session States
if "dataset_path" not in st.session_state:
    st.session_state["dataset_path"] = get_cached_dataset_path()
if "df_metadata" not in st.session_state:
    st.session_state["df_metadata"] = None
if "label_map" not in st.session_state:
    st.session_state["label_map"] = model_utils.LABEL_MAP

# Load metadata if dataset path is cached
if st.session_state["dataset_path"] and st.session_state["df_metadata"] is None:
    try:
        df, label_map = model_utils.load_metadata(st.session_state["dataset_path"])
        st.session_state["df_metadata"] = df
        st.session_state["label_map"] = label_map
    except Exception:
        pass

# ----------------- SECTION 1: Header -----------------
st.markdown('<div class="glowing-title">AI-Based Road Damage Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="glowing-subtitle">Smart City Infrastructure Monitoring using CNN</div>', unsafe_allow_html=True)

# Check model availability
model_exists = os.path.exists(MODEL_PATH)
if not model_exists:
    st.warning("⚠️ No trained CNN model found at `road_damage_cnn_model.keras`. Please run model training or place a trained model in the folder.")
    
    # Model downloader placeholder
    if st.session_state["dataset_path"]:
        if st.button("🚀 Run Quick Model Training"):
            with st.spinner("Training model..."):
                df = st.session_state["df_metadata"]
                model_utils.train_model_ui(df, MODEL_PATH, epochs=3, subset_size=150)
                st.success("Model trained successfully! Reloading...")
                st.rerun()
    else:
        st.info("If you have the dataset, click below to load/download it.")
        if st.button("📥 Download & Cache Dataset (~80MB)"):
            with st.spinner("Downloading dataset from Kaggle..."):
                path = model_utils.download_dataset()
                st.session_state["dataset_path"] = path
                set_cached_dataset_path(path)
                st.rerun()

st.markdown("---")

# ----------------- SECTION 2: About the Project -----------------
st.markdown("### About the Project")
st.markdown("""
<div class="glass-card">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
            <h5 style="color: #00f2fe; margin-top:0;">Why Road Monitoring is Important?</h5>
            <p style="font-size: 13.5px; line-height: 1.6; color: #cbd5e0; margin-bottom: 0;">
                Road surface deterioration causes billions in vehicle repairs annually and poses serious risks to public safety. Traditional inspections are manual, time-consuming, and labor-intensive. Automated image-based monitoring enables city councils to continuously inspect large road networks, saving repair costs and enhancing civic traffic safety.
            </p>
        </div>
        <div>
            <h5 style="color: #00f2fe; margin-top:0;">Role of CNNs in Computer Vision</h5>
            <p style="font-size: 13.5px; line-height: 1.6; color: #cbd5e0; margin-bottom: 0;">
                Convolutional Neural Networks (CNNs) emulate the human visual system by extracting spatial hierarchies of features—ranging from simple edges to complex shapes like potholes. The model automatically learns which texture variations indicate road defects, enabling automated classification with high accuracy.
            </p>
        </div>
    </div>
    <div style="margin-top: 18px;">
        <h5 style="color: #00f2fe; margin-top:0;">Practical Industry Applications</h5>
        <p style="font-size: 13.5px; line-height: 1.6; color: #cbd5e0; margin-bottom: 0;">
            This system powers real-world municipal technologies. It can be integrated into municipal dashcams on garbage trucks, public buses, or drone surveys to map pavement health. Cities can then automatically schedule repairs, assign priority indices, and track road health over time.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SECTION 3: Upload Area -----------------
st.markdown("### Upload Road Image")
st.markdown("<p style='font-size:14px; color:#8a99ad; margin-bottom: 10px;'>Drag and drop an image of a road surface or upload a file. Supported file formats: JPG, JPEG, PNG.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Upload road image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# Option to use a sample test image
sample_img_choice = st.checkbox("Or use a random test image from the local dataset")
image_bytes = None

if sample_img_choice:
    if st.session_state["df_metadata"] is not None:
        df = st.session_state["df_metadata"]
        sample_row = df.sample(1, random_state=None).iloc[0]
        sample_path = sample_row["path"]
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
        else:
            st.error("Sample image file not found on disk.")
    else:
        st.warning("Please download/load the dataset first to use sample images.")
elif uploaded_file is not None:
    image_bytes = uploaded_file.read()

# Run prediction pipeline if an image is provided
if image_bytes is not None:
    st.markdown("---")
    
    col_preview, col_predict = st.columns([1, 1])
    
    # ----------------- SECTION 4: Uploaded Image Preview -----------------
    with col_preview:
        st.markdown("### Uploaded Image Preview")
        st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
        pil_img = Image.open(io.BytesIO(image_bytes))
        st.image(pil_img, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Prediction loop
    with col_predict:
        if not model_exists:
            st.error("Cannot perform prediction: Model is not trained yet.")
        else:
            with st.spinner("Analyzing road anomalies..."):
                try:
                    label, confidence, detailed_probs, _ = model_utils.predict_damage(
                        MODEL_PATH,
                        image_bytes
                    )
                    
                    # Determine Severity & styling mapping
                    if label == "pothole":
                        severity = "High"
                        sev_class = "severity-high"
                        rec_class = "rec-high"
                        priority = "Immediate maintenance recommended."
                        warning = "High-risk road condition detected. Potential vehicle axle damage or accidents."
                    elif label == "crack":
                        severity = "Medium"
                        sev_class = "severity-medium"
                        rec_class = "rec-medium"
                        priority = "Scheduled sealing recommended."
                        warning = "Moderate-risk road condition detected. Requires maintenance within 3 months to prevent pothole formation."
                    else:  # manhole
                        severity = "Low"
                        sev_class = "severity-low"
                        rec_class = "rec-low"
                        priority = "Routine maintenance only."
                        warning = "Safe road condition detected. Regular annual audits are sufficient."
                        
                    # ----------------- SECTION 5: Prediction Area -----------------
                    st.markdown("### Prediction Results")
                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <table style="width:100%; font-size:16px; border-collapse: collapse;">
                                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 40px;">
                                    <td style="color:#8a99ad; font-weight:500;">Prediction</td>
                                    <td style="font-weight:700; text-transform: capitalize;">{label} Detected</td>
                                </tr>
                                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); height: 40px;">
                                    <td style="color:#8a99ad; font-weight:500;">Confidence</td>
                                    <td style="font-weight:700; color:#00f2fe;">{confidence * 100:.1f}%</td>
                                </tr>
                                <tr style="height: 40px;">
                                    <td style="color:#8a99ad; font-weight:500;">Severity Level</td>
                                    <td class="{sev_class}">{severity}</td>
                                </tr>
                            </table>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # ----------------- SECTION 6: Visualization Area -----------------
                    st.markdown("### Visualization Area")
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    
                    # Simple Plotly bar chart representing prediction probability
                    prob_df = pd.DataFrame({
                        "Anomaly": [c.capitalize() for c in detailed_probs.keys()],
                        "Confidence (%)": [v * 100 for v in detailed_probs.values()]
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x="Confidence (%)",
                        y="Anomaly",
                        orientation="h",
                        color="Anomaly",
                        color_discrete_sequence=["#ff9f43", "#2ed573", "#ff4757"],
                        template="plotly_dark",
                        range_x=[0, 100]
                    )
                    
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        height=160,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # ----------------- SECTION 7: Recommendations -----------------
                    st.markdown("### Recommendations")
                    st.markdown(
                        f"""
                        <div class="glass-card rec-box {rec_class}">
                            <h5 style="margin-top:0; margin-bottom: 8px;">Repair Priority: {priority}</h5>
                            <span style="font-size: 13.5px; line-height: 1.5; color: #e2e8f0; display: block;">
                                ⚠️ <b>Safety Warning</b>: {warning}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                except Exception as e:
                    st.error(f"Error during inference: {e}")

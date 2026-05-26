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

# Load Outfit Google Font and Premium custom CSS Stylesheet
CSS_STYLE = """
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}
.stApp {
    background: radial-gradient(circle at top, #191f35 0%, #090c14 100%) !important;
    color: #f1f5f9;
}
.premium-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.6) 100%);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 26px;
    margin-bottom: 25px;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.5);
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.premium-card:hover {
    border-color: rgba(0, 242, 254, 0.3);
    transform: translateY(-3px);
    box-shadow: 0 15px 50px 0 rgba(0, 242, 254, 0.15);
}
.glowing-title {
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 40px;
    margin-bottom: 4px;
    text-align: center;
    letter-spacing: -0.5px;
}
.glowing-subtitle {
    font-size: 18px;
    color: #94a3b8;
    font-weight: 400;
    margin-bottom: 30px;
    text-align: center;
    letter-spacing: 0.5px;
}
.card-section-title {
    color: #00f2fe;
    font-weight: 700;
    font-size: 18px;
    margin-top: 0;
    margin-bottom: 10px;
    border-bottom: 1px solid rgba(0, 242, 254, 0.1);
    padding-bottom: 6px;
}
.card-text {
    font-size: 14.5px;
    line-height: 1.6;
    color: #cbd5e1;
    margin-bottom: 0;
}
.metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.metric-row:last-child {
    border-bottom: none;
}
.metric-name {
    font-size: 15px;
    color: #94a3b8;
    font-weight: 500;
}
.metric-value {
    font-size: 17px;
    font-weight: 700;
}
.sev-pill {
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.5px;
}
.sev-pill-high {
    background-color: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.sev-pill-medium {
    background-color: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.sev-pill-low {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.rec-card {
    border-left: 5px solid;
    padding: 18px;
    border-radius: 8px;
    margin-top: 15px;
}
.rec-card-high {
    border-left-color: #ef4444;
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.08) 0%, rgba(15, 23, 42, 0.4) 100%);
}
.rec-card-medium {
    border-left-color: #f59e0b;
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.08) 0%, rgba(15, 23, 42, 0.4) 100%);
}
.rec-card-low {
    border-left-color: #10b981;
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.08) 0%, rgba(15, 23, 42, 0.4) 100%);
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 2px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 14px !important;
    padding: 12px !important;
}
.custom-progress-container {
    width: 100%;
    background-color: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    height: 8px;
    margin-top: 4px;
    overflow: hidden;
}
.custom-progress-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.8s ease-in-out;
}
</style>
""".replace('\n', ' ')
st.markdown(CSS_STYLE, unsafe_allow_html=True)

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

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- SECTION 2: About the Project -----------------
st.markdown("### 📋 About the Project")
st.markdown("""
<div class="premium-card">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
        <div>
            <h5 class="card-section-title">🛣️ Why Road Monitoring is Important?</h5>
            <p class="card-text">
                Road surface deterioration causes billions in vehicle repairs annually and poses serious risks to public safety. Traditional inspections are manual, time-consuming, and labor-intensive. Automated image-based monitoring enables city councils to continuously inspect large road networks, saving repair costs and enhancing civic traffic safety.
            </p>
        </div>
        <div>
            <h5 class="card-section-title">🧠 Role of CNNs in Computer Vision</h5>
            <p class="card-text">
                Convolutional Neural Networks (CNNs) emulate the human visual system by extracting spatial hierarchies of features—ranging from simple edges to complex shapes like potholes. The model automatically learns which texture variations indicate road defects, enabling automated classification with high accuracy.
            </p>
        </div>
    </div>
    <div style="margin-top: 20px;">
        <h5 class="card-section-title">🏗️ Practical Industry Applications</h5>
        <p class="card-text">
            This system powers real-world municipal technologies. It can be integrated into municipal dashcams on garbage trucks, public buses, or drone surveys to map pavement health. Cities can then automatically schedule repairs, assign priority indices, and track road health over time.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SECTION 3: Upload Area -----------------
st.markdown("### 📤 Upload Road Image")
st.markdown("<p style='font-size:14px; color:#94a3b8; margin-bottom: 15px;'>Drag and drop an image of a road surface or upload a file. Supported file formats: JPG, JPEG, PNG.</p>", unsafe_allow_html=True)

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
    
    col_preview, col_predict = st.columns([1, 1], gap="large")
    
    # ----------------- SECTION 4: Uploaded Image Preview -----------------
    with col_preview:
        st.markdown("### 📸 Image Preview")
        st.markdown("<div class='premium-card' style='text-align: center; padding: 12px;'>", unsafe_allow_html=True)
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
                        sev_class = "sev-pill sev-pill-high"
                        rec_class = "rec-card rec-card-high"
                        priority = "🚨 Immediate repair recommended."
                        warning = "High-risk road condition detected. Potential vehicle axle damage or accident hazard."
                        fill_color = "linear-gradient(90deg, #f87171, #ef4444)"
                    elif label == "crack":
                        severity = "Medium"
                        sev_class = "sev-pill sev-pill-medium"
                        rec_class = "rec-card rec-card-medium"
                        priority = "⚠️ Scheduled sealing recommended."
                        warning = "Moderate-risk road condition. Schedule bituminous crack sealing within the next 3 months to prevent future pothole formation."
                        fill_color = "linear-gradient(90deg, #fbbf24, #f59e0b)"
                    else:  # manhole
                        severity = "Low"
                        sev_class = "sev-pill sev-pill-low"
                        rec_class = "rec-card rec-card-low"
                        priority = "🟢 Routine auditing only."
                        warning = "Safe road condition detected. Standard sewer manhole/drain structure identified. Perform annual audits."
                        fill_color = "linear-gradient(90deg, #34d399, #10b981)"
                        
                    # ----------------- SECTION 5: Prediction Area -----------------
                    st.markdown("### 📊 Prediction Details")
                    st.markdown(
                        f"""
                        <div class="premium-card">
                            <div class="metric-row">
                                <span class="metric-name">Damage Type</span>
                                <span class="metric-value" style="text-transform: capitalize; color: #00f2fe;">{label} Detected</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Confidence</span>
                                <span class="metric-value">{confidence * 100:.2f}%</span>
                            </div>
                            <div class="metric-row">
                                <span class="metric-name">Severity Level</span>
                                <span class="{sev_class}">{severity}</span>
                            </div>
                            <div style="margin-top: 15px;">
                                <span class="metric-name">Confidence Bar Indicator</span>
                                <div class="custom-progress-container">
                                    <div class="custom-progress-fill" style="width: {confidence * 100:.1f}%; background: {fill_color};"></div>
                                </div>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # ----------------- SECTION 6: Visualization Area -----------------
                    st.markdown("### 📈 Confidence Visualizer")
                    st.markdown("<div class='premium-card' style='padding: 12px;'>", unsafe_allow_html=True)
                    
                    # Styled Plotly bar chart representing prediction probability
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
                        color_discrete_sequence=["#fbbf24", "#34d399", "#f87171"], # crack (yellow), manhole (green), pothole (red)
                        template="plotly_dark",
                        range_x=[0, 100]
                    )
                    
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        height=160,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, title=None)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # ----------------- SECTION 7: Recommendations -----------------
                    st.markdown("### 🛠️ Action Recommendations")
                    st.markdown(
                        f"""
                        <div class="{rec_class}">
                            <h5 style="margin-top:0; margin-bottom: 8px; font-weight: 700; font-size:16px;">{priority}</h5>
                            <span style="font-size: 13.5px; line-height: 1.5; color: #f1f5f9; display: block;">
                                {warning}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                except Exception as e:
                    st.error(f"Error during inference: {e}")

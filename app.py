import os
import io
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import plotly.express as px
from PIL import Image

import model_utils

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "road_damage_cnn_model.keras"))
CACHE_PATH_FILE = os.path.normpath(os.path.join(BASE_DIR, "dataset_path_cache.txt"))

# Page configuration - Set to WIDE mode as shown in the screenshot
st.set_page_config(
    page_title="AI-Based Road Damage Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Widescreen Custom Stylesheet matching the screenshot exactly
CSS_STYLE = """
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}
.stApp {
    background-color: #060b16 !important;
    color: #cbd5e1;
}
/* Style section headings with light blue/cyan color */
.section-heading {
    color: #38bdf8 !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 25px !important;
    margin-bottom: 12px !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    padding-bottom: 6px;
}
/* Clean thin border containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: rgba(15, 23, 42, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
    padding: 20px !important;
    box-shadow: none !important;
    transition: none !important;
    margin-bottom: 15px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(56, 189, 248, 0.2) !important;
}
/* About Section Grid Elements */
.about-col-title {
    font-size: 14px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
}
.about-col-text {
    font-size: 12.5px;
    line-height: 1.5;
    color: #94a3b8;
}
/* Custom Prediction Banner */
.pred-banner {
    background-color: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 8px;
    padding: 24px;
    text-align: center;
    margin-bottom: 15px;
}
.pred-banner-title {
    color: #38bdf8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.pred-banner-value {
    color: #f1f5f9;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
/* Pill badges */
.badge-row {
    display: flex;
    justify-content: center;
    gap: 15px;
    align-items: center;
}
.pill-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 12px;
    background-color: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
}
.pill-pothole {
    background-color: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border-color: rgba(239, 68, 68, 0.25);
}
.pill-crack {
    background-color: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border-color: rgba(245, 158, 11, 0.25);
}
.pill-manhole {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border-color: rgba(16, 185, 129, 0.25);
}
/* Hide Streamlit default branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stFileUploader"] {
    background-color: rgba(15, 23, 42, 0.2) !important;
    border: 1px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
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

# ----------------- SECTION 1: Header (Top-Left) -----------------
st.markdown('<h2 style="color: #f1f5f9; margin: 0; font-size: 22px; font-weight: 800; padding-bottom: 2px;">AI-Based Road Damage Detection System</h2>', unsafe_allow_html=True)
st.markdown('<p style="color: #94a3b8; margin: 0; font-size: 13px; margin-bottom: 20px;">Smart City Infrastructure Monitoring using CNN</p>', unsafe_allow_html=True)

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

# ----------------- SECTION 2: About the Project -----------------
st.markdown('<div class="section-heading">SECTION 2 -- About the Project</div>', unsafe_allow_html=True)
with st.container(border=True):
    col_about1, col_about2, col_about3 = st.columns(3, gap="large")
    with col_about1:
        st.markdown(
            '<div class="about-col-title">'
            '<span style="color:#00f2fe; margin-right:8px; font-size:12px;">■</span> Importance'
            '</div>'
            '<div class="about-col-text">'
            'Ensure public safety, reduce vehicle wear & accidents, lower city council maintenance costs, and prioritize high-risk repair zones.'
            '</div>', 
            unsafe_allow_html=True
        )
    with col_about2:
        st.markdown(
            '<div class="about-col-title">'
            '<span style="color:#ff007f; margin-right:8px; font-size:12px;">●</span> CNN Classification'
            '</div>'
            '<div class="about-col-text">'
            'Automatically extracts spatial hierarchies, analyzes textures, and performs classifications to replace manual inspection methods.'
            '</div>', 
            unsafe_allow_html=True
        )
    with col_about3:
        st.markdown(
            '<div class="about-col-title">'
            '<span style="color:#a855f7; margin-right:8px; font-size:12px;">▲</span> Practical Applications'
            '</div>'
            '<div class="about-col-text">'
            'Real-time automated road maintenance, integration with municipal street-sweeper cameras, and prioritized city repair budgets.'
            '</div>', 
            unsafe_allow_html=True
        )

# ----------------- SECTION 3: Upload Area -----------------
st.markdown('<div class="section-heading">SECTION 3 -- Upload Area</div>', unsafe_allow_html=True)
with st.container(border=True):
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
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_preview, col_predict = st.columns([1.2, 1.0], gap="large")
    
    # ----------------- LEFT COLUMN: SECTION 4 -- Image Preview -----------------
    with col_preview:
        st.markdown('<div class="section-heading">SECTION 4 -- Image Preview</div>', unsafe_allow_html=True)
        pil_img = Image.open(io.BytesIO(image_bytes))
        st.image(pil_img, use_column_width=True)
            
    # ----------------- RIGHT COLUMN: Prediction, Visualization & Recommendations -----------------
    with col_predict:
        if not model_exists:
            st.error("Cannot perform prediction: Model is not trained yet.")
        else:
            with st.spinner("Analyzing road anomalies..."):
                try:
                    label, confidence, detailed_probs, _ = model_utils.predict_damage(MODEL_PATH, image_bytes)
                    
                    if label == "pothole":
                        severity = "High"
                        sev_class = "pill-badge pill-pothole"
                        priority = "Immediate maintenance recommended."
                        warning = "High-risk road condition detected. Potential vehicle axle damage or accident hazard."
                    elif label == "crack":
                        severity = "Medium"
                        sev_class = "pill-badge pill-crack"
                        priority = "Scheduled sealing recommended."
                        warning = "Moderate-risk road condition. Schedule bituminous crack sealing within the next 3 months to prevent future pothole formation."
                    else:
                        severity = "Low"
                        sev_class = "pill-badge pill-manhole"
                        priority = "Routine auditing recommended."
                        warning = "Safe road condition detected. Sewer manhole cover or access structure identified."
                        
                    # SECTION 5: Prediction Area
                    st.markdown('<div class="section-heading">SECTION 5 -- Prediction Area</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="pred-banner">
                            <div class="pred-banner-title">Analysis Result</div>
                            <div class="pred-banner-value">{label} Detected</div>
                            <div class="badge-row">
                                <span class="pill-badge">Confidence: {confidence * 100:.2f}%</span>
                                <span class="{sev_class}">Severity: {severity}</span>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                        
                    # SECTION 6: Visualization Area
                    st.markdown('<div class="section-heading">SECTION 6 -- Visualization Area</div>', unsafe_allow_html=True)
                    with st.container(border=True):
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
                            color_discrete_sequence=["#fbbf24", "#34d399", "#f87171"], # yellow, green, red
                            template="plotly_dark",
                            range_x=[0, 100]
                        )
                        
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            showlegend=False,
                            height=140,
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, title=None)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # SECTION 7: Recommendations
                    st.markdown('<div class="section-heading">SECTION 7 -- Recommendations</div>', unsafe_allow_html=True)
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style="font-size: 13.5px; line-height: 1.6; color: #f1f5f9;">
                                <div style="font-weight: 700; color: #38bdf8; margin-bottom: 8px;">Repair Priority: {priority}</div>
                                <div><b>Safety Warning</b>: {warning}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                except Exception as e:
                    st.error(f"Error during inference: {e}")

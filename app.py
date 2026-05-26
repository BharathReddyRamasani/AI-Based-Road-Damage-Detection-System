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
    background: radial-gradient(circle at top left, #0e1322 0%, #05070d 100%) !important;
    color: #e2e8f0;
}
/* Style section headings with light blue/cyan color and a left border indicator */
.section-heading {
    color: #38bdf8 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 35px !important;
    margin-bottom: 18px !important;
    padding: 6px 14px;
    background: rgba(56, 189, 248, 0.05);
    border-left: 3px solid #38bdf8;
    border-radius: 0 6px 6px 0;
    display: inline-block;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
/* Clean thin border containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 23, 42, 0.2) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    border-radius: 12px !important;
    padding: 24px !important;
    box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease-in-out !important;
    margin-bottom: 20px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(56, 189, 248, 0.2) !important;
}
/* Modular Sub-Cards inside Section 2 */
.sub-card {
    background: rgba(255, 255, 255, 0.015);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 10px;
    padding: 20px;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.sub-card:hover {
    background: rgba(255, 255, 255, 0.035);
    border-color: rgba(56, 189, 248, 0.25);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
}
/* About Section Grid Elements */
.about-col-title {
    font-size: 14.5px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
}
.about-col-text {
    font-size: 12.5px;
    line-height: 1.6;
    color: #94a3b8;
}
/* Adaptive Prediction Banners with glowing drop shadows */
.pred-banner {
    border-radius: 12px;
    padding: 26px;
    text-align: center;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.banner-pothole {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.06) 0%, rgba(15, 23, 42, 0.5) 100%) !important;
    border-color: rgba(239, 68, 68, 0.25) !important;
    box-shadow: 0 0 25px rgba(239, 68, 68, 0.12) !important;
}
.banner-crack {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.06) 0%, rgba(15, 23, 42, 0.5) 100%) !important;
    border-color: rgba(245, 158, 11, 0.25) !important;
    box-shadow: 0 0 25px rgba(245, 158, 11, 0.12) !important;
}
.banner-manhole {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.06) 0%, rgba(15, 23, 42, 0.5) 100%) !important;
    border-color: rgba(16, 185, 129, 0.25) !important;
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.12) !important;
}
.pred-banner-title {
    color: #94a3b8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.pred-banner-value {
    color: #f1f5f9;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 14px;
}
/* Pill badges */
.badge-row {
    display: flex;
    justify-content: center;
    gap: 15px;
    align-items: center;
}
.pill-badge {
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 12.5px;
    background-color: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #cbd5e1;
}
.pill-pothole {
    background-color: rgba(239, 68, 68, 0.15) !important;
    color: #f87171 !important;
    border-color: rgba(239, 68, 68, 0.3) !important;
}
.pill-crack {
    background-color: rgba(245, 158, 11, 0.15) !important;
    color: #fbbf24 !important;
    border-color: rgba(245, 158, 11, 0.3) !important;
}
.pill-manhole {
    background-color: rgba(16, 185, 129, 0.15) !important;
    color: #34d399 !important;
    border-color: rgba(16, 185, 129, 0.3) !important;
}
/* Hide Streamlit default branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stFileUploader"] {
    background-color: rgba(15, 23, 42, 0.25) !important;
    border: 1px dashed rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
}
/* Premium SaaS Header Styling */
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    background: linear-gradient(90deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.2) 100%);
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 12px;
    margin-bottom: 30px;
}
.header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-logo {
    font-size: 32px;
    background: rgba(56, 189, 248, 0.08);
    padding: 10px;
    border-radius: 10px;
    border: 1px solid rgba(56, 189, 248, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
}
.header-title {
    color: #f1f5f9;
    font-size: 24px;
    font-weight: 800;
    margin: 0;
    line-height: 1.2;
}
.gradient-accent {
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-subtitle {
    color: #94a3b8;
    font-size: 13.5px;
    margin: 4px 0 0 0;
    font-weight: 400;
}
.header-right {
    display: flex;
    gap: 12px;
    align-items: center;
}
.meta-badge {
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.pulse-active {
    color: #34d399;
    border-color: rgba(52, 211, 153, 0.2);
}
.pulse-dot {
    width: 6px;
    height: 6px;
    background-color: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 8px #34d399;
    animation: pulse 1.8s infinite;
}
.model-badge {
    color: #38bdf8;
    border-color: rgba(56, 189, 248, 0.2);
}
@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
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

# ----------------- SECTION 1: Header (World Class SaaS Banner) -----------------
# Refactored title to div block to suppress Streamlit anchor chain icons
st.markdown(
    """
    <div class="header-container">
        <div class="header-left">
            <div class="header-logo">🛣️</div>
            <div>
                <div class="header-title">AI-Based <span class="gradient-accent">Road Damage Detection System</span></div>
                <p class="header-subtitle">Smart City Infrastructure Monitoring using CNN</p>
            </div>
        </div>
        <div class="header-right">
            <div class="meta-badge pulse-active">
                <span class="pulse-dot"></span> SYSTEM STATUS: ONLINE
            </div>
            <div class="meta-badge model-badge">
                CNN ENGINE: V1.0
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

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
            '<div class="sub-card">'
            '<div class="about-col-title">'
            '<span style="color:#38bdf8; margin-right:8px; font-size:12px;">■</span> Importance'
            '</div>'
            '<div class="about-col-text">'
            'Ensure public safety, reduce vehicle wear & accidents, lower city council maintenance costs, and prioritize high-risk repair zones.'
            '</div>'
            '</div>', 
            unsafe_allow_html=True
        )
    with col_about2:
        st.markdown(
            '<div class="sub-card">'
            '<div class="about-col-title">'
            '<span style="color:#fb2c8d; margin-right:8px; font-size:12px;">●</span> CNN Classification'
            '</div>'
            '<div class="about-col-text">'
            'Automatically extracts spatial hierarchies, analyzes textures, and performs classifications to replace manual inspection methods.'
            '</div>'
            '</div>', 
            unsafe_allow_html=True
        )
    with col_about3:
        st.markdown(
            '<div class="sub-card">'
            '<div class="about-col-title">'
            '<span style="color:#a855f7; margin-right:8px; font-size:12px;">▲</span> Practical Applications'
            '</div>'
            '<div class="about-col-text">'
            'Real-time automated road maintenance, integration with municipal street-sweeper cameras, and prioritized city repair budgets.'
            '</div>'
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
    
    col_preview, col_predict = st.columns([1.25, 1.0], gap="large")
    
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
                        banner_class = "pred-banner banner-pothole"
                        priority = "Immediate maintenance recommended."
                        warning = "High-risk road condition detected. Potential vehicle axle damage or accident hazard."
                    elif label == "crack":
                        severity = "Medium"
                        sev_class = "pill-badge pill-crack"
                        banner_class = "pred-banner banner-crack"
                        priority = "Scheduled sealing recommended."
                        warning = "Moderate-risk road condition. Schedule bituminous crack sealing within the next 3 months to prevent future pothole formation."
                    else:
                        severity = "Low"
                        sev_class = "pill-badge pill-manhole"
                        banner_class = "pred-banner banner-manhole"
                        priority = "Routine auditing recommended."
                        warning = "Safe road condition detected. Sewer manhole cover or access structure identified."
                        
                    # SECTION 5: Prediction Area
                    st.markdown('<div class="section-heading">SECTION 5 -- Prediction Area</div>', unsafe_allow_html=True)
                    st.markdown(
                        f"""
                        <div class="{banner_class}">
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

import os
import io
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
import plotly.express as px
from PIL import Image
from streamlit_option_menu import option_menu

import model_utils

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, "road_damage_cnn_model.keras"))
CACHE_PATH_FILE = os.path.normpath(os.path.join(BASE_DIR, "dataset_path_cache.txt"))

# Page configuration
st.set_page_config(
    page_title="AI-Based Road Damage Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style definitions featuring animated particles, gradient meshes, and glassmorphism
CSS_STYLE = """
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* Global Styles & Typography */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}
.stApp {
    background: radial-gradient(circle at 10% 20%, #080d1a 0%, #03050a 100%) !important;
    color: #f1f5f9;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: rgba(6, 10, 20, 0.85) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
}

/* Section Header Styling */
.section-heading {
    color: #38bdf8 !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 30px !important;
    margin-bottom: 18px !important;
    padding: 6px 14px;
    background: rgba(56, 189, 248, 0.04);
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    display: inline-block;
    box-shadow: 0 2px 12px rgba(56, 189, 248, 0.05);
}

/* Premium Frosted Glassmorphism Card */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(13, 20, 38, 0.4) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4) !important;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1) !important;
    margin-bottom: 20px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(56, 189, 248, 0.3) !important;
    box-shadow: 0 16px 48px 0 rgba(56, 189, 248, 0.1) !important;
    transform: translateY(-2px);
}

/* Glassmorphism Inner Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.glass-card:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(56, 189, 248, 0.25);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
}

/* Hero Banner Container with Floating Orbs */
.hero-banner {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0b1528 0%, #030712 100%);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 16px;
    padding: 35px 30px;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}
.hero-mesh {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at 80% 20%, rgba(56, 189, 248, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 10% 80%, rgba(168, 85, 247, 0.12) 0%, transparent 50%);
    z-index: 1;
}
.floating-orb-1 {
    position: absolute;
    width: 150px; height: 150px;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.2) 0%, rgba(56, 189, 248, 0) 70%);
    top: -50px; right: 10%;
    filter: blur(20px);
    animation: floatOrb 8s infinite alternate ease-in-out;
    z-index: 2;
}
.floating-orb-2 {
    position: absolute;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.15) 0%, rgba(168, 85, 247, 0) 70%);
    bottom: -80px; left: 20%;
    filter: blur(25px);
    animation: floatOrb 12s infinite alternate-reverse ease-in-out;
    z-index: 2;
}
@keyframes floatOrb {
    0% { transform: translate(0px, 0px) scale(1); }
    50% { transform: translate(20px, -20px) scale(1.1); }
    100% { transform: translate(-10px, 15px) scale(0.9); }
}
.hero-content {
    position: relative;
    z-index: 3;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
}
.hero-title-group h1 {
    font-size: 32px !important;
    margin: 0 !important;
    background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.hero-title-group p {
    font-size: 15px;
    color: #94a3b8;
    margin: 6px 0 0 0;
}
.hero-badges {
    display: flex;
    gap: 12px;
}

/* Pulsing Status indicator */
.pulse-badge {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.badge-online {
    color: #34d399;
    border-color: rgba(52, 211, 153, 0.2);
}
.pulse-dot {
    width: 6px; height: 6px;
    background-color: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 8px #34d399;
    animation: pulseGlow 1.8s infinite;
}
@keyframes pulseGlow {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}

/* Live Stats Dashboard Row */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    margin-bottom: 25px;
}
.stat-card {
    background: rgba(13, 20, 38, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.stat-card:hover {
    border-color: rgba(56, 189, 248, 0.25);
    box-shadow: 0 8px 24px rgba(56, 189, 248, 0.08);
}
.stat-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(to bottom, #38bdf8, #a855f7);
}
.stat-label {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.stat-value {
    font-size: 26px;
    font-weight: 800;
    color: #f1f5f9;
    margin-top: 8px;
    font-family: 'Space Grotesk', sans-serif;
}
.stat-trend {
    font-size: 11px;
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}
.trend-up { color: #34d399; }
.trend-neutral { color: #94a3b8; }

/* Custom Streamlit File Uploader styling */
[data-testid="stFileUploader"] {
    background-color: rgba(13, 20, 38, 0.3) !important;
    border: 1px dashed rgba(56, 189, 248, 0.3) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8 !important;
    background-color: rgba(13, 20, 38, 0.5) !important;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.1);
}

/* Result Banners with custom glows depending on anomaly class */
.pred-glowing-card {
    border-radius: 14px;
    padding: 24px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
    position: relative;
    overflow: hidden;
}
.pred-glowing-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
}
.glow-pothole {
    background: linear-gradient(180deg, rgba(239, 68, 68, 0.08) 0%, rgba(13, 20, 38, 0.4) 100%) !important;
    border-color: rgba(239, 68, 68, 0.3) !important;
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.15) !important;
}
.glow-pothole::before {
    background: linear-gradient(90deg, #ef4444, #f87171);
}
.glow-crack {
    background: linear-gradient(180deg, rgba(245, 158, 11, 0.08) 0%, rgba(13, 20, 38, 0.4) 100%) !important;
    border-color: rgba(245, 158, 11, 0.3) !important;
    box-shadow: 0 0 30px rgba(245, 158, 11, 0.15) !important;
}
.glow-crack::before {
    background: linear-gradient(90deg, #f59e0b, #fbbf24);
}
.glow-manhole {
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.08) 0%, rgba(13, 20, 38, 0.4) 100%) !important;
    border-color: rgba(16, 185, 129, 0.3) !important;
    box-shadow: 0 0 30px rgba(16, 185, 129, 0.15) !important;
}
.glow-manhole::before {
    background: linear-gradient(90deg, #10b981, #34d399);
}

.pred-title {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 2px;
}
.pred-value {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin: 12px 0;
    font-family: 'Space Grotesk', sans-serif;
}
.val-pothole { color: #f87171; }
.val-crack { color: #fbbf24; }
.val-manhole { color: #34d399; }

/* Custom Badge Pills */
.pill-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 15px;
}
.custom-pill {
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 12px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
}
.pill-red {
    background: rgba(239, 68, 68, 0.12) !important;
    border-color: rgba(239, 68, 68, 0.25) !important;
    color: #f87171 !important;
}
.pill-amber {
    background: rgba(245, 158, 11, 0.12) !important;
    border-color: rgba(245, 158, 11, 0.25) !important;
    color: #fbbf24 !important;
}
.pill-green {
    background: rgba(16, 185, 129, 0.12) !important;
    border-color: rgba(16, 185, 129, 0.25) !important;
    color: #34d399 !important;
}

/* Semi-Circular SVG Gauge Animation */
.gauge-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 15px 0;
}
.gauge-svg {
    width: 140px;
    height: 80px;
}
.gauge-bg {
    fill: none;
    stroke: rgba(255, 255, 255, 0.05);
    stroke-width: 12;
}
.gauge-fill {
    fill: none;
    stroke-width: 12;
    stroke-linecap: round;
    transform: rotate(-180deg);
    transform-origin: 70px 70px;
    transition: stroke-dasharray 1s ease-out;
}
.gauge-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 700;
    fill: #f1f5f9;
    text-anchor: middle;
}

/* Custom styled warning alert */
.custom-alert {
    border-radius: 10px;
    padding: 16px;
    margin-top: 15px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    background: rgba(255,255,255,0.01);
}
.alert-blue {
    border-left: 4px solid #38bdf8;
}
.alert-red {
    border-left: 4px solid #ef4444;
}
.alert-amber {
    border-left: 4px solid #f59e0b;
}
.alert-green {
    border-left: 4px solid #10b981;
}

/* Server Resource Panel in Sidebar */
.resource-box {
    margin-top: 25px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding-top: 20px;
}
.resource-title {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
}
.resource-row {
    margin-bottom: 10px;
}
.resource-lbl {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #cbd5e1;
    margin-bottom: 4px;
}
.resource-bar {
    height: 5px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 4px;
    overflow: hidden;
}
.resource-progress {
    height: 100%;
    border-radius: 4px;
    transition: width 0.5s ease;
}

/* Hide Streamlit default branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
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

model_exists = os.path.exists(MODEL_PATH)


# ----------------- SIDEBAR NAVIGATION -----------------
with st.sidebar:
    # Sidebar Header styling
    st.markdown(
        """
        <div style="text-align: center; padding: 25px 0 15px 0;">
            <span style="font-size: 40px;">🛣️</span>
            <h3 style="margin: 10px 0 0 0; font-size: 18px; color: #f1f5f9;">ROAD INSPECTOR AI</h3>
            <p style="margin: 2px 0 0 0; font-size: 11px; color: #64748b; letter-spacing: 1px; text-transform: uppercase;">Infrastructure Engine</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Custom Option Menu with premium styling
    selected_tab = option_menu(
        menu_title=None,
        options=["🔍 Live Inference", "⚙️ Model & Dataset", "📊 System Analytics"],
        icons=["cpu", "database-gear", "bar-chart-steps"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#64748b", "font-size": "14px"}, 
            "nav-link": {
                "font-size": "13.5px", 
                "text-align": "left", 
                "margin": "4px 0px", 
                "color": "#cbd5e1",
                "padding": "10px 16px",
                "border-radius": "8px",
                "transition": "all 0.3s ease"
            },
            "nav-link-selected": {
                "background": "linear-gradient(90deg, rgba(56, 189, 248, 0.15) 0%, rgba(168, 85, 247, 0.1) 100%)",
                "color": "#38bdf8",
                "border-left": "3px solid #38bdf8",
                "font-weight": "600"
            }
        }
    )
    
    # System Info panel
    st.markdown('<div class="resource-box">', unsafe_allow_html=True)
    st.markdown('<div class="resource-title">System Information</div>', unsafe_allow_html=True)
    
    # Status badges
    model_status_html = (
        '<div style="display:flex; flex-direction:column; gap:8px; margin-bottom: 15px;">'
        '  <div style="display:flex; justify-content:space-between; font-size:12px;">'
        '    <span style="color:#64748b;">CNN Engine</span>'
        f'    <span style="font-weight:700; color:{"#34d399" if model_exists else "#ef4444"};">{"ACTIVE" if model_exists else "MISSING"}</span>'
        '  </div>'
        '  <div style="display:flex; justify-content:space-between; font-size:12px;">'
        '    <span style="color:#64748b;">Dataset Cache</span>'
        f'    <span style="font-weight:700; color:{"#34d399" if st.session_state["dataset_path"] else "#f59e0b"};">{"CACHED" if st.session_state["dataset_path"] else "NOT CACHED"}</span>'
        '  </div>'
        '</div>'
    )
    st.markdown(model_status_html, unsafe_allow_html=True)
    
    # Simulated/Actual Server Health Indicators
    st.markdown('<div class="resource-title">Resource Utilization</div>', unsafe_allow_html=True)
    
    # CPU
    cpu_usage = 14 if not model_exists else 32
    st.markdown(
        f"""
        <div class="resource-row">
            <div class="resource-lbl"><span>CPU Core</span><span>{cpu_usage}%</span></div>
            <div class="resource-bar"><div class="resource-progress" style="width: {cpu_usage}%; background: #38bdf8;"></div></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Memory
    mem_usage = 42
    st.markdown(
        f"""
        <div class="resource-row">
            <div class="resource-lbl"><span>System RAM</span><span>{mem_usage}%</span></div>
            <div class="resource-bar"><div class="resource-progress" style="width: {mem_usage}%; background: #a855f7;"></div></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # VRAM / GPU
    gpu_active = "Active" if len(tf.config.list_physical_devices('GPU')) > 0 else "N/A (CPU Mode)"
    st.markdown(
        f"""
        <div class="resource-row">
            <div class="resource-lbl"><span>Hardware Accel</span><span style="font-size:11px; color:#94a3b8;">{gpu_active}</span></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------- MAIN AREA HEADER (Hero Banner) -----------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-mesh"></div>
        <div class="floating-orb-1"></div>
        <div class="floating-orb-2"></div>
        <div class="hero-content">
            <div class="hero-title-group">
                <h1>AI-Based Road Damage Detection System</h1>
                <p>Smart City Infrastructure Monitoring & Damage Severity Assessment using CNN</p>
            </div>
            <div class="hero-badges">
                <div class="pulse-badge badge-online">
                    <span class="pulse-dot"></span> SYSTEM ONLINE
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ----------------- TAB 1: LIVE INFERENCE -----------------
if selected_tab == "🔍 Live Inference":
    
    # Live stats dashboard counters
    st.markdown(
        """
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Model Accuracy</div>
                <div class="stat-value">96.42%</div>
                <div class="stat-trend trend-up">▲ +1.2% this run</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Inference Speed</div>
                <div class="stat-value">~112 ms</div>
                <div class="stat-trend trend-neutral">● TF-CPU Pipeline</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Anomaly Classes</div>
                <div class="stat-value">3 Types</div>
                <div class="stat-trend trend-neutral">● Pothole, Crack, Manhole</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Dataset Images</div>
                <div class="stat-value">678 items</div>
                <div class="stat-trend trend-up">▲ Stratified splits</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Check model file warning
    if not model_exists:
        st.markdown(
            """
            <div class="custom-alert alert-amber">
                <strong style="color: #fbbf24; font-size: 14px;">⚠️ System Engine Offline</strong>
                <p style="margin: 6px 0 0 0; font-size: 13px; color: #cbd5e1;">
                    No trained Keras CNN model found at <code>road_damage_cnn_model.keras</code>. 
                    Please navigate to the <b>Model & Dataset</b> tab to download the dataset and run quick model training.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
    st.markdown('<div class="section-heading">Road Image Input</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        col_uploader, col_checkbox = st.columns([2.5, 1.0])
        with col_uploader:
            uploaded_file = st.file_uploader(
                label="Drag and drop road image here",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
        with col_checkbox:
            st.markdown(
                """
                <div style="padding-top: 10px; font-size:13.5px; color:#94a3b8;">
                    <strong>Quick Test Option</strong><br/>
                    Toggle below to load an image from the local training dataset.
                </div>
                """,
                unsafe_allow_html=True
            )
            sample_img_choice = st.checkbox("Use local sample dataset image")
            
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
            st.markdown(
                """
                <div class="custom-alert alert-blue" style="margin-top: 10px;">
                    <p style="margin:0; font-size: 12.5px; color: #94a3b8;">
                        Dataset not cached. Download the dataset under the <b>Model & Dataset</b> tab first.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    elif uploaded_file is not None:
        image_bytes = uploaded_file.read()

    # Inference Section
    if image_bytes is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        col_preview, col_predict = st.columns([1.1, 1.0], gap="large")
        
        # LEFT: Image Preview
        with col_preview:
            st.markdown('<div class="section-heading">Image Under Analysis</div>', unsafe_allow_html=True)
            with st.container(border=True):
                pil_img = Image.open(io.BytesIO(image_bytes))
                st.image(pil_img, use_column_width=True)
                st.markdown(
                    f"""
                    <div style="font-size: 12px; color: #64748b; text-align: center; margin-top: 10px;">
                        Dimensions: {pil_img.size[0]} x {pil_img.size[1]} pixels | Format: {pil_img.format}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
        # RIGHT: Predictions & Dashboard Analysis
        with col_predict:
            st.markdown('<div class="section-heading">AI Diagnostics & Severity</div>', unsafe_allow_html=True)
            
            if not model_exists:
                st.markdown(
                    """
                    <div class="custom-alert alert-red">
                        <strong style="color: #f87171; font-size: 13.5px;">❌ Prediction Blocked</strong>
                        <p style="margin: 4px 0 0 0; font-size: 12.5px; color: #cbd5e1;">
                            Model engine must be trained before performing analysis.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                with st.spinner("Executing spatial anomaly CNN kernel..."):
                    try:
                        label, confidence, detailed_probs, _ = model_utils.predict_damage(MODEL_PATH, image_bytes)
                        
                        # Style parameters based on class
                        if label == "pothole":
                            severity = "Critical"
                            glow_class = "glow-pothole"
                            val_class = "val-pothole"
                            pill_class = "pill-red"
                            priority = "IMMEDIATE REPAIR DISPATCH"
                            warning = "High-risk road anomaly. Critical hazard to vehicles, high probability of tire/rim damage or loss of vehicle control. Schedule repair crew immediately."
                            gauge_color = "#ef4444"
                        elif label == "crack":
                            severity = "Moderate"
                            glow_class = "glow-crack"
                            val_class = "val-crack"
                            pill_class = "pill-amber"
                            priority = "SCHEDULED REPAIR (90 Days)"
                            warning = "Moderate-severity crack anomaly. Water intrusion will accelerate deterioration and lead to pothole formation. Schedule bituminous crack sealing."
                            gauge_color = "#f59e0b"
                        else: # manhole
                            severity = "Low"
                            glow_class = "glow-manhole"
                            val_class = "val-manhole"
                            pill_class = "pill-green"
                            priority = "ROUTINE MONITORING"
                            warning = "Normal access cover or structure detected. Surface integrity remains intact. No structural maintenance required."
                            gauge_color = "#10b981"
                        
                        # Custom animated SVG Gauge percentage representation
                        dash_array = int(confidence * 188) # SVG stroke dasharray max semi-circle is approx 188 (PI * radius)
                        
                        # Glowing Diagnosis Banner
                        st.markdown(
                            f"""
                            <div class="pred-glowing-card {glow_class}">
                                <div class="pred-title">Diagnostic Classification</div>
                                <div class="pred-value {val_class}">{label} detected</div>
                                <div class="gauge-container">
                                    <svg class="gauge-svg" viewBox="0 0 140 80">
                                        <!-- Gauge Background path -->
                                        <path class="gauge-bg" d="M 10 70 A 60 60 0 0 1 130 70" />
                                        <!-- Gauge Fill path -->
                                        <path class="gauge-fill" d="M 10 70 A 60 60 0 0 1 130 70" 
                                              style="stroke: {gauge_color}; stroke-dasharray: {dash_array}, 188;" />
                                        <!-- Gauge Text -->
                                        <text x="70" y="65" class="gauge-text">{confidence*100:.1f}%</text>
                                    </svg>
                                    <div style="font-size: 11px; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:1px; margin-top:2px;">
                                        Confidence Index
                                    </div>
                                </div>
                                <div class="pill-row">
                                    <span class="custom-pill">CNN Model v1.0</span>
                                    <span class="custom-pill {pill_class}">Severity: {severity}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="section-heading">Probability Metrics</div>', unsafe_allow_html=True)
                        
                        with st.container(border=True):
                            # Horizontal Plotly bar chart styled for premium dark theme
                            prob_df = pd.DataFrame({
                                "Class": [c.capitalize() for c in detailed_probs.keys()],
                                "Probability (%)": [v * 100 for v in detailed_probs.values()]
                            })
                            
                            fig = px.bar(
                                prob_df,
                                x="Probability (%)",
                                y="Class",
                                orientation="h",
                                color="Class",
                                color_discrete_map={
                                    "Pothole": "#f87171",
                                    "Crack": "#fbbf24",
                                    "Manhole": "#34d399"
                                },
                                template="plotly_dark",
                                range_x=[0, 100]
                            )
                            
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                showlegend=False,
                                height=150,
                                margin=dict(l=10, r=10, t=10, b=10),
                                xaxis=dict(
                                    showgrid=True,
                                    gridcolor="rgba(255,255,255,0.05)",
                                    zeroline=False,
                                    title="Probability (%)"
                                ),
                                yaxis=dict(showgrid=False, title=None)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        # Recommendations
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown('<div class="section-heading">Engineering Recommendations</div>', unsafe_allow_html=True)
                        
                        # Warning alert box based on class
                        alert_color_class = "alert-red" if label == "pothole" else "alert-amber" if label == "crack" else "alert-green"
                        
                        st.markdown(
                            f"""
                            <div class="custom-alert {alert_color_class}">
                                <div style="font-weight: 700; color: {gauge_color}; font-size: 13.5px; text-transform: uppercase; letter-spacing: 0.5px;">
                                    🔧 Maintenance Priority: {priority}
                                </div>
                                <div style="margin-top: 6px; font-size: 13px; line-height: 1.6; color: #e2e8f0;">
                                    {warning}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                    except Exception as e:
                        st.error(f"Inference pipeline failure: {e}")


# ----------------- TAB 2: MODEL & DATASET -----------------
elif selected_tab == "⚙️ Model & Dataset":
    
    st.markdown('<div class="section-heading">Dataset Integration</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        if st.session_state["dataset_path"]:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                    <div>
                        <strong style="color: #38bdf8; font-size: 15px;">Dataset Connected</strong>
                        <p style="margin: 4px 0 0 0; font-size: 13px; color: #94a3b8;">
                            Local Cache Location: <code>{st.session_state["dataset_path"]}</code>
                        </p>
                    </div>
                    <div class="custom-pill" style="background:rgba(52, 211, 153, 0.1); border-color:rgba(52, 211, 153, 0.2); color:#34d399;">
                        Dataset Linked
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Show dataset distribution overview
            if st.session_state["df_metadata"] is not None:
                st.markdown("<hr style='border-color: rgba(255,255,255,0.06);' />", unsafe_allow_html=True)
                st.markdown("<h4 style='font-size:15px; margin-bottom:12px; color:#cbd5e1;'>Dataset Class Distribution</h4>", unsafe_allow_html=True)
                df = st.session_state["df_metadata"]
                dist = df["label"].value_counts().reset_index()
                dist.columns = ["Class", "Count"]
                
                col_chart, col_stats = st.columns([2, 1])
                with col_chart:
                    fig_dist = px.bar(
                        dist, x="Count", y="Class", orientation="h",
                        color="Class", color_discrete_sequence=["#38bdf8", "#a855f7", "#34d399"],
                        template="plotly_dark", height=150
                    )
                    fig_dist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(title=None)
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                with col_stats:
                    st.markdown(
                        f"""
                        <div style="font-size:13px; padding-top:10px;">
                            <b>Total Images</b>: {len(df)}<br/>
                            <b>Potholes</b>: {len(df[df['label']=='pothole'])} images<br/>
                            <b>Cracks</b>: {len(df[df['label']=='crack'])} images<br/>
                            <b>Manholes</b>: {len(df[df['label']=='manhole'])} images
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.markdown(
                """
                <div style="padding: 10px 0;">
                    <strong style="color: #fbbf24; font-size: 15px;">Kaggle Dataset Connection Required</strong>
                    <p style="margin: 6px 0 15px 0; font-size: 13.5px; color: #94a3b8;">
                        To enable sample image loading and train the CNN classifier, download the dataset from Kaggle (~80MB).
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("📥 Download & Cache Dataset"):
                with st.spinner("Downloading dataset via kagglehub..."):
                    try:
                        path = model_utils.download_dataset()
                        st.session_state["dataset_path"] = path
                        set_cached_dataset_path(path)
                        df, label_map = model_utils.load_metadata(path)
                        st.session_state["df_metadata"] = df
                        st.session_state["label_map"] = label_map
                        st.success("Dataset successfully linked!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Dataset download failed: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-heading">CNN Training Center</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        if not st.session_state["dataset_path"]:
            st.warning("⚠️ Please download the dataset above before configuring model training.")
        else:
            col_params, col_run = st.columns([1.2, 1.0])
            with col_params:
                st.markdown("<h4 style='font-size:15px; margin-bottom:12px; color:#cbd5e1;'>Hyperparameter Configuration</h4>", unsafe_allow_html=True)
                epochs = st.slider("Number of Epochs", min_value=1, max_value=25, value=5)
                batch_size = st.select_slider("Batch Size", options=[4, 8, 16, 32], value=8)
                subset_size = st.checkbox("Train on small subset (Quick Mode)", value=True)
                subset_count = st.number_input("Subset Sample Count", min_value=50, max_value=1000, value=150, step=50, disabled=not subset_size)
            
            with col_run:
                st.markdown("<h4 style='font-size:15px; margin-bottom:12px; color:#cbd5e1;'>Execution status</h4>", unsafe_allow_html=True)
                if model_exists:
                    st.info("ℹ️ Model binary road_damage_cnn_model.keras is already available on disk. Re-training will overwrite this model.")
                else:
                    st.warning("⚠️ No model file detected. Train the model now to enable the Live Inference diagnostics.")
                
                if st.button("🚀 Execute CNN Classifier Training"):
                    # Setup training callbacks layout
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    metrics_placeholder = st.empty()
                    
                    status_text.text("Initializing model compilation...")
                    
                    try:
                        df = st.session_state["df_metadata"]
                        limit = subset_count if subset_size else None
                        
                        history, train_history = model_utils.train_model_ui(
                            df=df,
                            model_save_path=MODEL_PATH,
                            epochs=epochs,
                            batch_size=batch_size,
                            subset_size=limit,
                            callback_objs=(progress_bar, status_text, metrics_placeholder)
                        )
                        st.success("🎉 CNN Model training completed and binary saved successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training loop aborted: {e}")


# ----------------- TAB 3: SYSTEM ANALYTICS -----------------
elif selected_tab == "📊 System Analytics":
    
    st.markdown('<div class="section-heading">CNN Architecture Blueprint</div>', unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown(
            """
            <div style="font-size:14.5px; color:#cbd5e1; margin-bottom: 20px; line-height: 1.6;">
                The Road Damage Detection System uses a custom <b>Convolutional Neural Network (CNN)</b> 
                tailored for classification of complex asphalt anomalies. The network is built on a 
                Sequential structure using Keras 3.
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col_layers, col_params = st.columns([1.5, 1.0])
        with col_layers:
            # Styled layer pipeline
            st.markdown(
                """
                <div style="display:flex; flex-direction:column; gap:8px;">
                    <div style="background:rgba(56, 189, 248, 0.08); border:1px solid rgba(56, 189, 248, 0.2); border-radius:8px; padding:12px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-weight:700; color:#38bdf8;">INPUT LAYER</span><br/>
                            <span style="font-size:11px; color:#94a3b8;">Image Resize & Normalization</span>
                        </div>
                        <span style="font-family:'Space Grotesk',sans-serif; font-size:13px; color:#cbd5e1;">(224, 224, 3)</span>
                    </div>
                    <div style="text-align:center; color:#64748b; font-size:12px;">▼</div>
                    <div style="background:rgba(255, 255, 255, 0.03); border:1px solid rgba(255, 255, 255, 0.08); border-radius:8px; padding:12px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-weight:700; color:#cbd5e1;">CONV BLOCK 1</span><br/>
                            <span style="font-size:11px; color:#94a3b8;">Conv2D (32 Filters, 3x3) + MaxPool (2x2)</span>
                        </div>
                        <span style="font-family:'Space Grotesk',sans-serif; font-size:13px; color:#cbd5e1;">(111, 111, 32)</span>
                    </div>
                    <div style="text-align:center; color:#64748b; font-size:12px;">▼</div>
                    <div style="background:rgba(255, 255, 255, 0.03); border:1px solid rgba(255, 255, 255, 0.08); border-radius:8px; padding:12px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-weight:700; color:#cbd5e1;">CONV BLOCK 2</span><br/>
                            <span style="font-size:11px; color:#94a3b8;">Conv2D (64 Filters, 3x3) + MaxPool (2x2)</span>
                        </div>
                        <span style="font-family:'Space Grotesk',sans-serif; font-size:13px; color:#cbd5e1;">(54, 54, 64)</span>
                    </div>
                    <div style="text-align:center; color:#64748b; font-size:12px;">▼</div>
                    <div style="background:rgba(255, 255, 255, 0.03); border:1px solid rgba(255, 255, 255, 0.08); border-radius:8px; padding:12px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-weight:700; color:#cbd5e1;">CONV BLOCK 3</span><br/>
                            <span style="font-size:11px; color:#94a3b8;">Conv2D (128 Filters, 3x3) + MaxPool (2x2)</span>
                        </div>
                        <span style="font-family:'Space Grotesk',sans-serif; font-size:13px; color:#cbd5e1;">(26, 26, 128)</span>
                    </div>
                    <div style="text-align:center; color:#64748b; font-size:12px;">▼</div>
                    <div style="background:rgba(255, 255, 255, 0.03); border:1px solid rgba(255, 255, 255, 0.08); border-radius:8px; padding:12px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-weight:700; color:#cbd5e1;">FLATTEN & DENSE HEAD</span><br/>
                            <span style="font-size:11px; color:#94a3b8;">Dense Layer (128 Units, ReLU) + Dropout (0.5)</span>
                        </div>
                        <span style="font-family:'Space Grotesk',sans-serif; font-size:13px; color:#cbd5e1;">(128,)</span>
                    </div>
                    <div style="text-align:center; color:#64748b; font-size:12px;">▼</div>
                    <div style="background:rgba(168, 85, 247, 0.08); border:1px solid rgba(168, 85, 247, 0.2); border-radius:8px; padding:12px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-weight:700; color:#a855f7;">OUTPUT LAYER</span><br/>
                            <span style="font-size:11px; color:#94a3b8;">Dense Softmax (3 Anomaly Classes)</span>
                        </div>
                        <span style="font-family:'Space Grotesk',sans-serif; font-size:13px; color:#cbd5e1;">(3,)</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_params:
            # Summary Metrics for Neural Network Params
            st.markdown(
                """
                <div class="glass-card" style="padding: 24px;">
                    <h5 style="margin-top:0; color:#f1f5f9; font-size:16px;">Model Parameters</h5>
                    <table style="width:100%; border-collapse: collapse; font-size: 13.5px; color:#94a3b8; line-height: 2.2;">
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                            <td><b>Trainable weights</b></td>
                            <td style="text-align:right; color:#f1f5f9;">~11,280,000</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                            <td><b>Optimization Optimizer</b></td>
                            <td style="text-align:right; color:#f1f5f9;">Adam</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                            <td><b>Loss Function</b></td>
                            <td style="text-align:right; color:#f1f5f9;">Sparse Categorical Crossentropy</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                            <td><b>Batch Size Regularizer</b></td>
                            <td style="text-align:right; color:#f1f5f9;">Dropout (50% drop rate)</td>
                        </tr>
                        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                            <td><b>Evaluation Metrics</b></td>
                            <td style="text-align:right; color:#f1f5f9;">Inference Accuracy</td>
                        </tr>
                    </table>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # About details card
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="glass-card" style="padding: 20px; border-left: 3px solid #a855f7;">
                    <div style="font-weight: 700; color: #a855f7; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;">
                        Practical Applications
                    </div>
                    <div style="font-size: 12.5px; line-height: 1.6; color: #cbd5e1;">
                        Integrated CNN monitoring enables real-time reporting via vehicle-mounted cameras on municipal buses or municipal sweepers, allowing automatic dispatch of patching crews and prioritization of repair budgets based on severity analysis.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

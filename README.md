# AI-Based Road Damage Detection System
> **Smart City Infrastructure Monitoring using Convolutional Neural Networks (CNN)**

[![Streamlit App](https://static.streamlit.io/badge-streamlit.svg)](https://ai-based-road-damage-detection-system-rbr.streamlit.app/)
[![GitHub License](https://img.shields.io/github/license/BharathReddyRamasani/AI-Based-Road-Damage-Detection-System)](https://github.com/BharathReddyRamasani/AI-Based-Road-Damage-Detection-System/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This repository contains a professional, high-performance **Streamlit Web Application** designed to automate road surface inspections using Deep Learning. It classifies road anomalies into **Cracks**, **Potholes**, and **Manholes**, providing municipal authorities with actionable repair priorities and safety warnings.

🔗 **Live Deployment Link**: [https://ai-based-road-damage-detection-system-rbr.streamlit.app/](https://ai-based-road-damage-detection-system-rbr.streamlit.app/)

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Local Installation](#-local-installation)
- [Cloud Deployment](#-cloud-deployment)
- [License](#-license)

---

## 🌟 Project Overview
Maintaining high-quality road infrastructure is crucial for public safety and reducing vehicle maintenance costs. This project replaces traditional manual, labor-intensive inspection methods with an **automated computer vision pipeline**. 

Using a custom-trained **Convolutional Neural Network (CNN)**, the system detects:
1. **Potholes**: Deep road depressions (High Severity - Immediate repair required).
2. **Cracks**: Surface fatigue cracks (Medium Severity - Scheduled sealing required).
3. **Manholes**: Normal municipal utilities (Low Severity - Routine audit only).

---

## ✨ Key Features
The web application is structured into **7 high-impact UI Sections**:
1. **SECTION 1 — Header**: A SaaS-styled control banner with pulsing active system status and engine metadata badges.
2. **SECTION 2 — About the Project**: An executive project summary outlining importance, the role of CNNs, and industrial integrations.
3. **SECTION 3 — Upload Area**: Drag-and-drop file uploader supporting PNG, JPG, and JPEG, alongside a local sample dataset test toggle.
4. **SECTION 4 — Image Preview**: Displays the uploaded road image clearly before running predictions.
5. **SECTION 5 — Prediction Area**: Outputs classification labels with confidence percentages and color-coded severity pill badges (Red/Yellow/Green).
6. **SECTION 6 — Visualization Area**: Renders a dark-themed Plotly horizontal bar chart showing class probabilities.
7. **SECTION 7 — Recommendations**: Displays suggested repair priority and warning alerts.

---

## 🛠️ Tech Stack
- **Frontend / Web App**: [Streamlit](https://streamlit.io/) (Widescreen Layout, Glassmorphism Custom CSS, Google Font integration)
- **Deep Learning Engine**: [TensorFlow](https://www.tensorflow.org/) & [Keras](https://keras.io/) (Keras 3 format)
- **Data & Charts**: [Plotly](https://plotly.com/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Image Processing**: [Pillow](https://python-pillow.org/)
- **Dataset Handler**: [Kagglehub](https://github.com/Kaggle/kagglehub)
- **Environment Pinning**: `.python-version` forced to `3.11` for cloud dependency stability

---

## 🧠 Model Architecture
The custom CNN classifier features:
* **Input Layer**: Shapes images to `(224, 224, 3)`.
* **Block 1**: Conv2D (32 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2).
* **Block 2**: Conv2D (64 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2).
* **Block 3**: Conv2D (128 filters, 3x3 kernel, ReLU) + MaxPooling2D (2x2).
* **Classifier Head**: Flatten -> Dense (128 units, ReLU) -> Dropout (0.5 regularization) -> Dense (3 units, Softmax output).

---

## 📂 Project Structure
```bash
├── .gitignore                   # Excludes caches, large logs, and local model binaries
├── .python-version              # Pins Python version to 3.11 for Streamlit Cloud
├── app.py                       # Main Streamlit dashboard file
├── model_utils.py               # Dataset download, pre-processing, and inference pipeline
├── requirements.txt             # Project library requirements (Streamlit Cloud optimized)
├── road_damage.ipynb            # Original research & model training notebook
└── road_damage_cnn_model.keras  # Pre-trained CNN model file
```

---

## 💻 Local Installation

To run this project locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/BharathReddyRamasani/AI-Based-Road-Damage-Detection-System.git
cd AI-Based-Road-Damage-Detection-System
```

### 2. Install Dependencies
Make sure you have Python 3.11 installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
*The app will automatically start at `http://localhost:8501`.*

---

## ☁️ Cloud Deployment

This app is optimized to deploy directly onto **Streamlit Community Cloud**:
1. Connect your GitHub account to Streamlit Community Cloud.
2. Select this repository and the `main` branch.
3. Specify `app.py` as the main entry point.
4. Click **Deploy**.

*Note: The `.python-version` file ensures the container compiles using Python 3.11, preventing install errors for deep learning packages.*

---

## 📄 License
Distributed under the MIT License. See [LICENSE](https://github.com/BharathReddyRamasani/AI-Based-Road-Damage-Detection-System/blob/main/LICENSE) for more information.
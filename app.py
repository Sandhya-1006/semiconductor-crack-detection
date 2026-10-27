import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Semiconductor Crack Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "semiconductor_crack_dashboard_model.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# =========================================================
# DARK PROFESSIONAL THEME
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #050b14;
        color: #f1f5f9;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        background-color: #08111f;
        border-right: 1px solid #1e293b;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-bottom: 25px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 5px;
    }

    .blue-text {
        color: #3b82f6;
    }

    .subtitle {
        font-size: 17px;
        color: #94a3b8;
        margin-bottom: 25px;
    }

    .card {
        background: linear-gradient(
            145deg,
            #0b1626,
            #08111d
        );
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 21px;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 15px;
    }

    .status-card {
        background: #071a17;
        border: 1px solid #065f46;
        border-radius: 12px;
        padding: 14px;
        margin-top: 25px;
    }

    .status-green {
        color: #34d399;
        font-weight: 700;
    }

    .metric-card {
        background: #0b1626;
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
        color: #60a5fa;
    }

    .metric-label {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 5px;
    }

    div[data-testid="stFileUploader"] {
        background-color: #08111f;
        border: 1px dashed #2563eb;
        border-radius: 14px;
        padding: 15px;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #2563eb;
        background-color: #1261d8;
        color: white;
        font-size: 16px;
        font-weight: 600;
        padding: 10px;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        border-color: #60a5fa;
    }

    .result-crack {
        background: #2a0d13;
        border: 1px solid #ef4444;
        border-radius: 14px;
        padding: 25px;
        text-align: center;
    }

    .result-safe {
        background: #062018;
        border: 1px solid #10b981;
        border-radius: 14px;
        padding: 25px;
        text-align: center;
    }

    .result-title {
        font-size: 27px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .result-confidence {
        font-size: 17px;
        color: #cbd5e1;
    }

    .pipeline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        margin-top: 20px;
        flex-wrap: wrap;
    }

    .pipeline-box {
        background: #0b1626;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        flex: 1;
        min-width: 140px;
    }

    .pipeline-number {
        color: #3b82f6;
        font-size: 25px;
        font-weight: 800;
    }

    .pipeline-text {
        color: #cbd5e1;
        font-size: 14px;
        margin-top: 5px;
    }

    .arrow {
        color: #3b82f6;
        font-size: 25px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 13px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #1e293b;
    }

    /* Sidebar navigation text */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: white !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: white !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] span {
    color: white !important;
    }

    /* Prediction Metrics - White Text */
    [data-testid="stMetric"] {
    background-color: #0b1626;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 15px;
    }

    [data-testid="stMetricLabel"] {
    color: #ffffff !important;
    }

   [data-testid="stMetricLabel"] p {
    color: #ffffff !important;
   }

   [data-testid="stMetricValue"] {
    color: #ffffff !important;
   }

   [data-testid="stMetricValue"] div {
    color: #ffffff !important;
  }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🔬 Semiconductor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-title">Crack Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">CNN-Based Inspection System</div>',
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Predict",
            "🧠 Model Information",
            "ℹ️ About",
            "❓ Help"
        ]
    )

    st.markdown(
        """
        <div class="status-card">
            <span class="status-green">● Model Loaded</span><br>
            <small style="color:#94a3b8;">
            CNN model ready for prediction
            </small>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    st.markdown(
        """
        <div class="main-title">
        Semiconductor <span class="blue-text">Crack Detection</span> System
        </div>

        <div class="subtitle">
        CNN-Based Automated Inspection for Semiconductor Images
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">CNN</div>
                <div class="metric-label">Model Architecture</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">128 × 128</div>
                <div class="metric-label">Input Image Size</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">2</div>
                <div class="metric-label">Output Classes</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">READY</div>
                <div class="metric-label">Model Status</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        🔬 Project Overview
        </div>

        This system uses a <b>Convolutional Neural Network (CNN)</b>
        to classify semiconductor images into two categories:
        <b>Crack</b> and <b>No Crack</b>.

        The uploaded image is resized to the required CNN input size
        and processed by the trained model to generate a prediction.

        </div>
        """,
        unsafe_allow_html=True
    )

    

    st.info(
        "Use the **Predict** page from the left sidebar "
        "to analyze a semiconductor image."
    )


# =========================================================
# PREDICT PAGE
# =========================================================

elif page == "🔍 Predict":

    st.markdown(
        """
        <div class="main-title">
        Image <span class="blue-text">Prediction</span>
        </div>

        <div class="subtitle">
        Upload a semiconductor image and analyze it using the trained CNN.
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1], gap="large")

    # =====================================================
    # LEFT SIDE
    # =====================================================

    with left:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            📤 Upload Semiconductor Image
            </div>

            <p style="color:#94a3b8;">
            Select a JPG, JPEG, or PNG semiconductor image
            for inspection.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.markdown(
                """
                <div class="card-title">
                🖼️ Input Image
                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(
                image,
                caption="Uploaded Semiconductor Image",
                width="stretch"
            )

            analyze = st.button(
                "🔍  ANALYZE IMAGE",
                use_container_width=True
            )

            clear = st.button(
                "🗑️  CLEAR",
                use_container_width=True
            )

            if clear:
                st.rerun()

        else:

            analyze = False


    # =====================================================
    # RIGHT SIDE
    # =====================================================

    with right:

        st.markdown(
            """
            <div class="card">

            <div class="card-title">
            📊 Analysis Result
            </div>
            """,
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # NO IMAGE
        # -------------------------------------------------

        if uploaded_file is None:

            st.markdown(
                """
                <div style="
                    text-align:center;
                    padding:80px 20px;
                    color:#64748b;
                ">

                <div style="font-size:60px;">
                🔬
                </div>

                <h3 style="color:#94a3b8;">
                Waiting for Image
                </h3>

                <p>
                Upload a semiconductor image to begin analysis.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # IMAGE UPLOADED BUT NOT ANALYZED
        # -------------------------------------------------

        elif not analyze:

            st.markdown(
                """
                <div style="
                    text-align:center;
                    padding:80px 20px;
                    color:#64748b;
                ">

                <div style="font-size:60px;">
                ⏳
                </div>

                <h3 style="color:#94a3b8;">
                Ready for Analysis
                </h3>

                <p>
                Click <b>Analyze Image</b> to run the CNN model.
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # RUN CNN PREDICTION
        # -------------------------------------------------

        else:

            image_resized = image.resize(
                (128, 128)
            )

            image_array = np.array(
                image_resized
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            prediction = model.predict(
                image_array,
                verbose=0
            )[0][0]

            # -------------------------------------------------
            # CLASSIFICATION
            # -------------------------------------------------

            if prediction >= 0.5:

                result = "NO CRACK DETECTED"
                confidence = prediction * 100
                result_type = "safe"

            else:

                result = "CRACK DETECTED"
                confidence = (1 - prediction) * 100
                result_type = "crack"

            # -------------------------------------------------
            # RESULT CARD
            # -------------------------------------------------

            if result_type == "crack":

                st.markdown(
                    f"""
                    <div class="result-crack">

                    <div style="font-size:55px;">
                    ⚠️
                    </div>

                    <div class="result-title"
                    style="color:#f87171;">
                    {result}
                    </div>

                    <div class="result-confidence">
                    Confidence:
                    <b>{confidence:.2f}%</b>
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-safe">

                    <div style="font-size:55px;">
                    ✓
                    </div>

                    <div class="result-title"
                    style="color:#34d399;">
                    {result}
                    </div>

                    <div class="result-confidence">
                    Confidence:
                    <b>{confidence:.2f}%</b>
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # =================================================
            # PREDICTION METRICS
            # =================================================

            st.markdown(
                """
                <div style="
                    margin-top:25px;
                    margin-bottom:10px;
                    color:#cbd5e1;
                    font-size:18px;
                    font-weight:700;
                ">
                    📊 Prediction Metrics
                </div>
                """,
                unsafe_allow_html=True
            )

            metric1, metric2 = st.columns(2)

            with metric1:

                st.metric(
                    "Prediction Score",
                    f"{prediction:.4f}"
                )

            with metric2:

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            # =================================================
            # CONFIDENCE BAR
            # =================================================

            st.markdown(
                """
                <div style="
                    margin-top:25px;
                    margin-bottom:8px;
                    color:#94a3b8;
                    font-size:15px;
                    font-weight:600;
                ">
                    Classification Confidence
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(
                    max(
                        float(confidence / 100),
                        0.0
                    ),
                    1.0
                )
            )

            st.markdown(
                f"""
                <div style="
                    text-align:right;
                    color:#60a5fa;
                    font-size:14px;
                    margin-top:4px;
                ">
                    {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            # =================================================
            # RESULT INTERPRETATION
            # =================================================

            if result_type == "crack":

                st.error(
                    "The CNN classified this semiconductor image "
                    "as containing a crack."
                )

            else:

                st.success(
                    "The CNN classified this semiconductor image "
                    "as having no detected crack."
                )

            # =================================================
            # PROCESSING INFORMATION
            # =================================================

            st.markdown(
                """
                <div class="card">

                <div class="card-title">
                ⚙️ Processing Information
                </div>

                <p style="color:#cbd5e1;">
                <b>Original Image</b>
                &nbsp;→&nbsp;
                <b>128 × 128 Resize</b>
                &nbsp;→&nbsp;
                <b>CNN</b>
                &nbsp;→&nbsp;
                <b>Classification</b>
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

        # Close result card
        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# =========================================================
# MODEL INFORMATION PAGE
# =========================================================

elif page == "🧠 Model Information":

    st.markdown(
        """
        <div class="main-title">
        Model <span class="blue-text">Information</span>
        </div>

        <div class="subtitle">
        CNN architecture and technical configuration
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">CNN</div>
                <div class="metric-label">Architecture</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">128×128</div>
                <div class="metric-label">Input Resolution</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">3</div>
                <div class="metric-label">Conv Layers</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value">2</div>
                <div class="metric-label">Output Classes</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        🧠 CNN Architecture
        </div>

        <p style="color:#94a3b8;">
        The trained model consists of convolutional, pooling,
        flattening and fully connected layers.
        </p>

        <div style="
            background:#050b14;
            border:1px solid #1e3a5f;
            border-radius:12px;
            padding:20px;
            margin-top:20px;
            font-family:monospace;
            color:#60a5fa;
            line-height:2;
        ">

        Input Image<br>
        ↓<br>
        Rescaling (1/255)<br>
        ↓<br>
        Conv2D — 32 Filters<br>
        ↓<br>
        MaxPooling2D<br>
        ↓<br>
        Conv2D — 64 Filters<br>
        ↓<br>
        MaxPooling2D<br>
        ↓<br>
        Conv2D — 128 Filters<br>
        ↓<br>
        MaxPooling2D<br>
        ↓<br>
        Flatten<br>
        ↓<br>
        Dense — 128 Neurons<br>
        ↓<br>
        Dropout — 0.5<br>
        ↓<br>
        Dense — 1 Neuron<br>
        ↓<br>
        Sigmoid Output<br>
        ↓<br>
        Crack / No Crack

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        ⚙️ Layer Configuration
        </div>

        <table style="
            width:100%;
            border-collapse:collapse;
            color:#cbd5e1;
        ">

        <tr style="border-bottom:1px solid #1e293b;">
            <th style="padding:12px;text-align:left;">Layer</th>
            <th style="padding:12px;text-align:left;">Configuration</th>
            <th style="padding:12px;text-align:left;">Purpose</th>
        </tr>

        <tr>
            <td style="padding:12px;">Input</td>
            <td style="padding:12px;">128 × 128 × 3</td>
            <td style="padding:12px;">Image input</td>
        </tr>

        <tr>
            <td style="padding:12px;">Rescaling</td>
            <td style="padding:12px;">1 / 255</td>
            <td style="padding:12px;">Normalize pixel values</td>
        </tr>

        <tr>
            <td style="padding:12px;">Conv2D</td>
            <td style="padding:12px;">32 filters, 3 × 3</td>
            <td style="padding:12px;">Extract image features</td>
        </tr>

        <tr>
            <td style="padding:12px;">MaxPooling</td>
            <td style="padding:12px;">2 × 2</td>
            <td style="padding:12px;">Reduce feature size</td>
        </tr>

        <tr>
            <td style="padding:12px;">Conv2D</td>
            <td style="padding:12px;">64 filters, 3 × 3</td>
            <td style="padding:12px;">Extract deeper features</td>
        </tr>

        <tr>
            <td style="padding:12px;">MaxPooling</td>
            <td style="padding:12px;">2 × 2</td>
            <td style="padding:12px;">Reduce feature maps</td>
        </tr>

        <tr>
            <td style="padding:12px;">Conv2D</td>
            <td style="padding:12px;">128 filters, 3 × 3</td>
            <td style="padding:12px;">Learn advanced features</td>
        </tr>

        <tr>
            <td style="padding:12px;">MaxPooling</td>
            <td style="padding:12px;">2 × 2</td>
            <td style="padding:12px;">Reduce feature maps</td>
        </tr>

        <tr>
            <td style="padding:12px;">Flatten</td>
            <td style="padding:12px;">Feature vector</td>
            <td style="padding:12px;">Convert features to vector</td>
        </tr>

        <tr>
            <td style="padding:12px;">Dense</td>
            <td style="padding:12px;">128 neurons</td>
            <td style="padding:12px;">Classification learning</td>
        </tr>

        <tr>
            <td style="padding:12px;">Dropout</td>
            <td style="padding:12px;">0.5</td>
            <td style="padding:12px;">Reduce overfitting</td>
        </tr>

        <tr>
            <td style="padding:12px;">Output</td>
            <td style="padding:12px;">1 neuron, Sigmoid</td>
            <td style="padding:12px;">Binary classification</td>
        </tr>

        </table>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        📊 Model Summary
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "Input Shape",
            "128 × 128 × 3"
        )

    with p2:
        st.metric(
            "Trainable Parameters",
            "3.30 Million"
        )

    with p3:
        st.metric(
            "Output",
            "Binary"
        )

    st.info(
        "The dashboard uses the cleaned trained model "
        "`semiconductor_crack_dashboard_model.keras`."
    )


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "ℹ️ About":

    st.markdown(
        """
        <div class="main-title">
        About the <span class="blue-text">Project</span>
        </div>

        <div class="subtitle">
        Semiconductor Crack Detection using CNN
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        🎯 Project Objective
        </div>

        The objective of this project is to develop an image-based
        semiconductor inspection system using a Convolutional Neural
        Network.

        The system analyzes a semiconductor image and classifies it
        as either <b>Crack</b> or <b>No Crack</b>.

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        ⚙️ System Operation
        </div>

        <ol style="color:#cbd5e1; line-height:2;">

        <li>User uploads a semiconductor image.</li>

        <li>The image is converted to RGB format.</li>

        <li>The image is resized to 128 × 128 pixels.</li>

        <li>The trained CNN processes the image.</li>

        <li>The model generates a prediction score.</li>

        <li>The system displays Crack or No Crack.</li>

        </ol>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HELP PAGE
# =========================================================

elif page == "❓ Help":

    st.markdown(
        """
        <div class="main-title">
        <span class="blue-text">Help</span> & Instructions
        </div>

        <div class="subtitle">
        How to use the semiconductor inspection dashboard
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
        📋 How to Use
        </div>

        <ol style="color:#cbd5e1; line-height:2;">

        <li>Open the <b>Predict</b> page.</li>

        <li>Click the image upload area.</li>

        <li>Select a semiconductor JPG, JPEG, or PNG image.</li>

        <li>Check the uploaded image.</li>

        <li>Click <b>Analyze Image</b>.</li>

        <li>View the prediction result and confidence.</li>

        </ol>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "The system is intended for project demonstration and "
        "image classification purposes."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
    Semiconductor Crack Detection System
    | CNN-Based Image Classification
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
from PIL import Image
import time
from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO

# Load model
model = YOLO("best.pt", task="detect")

# Page config
st.set_page_config(
    page_title="📦 Pallet Detection",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark styling
st.markdown("""
    <style>
        body {
            background-color: #0f172a;
        }
        .main {
            background-color: #0f172a;
            color: #ffffff;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #38bdf8;
        }
        .stButton > button {
            background-color: #1d4ed8;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar: file uploader
with st.sidebar:
    st.title("📤 Upload Images")
    uploaded_files = st.file_uploader(
        "Select image files",
        type=["jpg", "jpeg", "png", "webp", "avif"],
        accept_multiple_files=True,
        help="You can upload multiple image files (each < 200MB)."
    )

# Title
st.markdown("<h1 style='text-align: center;'>📦 Pallet Detection and Counting</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by YOLOv8</p>", unsafe_allow_html=True)

def detect_pallets_yolo(image_pil):
    image_np = np.array(image_pil)
    results = model(image_np, verbose=False)[0]
    output_image = image_np.copy()

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

    count = len(results.boxes)
    return Image.fromarray(output_image), count

# Process each uploaded file
if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("🔍 Detecting pallets..."):
            start_time = time.time()
            detected_image, count = detect_pallets_yolo(image)
            detection_time = round(time.time() - start_time, 2)

        # Display original and result side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(detected_image, caption="Detected Image", use_column_width=True)

        # Show metrics
        st.markdown(f"""
        <div style='display: flex; gap: 2rem; justify-content: center; margin-top: 20px;'>
            <div style='padding: 10px 20px; background-color: #1b4332; color: white; border-radius: 8px;'>
                <strong>📦 Pallets Detected:</strong> {count}
            </div>
            <div style='padding: 10px 20px; background-color: #1d3557; color: white; border-radius: 8px;'>
                <strong>⏱ Detection Time:</strong> {detection_time} seconds
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Download button
        buf = BytesIO()
        detected_image.save(buf, format="PNG")
        st.download_button(
            label="📥 Download Detected Image",
            data=buf.getvalue(),
            file_name="pallet_detection.png",
            mime="image/png"
        )

        st.markdown("<hr>", unsafe_allow_html=True)
else:
    st.info("Please upload at least one image using the sidebar.")







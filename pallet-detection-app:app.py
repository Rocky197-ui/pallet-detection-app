import streamlit as st
from PIL import Image
import time
from ultralytics import YOLO
import numpy as np
import cv2

# Load YOLO model
model = YOLO('best.pt', task='detect')  # Replace with full path if needed

# Streamlit page setup
st.set_page_config(page_title="Pallet Detection", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: white;'>📦 Pallet Detection and Counting</h1>
    <p style='text-align: center;'>Upload an image to detect and count the number of pallets using YOLO.</p>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload Image(s)",
    type=["jpg", "jpeg", "png", "webp", "avif"],
    accept_multiple_files=True,
    help="Limit 200MB per file"
)

def detect_pallets_yolo(image_pil):
    image_np = np.array(image_pil)
    results = model(image_np, verbose=False)[0]

    # Create a copy for drawing
    output_image = image_np.copy()

    # Draw boxes (without labels/confidence)
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)

    count = len(results.boxes)
    return Image.fromarray(output_image), count

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        start_time = time.time()
        detected_image, count = detect_pallets_yolo(image)
        end_time = time.time()
        detection_time = round(end_time - start_time, 2)

        # Side-by-side display
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original image", use_column_width=True)
        with col2:
            st.image(detected_image, caption="Predicted image", use_column_width=True)

        # Result boxes
        st.markdown(f"""
        <div style='display: flex; gap: 2rem; justify-content: center; margin-top: 20px;'>
            <div style='padding: 10px 20px; background-color: #1b4332; color: white; border-radius: 8px;'>
                <strong>📦 Number of pallets detected:</strong> {count}
            </div>
            <div style='padding: 10px 20px; background-color: #1d3557; color: white; border-radius: 8px;'>
                <strong>⏱ Time taken for detection:</strong> {detection_time} seconds
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)




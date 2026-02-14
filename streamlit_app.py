import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.ocr_engine import adaptive_ocr


st.set_page_config(page_title="Handwritten OCR Prototype")

st.title("Automatic Handwritten Answer Sheet Digitizer")

uploaded_file = st.file_uploader(
    "Upload a handwritten image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Convert RGB (PIL) to BGR (OpenCV)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Running Adaptive OCR..."):
        text, confidence, mode = adaptive_ocr(image_np)

    st.success("Processing Complete")

    st.write("### Selected Preprocessing Mode:")
    st.write(mode)

    st.write("### Average Confidence:")
    st.write(f"{confidence}%")

    st.write("### Extracted Text:")
    st.write(text)

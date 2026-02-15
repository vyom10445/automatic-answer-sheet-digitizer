# Automatic Handwritten Answer Sheet Digitizer

## Overview

This project aims to digitize handwritten academic answer sheets using Computer Vision preprocessing and Tesseract LSTM-based OCR.

The system demonstrates how preprocessing significantly impacts OCR performance on handwritten text and implements an adaptive preprocessing strategy to improve recognition reliability.

---

## Problem Statement

Manual evaluation of handwritten answer sheets is time-consuming and inconsistent.  
Existing OCR systems perform well on printed text but struggle with natural handwritten academic notes.

This project explores:

- Image preprocessing techniques for handwriting enhancement
- LSTM-based OCR extraction
- Confidence-based evaluation
- Adaptive preprocessing selection

---

## System Architecture

### 1. Image Acquisition

- Scanned or camera-captured handwritten answer sheets

### 2. Preprocessing Pipelines

Multiple preprocessing modes are tested:

- Grayscale + Gaussian Blur
- Otsu Thresholding
- Adaptive Thresholding

### 3. OCR Engine

- Tesseract OCR (LSTM-based)
- Page Segmentation Mode tuning
- Word-level confidence extraction

### 4. Adaptive Selection

The system evaluates all preprocessing modes and selects the one with the highest average OCR confidence.

---

## Features

- Modular OCR engine (`src/ocr_engine.py`)
- CLI prototype for testing (`src/preprocessing_demo.py`)
- Interactive web interface using Streamlit
- Confidence-based evaluation metric
- Adaptive preprocessing selection

---

## Project Structure

```
automatic-answer-sheet-digitizer/
│
├── src/
│ ├── preprocessing_demo.py
│ └── ocr_engine.py
│
├── streamlit_app.py
├── samples/
├── uploads/
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Install Tesseract OCR

Download and install from:
https://github.com/UB-Mannheim/tesseract/wiki

Ensure Tesseract is added to system PATH.

Verify installation:

tesseract --version

---

### 2. Install Python Dependencies

pip install -r requirements.txt

---

## Running the CLI Prototype

From the `src` folder:

python preprocessing_demo.py

This will:

- Run adaptive OCR on sample image
- Display extracted text
- Show average confidence score

---

## Running the Web Interface

From project root:

streamlit run streamlit_app.py

Then open browser:

http://localhost:8501

Upload a handwritten image to:

- Extract text
- View selected preprocessing mode
- See average OCR confidence

---

## Evaluation

Preliminary testing shows:

- Baseline OCR performance varies depending on handwriting style.
- Natural cursive handwriting presents challenges for standard OCR engines.
- Adaptive preprocessing improves confidence compared to static pipelines.

This validates the need for more specialized handwriting models in future work.

---

## Future Improvements

- Fine-tuning deep learning models (CRNN)
- Line-level segmentation
- Confidence-based word highlighting
- Structured output generation (JSON / PDF)
- Deployment as web service

---

## Author

Vyom Chaturvedi  
2427030058
B.Tech Computer Science Engineering  
Manipal University Jaipur

---

## Disclaimer

This project demonstrates a prototype OCR pipeline for academic exploration. Performance may vary depending on handwriting style, lighting conditions, and image quality.

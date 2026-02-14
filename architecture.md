# System Architecture

## 1. Image Acquisition

- Scanned or camera-captured handwritten answer sheets.

## 2. Preprocessing (OpenCV)

- Grayscale conversion
- Noise reduction (Gaussian Blur)
- Adaptive thresholding
- Deskewing
- Line segmentation

## 3. OCR Engine

- Tesseract LSTM-based OCR
- Page Segmentation Mode tuning

## 4. Post Processing

- Confidence scoring
- Low-confidence highlighting

## 5. Output

- Editable text format
- Structured digital representation

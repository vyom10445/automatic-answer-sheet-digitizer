import cv2
import numpy as np
import pytesseract
from pytesseract import Output


# PREPROCESSING PIPELINE

def preprocess_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError("Image not found.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return thresh


# OCR WITH CONFIDENCE

def perform_ocr(image):
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config='--psm 6'
    )

    extracted_text = ""
    confidences = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()
        conf = int(data["conf"][i])

        if word != "":
            extracted_text += word + " "
            confidences.append(conf)

    return extracted_text.strip(), confidences


# CONFIDENCE ANALYSIS

def calculate_average_confidence(confidences):
    if len(confidences) == 0:
        return 0

    return round(sum(confidences) / len(confidences), 2)


# MAIN EXECUTION

def main():

    image_path = "../samples/sample_input.jpg"

    print("Preprocessing image...")
    processed_image = preprocess_image(image_path)

    cv2.imwrite("../samples/processed_output.jpg", processed_image)

    print("Running OCR...")
    text, confidences = perform_ocr(processed_image)

    print("\n--- Extracted Text ---")
    print(text)

    avg_conf = calculate_average_confidence(confidences)

    print("\nAverage OCR Confidence:", avg_conf, "%")

    print("\nWorking prototype executed successfully.")


if __name__ == "__main__":
    main()

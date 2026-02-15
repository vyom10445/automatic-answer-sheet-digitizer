"""
CLI Prototype Version
Runs adaptive OCR on a sample image.
"""

import cv2
from ocr_engine import adaptive_ocr


def main():
    image_path = "../samples/sample_input.jpg"

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found.")
        return

    print("Running Adaptive OCR...\n")

    text, confidence, mode = adaptive_ocr(image)

    print("Selected Mode:", mode)
    print("Average Confidence:", confidence, "%")
    print("\nExtracted Text:\n")
    print(text)


if __name__ == "__main__":
    main()

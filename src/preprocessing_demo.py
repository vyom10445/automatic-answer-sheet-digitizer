import cv2
import numpy as np
import os

# STEP 1: Load Image
def load_image(path):

    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    return image

# STEP 2: Convert to Grayscale
def convert_to_grayscale(image):

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



# STEP 3: Noise Reduction

def remove_noise(gray_image):

    return cv2.GaussianBlur(gray_image, (5, 5), 0)



# STEP 4: Adaptive Thresholding

def apply_threshold(blurred_image):

    return cv2.adaptiveThreshold(
        blurred_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )



# STEP 5: Automatic Deskew

def deskew(image):

    # Get coordinates of all non-zero pixels
    coords = np.column_stack(np.where(image > 0))

    # Determine minimum area rectangle
    angle = cv2.minAreaRect(coords)[-1]

    # Adjust angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Get image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate image
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated



# STEP 6: Save Processed Image

def save_image(output_path, image):

    cv2.imwrite(output_path, image)



# MAIN EXECUTION PIPELINE

def main():

    # Define paths
    input_path = "../samples/sample_input.jpg"
    output_path = "../samples/processed_output.jpg"

    print("Loading image...")
    image = load_image(input_path)

    print("Converting to grayscale...")
    gray = convert_to_grayscale(image)

    print("Reducing noise...")
    blurred = remove_noise(gray)

    print("Applying adaptive threshold...")
    thresholded = apply_threshold(blurred)

    print("Correcting skew...")
    deskewed = deskew(thresholded)

    print("Saving processed image...")
    save_image(output_path, deskewed)

    print("\nPreprocessing completed successfully.")
    print(f"Processed image saved at: {output_path}")


if __name__ == "__main__":
    main()

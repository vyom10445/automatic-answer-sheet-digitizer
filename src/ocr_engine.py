import cv2
import pytesseract
from pytesseract import Output



# PREPROCESSING MODES


def preprocess_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def preprocess_otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh


def preprocess_adaptive(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )



# OCR FUNCTION


def perform_ocr(image):
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config='--oem 3 --psm 6'
    )

    text = ""
    confidences = []

    for i in range(len(data["text"])):
        word = data["text"][i].strip()

        try:
            conf = int(data["conf"][i])
        except:
            conf = -1

        if word != "" and conf > 0:
            text += word + " "
            confidences.append(conf)

    if len(confidences) == 0:
        return "", 0

    avg_conf = sum(confidences) / len(confidences)
    return text.strip(), round(avg_conf, 2)


# ADAPTIVE PIPELINE


def adaptive_ocr(image):

    modes = {
        "Blur": preprocess_blur(image),
        "Otsu": preprocess_otsu(image),
        "Adaptive Threshold": preprocess_adaptive(image)
    }

    best_conf = 0
    best_text = ""
    best_mode = ""

    for mode_name, processed in modes.items():
        text, conf = perform_ocr(processed)

        if conf > best_conf:
            best_conf = conf
            best_text = text
            best_mode = mode_name

    return best_text, best_conf, best_mode

from pathlib import Path
import pytesseract
from PIL import Image


WINDOWS_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


if WINDOWS_TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT_PATH)


def extract_text_from_image(image_path: str) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(path)
    text = pytesseract.image_to_string(image)

    return text.strip()

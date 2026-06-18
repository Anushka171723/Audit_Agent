from pathlib import Path
import pytesseract
from PIL import Image


WINDOWS_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


if WINDOWS_TESSERACT_PATH.exists():
    pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT_PATH)


def _extract_text_from_pdf(path: Path) -> str:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PDF support requires PyMuPDF. Install project requirements and try again."
        ) from error

    with fitz.open(str(path)) as document:
        page_text = []

        for page in document:
            text = page.get_text("text").strip()
            if text:
                page_text.append(text)

        if page_text:
            return "\n\n".join(page_text).strip()

        ocr_text = []
        matrix = fitz.Matrix(2, 2)

        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes(
                "RGB",
                [pixmap.width, pixmap.height],
                pixmap.samples,
            )
            text = pytesseract.image_to_string(image).strip()
            if text:
                ocr_text.append(text)

    return "\n\n".join(ocr_text).strip()


def extract_text_from_image(image_path: str) -> str:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Invoice file not found: {image_path}")

    if path.suffix.lower() == ".pdf":
        return _extract_text_from_pdf(path)

    image = Image.open(path)
    text = pytesseract.image_to_string(image)

    return text.strip()


def extract_text_from_file(file_path: str) -> str:
    return extract_text_from_image(file_path)

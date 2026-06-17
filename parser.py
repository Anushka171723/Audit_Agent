import csv
import json
import os
import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


GROQ_EXTRACTION_MODEL = "llama-3.3-70b-versatile"


class InvoiceExtraction(BaseModel):
    invoice_no: str = Field(default="", description="Invoice number or invoice ID.")
    date: str = Field(default="", description="Invoice date exactly as written or normalized.")
    vendor: str = Field(default="", description="Seller, supplier, vendor, or issuing company name.")
    amount: float = Field(default=0.0, description="Subtotal or taxable amount before tax.")
    tax: float = Field(default=0.0, description="Total tax amount from all tax lines.")
    total: float = Field(default=0.0, description="Final payable amount, total due, or grand total.")


GROQ_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract clean invoice JSON from OCR text. Use only the requested schema. "
            "Prefer final payable total over subtotal. Add all tax-like rows into tax. "
            "If OCR has a likely tax row typo, infer it only when the amounts and final total support it. "
            "If a value is missing, use an empty string for text or 0 for numbers.",
        ),
        ("human", "OCR text:\n{ocr_text}"),
    ]
)


def _normalise_invoice_data(data: dict) -> dict:
    return {
        "invoice_no": str(data.get("invoice_no") or "").strip(),
        "date": str(data.get("date") or "").strip(),
        "vendor": str(data.get("vendor") or "").strip(),
        "amount": _to_number(data.get("amount")),
        "tax": _to_number(data.get("tax")),
        "total": _to_number(data.get("total")),
    }


def _to_number(value: object) -> float:
    if value in [None, ""]:
        return 0.0

    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return 0.0

def _find_value(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return ""


def _find_amount(pattern: str, text: str) -> float:
    value = _find_value(pattern, text)

    if not value:
        return 0.0

    cleaned_value = value.replace(",", "")

    try:
        return float(cleaned_value)
    except ValueError:
        return 0.0


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _find_line_amount(labels: list[str], text: str) -> float:
    for line in _lines(text):
        lower_line = line.lower()

        if any(lower_line.startswith(label.lower()) for label in labels):
            amounts = re.findall(r"[0-9,]+(?:\.[0-9]{1,2})?", line)

            if amounts:
                return float(amounts[-1].replace(",", ""))

    return 0.0


def _find_tax_total(text: str) -> float:
    tax_labels = ["tax", "gst", "cgst", "sgst", "igst", "pst", "qst", "hst", "vat"]
    tax_total = 0.0

    for line in _lines(text):
        lower_line = line.lower()

        if any(lower_line.startswith(label) for label in tax_labels):
            amounts = re.findall(r"[0-9,]+(?:\.[0-9]{1,2})?", line)

            if amounts:
                tax_total += float(amounts[-1].replace(",", ""))

    return round(tax_total, 2)


def parse_invoice_text_with_rules(text: str) -> dict:
    invoice_data = {
        "invoice_no": _find_value(r"invoice\s*(?:no|number|#)\s*[:\-]?\s*([A-Za-z0-9\-\/]+)", text),
        "date": _find_value(r"(?:date|invoice date)\s*[:\-]?\s*([A-Za-z]+\s+[0-9]{1,2},\s+[0-9]{4}|[0-9]{1,2}[\/\-][0-9]{1,2}[\/\-][0-9]{2,4})", text),
        "vendor": _find_value(r"(?:vendor|seller|supplier)\s*[:\-]?\s*(.+)", text) or (_lines(text)[0] if _lines(text) else ""),
        "amount": _find_line_amount(["subtotal", "amount", "taxable value"], text),
        "tax": _find_tax_total(text),
        "total": _find_line_amount(["total due", "grand total", "net amount", "total"], text),
    }

    return invoice_data


def parse_invoice_text_with_groq(text: str) -> dict:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=GROQ_EXTRACTION_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(InvoiceExtraction)
    messages = GROQ_EXTRACTION_PROMPT.format_messages(ocr_text=text)
    result = structured_llm.invoke(messages)

    if hasattr(result, "model_dump"):
        result = result.model_dump()

    return _normalise_invoice_data(result)


def parse_invoice_text(text: str, use_llm: bool = True) -> dict:
    if use_llm and os.getenv("GROQ_API_KEY"):
        try:
            return parse_invoice_text_with_groq(text)
        except Exception as error:
            print(f"Groq extraction failed, using rule parser instead: {error}")

    return parse_invoice_text_with_rules(text)


def save_json(data: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_csv(data: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

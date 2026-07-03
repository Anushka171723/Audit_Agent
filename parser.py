import csv
import json
import os
import re
from datetime import date, datetime
from pathlib import Path


GROQ_EXTRACTION_MODEL = "llama-3.3-70b-versatile"


GROQ_EXTRACTION_PROMPT = """You are an expert Invoice Intelligence and Audit Extraction System.

Analyze the provided OCR text and extract structured invoice information.

Your goal is to correctly understand invoices from any industry including:

* GST Invoices
* Hospital Bills
* Hotel Bills
* Restaurant Bills
* Amazon Invoices
* Flipkart Invoices
* Retail Invoices
* Service Invoices
* Travel Bills
* Utility Bills

Return ONLY valid JSON.

Rules:

1. Extract invoice number even if labeled as:

   * Invoice No
   * Invoice Number
   * Invoice #
   * Bill No
   * Bill Number
   * Ref No
   * Reference Number
   * Document Number
   * Tax Invoice Number

2. Extract:

   * vendor/company/supplier/seller name
   * customer name if present
   * invoice date
   * GSTIN if present
   * category
   * classification (Determine if it is a 'Purchase', 'Sales', or 'Expense'. 'Sales' if issued by us to a customer; 'Purchase' if B2B inventory, services, raw materials, or trade; 'Expense' if travel, utility bills, food/restaurant, medical/hospital, operational consumables, office supplies, or retail items).
   * subtotal amount
   * discount if present
   * total tax
   * grand total

3. Detect document type:

Possible values:

* gst_invoice
* hospital_invoice
* hotel_invoice
* restaurant_invoice
* amazon_invoice
* flipkart_invoice
* retail_invoice
* service_invoice
* travel_invoice
* utility_invoice
* other

4. Extract line items if present, including the product name, HSN/SAC code if present, quantity, unit price, and total amount.

5. Calculate total tax using:

   * GST
   * CGST
   * SGST
   * IGST
   * VAT
   * Service Tax

6. Determine OCR quality:

Possible values:

* good
* moderate
* poor

Mark OCR quality as:

* poor if text appears corrupted or unreadable
* moderate if some fields are unclear
* good if text is clear

7. Detect potential audit risks:

Possible values:

* missing_invoice_no
* missing_vendor
* missing_date
* missing_total
* missing_gstin
* tax_mismatch
* suspicious_amount
* poor_ocr
* duplicate_candidate
* future_date
* invalid_document

8. If a field is unavailable:

   * text fields = ""
   * numeric fields = 0
   * arrays = []

9. Convert monetary values to numbers only.

10. Do not explain anything.

11. Do not return markdown.

12. Return ONLY valid JSON.

Required Output Schema:

{{
"document_type": "",
"invoice_no": "",
"date": "",
"vendor": "",
"customer_name": "",
"gstin": "",
"category": "",
"classification": "Purchase",
"amount": 0,
"discount": 0,
"tax": 0,
"total": 0,
"payment_method": "",
"ocr_quality": "",
"items": [
{{
"product": "",
"hsn_sac": "",
"quantity": 0,
"unit_price": 0,
"amount": 0
}}
],
"audit_flags": []
}}

OCR Text:

{ocr_text}"""



DOCUMENT_TYPES = {
    "gst_invoice",
    "hospital_invoice",
    "hotel_invoice",
    "restaurant_invoice",
    "amazon_invoice",
    "flipkart_invoice",
    "retail_invoice",
    "service_invoice",
    "travel_invoice",
    "utility_invoice",
    "other",
}
OCR_QUALITIES = {"good", "moderate", "poor"}
AUDIT_FLAGS = {
    "missing_invoice_no",
    "missing_vendor",
    "missing_date",
    "missing_total",
    "missing_gstin",
    "tax_mismatch",
    "suspicious_amount",
    "poor_ocr",
    "duplicate_candidate",
    "future_date",
    "invalid_document",
}


def _to_number(value: object) -> float:
    if value in [None, ""]:
        return 0.0

    try:
        cleaned_value = re.sub(r"[^\d.\-]", "", str(value).replace(",", ""))
        if cleaned_value in ["", ".", "-", "-."]:
            return 0.0
        return float(cleaned_value)
    except ValueError:
        return 0.0


def _normalise_item(item: dict) -> dict:
    description = str(item.get("description") or item.get("product") or "").strip()
    return {
        "product": description,
        "hsn_sac": str(item.get("hsn_sac") or "").strip(),
        "quantity": _to_number(item.get("quantity")),
        "unit_price": _to_number(item.get("unit_price")),
        "amount": _to_number(item.get("amount")),
    }


def _classify_invoice(data: dict, text: str) -> str:
    classification = str(data.get("classification") or "").strip().title()
    if classification in ["Purchase", "Sales", "Expense"]:
        return classification
    
    doc_type = str(data.get("document_type") or "").strip().lower()
    category = str(data.get("category") or "").strip().lower()
    text_lower = text.lower()
    
    # 1. Expense: Electricity bill, Internet bill, Hotel bill, Restaurant bill, Fuel bill, Travel/Cab/Flight bill, Hospital/Medical
    expense_keywords = [
        "electricity", "power bill", "water bill", "gas bill", "broadband", "internet", "telecom", "telephonic",
        "hotel", "stay", "room charge", "lodging", "restaurant", "cafe", "food", "dining", "meal",
        "fuel", "petrol", "diesel", "cng", "travel", "flight", "boarding pass", "cab", "taxi", "uber", "ola",
        "hospital", "medical", "pharmacy", "medicine", "doctor"
    ]
    if doc_type in ["hotel_invoice", "restaurant_invoice", "travel_invoice", "utility_invoice", "hospital_invoice"]:
        return "Expense"
    if category in ["food", "hospitality", "travel", "utilities", "healthcare"]:
        return "Expense"
    if any(kw in text_lower for kw in expense_keywords):
        return "Expense"
        
    # 2. Sales: Invoices issued to customers (we are selling goods/services)
    sales_indicators = [
        "sales invoice", "tax invoice to", "sold to", "bill to", "invoice to customer", "sales receipt"
    ]
    # If the vendor name is our default business or if there are customer details but we are the vendor
    if any(kw in text_lower for kw in sales_indicators):
        return "Sales"
        
    # 3. Purchase: Invoices received from suppliers (e.g. Amazon, Flipkart, wholesalers, distributors, retail vendors)
    purchase_keywords = [
        "amazon", "flipkart", "supplier", "wholesaler", "distributor", "purchase order", "po no", 
        "dealer", "trader", "manufacturer", "distributors", "sold by", "shipped from"
    ]
    if doc_type in ["amazon_invoice", "flipkart_invoice", "retail_invoice"]:
        return "Purchase"
    if category in ["retail", "office supplies"]:
        return "Purchase"
    if any(kw in text_lower for kw in purchase_keywords):
        return "Purchase"
        
    return "Purchase"  # default fallback



def _normalise_invoice_data(data: dict, raw_text: str = "") -> dict:
    document_type = str(data.get("document_type") or "").strip().lower()
    ocr_quality = str(data.get("ocr_quality") or "").strip().lower()
    items = data.get("items") if isinstance(data.get("items"), list) else []
    audit_flags = data.get("audit_flags") if isinstance(data.get("audit_flags"), list) else []
    normalised_document_type = document_type if document_type in DOCUMENT_TYPES else "other"
    category = str(data.get("category") or "").strip()
    document_category = _category_from_document_type(normalised_document_type)
    if not category or (category.lower() == "other" and document_category != "Other"):
        category = document_category

    classification = _classify_invoice(data, raw_text)

    normalised = {
        "document_type": normalised_document_type,
        "invoice_no": str(data.get("invoice_no") or "").strip(),
        "date": str(data.get("date") or "").strip(),
        "vendor": str(data.get("vendor") or "").strip(),
        "customer_name": str(data.get("customer_name") or "").strip(),
        "gstin": str(data.get("gstin") or "").strip(),
        "category": category,
        "classification": classification,
        "amount": _to_number(data.get("amount")),
        "discount": _to_number(data.get("discount")),
        "tax": _to_number(data.get("tax")),
        "total": _to_number(data.get("total")),
        "payment_method": str(data.get("payment_method") or "").strip(),
        "ocr_quality": ocr_quality if ocr_quality in OCR_QUALITIES else "moderate",
        "items": [_normalise_item(item) for item in items if isinstance(item, dict)],
        "audit_flags": [flag for flag in audit_flags if flag in AUDIT_FLAGS],
    }

    return _add_derived_fields(normalised)



def _money_pattern() -> str:
    return r"(?:Rs\.?|INR|USD|EUR|GBP|\$)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)"


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _find_value(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _clean_label_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" :-\t")


def _amounts_in_line(line: str) -> list[float]:
    return [float(amount.replace(",", "")) for amount in re.findall(_money_pattern(), line, re.IGNORECASE)]


def _find_line_amount(labels: list[str], text: str) -> float:
    for line in _lines(text):
        lower_line = line.lower()
        if any(re.search(rf"\b{re.escape(label.lower())}\b", lower_line) for label in labels):
            amounts = _amounts_in_line(line)
            if amounts:
                return amounts[-1]

    return 0.0


def _find_invoice_number(text: str) -> str:
    labels = [
        r"tax\s*invoice\s*(?:no\.?|number|#)",
        r"invoice\s*(?:no\.?|number|#)",
        r"bill\s*(?:no\.?|number|#)",
        r"ref(?:erence)?\s*(?:no\.?|number|#)",
        r"document\s*(?:no\.?|number|#)",
        r"doc\s*(?:no\.?|number|#)",
    ]

    for label in labels:
        value = _find_value(rf"\b{label}\s*[:\-#]?\s*([A-Za-z0-9][A-Za-z0-9\-\/_.]*)", text)
        if value:
            return value

    return ""


def _find_invoice_date(text: str) -> str:
    date_value = (
        r"([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4}|"
        r"[0-9]{4}[\/\-.][0-9]{1,2}[\/\-.][0-9]{1,2}|"
        r"[A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4}|"
        r"[0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})"
    )

    for label in [r"invoice\s*date", r"bill\s*date", r"date"]:
        value = _find_value(rf"\b{label}\s*[:\-]?\s*{date_value}", text)
        if value:
            return value

    return _find_value(date_value, text)


def _parse_invoice_date(value: object) -> date | None:
    if not value:
        return None

    text = str(value).strip()
    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d.%m.%Y",
        "%Y/%m/%d",
        "%Y-%m-%d",
        "%Y.%m.%d",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d.%m.%y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]

    for date_format in formats:
        try:
            return datetime.strptime(text, date_format).date()
        except ValueError:
            continue

    return None


def _find_vendor(text: str) -> str:
    vendor = _find_value(
        r"\b(?:vendor|seller|supplier|merchant|company|business|from|billed\s*by)\s*[:\-]\s*([^\n\r]+)",
        text,
    )
    if vendor:
        return _clean_label_value(vendor)

    ignored_starts = (
        "invoice",
        "bill",
        "date",
        "gst",
        "tax",
        "subtotal",
        "sub total",
        "total",
        "amount",
        "qty",
        "quantity",
        "customer",
    )

    for line in _lines(text):
        clean_line = _clean_label_value(line)
        lower_line = clean_line.lower()
        if lower_line.startswith(ignored_starts):
            continue
        if re.search(r"\d{4,}|[$]", clean_line):
            continue
        if len(clean_line) >= 3:
            return clean_line

    return ""


def _find_customer_name(text: str) -> str:
    return _clean_label_value(
        _find_value(
            r"\b(?:customer|customer\s*name|bill\s*to|billed\s*to|patient|guest|ship\s*to)\s*[:\-]\s*([^\n\r]+)",
            text,
        )
    )


def _find_gstin(text: str) -> str:
    return _find_value(
        r"\b(?:GSTIN|GSTIN\/UIN|GST\s*(?:No\.?|Number|#))\s*[:\-]?\s*([0-9A-Z]{15})\b",
        text,
    ).upper()


def _find_payment_method(text: str) -> str:
    payment_method = _find_value(
        r"\b(?:payment\s*method|paid\s*by|mode\s*of\s*payment|payment\s*mode)\s*[:\-]\s*([^\n\r]+)",
        text,
    )
    if payment_method:
        return _clean_label_value(payment_method)

    lower_text = text.lower()
    for method in ["upi", "credit card", "debit card", "cash", "net banking", "bank transfer", "wallet"]:
        if method in lower_text:
            return method.title()

    return ""


def _find_category(text: str) -> str:
    category = _clean_label_value(
        _find_value(r"\b(?:category|expense\s*category|type|service\s*category)\s*[:\-]\s*([^\n\r]+)", text)
    )
    if category:
        return category

    category_keywords = {
        "Electronics": ["laptop", "computer", "mobile", "phone", "charger", "electronics", "amazon", "flipkart"],
        "Healthcare": ["hospital", "clinic", "medical", "medicine", "pharmacy", "patient"],
        "Food": ["restaurant", "food", "meal", "grocery", "cafe", "dining"],
        "Hospitality": ["hotel", "lodging", "resort", "room", "guest"],
        "Travel": ["flight", "airline", "taxi", "cab", "travel", "ticket", "booking"],
        "Retail": ["retail", "store", "shop", "mall"],
        "Manufacturing": ["manufacturing", "factory", "parts", "machinery"],
        "Services": ["service", "consulting", "maintenance", "professional fee"],
        "Utilities": ["electricity", "water", "gas", "utility", "internet", "telecom", "broadband"],
        "Office Supplies": ["stationery", "paper", "printer", "office supplies"],
    }
    lower_text = text.lower()

    for category_name, keywords in category_keywords.items():
        if any(keyword in lower_text for keyword in keywords):
            return category_name

    return "Other" if text.strip() else ""


def _category_from_document_type(document_type: str) -> str:
    categories = {
        "amazon_invoice": "Retail",
        "flipkart_invoice": "Retail",
        "hospital_invoice": "Healthcare",
        "hotel_invoice": "Hospitality",
        "restaurant_invoice": "Food",
        "service_invoice": "Services",
        "travel_invoice": "Travel",
        "utility_invoice": "Utilities",
        "retail_invoice": "Retail",
        "gst_invoice": "Services",
    }
    return categories.get(document_type, "Other")


def _detect_document_type(text: str) -> str:
    lower_text = text.lower()
    checks = [
        ("amazon_invoice", ["amazon", "amazon.in", "sold by", "order id"]),
        ("flipkart_invoice", ["flipkart", "ekart", "seller registered address"]),
        ("hospital_invoice", ["hospital", "patient", "doctor", "medical"]),
        ("hotel_invoice", ["hotel", "room", "guest", "check-in", "check out", "checkout"]),
        ("restaurant_invoice", ["restaurant", "cafe", "table no", "food", "dining"]),
        ("travel_invoice", ["flight", "airline", "boarding", "ticket", "pnr", "cab", "taxi", "travel"]),
        ("utility_invoice", ["electricity", "water bill", "gas bill", "broadband", "internet bill", "utility"]),
        ("service_invoice", ["service invoice", "consulting", "professional fee", "maintenance"]),
        ("retail_invoice", ["retail", "store", "shop", "cash memo"]),
        ("gst_invoice", ["gstin", "cgst", "sgst", "igst", "tax invoice"]),
    ]

    for document_type, keywords in checks:
        if any(keyword in lower_text for keyword in keywords):
            return document_type

    return "other"


def _find_total_amount(text: str) -> float:
    labels = [
        "grand total",
        "final total",
        "final amount",
        "net payable amount",
        "net payable",
        "payable amount",
        "total due",
        "amount due",
        "balance due",
        "total payable",
        "invoice total",
        "total amount",
        "total",
    ]
    tax_words = ["tax", "gst", "cgst", "sgst", "igst", "vat", "service tax"]

    for label in labels:
        for line in _lines(text):
            lower_line = line.lower()
            if not re.search(rf"\b{re.escape(label)}\b", lower_line):
                continue
            if any(re.search(rf"\b{tax_word}\b", lower_line) for tax_word in tax_words):
                continue

            amounts = _amounts_in_line(line)
            if amounts:
                return amounts[-1]

    return 0.0


def _find_tax_total(text: str) -> float:
    total_tax_labels = [
        "total tax",
        "tax total",
        "total gst",
        "gst total",
        "total vat",
        "vat total",
        "total sales tax",
        "total service tax",
    ]
    explicit_tax_total = _find_line_amount(total_tax_labels, text)
    if explicit_tax_total > 0:
        return round(explicit_tax_total, 2)

    tax_labels = ["tax", "gst", "cgst", "sgst", "igst", "vat", "sales tax", "service tax"]
    tax_total = 0.0

    for line in _lines(text):
        lower_line = line.lower()
        if any(token in lower_line for token in ["gstin", "tax invoice", "invoice no", "invoice number"]):
            continue
        if any(re.search(rf"\b{label}\b", lower_line) for label in tax_labels):
            amounts = _amounts_in_line(line)
            if amounts:
                tax_total += amounts[-1]

    return round(tax_total, 2)


def _find_items(text: str) -> list[dict]:
    items = []
    skip_words = [
        "invoice",
        "bill",
        "date",
        "gst",
        "tax",
        "subtotal",
        "sub total",
        "total",
        "amount due",
        "balance",
        "net payable",
        "payable amount",
        "ref no",
        "reference number",
        "document number",
        "doc no",
        "customer",
        "patient",
        "guest",
        "payment",
    ]

    for line in _lines(text):
        lower_line = line.lower()
        if any(word in lower_line for word in skip_words):
            continue

        amounts = _amounts_in_line(line)
        if not amounts:
            continue

        description = re.sub(_money_pattern(), "", line, flags=re.IGNORECASE)
        description = _clean_label_value(re.sub(r"\s{2,}", " ", description))
        if not description or len(description) < 3:
            continue

        quantity = 0.0
        unit_price = 0.0
        if len(amounts) >= 3:
            quantity = amounts[-3]
            unit_price = amounts[-2]

        items.append(
            {
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
                "amount": amounts[-1]
            }
        )

    return items


def _ocr_quality(text: str, invoice_data: dict) -> str:
    if not text.strip():
        return "poor"

    readable_chars = sum(1 for char in text if char.isalnum() or char.isspace() or char in ".,:/-#")
    quality_ratio = readable_chars / max(len(text), 1)
    missing_core = sum(
        1
        for field in ["invoice_no", "date", "vendor", "total"]
        if not invoice_data.get(field)
    )

    if quality_ratio < 0.65 or missing_core >= 3:
        return "poor"
    if quality_ratio < 0.85 or missing_core:
        return "moderate"
    return "good"


def _is_duplicate_candidate_text(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:duplicate|copy\s+of\s+invoice|invoice\s+copy|reprint|reprinted|copy)\b",
            text,
            re.IGNORECASE,
        )
    )


def _audit_flags(invoice_data: dict) -> list[str]:
    flags = []
    document_type = str(invoice_data.get("document_type") or "").strip().lower()

    if not invoice_data.get("invoice_no"):
        flags.append("missing_invoice_no")
    if not invoice_data.get("vendor"):
        flags.append("missing_vendor")
    if not invoice_data.get("date"):
        flags.append("missing_date")
    if not invoice_data.get("total"):
        flags.append("missing_total")
    if not invoice_data.get("gstin"):
        flags.append("missing_gstin")
    if invoice_data.get("ocr_quality") == "poor":
        flags.append("poor_ocr")
    invoice_date = _parse_invoice_date(invoice_data.get("date"))
    if invoice_date and invoice_date > date.today():
        flags.append("future_date")

    amount = _to_number(invoice_data.get("amount"))
    tax = _to_number(invoice_data.get("tax"))
    total = _to_number(invoice_data.get("total"))
    if (
        not invoice_data.get("invoice_no")
        or not invoice_data.get("vendor")
        or total <= 0
        or document_type in ["other", "unknown"]
    ):
        flags.append("invalid_document")

    if total > 0 and amount > 0 and abs((amount + tax) - total) > 1:
        flags.append("tax_mismatch")
    if amount > 100000 or total > 100000:
        flags.append("suspicious_amount")

    return flags


def _add_derived_fields(invoice_data: dict) -> dict:
    amount = _to_number(invoice_data.get("amount"))
    tax = _to_number(invoice_data.get("tax"))
    total = _to_number(invoice_data.get("total"))

    if amount <= 0 and total > 0:
        invoice_data["amount"] = round(total - tax, 2) if total >= tax else total
    if total <= 0 and amount > 0:
        invoice_data["total"] = round(amount + tax, 2)

    existing_flags = list(invoice_data.get("audit_flags", []))
    for flag in _audit_flags(invoice_data):
        if flag not in existing_flags:
            existing_flags.append(flag)
    invoice_data["audit_flags"] = existing_flags

    return invoice_data


def parse_invoice_text_with_rules(text: str) -> dict:
    document_type = _detect_document_type(text)
    category = _find_category(text)
    if category == "Other":
        category = _category_from_document_type(document_type)

    invoice_data = {
        "document_type": document_type,
        "invoice_no": _find_invoice_number(text),
        "date": _find_invoice_date(text),
        "vendor": _find_vendor(text),
        "customer_name": _find_customer_name(text),
        "gstin": _find_gstin(text),
        "category": category,
        "amount": _find_line_amount(
            [
                "subtotal",
                "sub total",
                "amount before tax",
                "taxable amount",
                "taxable value",
                "net amount",
                "base amount",
            ],
            text,
        ),
        "tax": _find_tax_total(text),
        "total": _find_total_amount(text),
        "payment_method": _find_payment_method(text),
        "ocr_quality": "",
        "items": _find_items(text),
        "audit_flags": ["duplicate_candidate"] if _is_duplicate_candidate_text(text) else [],
    }
    invoice_data["ocr_quality"] = _ocr_quality(text, invoice_data)

    return _normalise_invoice_data(invoice_data, text)


def _load_llm_json(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    return json.loads(cleaned)


def parse_invoice_text_with_groq(text: str) -> dict:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=GROQ_EXTRACTION_MODEL, temperature=0)
    prompt = GROQ_EXTRACTION_PROMPT.format(ocr_text=text)
    response = llm.invoke(prompt)
    invoice_data = _load_llm_json(response.content)

    return _normalise_invoice_data(invoice_data, text)


def is_valid_invoice(invoice_data: dict) -> bool:
    """
    Validate if the extracted data represents a valid invoice.
    
    Returns True only if:
    - invoice_no exists (not empty)
    - vendor exists (not empty)
    - total > 0
    - document_type is not "other"
    - document_type is not "unknown"
    
    Otherwise returns False.
    """
    invoice_no = str(invoice_data.get("invoice_no") or "").strip()
    vendor = str(invoice_data.get("vendor") or "").strip()
    document_type = str(invoice_data.get("document_type") or "").strip().lower()
    
    try:
        total = float(invoice_data.get("total") or 0)
    except (TypeError, ValueError):
        total = 0
    
    # Check all required conditions
    if not invoice_no:
        return False
    
    if not vendor:
        return False
    
    if total <= 0:
        return False
    
    if document_type in ("other", "unknown"):
        return False
    
    return True


def parse_invoice_text(text: str, use_llm: bool = True) -> dict:
    if use_llm and os.getenv("GROQ_API_KEY"):
        try:
            return parse_invoice_text_with_groq(text)
        except Exception as error:
            print(f"Groq extraction failed, using rule parser instead: {error}")

    return parse_invoice_text_with_rules(text)


def parse_natural_language_invoice(text: str) -> dict:
    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model=GROQ_EXTRACTION_MODEL, temperature=0)
            
            prompt = f"""You are an AI Invoice Creation System.
Given a natural language description of a purchase or sale transaction, generate a complete structured invoice JSON.
If details like invoice number, vendor, date, or unit prices are missing, generate realistic values (e.g., today's date for date, a sequential looking invoice number like 'INV-2026-1001', a generic vendor like 'Tech Retailers', and logical pricing for items if not specified).
Calculate the totals, tax, and amounts correctly based on the items and GST percentage mentioned.

Required Output Schema:
{{
"document_type": "retail_invoice",
"invoice_no": "INV-YYYY-XXXX",
"date": "YYYY-MM-DD",
"vendor": "Vendor Name",
"customer_name": "Customer Name",
"gstin": "GSTIN (generate realistic 15-character Indian GSTIN if not specified but GST is mentioned)",
"category": "Retail",
"classification": "Sales",
"amount": 0,
"discount": 0,
"tax": 0,
"total": 0,
"payment_method": "Cash",
"ocr_quality": "good",
"items": [
  {{
    "product": "Product/Item Name",
    "hsn_sac": "HSN Code",
    "quantity": 1,
    "unit_price": 0,
    "amount": 0
  }}
],
"audit_flags": []
}}

Input Text:
{text}"""
            response = llm.invoke(prompt)
            invoice_data = _load_llm_json(response.content)
            return _normalise_invoice_data(invoice_data, text)
        except Exception as error:
            print(f"Groq natural language generation failed: {error}")
            
    # Fallback / Dummy invoice generation if Groq fails or API key not present
    today_str = datetime.today().strftime("%Y-%m-%d")
    fallback_data = {
        "document_type": "retail_invoice",
        "invoice_no": "INV-" + datetime.today().strftime("%Y%m%d") + "-01",
        "date": today_str,
        "vendor": "General Retailer Ltd",
        "customer_name": "Customer",
        "gstin": "27AAAAA1111A1Z5",
        "category": "Retail",
        "classification": "Sales",
        "amount": 100.0,
        "discount": 0.0,
        "tax": 18.0,
        "total": 118.0,
        "payment_method": "Cash",
        "ocr_quality": "good",
        "items": [
            {
                "product": "Standard Item",
                "hsn_sac": "8471",
                "quantity": 1,
                "unit_price": 100.0,
                "amount": 100.0
            }
        ],
        "audit_flags": []
    }
    return _normalise_invoice_data(fallback_data, text)


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


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

4. Extract line items if present, including the product name, description, HSN/SAC code if present, quantity, unit price, tax, and total amount.

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
"description": "",
"hsn_sac": "",
"quantity": 0,
"unit_price": 0,
"tax": 0,
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
BUSINESS_CLASSIFICATIONS = {"Purchase", "Sales", "Expense"}
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

GSTIN_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")


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
    product = str(item.get("product") or item.get("item") or item.get("name") or "").strip()
    description = str(item.get("description") or item.get("details") or "").strip()
    if not product:
        product = description
    if not description:
        description = product

    return {
        "product": product,
        "description": description,
        "hsn_sac": str(item.get("hsn_sac") or "").strip(),
        "quantity": _to_number(item.get("quantity")),
        "unit_price": _to_number(item.get("unit_price")),
        "tax": _to_number(item.get("tax") or item.get("tax_amount") or item.get("gst")),
        "amount": _to_number(item.get("amount")),
    }


def _normalise_business_classification(value: object) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "purchase": "Purchase",
        "purchases": "Purchase",
        "procurement": "Purchase",
        "sales": "Sales",
        "sale": "Sales",
        "revenue": "Sales",
        "income": "Sales",
        "expense": "Expense",
        "expenses": "Expense",
        "operating expense": "Expense",
        "opex": "Expense",
    }
    return aliases.get(text, "")


def _classify_invoice(data: dict, text: str) -> str:
    classification = _normalise_business_classification(data.get("classification"))
    if classification in BUSINESS_CLASSIFICATIONS:
        return classification

    category_classification = _normalise_business_classification(data.get("category"))
    if category_classification in BUSINESS_CLASSIFICATIONS:
        return category_classification

    business_type = _normalise_business_classification(data.get("business_type"))
    if business_type in BUSINESS_CLASSIFICATIONS:
        return business_type

    transaction_type = _normalise_business_classification(data.get("transaction_type"))
    if transaction_type in BUSINESS_CLASSIFICATIONS:
        return transaction_type

    classification = str(data.get("classification") or "").strip().title()
    if classification in BUSINESS_CLASSIFICATIONS:
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
        "sales invoice", "tax invoice to", "invoice to customer", "sales receipt",
        "our invoice", "issued to customer", "customer invoice", "receipt from customer",
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
    if _normalise_business_classification(category):
        category = document_category
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

        raw_amounts = _amounts_in_line(line)
        hsn_sac = ""
        hsn_match = re.search(r"\b(?:HSN|SAC)\s*[:#\-]?\s*([0-9]{4,8})\b", line, re.IGNORECASE)
        if hsn_match:
            hsn_sac = hsn_match.group(1)
        else:
            numeric_tokens = re.findall(r"\b[0-9]{4,8}\b", line)
            if len(raw_amounts) >= 4 and numeric_tokens and raw_amounts[0] == _to_number(numeric_tokens[0]):
                hsn_sac = numeric_tokens[0]

        amount_line = line
        if hsn_sac:
            amount_line = re.sub(rf"\b(?:HSN|SAC)?\s*{re.escape(hsn_sac)}\b", "", amount_line, flags=re.IGNORECASE)

        amounts = _amounts_in_line(amount_line)
        if not amounts:
            continue

        description = line
        if hsn_sac:
            description = re.sub(rf"\b(?:HSN|SAC)?\s*{re.escape(hsn_sac)}\b", "", description, flags=re.IGNORECASE)
        description = re.sub(_money_pattern(), "", description, flags=re.IGNORECASE)
        description = re.sub(r"\b(?:HSN|SAC)\b", "", description, flags=re.IGNORECASE)
        description = _clean_label_value(re.sub(r"\s{2,}", " ", description))
        if not description or len(description) < 3:
            continue

        quantity = 0.0
        unit_price = 0.0
        tax = 0.0
        if len(amounts) >= 4:
            quantity = amounts[-4]
            unit_price = amounts[-3]
            tax = amounts[-2]
        elif len(amounts) >= 3:
            quantity = amounts[-3]
            unit_price = amounts[-2]
        elif len(amounts) == 2:
            unit_price = amounts[-2]
        if tax <= 0 and ("tax" in lower_line or "gst" in lower_line):
            tax = amounts[-2] if len(amounts) >= 2 else 0.0

        items.append(
            {
                "product": description,
                "description": description,
                "hsn_sac": hsn_sac,
                "quantity": quantity,
                "unit_price": unit_price,
                "tax": tax,
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
    """Fill in missing amount/tax/total using the single business rule:
       taxable = amount - discount
       total   = taxable + tax
    """
    amount   = _to_number(invoice_data.get("amount"))
    discount = _to_number(invoice_data.get("discount"))
    tax      = _to_number(invoice_data.get("tax"))
    total    = _to_number(invoice_data.get("total"))

    taxable = max(round(amount - discount, 2), 0.0)

    # Derive missing fields — don't overwrite values already present
    if amount <= 0 and total > 0 and tax >= 0:
        # amount = total - tax + discount  (reverse the rule)
        invoice_data["amount"] = round(total - tax + discount, 2)

    if total <= 0 and amount > 0:
        invoice_data["total"] = round(taxable + tax, 2)

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


def _default_unit_price(product_name: str) -> float:
    name = product_name.lower()
    price_map = [
        (["samsung", "phone", "mobile"], 20000.0),
        (["iphone"], 70000.0),
        (["charger"], 500.0),
        (["earphone", "earbud", "headphone"], 1500.0),
        (["laptop"], 55000.0),
        (["keyboard"], 1000.0),
        (["mouse"], 600.0),
        (["monitor"], 12000.0),
        (["printer"], 9000.0),
    ]

    for keywords, price in price_map:
        if any(keyword in name for keyword in keywords):
            return price

    return 1000.0


def _singular_product_name(product_name: str) -> str:
    words = product_name.strip().split()
    if words and words[-1].lower().endswith("s") and len(words[-1]) > 3:
        words[-1] = words[-1][:-1]
    return " ".join(words).title()


def _find_natural_customer(text: str) -> str:
    match = re.search(r"\b([A-Z][a-zA-Z]+)\s+(?:bought|purchased|ordered|took)\b", text)
    if match:
        return match.group(1)

    match = re.search(r"\b(?:customer|buyer|client)\s*[:\-]\s*([A-Za-z][A-Za-z ]+)", text, re.IGNORECASE)
    if match:
        return _clean_label_value(match.group(1)).title()

    return "Customer"


def _generate_gstin(vendor_name: str = "") -> str:
    """Generate a realistic, valid-format Indian GSTIN.

    Format: SS AAAAA NNNN A N Z C  (15 chars, no spaces)
      SS   — 2-digit state code (01-37)
      AAAAA — 5 uppercase alpha chars derived from vendor name
      NNNN  — 4 digits
      A     — 1 alpha char
      N     — 1 alphanumeric
      Z     — literal 'Z'
      C     — 1 alphanumeric check character

    The output is deterministic for a given vendor name so the same
    vendor always gets the same generated GSTIN across runs.
    """
    import hashlib

    seed = vendor_name.upper().strip() or "DEFAULT VENDOR"
    digest = hashlib.md5(seed.encode()).hexdigest().upper()  # 32 hex chars

    # State code: 01-37, derived from digest
    hex_state = int(digest[:4], 16) % 37 + 1
    state_code = f"{hex_state:02d}"

    # 5-letter PAN-style prefix from vendor name initials / letters
    letters_only = re.sub(r"[^A-Z]", "", seed)
    # Pad or trim to exactly 5 uppercase alpha chars
    while len(letters_only) < 5:
        letters_only += digest[len(letters_only) % 32]
    pan_alpha = letters_only[:5]

    # 4-digit number
    digits_4 = f"{int(digest[5:9], 16) % 10000:04d}"

    # Single alpha (entity type)
    entity_alpha = chr(ord("A") + int(digest[9], 16) % 26)

    # Alphanumeric check char
    alphanum = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    check1 = alphanum[int(digest[10], 16) % len(alphanum)]

    # Literal Z + final check digit
    check2 = alphanum[int(digest[11], 16) % len(alphanum)]

    return f"{state_code}{pan_alpha}{digits_4}{entity_alpha}{check1}Z{check2}"
    match = re.search(r"\b([A-Z][a-zA-Z]+)\s+(?:bought|purchased|ordered|took)\b", text)
    if match:
        return match.group(1)

    match = re.search(r"\b(?:customer|buyer|client)\s*[:\-]\s*([A-Za-z][A-Za-z ]+)", text, re.IGNORECASE)
    if match:
        return _clean_label_value(match.group(1)).title()

    return "Customer"


def _find_gst_rate(text: str) -> float:
    match = re.search(r"\b(?:gst|tax)\s*(?:rate)?\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*%?", text, re.IGNORECASE)
    if match:
        return _to_number(match.group(1))
    return 18.0


def _extract_hsn_sac(text: str) -> str:
    match = re.search(r"\b(?:HSN/SAC|HSN|SAC)\s*[:#\-]?\s*([A-Za-z0-9]{4,8})\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else ""


def _extract_unit_price(text: str) -> float:
    price_patterns = [
        r"(?:unit\s*price|price|rate|@)\s*[:\-]?\s*(?:rs\.?|inr)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        r"(?:rs\.?|inr)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(?:each|per\s*unit|/unit)?",
        r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)\s*(?:each|per\s*unit|/unit)",
    ]

    for pattern in price_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _to_number(match.group(1))

    return 0.0


def _clean_natural_product_name(text: str, hsn_sac: str) -> str:
    product = text
    if hsn_sac:
        product = re.sub(rf"\b(?:HSN/SAC|HSN|SAC)\s*[:#\-]?\s*{re.escape(hsn_sac)}\b", "", product, flags=re.IGNORECASE)

    product = re.sub(r"(?:unit\s*price|price|rate|@)\s*[:\-]?\s*(?:rs\.?|inr)?\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?", "", product, flags=re.IGNORECASE)
    product = re.sub(r"(?:rs\.?|inr)\s*[0-9][0-9,]*(?:\.[0-9]{1,2})?\s*(?:each|per\s*unit|/unit)?", "", product, flags=re.IGNORECASE)
    product = re.sub(r"\b[0-9][0-9,]*(?:\.[0-9]{1,2})?\s*(?:each|per\s*unit|/unit)\b", "", product, flags=re.IGNORECASE)
    product = re.sub(r"\b(?:gst|tax)\s*(?:rate)?\s*[:\-]?\s*[0-9]+(?:\.[0-9]+)?\s*%?\b", "", product, flags=re.IGNORECASE)
    product = re.sub(r"\b(?:and|with|plus)$", "", product, flags=re.IGNORECASE)

    return _clean_label_value(product)


def _looks_like_placeholder_item(item: dict) -> bool:
    product = str(item.get("product") or item.get("description") or "").strip().lower()
    return product in {"", "item", "product", "standard item", "product/item name"}


def _parse_natural_language_items(text: str, gst_rate: float) -> list[dict]:
    """Extract line items from a natural-language transaction description.

    Handles three input patterns:
    1. "<qty> <product> [at Rs <price>]"   — "2 Samsung phones at Rs 20000"
    2. "<product> [at/worth] Rs <price>"   — "web development services Rs 85000"
    3. Structured label block:
           Samsung Galaxy M06
           Qty: 2
           Price: 10499
    """
    items = []
    stop_words = {
        "gst", "tax", "discount", "cash", "upi", "card", "invoice", "bill",
        "from", "for", "to", "and", "with", "at", "rs", "inr",
    }

    seen_products: set[str] = set()

    def _add_item(product: str, qty: float, unit_price: float) -> None:
        product_words = [w for w in product.split() if w.lower() not in stop_words]
        cleaned = " ".join(product_words).strip()
        if not cleaned or len(cleaned) < 2:
            return
        key = cleaned.lower()
        if key in seen_products:
            return
        seen_products.add(key)

        if unit_price <= 0:
            unit_price = _default_unit_price(cleaned)
        if qty <= 0:
            qty = 1.0

        taxable_amount = round(qty * unit_price, 2)
        tax_amount = round(taxable_amount * gst_rate / 100, 2)
        product_name = _singular_product_name(cleaned)
        items.append({
            "product": product_name,
            "description": product_name,
            "hsn_sac": "",
            "quantity": qty,
            "unit_price": unit_price,
            "tax": tax_amount,
            "amount": taxable_amount,
        })

    # ── Pattern 3: structured label block (try first — highest precision) ──
    # Matches blocks like:
    #   Product Name\nQty: 2\nPrice: 10499
    #   Product Name\nQuantity: 2\nRate: 10499
    structured_pattern = re.compile(
        r"^([A-Za-z][A-Za-z0-9 /&().:#@\-]+?)\s*\n"   # product name on its own line
        r"(?:qty|quantity)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*\n"  # Qty: N
        r"(?:price|rate|unit\s*price|cost)\s*[:\-]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",  # Price: P
        re.IGNORECASE | re.MULTILINE,
    )
    for match in structured_pattern.finditer(text):
        product = _clean_label_value(match.group(1))
        qty = _to_number(match.group(2))
        price = _to_number(match.group(3))
        if product:
            _add_item(product, qty, price)

    if items:
        return items

    # ── Pattern 4: inline Qty/Price labels on the same line ──────────
    # Matches: "Samsung Galaxy S25   Qty:2   Price:72000"
    #          "USB Charger Qty: 1 Price: 500"
    #          "Laptop Stand  qty:3  rate:1499"
    inline_pattern = re.compile(
        r"^([A-Za-z][A-Za-z0-9 /&().:#@\-]{1,}?)\s+"   # product name
        r"(?:qty|quantity)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s+"  # Qty:N
        r"(?:price|rate|unit\s*price|cost)\s*[:\-]?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",  # Price:P
        re.IGNORECASE | re.MULTILINE,
    )
    for match in inline_pattern.finditer(text):
        product = _clean_label_value(match.group(1))
        qty = _to_number(match.group(2))
        price = _to_number(match.group(3))
        if product:
            _add_item(product, qty, price)

    if items:
        return items

    # ── Pattern 1: leading quantity — "2 Samsung phones at Rs 20000" ──
    normalised_text = re.sub(r"[\r\n]+", "; ", text)
    pattern_qty_first = re.compile(
        r"\b([0-9]+(?:\.[0-9]+)?)\s+([A-Za-z][A-Za-z0-9 /&().:#@-]*?)(?=(?:[;,]|\band\b)\s*[0-9]+(?:\.[0-9]+)?\s+[A-Za-z]|\s+\b(?:gst|tax|discount|total|payment|paid)\b|$)",
        re.IGNORECASE,
    )
    for match in pattern_qty_first.finditer(normalised_text):
        quantity = _to_number(match.group(1))
        raw_product = _clean_label_value(match.group(2))
        hsn_sac = _extract_hsn_sac(raw_product)
        unit_price = _extract_unit_price(raw_product)
        product = _clean_natural_product_name(raw_product, hsn_sac)
        if product and quantity > 0:
            _add_item(product, quantity, unit_price)

    if items:
        return items

    # ── Pattern 2: product name then price — "services Rs 85000" ──
    pattern_name_price = re.compile(
        r"([A-Za-z][A-Za-z0-9 /&().:#@-]{3,}?)\s*(?:[-–@]|at|worth|for|:)?\s*(?:rs\.?|inr)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
        re.IGNORECASE,
    )
    for match in pattern_name_price.finditer(normalised_text):
        raw_product = _clean_label_value(match.group(1))
        price = _to_number(match.group(2))
        skip_words = {"gst", "tax", "total", "amount", "discount", "rs", "inr", "payment"}
        if raw_product.lower() in skip_words or len(raw_product) < 3:
            continue
        product = _clean_natural_product_name(raw_product, "")
        _add_item(product, 1.0, price)

    return items


def parse_natural_language_invoice_with_rules(text: str) -> dict:
    today_str = datetime.today().strftime("%Y-%m-%d")
    gst_rate = _find_gst_rate(text)
    items = _parse_natural_language_items(text, gst_rate)
    customer = _find_natural_customer(text)

    # Business rule: GST applied on post-discount taxable amount
    subtotal = round(sum(_to_number(item.get("amount")) for item in items), 2)
    discount = 0.0
    taxable_after_discount = round(subtotal - discount, 2)
    gst_amount = round(taxable_after_discount * gst_rate / 100, 2)
    grand_total = round(taxable_after_discount + gst_amount, 2)

    fallback_data = {
        "document_type": "retail_invoice",
        "invoice_no": "INV-" + datetime.today().strftime("%Y%m%d%H%M%S"),
        "date": today_str,
        "vendor": "AI Generated Store",
        "customer_name": customer,
        "gstin": _generate_gstin("AI Generated Store"),
        "category": "Retail",
        "classification": "Sales",
        "amount": subtotal,
        "discount": discount,
        "tax": gst_amount,
        "total": grand_total,
        "payment_method": "Cash",
        "ocr_quality": "good",
        "items": items,
        "audit_flags": [],
    }
    return _normalise_invoice_data(fallback_data, text)


NATURAL_LANGUAGE_INVOICE_PROMPT = """You are an expert AI Invoice Generation Assistant for Indian businesses.

A user has described a business transaction in plain English. Your job is to understand the intent,
extract all relevant details, fill in any missing information intelligently, calculate all financial
figures correctly, and return a complete, ready-to-use invoice in JSON format.

## REFUSAL RULE — read this first
If the input does NOT describe a real business transaction and is missing ALL of the following
essential fields, you MUST refuse and return the refusal JSON below instead of inventing an invoice:

  Essential fields (at least ONE product/service is always required):
  - vendor  OR  a business name that is selling something
  - at least one product or service being sold/purchased
  - at least one quantity or price (even approximate)

If any combination of the above can be reasonably inferred from the text, proceed normally.
Only refuse when the input is completely uninformative (e.g. "hello", "test", "what is GST?",
a random sentence with no transaction context).

Refusal JSON format (return ONLY this, no other keys):
{{"status": "__INSUFFICIENT_DATA__", "missing": ["<field1>", "<field2>"], "suggestion": "<one sentence telling the user what to add>"}}

## Accepted input formats
The user may describe a transaction in ANY of these formats. Understand all of them equally well:

### Format A — Natural language
"TechMart Electronics sold 2 Samsung Galaxy S25 phones at ₹72,000 each and 1 charger at ₹500 to Rahul Sharma. GST 18%. Payment by UPI."

### Format B — Structured labels (multi-line)
Vendor: TechMart Electronics
Customer: Rahul Sharma
Samsung Galaxy S25   Qty: 2   Price: 72000
USB Charger          Qty: 1   Price: 500
GST: 18%
Payment: UPI

### Format C — Mixed / semi-structured
Vendor: TechMart Electronics Pvt. Ltd.
Customer: Rahul Sharma
Products:
Samsung Galaxy S25
Qty: 2
Price: 72000
USB Charger
Qty: 1
Price: 500
Discount: 1000
GST: 18%
Payment: Credit Card

In all cases, extract vendor, customer, each product with quantity and price, GST rate, discount, and payment method.
- Identify products/services, quantities, and prices
- Detect the GST rate (default 18% if not mentioned)
- Infer the type of transaction (Sales / Purchase / Expense)
- Infer the document category (Retail / Services / Food / Healthcare / etc.)

## Rules for filling missing information
- If invoice number is missing: generate one like "INV-{{year}}-{{4-digit-random}}" (e.g. INV-2026-4821)
- If date is missing: use today's date in YYYY-MM-DD format
- If vendor is missing but context implies a shop/store: infer a realistic business name (e.g. "Tech Hub Electronics", "City Medical Store")
- If customer name is missing: extract from sentence subject (e.g. "Rahul bought..." → customer = "Rahul") or use "Walk-in Customer"
- If unit price is missing: use realistic Indian market prices for the item
- If GSTIN is not mentioned but GST is charged: generate a realistic 15-character Indian GSTIN that matches the vendor name (e.g. for "TechMart Electronics Pvt. Ltd." use "27TECME1234P1Z5", for "BuildPro Construction" use "29BUICP5678K1Z3"). Always follow the format: 2-digit state code + 5 alpha PAN prefix from vendor initials + 4 digits + 1 alpha + 1 alphanumeric + Z + 1 alphanumeric
- If payment method is missing: infer from context or use "Cash"

## Financial calculation rules (CRITICAL - always follow exactly)
For EACH line item:
  item_taxable_amount = quantity × unit_price
  item_tax = round(item_taxable_amount × gst_rate / 100, 2)
  item_amount = item_taxable_amount  (taxable amount before tax, NOT including tax)

For the invoice totals:
  amount (subtotal) = sum of all item_amount values
  discount = as mentioned, else 0
  taxable_after_discount = amount - discount
  tax = sum of all item_tax values (or taxable_after_discount × gst_rate / 100 if computed at invoice level)
  total (grand total) = taxable_after_discount + tax

IMPORTANT: item.amount = quantity × unit_price (before tax). Do NOT add tax to item.amount.

## Document type inference
Pick the best match from: gst_invoice, retail_invoice, service_invoice, restaurant_invoice,
hospital_invoice, hotel_invoice, travel_invoice, utility_invoice, amazon_invoice, flipkart_invoice

## Classification rules
- Sales: user/store is selling to a customer
- Purchase: buying from a supplier/vendor for resale or production
- Expense: operational costs (food, travel, utilities, medical, hotel)

Today's date: {today}

## Input description from user:
{text}

## Output
Return ONLY valid JSON. No explanation. No markdown. No code fences.
If the input is insufficient, return the refusal JSON shown above.
Otherwise return the full invoice JSON below.

{{
  "document_type": "",
  "invoice_no": "",
  "date": "",
  "vendor": "",
  "customer_name": "",
  "gstin": "",
  "category": "",
  "classification": "Sales",
  "amount": 0,
  "discount": 0,
  "tax": 0,
  "total": 0,
  "payment_method": "",
  "ocr_quality": "good",
  "items": [
    {{
      "product": "",
      "description": "",
      "hsn_sac": "",
      "quantity": 1,
      "unit_price": 0,
      "tax": 0,
      "amount": 0
    }}
  ],
  "audit_flags": []
}}"""


class InsufficientInvoiceDataError(ValueError):
    """Raised when the LLM determines the input lacks enough data to build an invoice."""
    def __init__(self, missing: list[str], suggestion: str) -> None:
        self.missing    = missing      # list of missing field names
        self.suggestion = suggestion   # one-sentence human-readable hint
        super().__init__(f"Insufficient invoice data. Missing: {missing}")


def parse_natural_language_invoice(text: str) -> dict:
    today_str = datetime.today().strftime("%Y-%m-%d")

    if os.getenv("GROQ_API_KEY"):
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model=GROQ_EXTRACTION_MODEL, temperature=0)

            prompt = NATURAL_LANGUAGE_INVOICE_PROMPT.format(
                text=text,
                today=today_str,
            )
            response = llm.invoke(prompt)

            # Log raw response for debugging
            raw_content = response.content
            print(f"[NL Invoice] LLM raw response (first 400 chars): {raw_content[:400]}")

            invoice_data = _load_llm_json(raw_content)

            # ── Refusal sentinel: LLM decided input is insufficient ──────
            if invoice_data.get("status") == "__INSUFFICIENT_DATA__":
                missing    = invoice_data.get("missing", [])
                suggestion = invoice_data.get("suggestion", "Please provide more details.")
                print(f"[NL Invoice] LLM refused — missing: {missing}")
                raise InsufficientInvoiceDataError(missing, suggestion)
            normalised_data = _normalise_invoice_data(invoice_data, text)

            # ── Step 1: collect valid items ──────────────────────────────
            llm_items = normalised_data.get("items") if isinstance(normalised_data.get("items"), list) else []
            valid_items = [
                item for item in llm_items
                if isinstance(item, dict) and not _looks_like_placeholder_item(item)
            ]

            print(f"[NL Invoice] LLM items count: {len(llm_items)}, valid: {len(valid_items)}")

            # ── Step 2: heal each item — recompute amount/tax when zero ─────
            gst_rate = _find_gst_rate(text)
            healed_items: list[dict] = []
            for item in valid_items:
                qty        = _to_number(item.get("quantity"))
                unit_price = _to_number(item.get("unit_price"))
                item_amount = _to_number(item.get("amount"))

                if unit_price <= 0:
                    unit_price = _default_unit_price(item.get("product") or item.get("description") or "")
                    item["unit_price"] = unit_price

                if qty <= 0:
                    qty = 1.0
                    item["quantity"] = qty

                # item.amount = qty × unit_price  (pre-tax subtotal per line)
                taxable = round(qty * unit_price, 2)
                item["amount"] = taxable

                # item.tax = item.amount × gst_rate / 100
                item["tax"] = round(taxable * gst_rate / 100, 2)

                healed_items.append(item)

            if healed_items:
                # ── Step 3: invoice totals — GST on post-discount taxable amount ──
                subtotal = round(sum(_to_number(i.get("amount")) for i in healed_items), 2)
                discount = _to_number(normalised_data.get("discount"))
                taxable_after_discount = round(subtotal - discount, 2)
                gst_amount = round(taxable_after_discount * gst_rate / 100, 2)
                grand_total = round(taxable_after_discount + gst_amount, 2)

                normalised_data["items"]    = healed_items
                normalised_data["amount"]   = subtotal          # pre-discount subtotal
                normalised_data["discount"] = discount
                normalised_data["tax"]      = gst_amount        # GST on taxable amount
                normalised_data["total"]    = grand_total
            else:
                # ── Fallback: LLM returned no usable items — use rule parser
                print("[NL Invoice] No valid LLM items, falling back to rule parser")
                rule_items = _parse_natural_language_items(text, gst_rate)
                print(f"[NL Invoice] Rule parser found {len(rule_items)} items")
                if rule_items:
                    subtotal = round(sum(_to_number(i.get("amount")) for i in rule_items), 2)
                    discount = _to_number(normalised_data.get("discount"))
                    taxable_after_discount = round(subtotal - discount, 2)
                    gst_amount = round(taxable_after_discount * gst_rate / 100, 2)
                    grand_total = round(taxable_after_discount + gst_amount, 2)

                    normalised_data["items"]    = rule_items
                    normalised_data["amount"]   = subtotal
                    normalised_data["discount"] = discount
                    normalised_data["tax"]      = gst_amount
                    normalised_data["total"]    = grand_total

            # ── Step 4: ensure invoice_no, date, vendor, gstin are always present
            if not normalised_data.get("invoice_no"):
                normalised_data["invoice_no"] = "INV-" + datetime.today().strftime("%Y%m%d%H%M%S")
            if not normalised_data.get("date"):
                normalised_data["date"] = today_str
            if not normalised_data.get("vendor"):
                normalised_data["vendor"] = "AI Generated Store"
            # Generate a vendor-specific GSTIN when the LLM left it blank or used a placeholder
            vendor = normalised_data.get("vendor", "")
            gstin  = str(normalised_data.get("gstin") or "").strip().upper()
            _placeholder_gstins = {"27AAAAA1111A1Z5", "22AAAAA0000A1Z5", ""}
            if gstin in _placeholder_gstins or not GSTIN_PATTERN.match(gstin):
                normalised_data["gstin"] = _generate_gstin(vendor)

            # ── Step 5: re-run audit flags with correct totals ───────────
            existing = [
                f for f in normalised_data.get("audit_flags", [])
                if f not in {"missing_total", "tax_mismatch", "invalid_document", "suspicious_amount"}
            ]
            normalised_data["audit_flags"] = existing
            for flag in _audit_flags(normalised_data):
                if flag not in existing:
                    existing.append(flag)
            normalised_data["audit_flags"] = existing

            print(f"[NL Invoice] Final: items={len(normalised_data.get('items', []))}, total={normalised_data.get('total')}")
            return normalised_data

        except InsufficientInvoiceDataError:
            raise  # propagate to caller — do NOT fall through to rule parser
        except Exception as error:
            print(f"[NL Invoice] Groq generation failed: {error}")
            import traceback
            traceback.print_exc()

    print("[NL Invoice] Falling back to rule-based parser")
    return parse_natural_language_invoice_with_rules(text)


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

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, date

import database
from database import (
    save_invoice_record,
    get_invoice_by_number,
    update_invoice_record,
    get_all_invoices,
    search_invoices,
)

from audit import audit_invoice, create_audit_summary
from chat import answer_audit_question
from ocr import extract_text_from_file
from parser import (
    parse_invoice_text,
    parse_natural_language_invoice,
    save_csv,
    save_json,
    is_valid_invoice,
    _classify_invoice,
    InsufficientInvoiceDataError,
)




load_dotenv()

UPLOADS_DIR = Path("data/uploads")


def find_duplicate_invoice(record: dict) -> dict | None:
    if hasattr(database, "find_duplicate_invoice"):
        return database.find_duplicate_invoice(record)

    invoice_no = str(record.get("invoice_no") or "").strip()
    if invoice_no and database.check_duplicate_invoice(invoice_no):
        return database.get_invoice_by_number(invoice_no)

    vendor = str(record.get("vendor") or "").strip()
    invoice_date = str(record.get("date") or "").strip()

    try:
        amount = float(record.get("amount") or 0)
        total = float(record.get("total") or 0)
    except (TypeError, ValueError):
        return None

    if not vendor or not invoice_date or amount <= 0 or total <= 0:
        return None

    for invoice in database.get_all_invoices():
        try:
            invoice_amount = float(invoice.get("amount") or 0)
            invoice_total = float(invoice.get("total") or 0)
        except (TypeError, ValueError):
            continue

        if (
            invoice.get("vendor") == vendor
            and invoice.get("date") == invoice_date
            and invoice_amount == amount
            and invoice_total == total
        ):
            return invoice

    return None


def save_text(text: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def has_invoice_keywords(raw_text: str) -> bool:
    keywords = [
        "invoice",
        "bill",
        "receipt",
        "gstin",
        "tax",
        "subtotal",
        "total",
        "amount due",
        "grand total",
        "vendor",
        "supplier",
        "seller",
        "customer",
        "patient",
        "guest",
    ]
    text = raw_text.lower()
    return any(keyword in text for keyword in keywords)


def invoice_validation_errors(invoice_data: dict, raw_text: str) -> list[str]:
    errors = []
    document_type = str(invoice_data.get("document_type") or "").strip().lower()
    invoice_no = str(invoice_data.get("invoice_no") or "").strip()
    date = str(invoice_data.get("date") or "").strip()
    vendor = str(invoice_data.get("vendor") or "").strip()

    try:
        total = float(invoice_data.get("total") or 0)
    except (TypeError, ValueError):
        total = 0

    if not raw_text.strip():
        errors.append("No readable text was found in the uploaded file.")

    if document_type == "other" and not has_invoice_keywords(raw_text):
        errors.append("The uploaded file does not appear to be an invoice or bill.")

    if not vendor:
        errors.append("Vendor or supplier name could not be detected.")

    if not invoice_no and not date:
        errors.append("Invoice number or invoice date could not be detected.")

    if total <= 0:
        errors.append("Invoice total could not be detected.")

    return errors


def run_audit_pipeline(file_path: str) -> tuple[dict, dict | None, str, list[str]]:
    raw_text = extract_text_from_file(file_path)
    invoice_data = parse_invoice_text(raw_text)
    validation_errors = invoice_validation_errors(invoice_data, raw_text)

    if validation_errors:
        return invoice_data, None, "", validation_errors

    # Check if the extracted data represents a valid invoice
    if not is_valid_invoice(invoice_data):
        # Add invalid_document flag
        if "audit_flags" not in invoice_data:
            invoice_data["audit_flags"] = []
        if "invalid_document" not in invoice_data["audit_flags"]:
            invoice_data["audit_flags"].append("invalid_document")
        
        return invoice_data, None, "", ["Uploaded file does not appear to be a valid invoice."]

    audit_result = audit_invoice(invoice_data)
    audit_summary = create_audit_summary(invoice_data, audit_result)

    save_text(raw_text, "outputs/raw_ocr_text.txt")
    save_json(invoice_data, "outputs/extracted_data.json")
    save_csv(invoice_data, "outputs/extracted_data.csv")
    save_json(audit_result, "outputs/audit_report.json")
    save_text(audit_summary, "outputs/audit_report.txt")

    return invoice_data, audit_result, audit_summary, []


def run_natural_language_pipeline(text: str) -> tuple[dict, dict | None, str, list[str]]:
    """Parse NL text → validate → audit only if valid."""
    try:
        invoice_data = parse_natural_language_invoice(text)
    except InsufficientInvoiceDataError as e:
        # LLM explicitly refused — return field-level errors and a suggestion
        missing_labels = {
            "vendor":   "Vendor / seller name",
            "customer": "Customer / buyer name",
            "products": "At least one product or service",
            "quantity": "Quantity for each item",
            "price":    "Price or amount for each item",
        }
        errors = [
            f"Missing: {missing_labels.get(f, f.replace('_', ' ').title())}"
            for f in e.missing
        ]
        if e.suggestion:
            errors.append(f"💡 {e.suggestion}")
        return {}, None, "", errors

    errors: list[str] = []

    if not invoice_data.get("vendor"):
        errors.append("Vendor name could not be determined from the description.")

    items = invoice_data.get("items") if isinstance(invoice_data.get("items"), list) else []
    valid_items = [i for i in items if isinstance(i, dict) and (i.get("product") or i.get("description"))]
    if not valid_items:
        errors.append(
            "No line items could be extracted. "
            "Make sure your description includes product names "
            "(e.g. 'Samsung phone', 'consulting services') and optionally quantities and prices."
        )

    try:
        total = float(invoice_data.get("total") or 0)
    except (TypeError, ValueError):
        total = 0.0
    if total <= 0 and not errors:
        errors.append("Invoice total is zero — try including prices, e.g. 'Samsung phone at Rs 20000'.")

    if errors:
        return invoice_data, None, "", errors

    audit_result = audit_invoice(invoice_data)
    audit_summary = create_audit_summary(invoice_data, audit_result)

    save_text(text, "outputs/generated_invoice_prompt.txt")
    save_json(invoice_data, "outputs/extracted_data.json")
    save_csv(invoice_data, "outputs/extracted_data.csv")
    save_json(audit_result, "outputs/audit_report.json")
    save_text(audit_summary, "outputs/audit_report.txt")

    return invoice_data, audit_result, audit_summary, []


def save_uploaded_file(uploaded_file) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOADS_DIR / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())

    return file_path


def persist_invoice(invoice_data: dict, audit_result: dict) -> tuple[dict, dict, str, str]:
    invoice_no = (
        invoice_data.get("invoice_no")
        or invoice_data.get("invoice_number")
        or ""
    )

    invoice_date = (
        invoice_data.get("date")
        or invoice_data.get("invoice_date")
        or ""
    )

    record = {
        "document_type": invoice_data.get("document_type", ""),
        "invoice_no": invoice_no,
        "date": invoice_date,
        "vendor": invoice_data.get("vendor", ""),
        "customer_name": invoice_data.get("customer_name", ""),
        "gstin": invoice_data.get("gstin", ""),
        "category": invoice_data.get("category", ""),
        "classification": invoice_data.get("classification", "Expense"),
        "amount": invoice_data.get("amount", 0),
        "discount": invoice_data.get("discount", 0),
        "tax": invoice_data.get("tax", 0),
        "total": invoice_data.get("total", 0),
        "payment_method": invoice_data.get("payment_method", ""),
        "ocr_quality": invoice_data.get("ocr_quality", ""),
        "items": invoice_data.get("items", []),
        "audit_flags": invoice_data.get("audit_flags", []),
        "audit_status": audit_result.get("status"),
        "risk_score": audit_result.get("risk_score", 100),
        "issue_count": audit_result.get("issue_count", 0),
        "issues": audit_result.get("issues", []),
        "debugging_info": audit_result.get("debugging_info", {}),
        "created_at": datetime.utcnow(),
    }

    duplicate = find_duplicate_invoice(record)
    if duplicate:
        audit_flags = invoice_data.setdefault("audit_flags", [])
        if "duplicate_candidate" not in audit_flags:
            audit_flags.append("duplicate_candidate")

        # Build an informative message about how the duplicate was detected
        dup_inv_no = str(duplicate.get("invoice_no") or "").strip()
        dup_vendor = str(duplicate.get("vendor") or "").strip()
        dup_date   = str(duplicate.get("date") or "").strip()

        if dup_inv_no and dup_inv_no == invoice_no:
            dup_reason = f"Invoice number **{dup_inv_no}** already exists in the database."
        elif dup_vendor and dup_date:
            dup_reason = (
                f"A transaction from **{dup_vendor}** on **{dup_date}** "
                f"with the same amount was already recorded."
            )
        else:
            dup_reason = (
                f"Same vendor, customer, and line items already exist in the database "
                f"(content fingerprint match — possible re-submission or fraud attempt)."
            )

        audit_result = audit_invoice(invoice_data)
        audit_summary = create_audit_summary(invoice_data, audit_result)
        save_json(invoice_data, "outputs/extracted_data.json")
        save_json(audit_result, "outputs/audit_report.json")
        save_text(audit_summary, "outputs/audit_report.txt")
        return invoice_data, audit_result, audit_summary, f"⚠ Duplicate detected — {dup_reason}"

    save_invoice_record(record)
    return invoice_data, audit_result, create_audit_summary(invoice_data, audit_result), "Invoice saved successfully."


def status_label(status: str) -> str:
    if status == "passed":
        return "Passed"

    if status == "failed":
        return "Failed"

    if status == "waiting":
        return "Waiting"

    return "Warning"


def status_icon(status: str) -> str:
    if status == "passed":
        return "&#9989;"

    if status == "failed":
        return "&#10060;"

    if status == "waiting":
        return ""

    return "&#9888;"


def status_tone(status: str) -> str:
    if status == "passed":
        return "passed"

    if status == "failed":
        return "failed"

    if status == "waiting":
        return "waiting"

    return "warning"


def decision_text(audit_result: dict) -> str:
    if audit_result.get("status") == "passed":
        return "No financial inconsistencies detected."

    return "Manual review required before approval."


def _render_high_value_card(message: str) -> None:
    """Render the high-value invoice warning card with badge, coloured metrics, and action list."""
    parts     = [p.strip() for p in message.split(" | ")]
    title_txt = parts[0] if len(parts) > 0 else "High Value Invoice"
    inv_amt   = parts[1].split(": ", 1)[-1] if len(parts) > 1 else ""
    threshold = parts[2].split(": ", 1)[-1] if len(parts) > 2 else ""

    # ── Badge (replaces the large olive st.warning container) ────────
    st.markdown("🟡 **HIGH VALUE INVOICE**")

    with st.container(border=True):
        c1, c2 = st.columns(2)
        # Colour the invoice amount amber to signal the trigger
        c1.markdown("**Invoice Amount**")
        c1.markdown(
            f"<span style='color:#f59e0b; font-size:1.3rem; font-weight:800;'>{inv_amt}</span>",
            unsafe_allow_html=True,
        )
        c2.markdown("**Configured Threshold**")
        c2.markdown(
            f"<span style='color:#94a3b8; font-size:1.3rem; font-weight:700;'>{threshold}</span>",
            unsafe_allow_html=True,
        )

        st.caption("📋 Invoices above this limit require manual approval before processing.")

        # Recommended actions
        st.markdown("**📌 Recommended Action**")
        for action in [
            "Verify Purchase Order matches this invoice",
            "Obtain Manager / Finance approval",
            "Verify Vendor details and GSTIN",
        ]:
            st.markdown(f"• {action}")


def render_issue_list(audit_result: dict) -> None:
    """Render audit issues with coloured severity badges."""
    issues = audit_result.get("issues", [])
    if not issues:
        st.success("✅ No audit issues detected.")
        return
    for issue in issues:
        severity = str(issue.get("severity", "medium")).lower()
        field    = issue.get("field", "")
        message  = issue.get("message", "")

        if field == "high_value":
            _render_high_value_card(message)
            continue

        msg = f"• {message}"
        if severity == "high":
            st.error(msg)
        elif severity == "medium":
            st.warning(msg)
        else:
            st.info(msg)


def render_issue_html(audit_result: dict) -> str:
    """Legacy helper kept for the upload-audit status card which uses unsafe_allow_html.
    Only called inside the upload-audit section — not for generated invoices."""
    issues = audit_result.get("issues", [])

    if not issues:
        return '<div class="issue-ok">No audit issues were found.</div>'

    issue_lines = []
    for issue in issues:
        issue_lines.append(f'<div class="issue-row">&#8226; {issue["message"]}</div>')

    return "".join(issue_lines)


def risk_score(audit_result: dict) -> int:
    if "risk_score" in audit_result:
        return int(audit_result["risk_score"])

    score = 100

    for issue in audit_result.get("issues", []):
        severity = issue.get("severity")

        if severity == "high":
            score -= 25
        elif severity == "medium":
            score -= 10
        else:
            score -= 5

    return max(score, 0)


def ai_summary_text(audit_result: dict, invoice_data: dict | None) -> str:
    if not invoice_data:
        return "Upload an invoice to generate an audit summary."

    if audit_result.get("status") == "passed":
        return (
            "The invoice was successfully validated. The total amount matches the "
            "tax calculation and all required fields are present."
        )

    issue_count = audit_result.get("issue_count", 0)
    return (
        f"The audit found {issue_count} issue(s). Review the highlighted findings "
        "before approving this invoice."
    )


def recommendation_text(audit_result: dict) -> str:
    if audit_result.get("status") == "passed":
        return "Recommendation: approve or move to the next review stage."

    return "Recommendation: hold approval until the listed issues are checked."


def money_value(value: object) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "0.00"


def money_label(value: object) -> str:
    return f"₹{money_value(value)}"


def title_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "Missing"

    return text.replace("_", " ").title()


def money_label(value: object) -> str:
    return f"Rs. {money_value(value)}"


def render_invoice_summary(invoice_data: dict) -> None:
    cls = str(invoice_data.get("classification") or "Expense").strip().title()
    cls_display = "🟠 Expense"
    if cls == "Purchase":
        cls_display = "🟢 Purchase"
    elif cls == "Sales":
        cls_display = "🔵 Sales"

    rows = [
        {"field": "document_type", "value": title_label(invoice_data.get("document_type"))},
        {"field": "invoice_no", "value": invoice_data.get("invoice_no") or "Missing"},
        {"field": "date", "value": invoice_data.get("date") or "Missing"},
        {"field": "vendor", "value": invoice_data.get("vendor") or "Missing"},
        {"field": "document_category", "value": invoice_data.get("category") or "Missing"},
        {"field": "business_classification", "value": cls_display},
        {"field": "amount", "value": money_label(invoice_data.get("amount"))},
        {"field": "tax", "value": money_label(invoice_data.get("tax"))},
        {"field": "total", "value": money_label(invoice_data.get("total"))},
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_line_items(invoice_data: dict) -> None:
    items = invoice_data.get("items") if isinstance(invoice_data.get("items"), list) else []

    # Hide section if no items exist
    if not items:
        return

    st.markdown("**Line Items**")
    rows = [
        {
            "Product": item.get("product") or item.get("description", "N/A"),
            "Description": item.get("description") or "N/A",
            "HSN/SAC": item.get("hsn_sac") or "N/A",
            "Qty": item.get("quantity", 0),
            "Unit Price": item.get("unit_price", 0),
            "Tax": item.get("tax", 0),
            "Amount": item.get("amount", 0),
        }
        for item in items
        if isinstance(item, dict)
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)



def flag_label(flag: str) -> str:
    labels = {
        "missing_invoice_no": "Invoice Number Missing",
        "missing_vendor": "Vendor Missing",
        "missing_date": "Invoice Date Missing",
        "missing_total": "Total Missing",
        "missing_gstin": "Missing GSTIN",
        "tax_mismatch": "Tax Mismatch",
        "suspicious_amount": "Suspicious Amount",
        "poor_ocr": "Poor OCR Quality",
        "duplicate_candidate": "Possible Duplicate Invoice",
    }
    return labels.get(flag, title_label(flag))


def render_audit_flags(invoice_data: dict) -> None:
    flags = invoice_data.get("audit_flags") if isinstance(invoice_data.get("audit_flags"), list) else []

    # Hide section if no flags exist
    if not flags:
        return

    st.markdown("**Audit Flags**")
    for flag in flags:
        st.warning(flag_label(str(flag)))


def render_risk_panel(invoice_data: dict | None, audit_result: dict) -> None:
    st.markdown("**Risk Panel**")
    st.write(f"Risk Score: {risk_score(audit_result)}/100")

    checks = [
        ("Invoice Number Found", bool(invoice_data and invoice_data.get("invoice_no"))),
        ("Vendor Found", bool(invoice_data and invoice_data.get("vendor"))),
        ("Customer Found", bool(invoice_data and invoice_data.get("customer_name"))),
        ("GSTIN Present", bool(invoice_data and invoice_data.get("gstin"))),
        ("Document Category Detected", bool(invoice_data and invoice_data.get("category"))),
        ("Business Classification Detected", bool(invoice_data and invoice_data.get("classification"))),
        ("Total Found", bool(invoice_data and invoice_data.get("total"))),
    ]

    for label, passed in checks:
        st.write(f"{'✓' if passed else '⚠'} {label if passed else label.replace(' Found', ' Missing').replace(' Present', ' Missing').replace(' Detected', ' Missing')}")

    for issue in audit_result.get("issues", []):
        if issue.get("field") in ["invoice_no", "vendor", "customer_name", "gstin", "category", "classification", "total"]:
            continue
        st.write(f"⚠ {issue.get('message')}")


def render_kpi_cards(audit_result: dict, invoice_data: dict | None) -> None:
    """Render KPI cards using native Streamlit metrics — no raw HTML."""
    total_amount = invoice_data.get("total") if invoice_data else 0
    score = risk_score(audit_result)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status_label(audit_result["status"]))
    c2.metric("Issues", audit_result["issue_count"])
    c3.metric("Risk Score", f"{score}/100")
    c4.metric("Amount", f"Rs. {money_value(total_amount)}")


def render_invoice_preview(uploaded_file) -> None:
    """Show uploaded file preview using native Streamlit — no raw HTML."""
    if uploaded_file:
        if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
            st.info(f"PDF uploaded: {uploaded_file.name}")
            return
        st.image(uploaded_file, use_container_width=True)
        return
    st.caption("No invoice preview yet.")


def render_chat() -> None:
    st.subheader("Ask the AI Invoice Audit Agent")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    _CHAT_HINTS = [
        "Why was this invoice flagged?",
        "Summarize this invoice.",
        "Explain the audit findings.",
        "Is this invoice suspicious?",
        "Why is the risk score high?",
        "Show all tax calculations.",
        "What issues were found?",
        "Is it safe to approve this invoice?",
    ]
    # Rotate placeholder based on message count so it changes after each reply
    hint = _CHAT_HINTS[len(st.session_state.messages) % len(_CHAT_HINTS)]

    question = st.chat_input(hint)

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing invoice..."):
                answer = answer_audit_question(question)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


st.set_page_config(page_title="AI Invoice Audit Agent", page_icon="AI", layout="centered")

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    body, .stApp, .main, .block-container, .css-1d391kg, .css-1v3fvcr, .css-k1vhr4 {
        background-color: #0b1221 !important;
        color: #e2e8f0 !important;
    }

    .block-container {
        max-width: 820px;
        padding-top: 24px;
        padding-bottom: 28px;
    }

    h1 {
        text-align: center;
        font-size: 2.15rem !important;
        font-weight: 760 !important;
        margin-bottom: 22px !important;
        letter-spacing: 0 !important;
        color: #f9fafb !important;
    }

    .section {
        border-top: 0;
        padding: 10px 0 18px;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 720;
        margin-bottom: 8px;
        color: #e2e8f0;
    }

    .status-card {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        padding: 14px;
        background: rgba(15, 23, 42, 0.95);
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.5);
        margin-bottom: 10px;
    }

    .analysis-hero {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.85);
        padding: 22px 18px 20px;
        text-align: center;
        margin-top: 8px;
    }

    .analysis-eyebrow {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
    }

    .hero-status {
        color: #f8fafc;
        font-size: 2.15rem;
        font-weight: 820;
        line-height: 1.1;
        margin-bottom: 8px;
    }

    .risk-score {
        color: #cbd5e1;
        font-size: 1rem;
        font-weight: 760;
        margin-bottom: 8px;
    }

    .hero-decision {
        color: #cbd5e1;
        font-size: 1rem;
        max-width: 520px;
        margin: 0 auto;
    }

    .status-card.passed {
        border-left: 5px solid #22c55e;
    }

    .status-card.warning {
        border-left: 5px solid #f59e0b;
    }

    .status-card.failed {
        border-left: 5px solid #ef4444;
    }

    .status-card.waiting {
        border-left: 5px solid #6b7280;
    }

    .status-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 8px;
    }

    .status-line {
        font-size: 1.08rem;
        font-weight: 760;
        color: #e2e8f0;
    }

    .status-badge {
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.78rem;
        font-weight: 760;
        white-space: nowrap;
    }

    .status-badge.passed {
        color: #22c55e;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .status-badge.warning {
        color: #f59e0b;
        background: rgba(251, 191, 36, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .status-badge.failed {
        color: #ef4444;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .issue-count {
        color: #cbd5e1;
        margin-bottom: 8px;
        font-weight: 650;
    }

    .decision-line {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-bottom: 8px;
    }

    .issue-row {
        padding: 4px 0;
        color: #f8fafc;
    }

    .issue-ok {
        padding: 4px 0;
        color: #22c55e;
        font-weight: 650;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 8px;
        margin: 14px 0 10px;
    }

    .kpi-card {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.9);
        padding: 10px;
        min-height: 72px;
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .preview-empty {
        border: 1px dashed rgba(148, 163, 184, 0.45);
        border-radius: 8px;
        min-height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 18px;
        color: #94a3b8;
        background: rgba(15, 23, 42, 0.8);
        text-align: center;
    }

    div[data-testid="stImage"] img {
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 8px;
        max-height: 220px;
        object-fit: contain;
        background: rgba(15, 23, 42, 0.95);
    }

    .summary-card {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        padding: 14px;
        background: rgba(15, 23, 42, 0.95);
        margin-top: 10px;
    }

    .summary-title {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 760;
        margin-bottom: 8px;
    }

    .summary-body {
        color: #cbd5e1;
        line-height: 1.5;
        margin-bottom: 10px;
    }

    .recommendation {
        color: #f8fafc;
        font-weight: 680;
        padding-top: 10px;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
    }

    div[data-testid="stHorizontalBlock"] {
        border: 0 !important;
    }

    div[data-testid="stFileUploader"] {
        margin-bottom: 8px;
    }

    div[data-testid="stFileUploader"] section {
        border-radius: 8px;
        border-color: rgba(148, 163, 184, 0.35);
        background: rgba(15, 23, 42, 0.9);
        padding: 12px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 8px;
        overflow: hidden;
        background: rgba(15, 23, 42, 0.9);
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }

    .stButton button {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid rgba(148, 163, 184, 0.4) !important;
    }

    .stButton button:hover {
        background-color: #334155 !important;
    }

    hr {
        display: none;
    }

    /* ---- Create Invoice from Text styles ---- */
    .nl-header {
        margin-bottom: 14px;
    }
    .nl-title {
        font-size: 1.08rem;
        font-weight: 760;
        color: #a5b4fc;
        margin-bottom: 5px;
    }
    .nl-subtitle {
        font-size: 0.84rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 10px;
    }
    .example-prompts-label {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    /* ---- Generated Invoice Card ---- */
    .gen-invoice-card {
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.98);
        padding: 22px 24px 18px;
        margin: 14px 0 8px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    }
    .gen-inv-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 18px;
        border-bottom: 1px solid rgba(148,163,184,0.15);
        padding-bottom: 14px;
    }
    .gen-inv-title {
        font-size: 0.75rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .gen-inv-number {
        font-size: 1.3rem;
        font-weight: 900;
        color: #a5b4fc;
        font-family: monospace;
    }
    .gen-inv-badge {
        display: inline-block;
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .gen-inv-meta {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 3px;
    }
    .gen-inv-parties {
        display: flex;
        justify-content: space-between;
        margin-bottom: 18px;
    }
    .gen-inv-party {}
    .gen-inv-party-label {
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 3px;
    }
    .gen-inv-party-name {
        font-size: 0.98rem;
        font-weight: 760;
        color: #f1f5f9;
    }
    .gen-inv-party-gstin {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    .gen-inv-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
        margin-bottom: 14px;
    }
    .gen-inv-table th {
        background: rgba(30, 41, 59, 0.9);
        color: #64748b;
        font-weight: 700;
        padding: 8px 10px;
        border-bottom: 1px solid rgba(148,163,184,0.15);
        font-size: 0.76rem;
        letter-spacing: 0.04em;
    }
    .gen-inv-table td {
        padding: 8px 10px;
        border-bottom: 1px solid rgba(148,163,184,0.08);
        color: #e2e8f0;
    }
    .gen-inv-table tr:last-child td {
        border-bottom: none;
    }
    .gen-inv-totals {
        border-top: 1px solid rgba(148,163,184,0.2);
        padding-top: 12px;
        max-width: 320px;
        margin-left: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Export Helpers ---

def _fmt_inr(value: object) -> str:
    """Format a number as Indian Rupees with Indian comma grouping (₹X,XX,XXX)."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "₹0"
    # Indian grouping: last 3 digits, then groups of 2
    negative = n < 0
    s = f"{abs(n):.0f}"
    if len(s) > 3:
        last3 = s[-3:]
        rest  = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        s = ",".join(groups) + "," + last3
    return ("−₹" if negative else "₹") + s


def _status_badge(status: str) -> str:
    """Return a coloured emoji badge string for a given audit status."""
    return {
        "passed":  "🟢 Passed",
        "warning": "🟡 Warning",
        "failed":  "🔴 Failed",
    }.get(status.lower(), "⚪ Unknown")


def _score_colour(score: int) -> str:
    if score >= 80:
        return "green"
    if score >= 60:
        return "orange"
    return "red"


def _render_generated_invoice_card(invoice_data: dict, audit_result: dict) -> None:
    """Render the full AI-generated invoice card with all 8 UI improvements."""

    # ── helpers ─────────────────────────────────────────────────────
    raw_items = invoice_data.get("items")
    items: list[dict] = [i for i in raw_items if isinstance(i, dict)] if isinstance(raw_items, list) else []

    status     = audit_result.get("status", "warning")
    score      = risk_score(audit_result)
    issues     = audit_result.get("issues", [])

    # totals (business rule: GST on post-discount taxable)
    if items:
        subtotal = round(sum(float(i.get("amount") or 0) for i in items), 2)
        discount = round(float(invoice_data.get("discount") or 0), 2)
        taxable  = round(subtotal - discount, 2)
        gst      = round(float(invoice_data.get("tax") or 0), 2)
        grand_total = round(taxable + gst, 2)
    else:
        subtotal    = round(float(invoice_data.get("amount") or 0), 2)
        discount    = round(float(invoice_data.get("discount") or 0), 2)
        taxable     = round(subtotal - discount, 2)
        gst         = round(float(invoice_data.get("tax") or 0), 2)
        grand_total = round(float(invoice_data.get("total") or 0), 2)

    cls = str(invoice_data.get("classification") or "Expense").strip().title()
    cls_icon = {"Purchase": "🟢", "Sales": "🔵", "Expense": "🟠"}.get(cls, "⚪")

    # ── ① AI Summary card — wording matches audit status exactly ────
    st.subheader("🤖 AI Summary")
    n_products = len(items)
    high_value = any(i.get("field") == "high_value" for i in issues)

    doc_type = str(invoice_data.get("document_type") or "").replace("_", " ").title()
    vendor   = invoice_data.get("vendor") or "Unknown vendor"

    summary_lines = [
        f"{doc_type} generated for **{vendor}**"
        + (f" containing **{n_products} product{'s' if n_products != 1 else ''}**." if n_products else "."),
        f"Total payable amount is **{_fmt_inr(grand_total)}**.",
    ]

    # Status-aligned line — mirrors _status_badge() exactly
    if status == "passed":
        summary_lines.append("✅ No audit issues found. Invoice is ready to process.")
    elif status == "warning":
        summary_lines.append("🟡 Review recommended before saving — minor issues detected.")
    else:
        summary_lines.append("🔴 Critical issues detected — do not process without manual review.")

    if high_value:
        summary_lines.append("⚠ High-value invoice — manager approval required before processing.")

    with st.container(border=True):
        for line in summary_lines:
            st.markdown(line)

    st.divider()

    # ── ① Header: invoice meta + ① coloured status badge ────────────
    col_inv, col_status = st.columns([2, 1])
    with col_inv:
        st.markdown(f"**Invoice No:** `{invoice_data.get('invoice_no') or '—'}`")
        st.markdown(f"📅 **Date:** {invoice_data.get('date') or '—'}")
        st.markdown(f"💳 **Payment:** {invoice_data.get('payment_method') or 'Cash'}")
    with col_status:
        st.markdown(f"**📊 Audit Status**")
        st.markdown(f"### {_status_badge(status)}")
        score_color = _score_colour(score)
        st.markdown(f"**Risk Score:** :{score_color}[{score}/100]")

    st.divider()

    # ── ⑥ Vendor / Customer with icons ──────────────────────────────
    col_from, col_to = st.columns(2)
    with col_from:
        st.markdown("🏢 **From (Vendor)**")
        st.write(invoice_data.get("vendor") or "—")
        if invoice_data.get("gstin"):
            st.caption(f"GSTIN: {invoice_data.get('gstin')}")
    with col_to:
        st.markdown("👤 **To (Customer)**")
        st.write(invoice_data.get("customer_name") or "—")
        st.caption(f"{cls_icon} {cls}  ·  {invoice_data.get('category') or '—'}")

    st.divider()

    # ── ⑧ Statistics card ───────────────────────────────────────────
    total_qty  = sum(float(i.get("quantity") or 0) for i in items)
    gst_rate_pct = ""
    if subtotal > 0 and gst > 0:
        rate = round(gst / taxable * 100) if taxable > 0 else 0
        gst_rate_pct = f"{rate}%"

    # ── ⑧ Statistics — two rows of 4 wider cards, values won't truncate ──
    stat_data = [
        ("📦 Products",        str(n_products)),
        ("🔢 Total Qty",       f"{total_qty:,.0f}"),
        ("🏷 Category",        invoice_data.get("category") or "—"),
        (f"{cls_icon} Classification", cls),
        ("💰 GST Rate",        gst_rate_pct or "—"),
        ("🏷 Discount",        _fmt_inr(discount) if discount > 0 else "None"),
        ("📅 Date",            invoice_data.get("date") or "—"),
        ("💳 Payment",         invoice_data.get("payment_method") or "—"),
    ]
    row1 = st.columns(4)
    row2 = st.columns(4)
    for col, (label, val) in zip(row1 + row2, stat_data):
        col.metric(label, val)

    st.divider()

    # ── ② Line items table with HSN/SAC + ③ Indian currency ─────────
    st.markdown("📦 **Products**")
    if items:
        rows = []
        for item in items:
            try:
                qty        = float(item.get("quantity") or 0)
                unit_price = float(item.get("unit_price") or 0)
                item_tax   = float(item.get("tax") or 0)
                item_amount= float(item.get("amount") or 0)
            except (TypeError, ValueError):
                qty, unit_price, item_tax, item_amount = 0.0, 0.0, 0.0, 0.0

            hsn = str(item.get("hsn_sac") or "").strip()
            rows.append({
                "Product":          item.get("product") or item.get("description") or "—",
                "HSN / SAC":        hsn if hsn else "—",          # ② always shown
                "Qty":              qty,
                "Unit Price":       _fmt_inr(unit_price),          # ③ Indian format
                "Tax":              _fmt_inr(item_tax),
                "Amount":           _fmt_inr(item_amount),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No line items found in this invoice.")

    st.divider()

    # ── ③ Totals — plain columns, no Markdown tricks needed ─────────
    st.markdown("**Totals**")
    def _total_row(label: str, amount: str, bold: bool = False) -> None:
        c1, c2 = st.columns([2, 1])
        if bold:
            c1.markdown(f"**{label}**")
            c2.markdown(f"**{amount}**")
        else:
            c1.write(label)
            c2.write(amount)

    _total_row("Subtotal", _fmt_inr(subtotal))
    if discount > 0:
        _total_row("Discount", f"−{_fmt_inr(discount)}")
        _total_row("Taxable Amount", _fmt_inr(taxable))
    _total_row("GST / Tax", _fmt_inr(gst))
    st.divider()
    _total_row("Grand Total", _fmt_inr(grand_total), bold=True)

    st.divider()

    # ── ⑤ Audit Findings card with Action section ───────────────────
    st.markdown("⚠ **Findings**")
    if not issues:
        st.success("✅ No audit issues detected. Invoice is ready to process.")
    else:
        for issue in issues:
            severity = str(issue.get("severity", "medium")).lower()
            field    = issue.get("field", "")
            message  = issue.get("message", "")

            if field == "high_value":
                _render_high_value_card(message)
                continue

            msg = f"• {message}"
            if severity == "high":
                st.error(msg)
            elif severity == "medium":
                st.warning(msg)
            else:
                st.info(msg)

    st.divider()

    # ── ⑦ Download buttons with icons, PDF primary ──────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📄 Download PDF Report",
            data=generate_invoice_pdf(invoice_data, audit_result),
            file_name=f"audit_report_{invoice_data.get('invoice_no', 'export')}.pdf",
            mime="application/pdf",
            type="primary",
            key="nl_pdf_download",
        )
    with col2:
        st.download_button(
            label="📊 Download CSV",
            data=generate_invoice_csv(invoice_data),
            file_name=f"invoice_{invoice_data.get('invoice_no', 'export')}.csv",
            mime="text/csv",
            key="nl_csv_download",
        )


def generate_invoice_csv(invoice_data: dict) -> str:
    import io, csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["INVOICE REPORT", invoice_data.get("invoice_no", "")])
    writer.writerow(["Vendor / Supplier", invoice_data.get("vendor", "")])
    writer.writerow(["Invoice Date", invoice_data.get("date", "")])
    writer.writerow(["Customer Name", invoice_data.get("customer_name", "")])
    writer.writerow(["GSTIN", invoice_data.get("gstin", "")])
    writer.writerow(["Document Category", invoice_data.get("category", "")])
    writer.writerow(["Business Classification", invoice_data.get("classification", "Expense")])
    writer.writerow([])
    writer.writerow(["Product", "Description", "HSN/SAC", "Qty", "Unit Price", "Tax", "Amount"])
    for item in invoice_data.get("items", []):
        writer.writerow([
            item.get("product") or item.get("description", ""),
            item.get("description", ""),
            item.get("hsn_sac", "N/A"),
            item.get("quantity", 0),
            item.get("unit_price", 0),
            item.get("tax", 0),
            item.get("amount", 0)
        ])
    writer.writerow([])
    writer.writerow(["Taxable Subtotal", invoice_data.get("amount", 0)])
    writer.writerow(["Discount", invoice_data.get("discount", 0)])
    writer.writerow(["Tax Amount", invoice_data.get("tax", 0)])
    writer.writerow(["Grand Total", invoice_data.get("total", 0)])
    return output.getvalue()


def generate_invoice_pdf(invoice_data: dict, audit_result: dict) -> bytes:
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    
    # Title
    page.insert_text((50, 60), "INVOICE AUDIT REPORT", fontsize=18, fontname="helvetica-bold", color=(0.1, 0.2, 0.4))
    page.draw_line((50, 75), (550, 75), color=(0.1, 0.2, 0.4), width=1.5)
    
    # Meta details
    y = 110
    details = [
        ("Invoice No:", invoice_data.get("invoice_no", "N/A")),
        ("Vendor:", invoice_data.get("vendor", "N/A")),
        ("Date:", invoice_data.get("date", "N/A")),
        ("Customer:", invoice_data.get("customer_name", "N/A")),
        ("GSTIN:", invoice_data.get("gstin", "N/A")),
        ("Document Category:", invoice_data.get("category", "N/A")),
        ("Business Classification:", invoice_data.get("classification", "Expense")),
        ("Audit Status:", str(audit_result.get("status", "Unknown")).upper()),
        ("Risk Score:", f"{audit_result.get('risk_score', 100)}/100"),
    ]
    
    for label, val in details:
        page.insert_text((50, y), label, fontsize=10, fontname="helvetica-bold")
        page.insert_text((150, y), str(val), fontsize=10, fontname="helvetica")
        y += 18
        
    y += 15
    page.insert_text((50, y), "LINE ITEMS / PRODUCT TABLE", fontsize=12, fontname="helvetica-bold", color=(0.1, 0.2, 0.4))
    page.draw_line((50, y+5), (550, y+5), color=(0.1, 0.2, 0.4), width=1.0)
    y += 25
    
    # Table header
    page.insert_text((50, y), "Product", fontsize=8, fontname="helvetica-bold")
    page.insert_text((135, y), "Description", fontsize=8, fontname="helvetica-bold")
    page.insert_text((250, y), "HSN/SAC", fontsize=8, fontname="helvetica-bold")
    page.insert_text((310, y), "Qty", fontsize=8, fontname="helvetica-bold")
    page.insert_text((350, y), "Unit Price", fontsize=8, fontname="helvetica-bold")
    page.insert_text((425, y), "Tax", fontsize=8, fontname="helvetica-bold")
    page.insert_text((485, y), "Amount", fontsize=8, fontname="helvetica-bold")
    y += 15
    page.draw_line((50, y), (550, y), color=(0.7, 0.7, 0.7), width=0.5)
    y += 15
    
    for item in invoice_data.get("items", []):
        if y > 700:
            page = doc.new_page()
            y = 50
        prod = (item.get("product") or "")[:16]
        desc = (item.get("description") or item.get("product") or "")[:24]
        page.insert_text((50, y), prod, fontsize=8, fontname="helvetica")
        page.insert_text((135, y), desc, fontsize=8, fontname="helvetica")
        page.insert_text((250, y), str(item.get("hsn_sac", "N/A"))[:10], fontsize=8, fontname="helvetica")
        page.insert_text((310, y), str(item.get("quantity", 0)), fontsize=8, fontname="helvetica")
        page.insert_text((350, y), f"Rs. {float(item.get('unit_price',0) or 0):,.2f}", fontsize=8, fontname="helvetica")
        page.insert_text((425, y), f"Rs. {float(item.get('tax',0) or 0):,.2f}", fontsize=8, fontname="helvetica")
        page.insert_text((485, y), f"Rs. {float(item.get('amount',0) or 0):,.2f}", fontsize=8, fontname="helvetica")
        y += 18
        
    y += 15
    page.draw_line((50, y), (550, y), color=(0.7, 0.7, 0.7), width=0.5)
    y += 15
    
    totals = [
        ("Taxable Subtotal:", invoice_data.get("amount", 0)),
        ("Discount:", invoice_data.get("discount", 0)),
        ("Tax Amount:", invoice_data.get("tax", 0)),
        ("Grand Total:", invoice_data.get("total", 0)),
    ]
    for label, val in totals:
        page.insert_text((330, y), label, fontsize=9, fontname="helvetica-bold")
        page.insert_text((470, y), f"Rs. {float(val):,.2f}", fontsize=9, fontname="helvetica")
        y += 15
        
    # Audit issues
    y += 15
    page.insert_text((50, y), "AUDIT FINDINGS & ISSUES", fontsize=12, fontname="helvetica-bold", color=(0.1, 0.2, 0.4))
    page.draw_line((50, y+5), (550, y+5), color=(0.1, 0.2, 0.4), width=1.0)
    y += 25
    
    issues = audit_result.get("issues", [])
    if not issues:
        page.insert_text((50, y), "No issues detected. Invoice is valid.", fontsize=10, fontname="helvetica", color=(0.1, 0.6, 0.1))
    else:
        for issue in issues:
            if y > 750:
                page = doc.new_page()
                y = 50
            severity = str(issue.get("severity", "medium")).upper()
            msg = issue.get("message", "")
            color = (0.8, 0.1, 0.1) if severity == "HIGH" else (0.7, 0.5, 0.1)
            page.insert_text((50, y), f"[{severity}] {msg}", fontsize=9, fontname="helvetica-bold", color=color)
            y += 15
            
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


# --- App Title ---
st.title("AI Invoice Audit Agent")

# --- Tab Setup ---
tab_upload, tab_search, tab_dashboard = st.tabs([
    "📤 Upload & Audit",
    "🔍 Search & Details",
    "📊 Dashboard"
])

# --- Tab 1: Upload & Audit ---
with tab_upload:
    st.subheader("Upload Invoice")
    uploaded_file = st.file_uploader(
        "Drag and drop invoice image or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        label_visibility="collapsed",
        key="uploader_tab1"
    )

    run_button = st.button("Run Audit", type="primary", use_container_width=True, key="run_btn_tab1")

    if run_button and uploaded_file:
        file_path = save_uploaded_file(uploaded_file)

        with st.spinner("Audit agent is reading and checking the invoice..."):
            invoice_data, audit_result, audit_summary, validation_errors = run_audit_pipeline(str(file_path))

        if validation_errors:
            file_path.unlink(missing_ok=True)
            st.session_state.invoice_data = None
            st.session_state.audit_result = None
            st.session_state.audit_summary = ""
            st.session_state.invoice_source = ""
            st.session_state.messages = []
            st.error("This document does not appear to be a valid invoice.")

            for error in validation_errors:
                st.warning(error)
        else:
            st.session_state.invoice_data = invoice_data
            st.session_state.audit_result = audit_result
            st.session_state.audit_summary = audit_summary
            st.session_state.invoice_source = "upload"
            st.session_state.messages = []

            try:
                invoice_data, audit_result, audit_summary, save_message = persist_invoice(invoice_data, audit_result)
                st.session_state.invoice_data = invoice_data
                st.session_state.audit_result = audit_result
                st.session_state.audit_summary = audit_summary

                if save_message.startswith("⚠"):
                    st.warning(save_message)
                else:
                    st.success(save_message)
            except Exception as e:
                st.error(f"Failed to save audit record to database: {e}")

    st.divider()
    st.subheader("✦ Create Invoice from Text")

    # Example prompts
    EXAMPLE_PROMPTS = [
        "TechMart sold 2 Samsung phones at ₹20,000 each and 1 charger at ₹500 to Rahul. GST 18%. Payment by UPI.",
        "Supply 50 Office Chairs at ₹4,500 each to Infosys Ltd. GST 12%. Payment by bank transfer.",
        "Hotel stay for 2 nights at ₹3,500 per night for Priya Sharma. GST 12%. Payment by credit card.",
        "City Pharmacy sold medicines worth ₹2,200 to Meera Joshi. GST 5%. Payment by cash.",
        "CodeCraft Solutions — web development services ₹85,000 for StartupXYZ. GST 18%. Payment by UPI.",
    ]

    st.caption("Try an example:")
    example_cols = st.columns(len(EXAMPLE_PROMPTS))
    for idx, (col, prompt) in enumerate(zip(example_cols, EXAMPLE_PROMPTS)):
        with col:
            short = prompt[:36] + "…" if len(prompt) > 36 else prompt
            if st.button(short, key=f"example_btn_{idx}", use_container_width=True):
                # Write directly to the widget's session state key so the
                # textarea re-renders with the selected text immediately.
                st.session_state["natural_invoice_text"] = prompt

    st.markdown("**Transaction Description**")
    natural_invoice_text = st.text_area(
        "Transaction Description",
        placeholder=(
            "Describe a transaction in plain English or structured format.\n\n"
            'Natural language: "TechMart sold 3 laptops at ₹55,000 each to Anjali. GST 18%. Payment via UPI."\n\n'
            "Structured:\n"
            "Vendor: TechMart Electronics\n"
            "Customer: Rahul Sharma\n"
            "Samsung Galaxy S25  Qty:2  Price:72000\n"
            "USB Charger  Qty:1  Price:500\n"
            "GST: 18%  Payment: UPI"
        ),
        height=140,
        label_visibility="collapsed",
        key="natural_invoice_text",
    )

    # Tip line under the textarea
    st.caption(
        "💡 Tip: Mention the vendor, customer, products/services, quantity, price, "
        "GST (optional), and payment method for the best results."
    )

    # Loading-state button
    generating = st.session_state.get("nl_generating", False)
    btn_label  = "🧠 AI is processing..." if generating else "⚡ Generate Invoice & Run Audit"
    generate_invoice_button = st.button(
        btn_label,
        type="primary",
        use_container_width=True,
        key="generate_invoice_btn",
        disabled=generating,
    )

    if generate_invoice_button:
        input_text = natural_invoice_text.strip()
        if not input_text:
            st.warning("Enter a transaction description first.")
        else:
            st.session_state["nl_generating"] = True
            with st.spinner("🧠 AI is understanding your text, building the invoice, and running the audit…"):
                invoice_data, audit_result, audit_summary, nl_errors = run_natural_language_pipeline(input_text)
            st.session_state["nl_generating"] = False

            if nl_errors:
                # LLM refused or validation failed — show structured guidance
                st.error("⚠️ Cannot generate an invoice from this description.")
                suggestion = next((e for e in nl_errors if e.startswith("💡")), None)
                field_errors = [e for e in nl_errors if not e.startswith("💡")]
                if field_errors:
                    with st.container(border=True):
                        st.markdown("**What's missing:**")
                        for err in field_errors:
                            st.markdown(f"- {err}")
                        if suggestion:
                            st.info(suggestion)
                        st.markdown(
                            "**Example of a complete description:**\n\n"
                            "> *TechMart Electronics sold 2 Samsung phones at ₹20,000 each "
                            "and 1 USB cable at ₹250 to Rahul Sharma. GST 18%. Payment by UPI.*"
                        )
                elif suggestion:
                    st.info(suggestion)
                st.session_state.invoice_data = None
                st.session_state.audit_result = None
                st.session_state.audit_summary = ""
                st.session_state.invoice_source = ""
            else:
                st.session_state.invoice_data = invoice_data
                st.session_state.audit_result = audit_result
                st.session_state.audit_summary = audit_summary
                st.session_state.invoice_source = "generated"
                st.session_state.messages = []

                try:
                    invoice_data, audit_result, audit_summary, save_message = persist_invoice(invoice_data, audit_result)
                    st.session_state.invoice_data = invoice_data
                    st.session_state.audit_result = audit_result
                    st.session_state.audit_summary = audit_summary

                    if save_message.startswith("⚠"):
                        st.warning(save_message)
                    else:
                        st.success(f"✅ {save_message} — Invoice saved to MongoDB and available for search, analytics & AI chat.")
                except Exception as e:
                    st.error(f"Failed to save generated invoice to database: {e}")

    # Output details
    audit_result_t1 = st.session_state.get("audit_result")
    invoice_data_t1 = st.session_state.get("invoice_data")
    invoice_source_t1 = st.session_state.get("invoice_source", "")

    if audit_result_t1 and invoice_data_t1:
        if invoice_source_t1 == "generated":
            # Generated invoices — fully native Streamlit card
            st.subheader("🤖 AI-Generated Invoice & Audit Result")
            _render_generated_invoice_card(invoice_data_t1, audit_result_t1)
        else:
            # Uploaded invoices — native Streamlit audit view
            st.subheader("📊 AI Invoice Audit Agent Analysis")

            render_kpi_cards(audit_result_t1, invoice_data_t1)
            st.markdown(f"**Status:** {_status_badge(audit_result_t1['status'])}")
            st.markdown(f"**Decision:** {decision_text(audit_result_t1)}")
            st.markdown("**Audit Findings**")
            render_issue_list(audit_result_t1)

            st.markdown(f"**AI Summary:** {ai_summary_text(audit_result_t1, invoice_data_t1)}")
            st.caption(recommendation_text(audit_result_t1))

            with st.expander("Invoice preview and fields", expanded=False):
                render_invoice_preview(uploaded_file)
                st.markdown("**Invoice Summary**")
                render_invoice_summary(invoice_data_t1)
                render_line_items(invoice_data_t1)
                render_audit_flags(invoice_data_t1)

            st.markdown("### Export Invoice")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download CSV",
                    data=generate_invoice_csv(invoice_data_t1),
                    file_name=f"invoice_{invoice_data_t1.get('invoice_no', 'export')}.csv",
                    mime="text/csv",
                    key="csv_download_t1",
                )
            with col2:
                st.download_button(
                    label="Download PDF Report",
                    data=generate_invoice_pdf(invoice_data_t1, audit_result_t1),
                    file_name=f"audit_report_{invoice_data_t1.get('invoice_no', 'export')}.pdf",
                    mime="application/pdf",
                    key="pdf_download_t1",
                )

    st.divider()
    render_chat()


# --- Tab 2: Search & Details ---
with tab_search:
    st.subheader("Search Invoices")
    
    col1, col2 = st.columns(2)
    with col1:
        s_inv_no = st.text_input("Invoice Number", placeholder="e.g., INV-001", key="s_inv_no")
        s_vendor = st.text_input("Vendor", placeholder="e.g., Amazon", key="s_vendor")
        s_cust = st.text_input("Customer", placeholder="e.g., Rahul", key="s_cust")
        s_gstin = st.text_input("GSTIN", placeholder="e.g., 27AAAAA1111A1Z5", key="s_gstin")
    with col2:
        s_cat = st.text_input("Document Category", placeholder="e.g., Retail", key="s_cat")
        s_status = st.selectbox("Status", ["All", "Passed", "Warning", "Failed"], key="s_status")
        s_class = st.selectbox("Business Classification", ["All", "Purchase", "Sales", "Expense"], key="s_class")
        s_date = st.text_input("Date", placeholder="YYYY-MM-DD", key="s_date")
        
    search_triggered = st.button("Search Database", type="primary", use_container_width=True, key="search_triggered_btn")
    
    # Run search query
    query_params = {}
    if s_inv_no: query_params["invoice_no"] = s_inv_no
    if s_vendor: query_params["vendor"] = s_vendor
    if s_cust: query_params["customer_name"] = s_cust
    if s_gstin: query_params["gstin"] = s_gstin
    if s_cat: query_params["category"] = s_cat
    if s_status != "All": query_params["status"] = s_status
    if s_class != "All": query_params["classification"] = s_class
    if s_date: query_params["date"] = s_date
    
    matching_invoices = []
    if search_triggered or s_inv_no or s_vendor or s_cust or s_gstin or s_cat or s_date or s_status != "All" or s_class != "All":
        matching_invoices = search_invoices(query_params)
    else:
        # Default load all invoices
        matching_invoices = get_all_invoices()

    db_error = database.get_database_error() if hasattr(database, "get_database_error") else ""
    if db_error:
        st.warning("Database connection failed. Check your MongoDB username, password, and Atlas access settings.")
        
    if matching_invoices:
        st.markdown(f"**Found {len(matching_invoices)} invoices:**")
        
        # Display simplified table
        table_rows = []
        for inv in matching_invoices:
            table_rows.append({
                "Invoice No": inv.get("invoice_no", "N/A"),
                "Vendor": inv.get("vendor", "N/A"),
                "Date": inv.get("date", "N/A"),
                "Document Category": inv.get("category", "N/A"),
                "Business Classification": inv.get("classification", "Expense"),
                "Total": f"Rs. {float(inv.get('total', 0)):,.2f}",
                "Status": str(inv.get("audit_status", "unknown")).upper()
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        
        # Select box for detailed page view
        dropdown_options = [f"{inv.get('invoice_no')} | {inv.get('vendor')} | Total: Rs. {float(inv.get('total',0)):,.2f}" for inv in matching_invoices]
        selected_option = st.selectbox("Select invoice to view complete details page:", dropdown_options, key="select_details_drop")
        
        if selected_option:
            selected_inv_no = selected_option.split(" | ")[0].strip()
            invoice = get_invoice_by_number(selected_inv_no)
            if invoice:
                audit_res = {
                    "status": invoice.get("audit_status", "passed"),
                    "risk_score": invoice.get("risk_score", 100),
                    "issue_count": invoice.get("issue_count", 0),
                    "issues": invoice.get("issues", []),
                }

                st.divider()
                st.subheader(f"📄 Invoice Details: {selected_inv_no}")

                # Status metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Audit Status", status_label(audit_res["status"]))
                m2.metric("Risk Score", f"{audit_res['risk_score']}/100")
                m3.metric("Issues", audit_res["issue_count"])

                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write(f"**Vendor:** {invoice.get('vendor', 'N/A')}")
                    st.write(f"**Customer Name:** {invoice.get('customer_name', 'N/A')}")
                    st.write(f"**Invoice Date:** {invoice.get('date', 'N/A')}")
                    st.write(f"**GSTIN:** {invoice.get('gstin', 'N/A')}")
                with col_d2:
                    cls_val = str(invoice.get("classification") or "Expense").strip().title()
                    cls_disp = {"Purchase": "🟢 Purchase", "Sales": "🔵 Sales", "Expense": "🟠 Expense"}.get(cls_val, "⚪ " + cls_val)
                    st.write(f"**Document Category:** {invoice.get('category', 'N/A')}")
                    st.write(f"**Business Classification:** {cls_disp}")
                    st.write(f"**Subtotal:** Rs. {float(invoice.get('amount', 0)):,.2f}")
                    st.write(f"**Total Amount:** Rs. {float(invoice.get('total', 0)):,.2f}")

                st.markdown("**Product Table**")
                render_line_items(invoice)

                st.markdown("**Audit Findings / Issues**")
                render_issue_list(audit_res)

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download CSV Export",
                        data=generate_invoice_csv(invoice),
                        file_name=f"invoice_{selected_inv_no}.csv",
                        mime="text/csv",
                        key="csv_download_t2",
                    )
                with col2:
                    st.download_button(
                        label="Download PDF Report",
                        data=generate_invoice_pdf(invoice, audit_res),
                        file_name=f"audit_report_{selected_inv_no}.pdf",
                        mime="application/pdf",
                        key="pdf_download_t2",
                    )
    else:
        st.info("No matching records found in database.")



# --- Tab 3: Dashboard ---
with tab_dashboard:
    st.subheader("📊 Analytics Dashboard")
    
    invoices = get_all_invoices()
    db_error = database.get_database_error() if hasattr(database, "get_database_error") else ""
    if db_error:
        st.warning("Database connection failed. Check your MongoDB username, password, and Atlas access settings.")

    if invoices:
        total_invs = len(invoices)
        passed_count = sum(1 for inv in invoices if str(inv.get("audit_status")).lower() == "passed")
        warning_count = sum(1 for inv in invoices if str(inv.get("audit_status")).lower() == "warning")
        failed_count = sum(1 for inv in invoices if str(inv.get("audit_status")).lower() == "failed")

        # KPI row — native st.metric
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Invoices", total_invs)
        k2.metric("✅ Passed", passed_count)
        k3.metric("⚠️ Warning", warning_count)
        k4.metric("❌ Failed", failed_count)

        # Analytics calculations
        cat_counts: dict[str, int] = {}
        class_counts = {"Purchase": 0, "Sales": 0, "Expense": 0}
        total_score = 0

        for inv in invoices:
            cat = str(inv.get("category") or "Uncategorized").strip()
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

            cls = inv.get("classification")
            if not cls:
                dummy_data = {
                    "document_type": inv.get("document_type", ""),
                    "category": inv.get("category", ""),
                }
                text_context = f"{inv.get('vendor', '')} {inv.get('category', '')}"
                cls = _classify_invoice(dummy_data, text_context)

            cls = str(cls).strip().title()
            if cls in class_counts:
                class_counts[cls] += 1
            else:
                class_counts["Expense"] += 1

            total_score += inv.get("risk_score", 100)

        avg_score = round(total_score / total_invs, 1)

        col_db1, col_db2 = st.columns(2)
        with col_db1:
            st.markdown("**Category-wise Distribution**")
            st.bar_chart(cat_counts)
        with col_db2:
            st.markdown("**Business Classification Metrics**")
            st.dataframe(
                [{"Business Classification": k, "Invoices": v} for k, v in class_counts.items()],
                use_container_width=True,
                hide_index=True,
            )
            st.metric("Average Risk Score", f"{avg_score} / 100")
    else:
        st.info("No analytics data available. Insert invoices to display metrics.")

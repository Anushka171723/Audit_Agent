from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

import database
from database import save_invoice_record, get_invoice_by_number, update_invoice_record

from audit import audit_invoice, create_audit_summary
from chat import answer_audit_question
from ocr import extract_text_from_file
from parser import parse_invoice_text, save_csv, save_json, is_valid_invoice


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


def save_uploaded_file(uploaded_file) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOADS_DIR / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())

    return file_path


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


def render_issue_list(audit_result: dict) -> None:
    st.markdown(render_issue_html(audit_result), unsafe_allow_html=True)


def render_issue_html(audit_result: dict) -> str:
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
    rows = [
        {"field": "document_type", "value": title_label(invoice_data.get("document_type"))},
        {"field": "invoice_no", "value": invoice_data.get("invoice_no") or "Missing"},
        {"field": "date", "value": invoice_data.get("date") or "Missing"},
        {"field": "vendor", "value": invoice_data.get("vendor") or "Missing"},
        {"field": "category", "value": invoice_data.get("category") or "Missing"},
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
            "description": item.get("description", ""),
            "quantity": item.get("quantity", 0),
            "unit_price": item.get("unit_price", 0),
            "amount": item.get("amount", 0),
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
        ("Category Detected", bool(invoice_data and invoice_data.get("category"))),
        ("Total Found", bool(invoice_data and invoice_data.get("total"))),
    ]

    for label, passed in checks:
        st.write(f"{'✓' if passed else '⚠'} {label if passed else label.replace(' Found', ' Missing').replace(' Present', ' Missing').replace(' Detected', ' Missing')}")

    for issue in audit_result.get("issues", []):
        if issue.get("field") in ["invoice_no", "vendor", "customer_name", "gstin", "category", "total"]:
            continue
        st.write(f"⚠ {issue.get('message')}")


def render_kpi_cards(audit_result: dict, invoice_data: dict | None) -> None:
    total_amount = invoice_data.get("total") if invoice_data else 0
    score = risk_score(audit_result)

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Status</div>
                <div class="kpi-value">{status_label(audit_result["status"])}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Issues</div>
                <div class="kpi-value">{audit_result["issue_count"]}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Risk Score</div>
                <div class="kpi-value">{score}/100</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Amount</div>
                <div class="kpi-value">{money_value(total_amount)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_invoice_preview(uploaded_file) -> None:
    if uploaded_file:
        if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
            st.markdown(
                f'<div class="preview-empty">PDF uploaded: {uploaded_file.name}</div>',
                unsafe_allow_html=True,
            )
            return

        st.image(uploaded_file, use_container_width=True)
        return

    st.markdown(
        '<div class="preview-empty">No invoice preview yet.</div>',
        unsafe_allow_html=True,
    )


def render_chat() -> None:
    st.markdown('<div class="section-title">Ask the Audit Agent</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Why did this invoice fail?")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing invoice..."):
                answer = answer_audit_question(question)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


st.set_page_config(page_title="AI Audit Agent", page_icon="AI", layout="centered")

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
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Audit Agent")

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Upload Invoice</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag and drop invoice image or PDF",
    type=["png", "jpg", "jpeg", "pdf"],
    label_visibility="collapsed",
)

run_button = st.button("Run Audit", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if run_button and uploaded_file:
    file_path = save_uploaded_file(uploaded_file)

    with st.spinner("Audit agent is reading and checking the invoice..."):
        invoice_data, audit_result, audit_summary, validation_errors = run_audit_pipeline(str(file_path))

    if validation_errors:
        file_path.unlink(missing_ok=True)
        st.session_state.invoice_data = None
        st.session_state.audit_result = None
        st.session_state.audit_summary = ""
        st.session_state.messages = []
        st.error("This document does not appear to be a valid invoice.")

        for error in validation_errors:
            st.warning(error)

        st.stop()

    st.session_state.invoice_data = invoice_data
    st.session_state.audit_result = audit_result
    st.session_state.audit_summary = audit_summary
    st.session_state.messages = []

    # Persist audit record to MongoDB (if configured)
    try:
        invoice_no = (
            invoice_data.get("invoice_no")
            or invoice_data.get("invoice_number")
            or invoice_data.get("number")
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
            "amount": invoice_data.get("amount", 0),
            "discount": invoice_data.get("discount", 0),
            "tax": invoice_data.get("tax", 0),
            "total": invoice_data.get("total", invoice_data.get("grand_total", 0)),
            "payment_method": invoice_data.get("payment_method", ""),
            "ocr_quality": invoice_data.get("ocr_quality", ""),
            "items": invoice_data.get("items", []),
            "audit_flags": invoice_data.get("audit_flags", []),
            "audit_status": audit_result.get("status"),
            "risk_score": audit_result.get("risk_score", 100),
            "issue_count": audit_result.get("issue_count", len(audit_result.get("issues", []))),
            "debugging_info": audit_result.get("debugging_info", {}),
            "created_at": datetime.utcnow(),
        }

        duplicate = find_duplicate_invoice(record)
        if duplicate:
            audit_flags = invoice_data.setdefault("audit_flags", [])
            if "duplicate_candidate" not in audit_flags:
                audit_flags.append("duplicate_candidate")

            audit_result = audit_invoice(invoice_data)
            audit_summary = create_audit_summary(invoice_data, audit_result)
            st.session_state.invoice_data = invoice_data
            st.session_state.audit_result = audit_result
            st.session_state.audit_summary = audit_summary
            save_json(invoice_data, "outputs/extracted_data.json")
            save_json(audit_result, "outputs/audit_report.json")
            save_text(audit_summary, "outputs/audit_report.txt")
            st.warning("Duplicate invoice detected - the record already exists in the database.")
        else:
            inserted_id = save_invoice_record(record)
            st.success("Invoice saved successfully.")
    except Exception as e:
        st.error(f"Failed to save audit record to database: {e}")

if run_button and not uploaded_file:
    st.warning("Please upload an invoice file first.")

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Edit Existing Invoice</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    search_invoice_no = st.text_input(
        "Enter invoice number to edit",
        value="",
        placeholder="e.g., INV001",
        key="search_invoice",
        help="Search for an existing invoice in the database.",
    )

with col2:
    search_button = st.button("Search", key="search_btn", use_container_width=True)

if search_button and search_invoice_no:
    found_invoice = get_invoice_by_number(search_invoice_no.strip())
    if found_invoice:
        st.session_state.editing_invoice = found_invoice
        st.success(f"Invoice {search_invoice_no} loaded for editing.")
    else:
        st.error(f"No invoice record found for {search_invoice_no}.")

if "editing_invoice" in st.session_state and st.session_state.editing_invoice is not None:
    editing_invoice = st.session_state.editing_invoice
    invoice_no_from_db = editing_invoice.get("invoice_no", "")
    st.markdown("**Current Invoice Data:**")
    
    edited_vendor = st.text_input(
        "Vendor",
        value=editing_invoice.get("vendor", ""),
        key="edit_vendor",
    )
    
    edited_date = st.text_input(
        "Date",
        value=editing_invoice.get("date", ""),
        key="edit_date",
    )
    
    edited_category = st.text_input(
        "Category",
        value=editing_invoice.get("category", ""),
        key="edit_category",
    )
    
    edited_amount = st.number_input(
        "Amount",
        value=float(editing_invoice.get("amount", 0)),
        key="edit_amount",
    )
    
    edited_tax = st.number_input(
        "Tax",
        value=float(editing_invoice.get("tax", 0)),
        key="edit_tax",
    )
    
    edited_total = st.number_input(
        "Total",
        value=float(editing_invoice.get("total", 0)),
        key="edit_total",
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        save_edit_button = st.button("Save Changes", type="primary", use_container_width=True)
    
    with col2:
        re_audit_button = st.button("Save & Re-Audit", use_container_width=True)
    
    if save_edit_button and invoice_no_from_db:
        updates = {
            "vendor": edited_vendor.strip(),
            "date": edited_date.strip(),
            "category": edited_category.strip(),
            "amount": float(edited_amount),
            "tax": float(edited_tax),
            "total": float(edited_total),
        }
        
        try:
            success = update_invoice_record(invoice_no_from_db, updates)
            if success:
                st.session_state.editing_invoice = {**editing_invoice, **updates}
                st.success("Invoice updated successfully!")
            else:
                st.error(f"Failed to update invoice {invoice_no_from_db}.")
        except Exception as e:
            st.error(f"Error updating invoice: {e}")
    
    if re_audit_button and invoice_no_from_db:
        updates = {
            "vendor": edited_vendor.strip(),
            "date": edited_date.strip(),
            "category": edited_category.strip(),
            "amount": float(edited_amount),
            "tax": float(edited_tax),
            "total": float(edited_total),
        }
        
        try:
            # Update invoice with edited data
            success = update_invoice_record(invoice_no_from_db, updates)
            if not success:
                st.error(f"Failed to save changes to invoice {invoice_no_from_db}.")
                st.stop()
            
            # Prepare invoice data for audit
            invoice_data = {
                "document_type": editing_invoice.get("document_type", ""),
                "invoice_no": invoice_no_from_db,
                "date": edited_date.strip(),
                "vendor": edited_vendor.strip(),
                "customer_name": editing_invoice.get("customer_name", ""),
                "gstin": editing_invoice.get("gstin", ""),
                "category": edited_category.strip(),
                "amount": float(edited_amount),
                "tax": float(edited_tax),
                "total": float(edited_total),
                "payment_method": editing_invoice.get("payment_method", ""),
                "ocr_quality": editing_invoice.get("ocr_quality", ""),
                "items": editing_invoice.get("items", []),
                "audit_flags": editing_invoice.get("audit_flags", []),
            }
            
            # Run audit
            with st.spinner("Running audit on updated invoice..."):
                audit_result = audit_invoice(invoice_data)
                audit_summary = create_audit_summary(invoice_data, audit_result)
            
            # Update audit results in database
            audit_updates = {
                "audit_status": audit_result.get("status"),
                "risk_score": audit_result.get("risk_score", 100),
                "issue_count": audit_result.get("issue_count", len(audit_result.get("issues", []))),
            }
            
            update_success = update_invoice_record(invoice_no_from_db, audit_updates)
            if not update_success:
                st.warning("Re-audit completed but could not save audit results to database.")
            
            # Update session state
            st.session_state.invoice_data = invoice_data
            st.session_state.audit_result = audit_result
            st.session_state.audit_summary = audit_summary
            st.session_state.messages = []
            st.session_state.editing_invoice = None
            
            st.success("Invoice updated and re-audited successfully!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error during re-audit: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">&#129302; Audit Agent Analysis</div>', unsafe_allow_html=True)

audit_result = st.session_state.get("audit_result")
invoice_data = st.session_state.get("invoice_data")

if audit_result:
    st.markdown(
        f"""
        <div class="analysis-hero">
            <div class="analysis-eyebrow">AI AUDIT ANALYSIS</div>
            <div class="hero-status">{status_icon(audit_result["status"])} {status_label(audit_result["status"])}</div>
            <div class="risk-score">Risk Score: {risk_score(audit_result)}/100</div>
            <div class="hero-decision">{decision_text(audit_result)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_kpi_cards(audit_result, invoice_data)
    st.markdown(
        f"""
        <div class="status-card {status_tone(audit_result["status"])}">
            <div class="status-head">
                <div class="status-line">Audit Findings</div>
                <div class="status-badge {status_tone(audit_result["status"])}">{audit_result["issue_count"]} issues</div>
            </div>
            {render_issue_html(audit_result)}
        </div>
        <div class="summary-card">
            <div class="summary-title">AI Summary</div>
            <div class="summary-body">{ai_summary_text(audit_result, invoice_data)}</div>
            <div class="recommendation">{recommendation_text(audit_result)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    waiting_result = {"status": "waiting", "issue_count": 0, "issues": []}
    st.markdown(
        """
        <div class="analysis-hero">
            <div class="analysis-eyebrow">AI AUDIT ANALYSIS</div>
            <div class="hero-status">Waiting</div>
            <div class="risk-score">Risk Score: --/100</div>
            <div class="hero-decision">Upload an invoice and run the audit.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_kpi_cards(waiting_result, invoice_data)

if invoice_data:
    with st.expander("Invoice preview and extracted fields", expanded=False):
        render_invoice_preview(uploaded_file)
        st.markdown("**Invoice Summary**")
        render_invoice_summary(invoice_data)
        render_line_items(invoice_data)
        render_audit_flags(invoice_data)
elif uploaded_file:
    with st.expander("Invoice preview", expanded=False):
        render_invoice_preview(uploaded_file)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section">', unsafe_allow_html=True)
render_chat()
st.markdown("</div>", unsafe_allow_html=True)

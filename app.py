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
    save_csv,
    save_json,
    is_valid_invoice,
    _classify_invoice,
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
        {"field": "category", "value": invoice_data.get("category") or "Missing"},
        {"field": "classification", "value": cls_display},
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
            "HSN/SAC": item.get("hsn_sac") or "N/A",
            "Qty": item.get("quantity", 0),
            "Unit Price": item.get("unit_price", 0),
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
            content = message["content"]
            if content.startswith("<"):
                st.markdown(content, unsafe_allow_html=True)
            else:
                st.write(content)

    question = st.chat_input("Why did this invoice fail?")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing invoice..."):
                answer = answer_audit_question(question)
            if answer.startswith("<"):
                st.markdown(answer, unsafe_allow_html=True)
            else:
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

# --- Export Helpers ---

def generate_invoice_csv(invoice_data: dict) -> str:
    import io, csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["INVOICE REPORT", invoice_data.get("invoice_no", "")])
    writer.writerow(["Vendor / Supplier", invoice_data.get("vendor", "")])
    writer.writerow(["Invoice Date", invoice_data.get("date", "")])
    writer.writerow(["Customer Name", invoice_data.get("customer_name", "")])
    writer.writerow(["GSTIN", invoice_data.get("gstin", "")])
    writer.writerow(["Category", invoice_data.get("category", "")])
    writer.writerow(["Classification", invoice_data.get("classification", "Expense")])
    writer.writerow([])
    writer.writerow(["Product / Description", "HSN/SAC", "Qty", "Unit Price", "Amount"])
    for item in invoice_data.get("items", []):
        writer.writerow([
            item.get("product") or item.get("description", ""),
            item.get("hsn_sac", "N/A"),
            item.get("quantity", 0),
            item.get("unit_price", 0),
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
        ("Category:", invoice_data.get("category", "N/A")),
        ("Classification:", invoice_data.get("classification", "Expense")),
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
    page.insert_text((50, y), "Product / Description", fontsize=9, fontname="helvetica-bold")
    page.insert_text((240, y), "HSN/SAC", fontsize=9, fontname="helvetica-bold")
    page.insert_text((330, y), "Qty", fontsize=9, fontname="helvetica-bold")
    page.insert_text((380, y), "Unit Price", fontsize=9, fontname="helvetica-bold")
    page.insert_text((470, y), "Amount", fontsize=9, fontname="helvetica-bold")
    y += 15
    page.draw_line((50, y), (550, y), color=(0.7, 0.7, 0.7), width=0.5)
    y += 15
    
    for item in invoice_data.get("items", []):
        if y > 700:
            page = doc.new_page()
            y = 50
        prod = (item.get("product") or item.get("description", ""))[:32]
        page.insert_text((50, y), prod, fontsize=9, fontname="helvetica")
        page.insert_text((240, y), str(item.get("hsn_sac", "N/A")), fontsize=9, fontname="helvetica")
        page.insert_text((330, y), str(item.get("quantity", 0)), fontsize=9, fontname="helvetica")
        page.insert_text((380, y), f"Rs. {float(item.get('unit_price',0)):,.2f}", fontsize=9, fontname="helvetica")
        page.insert_text((470, y), f"Rs. {float(item.get('amount',0)):,.2f}", fontsize=9, fontname="helvetica")
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
st.title("AI Audit Agent")

# --- Tab Setup ---
tab_upload, tab_search, tab_dashboard = st.tabs([
    "📤 Upload & Audit",
    "🔍 Search & Details",
    "📊 Dashboard"
])

# --- Tab 1: Upload & Audit ---
with tab_upload:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Upload Invoice</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag and drop invoice image or PDF",
        type=["png", "jpg", "jpeg", "pdf"],
        label_visibility="collapsed",
        key="uploader_tab1"
    )

    run_button = st.button("Run Audit", type="primary", use_container_width=True, key="run_btn_tab1")
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
        else:
            st.session_state.invoice_data = invoice_data
            st.session_state.audit_result = audit_result
            st.session_state.audit_summary = audit_summary
            st.session_state.messages = []

            # Persist audit record to MongoDB
            try:
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

    # Output details
    audit_result_t1 = st.session_state.get("audit_result")
    invoice_data_t1 = st.session_state.get("invoice_data")

    if audit_result_t1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">&#129302; Audit Agent Analysis</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="analysis-hero">
                <div class="analysis-eyebrow">AI AUDIT ANALYSIS</div>
                <div class="hero-status">{status_icon(audit_result_t1["status"])} {status_label(audit_result_t1["status"])}</div>
                <div class="risk-score">Risk Score: {risk_score(audit_result_t1)}/100</div>
                <div class="hero-decision">{decision_text(audit_result_t1)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_kpi_cards(audit_result_t1, invoice_data_t1)
        st.markdown(
            f"""
            <div class="status-card {status_tone(audit_result_t1["status"])}">
                <div class="status-head">
                    <div class="status-line">Audit Findings</div>
                    <div class="status-badge {status_tone(audit_result_t1["status"])}">{audit_result_t1["issue_count"]} issues</div>
                </div>
                {render_issue_html(audit_result_t1)}
            </div>
            <div class="summary-card">
                <div class="summary-title">AI Summary</div>
                <div class="summary-body">{ai_summary_text(audit_result_t1, invoice_data_t1)}</div>
                <div class="recommendation">{recommendation_text(audit_result_t1)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Invoice preview and fields", expanded=False):
            render_invoice_preview(uploaded_file)
            st.markdown("**Invoice Summary**")
            render_invoice_summary(invoice_data_t1)
            render_line_items(invoice_data_t1)
            render_audit_flags(invoice_data_t1)
        
        # Download/Export actions
        st.markdown("### Export Invoice")
        col1, col2 = st.columns(2)
        with col1:
            csv_data = generate_invoice_csv(invoice_data_t1)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"invoice_{invoice_data_t1.get('invoice_no', 'export')}.csv",
                mime="text/csv",
                key="csv_download_t1"
            )
        with col2:
            pdf_data = generate_invoice_pdf(invoice_data_t1, audit_result_t1)
            st.download_button(
                label="Download PDF Report",
                data=pdf_data,
                file_name=f"audit_report_{invoice_data_t1.get('invoice_no', 'export')}.pdf",
                mime="application/pdf",
                key="pdf_download_t1"
            )
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="section">', unsafe_allow_html=True)
    render_chat()
    st.markdown("</div>", unsafe_allow_html=True)


# --- Tab 2: Search & Details ---
with tab_search:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Search Invoices</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        s_inv_no = st.text_input("Invoice Number", placeholder="e.g., INV-001", key="s_inv_no")
        s_vendor = st.text_input("Vendor", placeholder="e.g., Amazon", key="s_vendor")
        s_cust = st.text_input("Customer", placeholder="e.g., Rahul", key="s_cust")
        s_gstin = st.text_input("GSTIN", placeholder="e.g., 27AAAAA1111A1Z5", key="s_gstin")
    with col2:
        s_cat = st.text_input("Category", placeholder="e.g., Retail", key="s_cat")
        s_status = st.selectbox("Status", ["All", "Passed", "Warning", "Failed"], key="s_status")
        s_class = st.selectbox("Classification", ["All", "Purchase", "Sales", "Expense"], key="s_class")
        s_date = st.text_input("Date", placeholder="YYYY-MM-DD", key="s_date")
        
    search_triggered = st.button("Search Database", type="primary", use_container_width=True, key="search_triggered_btn")
    st.markdown("</div>", unsafe_allow_html=True)
    
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
        
    if matching_invoices:
        st.markdown(f"**Found {len(matching_invoices)} invoices:**")
        
        # Display simplified table
        table_rows = []
        for inv in matching_invoices:
            table_rows.append({
                "Invoice No": inv.get("invoice_no", "N/A"),
                "Vendor": inv.get("vendor", "N/A"),
                "Date": inv.get("date", "N/A"),
                "Category": inv.get("category", "N/A"),
                "Classification": inv.get("classification", "Expense"),
                "Total": f"Rs. {float(inv.get('total', 0)):,.2f}",
                "Status": str(inv.get("audit_status", "unknown")).upper()
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        
        # Select box for detailed page view
        dropdown_options = [f"{inv.get('invoice_no')} | {inv.get('vendor')} | Total: Rs. {float(inv.get('total',0)):,.2f}" for inv in matching_invoices]
        selected_option = st.selectbox("Select invoice to view complete details page:", dropdown_options, key="select_details_drop")
        
        if selected_option:
            selected_inv_no = selected_option.split(" | ")[0].strip()
            # Fetch complete invoice from Mongo
            invoice = get_invoice_by_number(selected_inv_no)
            if invoice:
                # Re-run or construct audit report
                audit_res = {
                    "status": invoice.get("audit_status", "passed"),
                    "risk_score": invoice.get("risk_score", 100),
                    "issue_count": invoice.get("issue_count", 0),
                    "issues": invoice.get("issues", [])
                }
                
                st.markdown("---")
                st.subheader(f"📄 Invoice Details: {selected_inv_no}")
                
                # Render Detailed view
                st.markdown(
                    f"""
                    <div class="analysis-hero">
                        <div class="analysis-eyebrow">STORED AI AUDIT REPORT</div>
                        <div class="hero-status">{status_icon(audit_res["status"])} {status_label(audit_res["status"])}</div>
                        <div class="risk-score">Risk Score: {audit_res["risk_score"]}/100</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write(f"**Vendor:** {invoice.get('vendor', 'N/A')}")
                    st.write(f"**Customer Name:** {invoice.get('customer_name', 'N/A')}")
                    st.write(f"**Invoice Date:** {invoice.get('date', 'N/A')}")
                    st.write(f"**GSTIN:** {invoice.get('gstin', 'N/A')}")
                with col_d2:
                    cls_val = str(invoice.get("classification") or "Expense").strip().title()
                    cls_disp = "🟠 Expense"
                    if cls_val == "Purchase":
                        cls_disp = "🟢 Purchase"
                    elif cls_val == "Sales":
                        cls_disp = "🔵 Sales"
                    st.write(f"**Category:** {invoice.get('category', 'N/A')}")
                    st.write(f"**Classification:** {cls_disp}")
                    st.write(f"**Subtotal:** Rs. {float(invoice.get('amount', 0)):,.2f}")
                    st.write(f"**Total Amount:** Rs. {float(invoice.get('total', 0)):,.2f}")

                
                # Product Table
                st.markdown("**Product Table**")
                render_line_items(invoice)
                
                # Audit findings
                st.markdown(
                    f"""
                    <div class="status-card {status_tone(audit_res["status"])}">
                        <div class="status-head">
                            <div class="status-line">Audit Findings / Issues</div>
                            <div class="status-badge {status_tone(audit_res["status"])}">{audit_res["issue_count"]} issues</div>
                        </div>
                        {render_issue_html(audit_res)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Export options for stored record
                col1, col2 = st.columns(2)
                with col1:
                    csv_data = generate_invoice_csv(invoice)
                    st.download_button(
                        label="Download CSV Export",
                        data=csv_data,
                        file_name=f"invoice_{selected_inv_no}.csv",
                        mime="text/csv",
                        key="csv_download_t2"
                    )
                with col2:
                    pdf_data = generate_invoice_pdf(invoice, audit_res)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_data,
                        file_name=f"audit_report_{selected_inv_no}.pdf",
                        mime="application/pdf",
                        key="pdf_download_t2"
                    )
    else:
        st.info("No matching records found in database.")



# --- Tab 3: Dashboard ---
with tab_dashboard:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
    
    invoices = get_all_invoices()
    if invoices:
        total_invs = len(invoices)
        passed_count = sum(1 for inv in invoices if str(inv.get("audit_status")).lower() == "passed")
        warning_count = sum(1 for inv in invoices if str(inv.get("audit_status")).lower() == "warning")
        failed_count = sum(1 for inv in invoices if str(inv.get("audit_status")).lower() == "failed")
        
        # Display KPI cards
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Total Invoices</div>
                    <div class="kpi-value">{total_invs}</div>
                </div>
                <div class="kpi-card" style="border-left: 4px solid #22c55e;">
                    <div class="kpi-label">Passed</div>
                    <div class="kpi-value" style="color: #22c55e;">{passed_count}</div>
                </div>
                <div class="kpi-card" style="border-left: 4px solid #f59e0b;">
                    <div class="kpi-label">Warning</div>
                    <div class="kpi-value" style="color: #f59e0b;">{warning_count}</div>
                </div>
                <div class="kpi-card" style="border-left: 4px solid #ef4444;">
                    <div class="kpi-label">Failed</div>
                    <div class="kpi-value" style="color: #ef4444;">{failed_count}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Analytics breakdown calculations
        cat_counts = {}
        class_counts = {"Purchase": 0, "Sales": 0, "Expense": 0}
        total_score = 0
        
        for inv in invoices:
            cat = str(inv.get("category") or "Uncategorized").strip()
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            
            cls = inv.get("classification")
            if not cls:
                # Dynamically classify older stored database invoices
                dummy_data = {
                    "document_type": inv.get("document_type", ""),
                    "category": inv.get("category", "")
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
            st.markdown("**Classification Metrics**")
            st.dataframe([
                {"Classification": k, "Invoices": v} for k, v in class_counts.items()
            ], use_container_width=True, hide_index=True)
            
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 18px; text-align: center; margin-top: 15px;">
                    <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 700;">AVERAGE RISK SCORE</div>
                    <div style="font-size: 2rem; font-weight: 800; color: #a5b4fc;">{avg_score} / 100</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No analytics data available. Insert invoices to display metrics.")

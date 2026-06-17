from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from database import save_invoice_record, check_duplicate_invoice

from audit import audit_invoice, create_audit_summary
from chat import answer_audit_question
from ocr import extract_text_from_image
from parser import parse_invoice_text, save_csv, save_json


load_dotenv()

UPLOADS_DIR = Path("data/uploads")


def save_text(text: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_audit_pipeline(image_path: str) -> tuple[dict, dict, str]:
    raw_text = extract_text_from_image(image_path)
    invoice_data = parse_invoice_text(raw_text)
    audit_result = audit_invoice(invoice_data)
    audit_summary = create_audit_summary(invoice_data, audit_result)

    save_text(raw_text, "outputs/raw_ocr_text.txt")
    save_json(invoice_data, "outputs/extracted_data.json")
    save_csv(invoice_data, "outputs/extracted_data.csv")
    save_json(audit_result, "outputs/audit_report.json")
    save_text(audit_summary, "outputs/audit_report.txt")

    return invoice_data, audit_result, audit_summary


def save_uploaded_file(uploaded_file) -> Path:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOADS_DIR / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())

    return file_path


def status_label(status: str) -> str:
    if status == "passed":
        return "Passed"

    if status == "waiting":
        return "Waiting"

    return "Warning"


def status_icon(status: str) -> str:
    if status == "passed":
        return "&#9989;"

    if status == "waiting":
        return ""

    return "&#9888;"


def status_tone(status: str) -> str:
    if status == "passed":
        return "passed"

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
    score = 100

    for issue in audit_result.get("issues", []):
        severity = issue.get("severity")

        if severity == "high":
            score -= 25
        elif severity == "medium":
            score -= 12
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
    "Drag and drop invoice image",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)

category_input = st.text_input(
    "Invoice category",
    value="",
    placeholder="Electronics, Services, Office supplies",
    help="Enter a category for the invoice if it is not already extracted.",
)

run_button = st.button("Run Audit", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if run_button and uploaded_file:
    file_path = save_uploaded_file(uploaded_file)

    with st.spinner("Audit agent is reading and checking the invoice..."):
        invoice_data, audit_result, audit_summary = run_audit_pipeline(str(file_path))

        if category_input:
            invoice_data["category"] = category_input.strip()

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
            "invoice_no": invoice_no,
            "date": invoice_date,
            "vendor": invoice_data.get("vendor", ""),
            "category": invoice_data.get("category", ""),
            "amount": invoice_data.get("amount", 0),
            "tax": invoice_data.get("tax", 0),
            "total": invoice_data.get("total", invoice_data.get("grand_total", 0)),
            "audit_status": audit_result.get("status"),
            "issue_count": audit_result.get("issue_count", len(audit_result.get("issues", []))),
            "created_at": datetime.utcnow(),
        }

        if invoice_no and check_duplicate_invoice(invoice_no):
            st.warning("Duplicate invoice detected — the record already exists in the database.")
        else:
            inserted_id = save_invoice_record(record)
            st.success(f"Saved audit record to database (id={inserted_id})")
    except Exception as e:
        st.error(f"Failed to save audit record to database: {e}")

if run_button and not uploaded_file:
    st.warning("Please upload an invoice image first.")

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
        st.dataframe([invoice_data], use_container_width=True, hide_index=True)
elif uploaded_file:
    with st.expander("Invoice preview", expanded=False):
        render_invoice_preview(uploaded_file)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="section">', unsafe_allow_html=True)
render_chat()
st.markdown("</div>", unsafe_allow_html=True)

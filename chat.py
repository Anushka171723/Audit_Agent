import json
import os
import re
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

from audit import audit_invoice, create_audit_summary
from database import (
    get_all_invoices,
    get_failed_invoices,
    get_invoice_count,
    get_invoice_by_number,
    get_duplicate_invoices,
    check_duplicate_invoice,
)
from ocr import extract_text_from_image
from parser import parse_invoice_text, save_csv, save_json


def _save_raw_text(text: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@tool("OCRTool")
def ocr_tool(image_path: str) -> str:
    """Extract text from an invoice image, convert it into structured data, and save OCR outputs."""
    raw_text = extract_text_from_image(image_path)
    invoice_data = parse_invoice_text(raw_text)

    _save_raw_text(raw_text, "outputs/raw_ocr_text.txt")
    save_json(invoice_data, "outputs/extracted_data.json")
    save_csv(invoice_data, "outputs/extracted_data.csv")

    result = {
        "message": "OCR extraction completed",
        "raw_text_path": "outputs/raw_ocr_text.txt",
        "json_path": "outputs/extracted_data.json",
        "csv_path": "outputs/extracted_data.csv",
        "extracted_data": invoice_data,
    }

    return json.dumps(result, indent=2)


@tool("AuditTool")
def audit_tool(extracted_json_path: str = "outputs/extracted_data.json") -> str:
    """Audit extracted invoice JSON data and save audit results."""
    input_path = Path(extracted_json_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Extracted data file not found: {extracted_json_path}")

    invoice_data = json.loads(input_path.read_text(encoding="utf-8"))
    audit_result = audit_invoice(invoice_data)
    audit_summary = create_audit_summary(invoice_data, audit_result)

    save_json(audit_result, "outputs/audit_report.json")
    _save_raw_text(audit_summary, "outputs/audit_report.txt")

    result = {
        "message": "Audit completed",
        "audit_json_path": "outputs/audit_report.json",
        "audit_text_path": "outputs/audit_report.txt",
        "audit_result": audit_result,
    }

    return json.dumps(result, indent=2)


@tool("ReportTool")
def report_tool(audit_report_path: str = "outputs/audit_report.txt") -> str:
    """Read the saved audit report and return it as plain text."""
    report_path = Path(audit_report_path)

    if not report_path.exists():
        raise FileNotFoundError(f"Audit report file not found: {audit_report_path}")

    return report_path.read_text(encoding="utf-8")


@tool("ListInvoicesTool")
def list_invoices_tool(limit: int = 25) -> str:
    """Return a summary of recent invoices stored in MongoDB."""
    invoices = get_all_invoices()[:limit]
    if not invoices:
        return "No invoices are stored in the database yet."

    lines = [
        f"{invoice.get('invoice_no', 'UNKNOWN')} | {invoice.get('date', 'N/A')} | {invoice.get('vendor', 'N/A')} | {invoice.get('audit_status', 'unknown')} | {invoice.get('issue_count', 0)} issues"
        for invoice in invoices
    ]
    return "\n".join(lines)


@tool("GetInvoiceTool")
def get_invoice_tool(invoice_no: str) -> str:
    """Return the stored invoice record for the given invoice number."""
    invoice = get_invoice_by_number(invoice_no)
    if not invoice:
        return f"Invoice {invoice_no} was not found."

    return json.dumps(invoice, default=str, indent=2)


@tool("FailedInvoicesTool")
def failed_invoices_tool() -> str:
    """Return a list of failed or warning invoices from MongoDB."""
    invoices = get_failed_invoices()
    if not invoices:
        return "No failed invoices are stored in the database."

    lines = [
        f"{invoice.get('invoice_no', 'UNKNOWN')} | {invoice.get('date', 'N/A')} | {invoice.get('vendor', 'N/A')} | {invoice.get('audit_status', 'unknown')}"
        for invoice in invoices
    ]
    return "\n".join(lines)


@tool("InvoiceCountTool")
def invoice_count_tool() -> str:
    """Return the total count of stored invoices."""
    return str(get_invoice_count())


AUDIT_AGENT_TOOLS = [
    ocr_tool,
    audit_tool,
    report_tool,
    list_invoices_tool,
    get_invoice_tool,
    failed_invoices_tool,
    invoice_count_tool,
]
GROQ_MODEL = "llama-3.3-70b-versatile"


AUDIT_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful audit guide. Answer only from the invoice data and audit report context.",
        ),
        (
            "human",
            "Invoice data:\n{invoice_data}\n\nAudit report:\n{audit_report}\n\nUser question: {question}",
        ),
    ]
)


def _load_json(path: str) -> dict:
    file_path = Path(path)

    if not file_path.exists():
        return {}

    return json.loads(file_path.read_text(encoding="utf-8"))


def load_chat_context() -> dict:
    invoice_data = _load_json("outputs/extracted_data.json")
    audit_result = _load_json("outputs/audit_report.json")
    audit_report_path = Path("outputs/audit_report.txt")

    if audit_report_path.exists():
        audit_report = audit_report_path.read_text(encoding="utf-8")
    else:
        audit_report = ""

    return {
        "invoice_data": invoice_data,
        "audit_result": audit_result,
        "audit_report": audit_report,
    }


def _format_issues(issues: list[dict]) -> str:
    if not issues:
        return "No audit issues were found."

    lines = []

    for index, issue in enumerate(issues, start=1):
        lines.append(
            f"{index}. {issue['severity'].upper()} - {issue['field']}: {issue['message']}"
        )

    return "\n".join(lines)


def _answer_with_local_rules(question: str, invoice_data: dict, audit_result: dict, audit_report: str) -> str:
    question_text = question.lower()
    issues = audit_result.get("issues", [])
    status = audit_result.get("status", "unknown")

    if any(word in question_text for word in ["safe", "approve", "approval", "pass"]):
        if status == "passed":
            return "Yes. Based on the current audit checks, this invoice is safe to approve."

        return (
            "No. This invoice should not be approved yet because the audit found issues:\n"
            f"{_format_issues(issues)}"
        )

    if any(word in question_text for word in ["issue", "issues", "problem", "warning", "risk"]):
        return f"The audit status is {status.upper()}.\n\nIssues found:\n{_format_issues(issues)}"

    if any(word in question_text for word in ["total", "mismatch", "difference"]):
        total_issues = [issue for issue in issues if issue.get("field") == "total"]

        if total_issues:
            return _format_issues(total_issues)

        return "No total mismatch was found in the saved audit report."

    if any(word in question_text for word in ["invoice", "details", "data", "vendor", "amount", "tax", "date"]):
        return (
            f"Invoice No: {invoice_data.get('invoice_no')}\n"
            f"Date: {invoice_data.get('date')}\n"
            f"Vendor: {invoice_data.get('vendor')}\n"
            f"Category: {invoice_data.get('category', 'N/A')}\n"
            f"Amount: {invoice_data.get('amount')}\n"
            f"Tax: {invoice_data.get('tax')}\n"
            f"Total: {invoice_data.get('total')}"
        )

    if any(word in question_text for word in ["report", "summary"]):
        return audit_report

    return (
        f"The invoice audit status is {status.upper()} with {audit_result.get('issue_count', 0)} issue(s).\n"
        "Ask me about approval, issues, invoice details, total mismatch, or the full report."
    )


def _format_invoice_record(invoice: dict) -> str:
    return (
        f"Invoice No: {invoice.get('invoice_no')}\n"
        f"Date: {invoice.get('date')}\n"
        f"Vendor: {invoice.get('vendor')}\n"
        f"Category: {invoice.get('category', 'N/A')}\n"
        f"Amount: {invoice.get('amount')}\n"
        f"Tax: {invoice.get('tax')}\n"
        f"Total: {invoice.get('total')}\n"
        f"Status: {invoice.get('audit_status')}\n"
        f"Issues: {invoice.get('issue_count')}\n"
        f"Created: {invoice.get('created_at')}"
    )


def _extract_invoice_no(question: str) -> str:
    match = re.search(r"invoice\s*(?:number|no\.?|#)?\s*([A-Za-z0-9\-]+)", question, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    match = re.search(r"\b([A-Za-z0-9\-]{3,})\b", question)
    return match.group(1).strip() if match else ""


def _answer_with_db(question: str) -> str:
    question_text = question.lower()

    if any(phrase in question_text for phrase in ["list all invoices", "list invoices", "all invoices", "show all invoices"]):
        invoices = get_all_invoices()
        if not invoices:
            return "No invoices are stored in the database yet."

        lines = [
            f"{invoice.get('invoice_no', 'UNKNOWN')} | {invoice.get('date', 'N/A')} | {invoice.get('vendor', 'N/A')} | {invoice.get('audit_status', 'unknown')} | {invoice.get('issue_count', 0)} issues"
            for invoice in invoices[:25]
        ]
        return "\n".join(lines)

    if "show invoice" in question_text or "display invoice" in question_text:
        invoice_no = _extract_invoice_no(question)

        if not invoice_no:
            return "Please provide an invoice number like INV001 to look up."

        invoice = get_invoice_by_number(invoice_no)
        if not invoice:
            return f"No invoice record found for {invoice_no}."

        return _format_invoice_record(invoice)

    if "show failed invoices" in question_text or "failed invoices" in question_text:
        invoices = get_failed_invoices()
        if not invoices:
            return "No failed invoices are stored in the database."

        lines = [
            f"{invoice.get('invoice_no', 'UNKNOWN')} | {invoice.get('date', 'N/A')} | {invoice.get('vendor', 'N/A')} | {invoice.get('audit_status', 'unknown')}"
            for invoice in invoices
        ]
        return "\n".join(lines)

    if any(phrase in question_text for phrase in ["how many invoices", "how many are stored", "total invoices", "invoice count"]):
        count = get_invoice_count()
        return f"There are {count} invoices stored in the database."

    if "is invoice" in question_text and "approved" in question_text:
        invoice_no = _extract_invoice_no(question)
        if not invoice_no:
            return "Please provide an invoice number like INV001 to check approval."

        invoice = get_invoice_by_number(invoice_no)
        if not invoice:
            return f"No invoice record found for {invoice_no}."

        status = invoice.get("audit_status", "unknown")
        if status == "passed":
            return f"Yes. Invoice {invoice_no} is approved."
        if status == "waiting":
            return f"Invoice {invoice_no} is still pending review."
        return f"No. Invoice {invoice_no} is not approved. Status: {status}."

    if "duplicate invoices" in question_text or "list duplicate" in question_text:
        duplicates = get_duplicate_invoices()
        if not duplicates:
            return "No duplicate invoices were found in the database."

        invoice_numbers = sorted({invoice["invoice_no"] for invoice in duplicates})
        return "Duplicate invoices found: " + ", ".join(invoice_numbers)

    if "duplicate" in question_text:
        invoice_no = _extract_invoice_no(question)
        if invoice_no:
            exists = check_duplicate_invoice(invoice_no)
            return (
                f"Invoice {invoice_no} is a duplicate." if exists else f"Invoice {invoice_no} is not a duplicate."
            )

    return ""


def _answer_with_groq(question: str, invoice_data: dict, audit_report: str) -> str:
    from langchain_groq import ChatGroq

    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    messages = AUDIT_CHAT_PROMPT.format_messages(
        invoice_data=json.dumps(invoice_data, indent=2),
        audit_report=audit_report,
        question=question,
    )
    response = llm.invoke(messages)

    return response.content


def answer_audit_question(question: str, use_llm: bool = True) -> str:
    context = load_chat_context()
    invoice_data = context["invoice_data"]
    audit_result = context["audit_result"]
    audit_report = context["audit_report"]

    db_answer = _answer_with_db(question)
    if db_answer:
        return db_answer

    if not invoice_data or not audit_result:
        return (
            "I cannot find extracted data or an audit report yet for invoice-specific questions. "
            "Run the OCR and audit first or ask a MongoDB-backed query like 'List all invoices'."
        )

    if use_llm and os.getenv("GROQ_API_KEY"):
        try:
            return _answer_with_groq(question, invoice_data, audit_report)
        except Exception as error:
            return (
                f"Groq could not answer this time: {error}\n\n"
                "Local audit answer:\n"
                f"{_answer_with_local_rules(question, invoice_data, audit_result, audit_report)}"
            )

    return _answer_with_local_rules(question, invoice_data, audit_result, audit_report)


def start_chat() -> None:
    print("Audit chat started. Type 'exit' to stop.")
    print("Mode: Groq LLM" if os.getenv("GROQ_API_KEY") else "Mode: local audit guide")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Agent: Chat closed.")
            break

        if not question:
            continue

        answer = answer_audit_question(question)
        print(f"Agent: {answer}")

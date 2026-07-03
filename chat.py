import json
import os
import re
from datetime import datetime, date
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
    get_invoices_by_category,
    check_duplicate_invoice,
    get_invoices_by_date,
)
from ocr import extract_text_from_image
from parser import parse_invoice_text, save_csv, save_json



def _save_raw_text(text: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@tool("OCRTool")
def ocr_tool(image_path: str) -> str:
    """Extract text from an invoice image or PDF, convert it into structured data, and save OCR outputs."""
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
def list_invoices_tool() -> str:
    """Return every invoice stored in MongoDB."""
    return format_invoice_list(get_all_invoices())


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
    amount = invoice.get('amount', 0)
    tax = invoice.get('tax', 0)
    total = invoice.get('total', 0)
    
    return (
        f"Invoice No: {invoice.get('invoice_no')}\n"
        f"Vendor: {invoice.get('vendor')}\n"
        f"Date: {invoice.get('date')}\n"
        f"Customer: {invoice.get('customer_name', 'N/A')}\n"
        f"Amount: ₹{amount:,.2f}\n"
        f"Tax: ₹{tax:,.2f}\n"
        f"Total: ₹{total:,.2f}\n"
        f"Audit Status: {_display_status(invoice.get('audit_status'))}\n"
        f"Issues: {invoice.get('issue_count', 0)}\n"
        f"Created: {invoice.get('created_at')}"
    )


def _display_status(status: object) -> str:
    status_text = str(status or "unknown").strip()

    if not status_text:
        return "Unknown"

    if status_text.lower() == "passed":
        return "Passed"

    if status_text.lower() == "warning":
        return "Warning"

    if status_text.lower() == "failed":
        return "Failed"

    if status_text.lower() == "waiting":
        return "Waiting"

    return status_text[:1].upper() + status_text[1:]


def format_invoice_list(invoices: list[dict], title: str = "Invoices in database:") -> str:
    if not invoices:
        return "No invoices are stored in the database yet."

    lines = [title, ""]

    for index, invoice in enumerate(invoices, start=1):
        lines.append(
            f"{index}. {invoice.get('invoice_no', 'UNKNOWN')} | "
            f"{invoice.get('vendor', 'N/A')} | "
            f"{_display_status(invoice.get('audit_status'))}"
        )

    return "\n".join(lines)


def format_invoice_list_with_category(invoices: list[dict], title: str = "Invoices grouped by category:") -> str:
    """Return styled HTML table of invoices grouped by category."""
    if not invoices:
        return "No invoices are stored in the database yet."

    from collections import defaultdict
    grouped: dict = defaultdict(list)
    for invoice in invoices:
        category = (invoice.get("category") or "Uncategorized").strip()
        grouped[category].append(invoice)

    STATUS_COLORS = {
        "passed":  ("#16a34a", "#dcfce7"),
        "failed":  ("#dc2626", "#fee2e2"),
        "warning": ("#d97706", "#fef3c7"),
    }

    def badge(status: str) -> str:
        key = str(status or "").lower()
        color, bg = STATUS_COLORS.get(key, ("#64748b", "#f1f5f9"))
        label = _display_status(status)
        return (
            f'<span style="background:{bg};color:{color};border:1px solid {color};'
            f'border-radius:4px;padding:2px 8px;font-size:0.78rem;font-weight:700;">'
            f'{label}</span>'
        )

    html_parts = [
        '<style>'
        '.inv-table{width:100%;border-collapse:collapse;font-size:0.88rem;margin-bottom:10px;}'
        '.inv-table th{background:#1e293b;color:#94a3b8;padding:7px 10px;text-align:left;font-weight:600;}'
        '.inv-table td{padding:7px 10px;border-bottom:1px solid #1e293b;color:#e2e8f0;}'
        '.inv-table tr:hover td{background:rgba(99,102,241,0.08);}'
        '.cat-heading{color:#a5b4fc;font-weight:700;font-size:0.95rem;margin:14px 0 4px;}'
        '.inv-summary{color:#94a3b8;font-size:0.82rem;margin-top:8px;}'
        '</style>',
        f'<div style="font-weight:700;font-size:1rem;color:#e2e8f0;margin-bottom:8px;">{title}</div>',
    ]

    total = 0
    for category in sorted(grouped.keys()):
        cat_invoices = grouped[category]
        total += len(cat_invoices)
        html_parts.append(
            f'<div class="cat-heading">📂 {category} &nbsp;'
            f'<span style="color:#64748b;font-size:0.8rem;font-weight:400;">'
            f'({len(cat_invoices)} invoice{"s" if len(cat_invoices) != 1 else ""})</span></div>'
        )
        html_parts.append(
            '<table class="inv-table">'
            '<thead><tr><th>#</th><th>Invoice No</th><th>Vendor</th><th>Status</th></tr></thead><tbody>'
        )
        for i, inv in enumerate(cat_invoices, 1):
            html_parts.append(
                f'<tr>'
                f'<td style="color:#64748b;">{i}</td>'
                f'<td style="font-family:monospace;color:#818cf8;">{inv.get("invoice_no","UNKNOWN")}</td>'
                f'<td>{inv.get("vendor","N/A")}</td>'
                f'<td>{badge(inv.get("audit_status"))}</td>'
                f'</tr>'
            )
        html_parts.append('</tbody></table>')

    html_parts.append(
        f'<div class="inv-summary">Total: {total} invoices across {len(grouped)} categories.</div>'
    )
    return "".join(html_parts)




def _format_invoice_numbers(invoices: list[dict]) -> str:
    if not invoices:
        return "No invoices are stored in the database yet."

    numbers = [str(invoice.get("invoice_no", "UNKNOWN")) for invoice in invoices]
    return "Invoice numbers in database:\n\n" + "\n".join(
        f"{index}. {invoice_no}" for index, invoice_no in enumerate(numbers, start=1)
    )


def _is_invoice_collection_question(question_text: str) -> bool:
    return bool(re.search(r"\binvoices\b", question_text)) and any(
        word in question_text
        for word in ["list", "show", "display", "give", "get", "fetch", "see"]
    )


def _is_invoice_number_collection_question(question_text: str) -> bool:
    return _is_invoice_collection_question(question_text) and bool(
        re.search(r"\binvoice\s*(?:numbers?|nos?)\b|\binv\s*(?:numbers?|nos?)\b", question_text)
    )


def _extract_invoice_no(question: str) -> str:
    match = re.search(
        r"(?:find|show|display|get|lookup|search for)?\s*invoice\s*(?:number|no\.?|#)?\s*([A-Za-z0-9\-]+)",
        question,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()

    match = re.search(r"\b([A-Za-z]{2,}\d+[A-Za-z0-9\-]*)\b", question)
    return match.group(1).strip() if match else ""


def _extract_invoice_number_from_query(question: str) -> str:
    """Extract invoice number from patterns like 'invoice number XYZ', 'having invoice XYZ', etc."""
    # Pattern: "invoice number", "invoice no", "invoice no.", "having invoice number", "having invoice"
    # Captures everything after the pattern until end of sentence (handles spaces, slashes, dashes)
    match = re.search(
        r"(?:invoice\s*(?:number|no\.?)|having\s+invoice\s*(?:number|no\.?)?)\s+([A-Za-z0-9\s/\-]+?)(?:\s+(?:in|from|database|stored|to|for|check)|$|\.|\?)",
        question,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


def answer_database_question(question: str) -> str:
    question_text = question.lower()

    # 1. Handle specific failure reason queries first
    if "why did" in question_text and ("fail" in question_text or "warning" in question_text or "passed" in question_text or "audit" in question_text or "issue" in question_text):
        invoice_no = _extract_invoice_number_from_query(question) or _extract_invoice_no(question)
        if invoice_no:
            invoice = get_invoice_by_number(invoice_no)
            if invoice:
                status = _display_status(invoice.get("audit_status"))
                issues = invoice.get("issues", [])
                if not issues and status == "Passed":
                    return f"Invoice {invoice_no} passed the audit with a risk score of {invoice.get('risk_score', 100)}/100 and has no issues."
                
                issue_lines = []
                for idx, issue in enumerate(issues, 1):
                    severity = str(issue.get("severity", "unknown")).upper()
                    field = str(issue.get("field", "unknown")).replace("_", " ").title()
                    msg = issue.get("message", "")
                    issue_lines.append(f"{idx}. **[{severity}]** {field}: {msg}")
                
                issues_str = "\n".join(issue_lines)
                return (
                    f"Invoice {invoice_no} has status **{status}** (Risk Score: {invoice.get('risk_score', 100)}/100).\n\n"
                    f"**Audit Findings/Issues:**\n{issues_str}"
                )
            else:
                return f"Invoice {invoice_no} was not found in the database."

    # 2. Handle invoice number lookups (before category and generic checks)
    if any(phrase in question_text for phrase in ["invoice number", "invoice no", "having invoice", "find invoice", "show invoice", "display invoice", "get invoice", "lookup invoice", "search invoice"]):
        invoice_no = _extract_invoice_number_from_query(question)
        if not invoice_no:
            # Try alternative extraction
            invoice_no = _extract_invoice_no(question)
        
        if invoice_no:
            invoice = get_invoice_by_number(invoice_no)
            if invoice:
                # Format full invoice details
                lines = [
                    f"Invoice Number: {invoice.get('invoice_no', 'N/A')}",
                    f"Vendor: {invoice.get('vendor', 'N/A')}",
                    f"Date: {invoice.get('date', 'N/A')}",
                    f"Category: {invoice.get('category', 'N/A')}",
                    f"Status: {_display_status(invoice.get('audit_status'))}",
                    f"Amount: ₹{invoice.get('amount', 0):,.2f}",
                    f"Tax: ₹{invoice.get('tax', 0):,.2f}",
                    f"Total: ₹{invoice.get('total', 0):,.2f}",
                ]
                return "\n".join(lines)
            else:
                return f"Invoice {invoice_no} not found in database."

    # 3. Handle today's invoices queries
    if "today" in question_text:
        today_str = datetime.today().strftime("%Y-%m-%d")
        invoices = get_invoices_by_date(today_str)
        if not invoices:
            # Fallback check on datetime created_at timestamps
            all_invoices = get_all_invoices()
            today_invs = []
            for inv in all_invoices:
                created = inv.get("created_at")
                if created:
                    if isinstance(created, datetime):
                        if created.date() == date.today():
                            today_invs.append(inv)
                    elif today_str in str(created):
                        today_invs.append(inv)
            invoices = today_invs

        if invoices:
            return format_invoice_list_with_category(invoices, f"Invoices recorded today ({today_str}):")
        return f"No invoices have been recorded today ({today_str})."

    # 4. Intercept vendor-specific query (e.g. "Show Amazon invoices")
    vendor_match = re.search(r"(?:show|list|get|find|display)\s+(?:me\s+)?([A-Za-z0-9\-]+?)\s+invoices", question_text)
    if vendor_match:
        vendor_name = vendor_match.group(1).strip()
        if vendor_name not in ["all", "retail", "medical", "hospital", "pharmacy", "healthcare", "hotel", "hospitality", "restaurant", "cafe", "food", "travel", "flight", "airline", "service", "services", "utility", "utilities", "failed", "warning", "passed", "approved", "duplicate", "today", "todays"]:
            invoices = get_invoices_by_category(vendor_name)
            if invoices:
                return format_invoice_list_with_category(invoices, f"Invoices matching vendor/keyword '{vendor_name}':")
            else:
                return f"No invoices found matching vendor/keyword '{vendor_name}'."

    # Map keywords to categories (removed vendor specific keywords like amazon, flipkart)
    category_keywords = {
        r"\bretail\b": "Retail",
        r"\bmedical\b|\bhospital\b|\bpharmacy\b|\bhealthcare\b|\bhealth\s*care\b|\bhealth\b|\bclinic\b|\bmedicine\b|\bdrug\b": ["Medical", "Healthcare", "Pharmacy", "Health"],
        r"\bhotel\b|\bhospitality\b": "Hospitality",
        r"\brestaurant\b|\bcafe\b|\bfood\b": "Food",
        r"\btravel\b|\bflight\b|\bairline\b": "Travel",
        r"\bservice\b|\bservices\b": "Services",
        r"\butility\b|\butilities\b|\belectricity\b|\bwater\b|\bgas\b": "Utilities",
    }

    
    matched_categories = []
    for keyword_pattern, categories in category_keywords.items():
        if re.search(keyword_pattern, question_text, re.IGNORECASE):
            if isinstance(categories, list):
                matched_categories.extend(categories)
            else:
                matched_categories.append(categories)
    
    if matched_categories:
        # Search for invoices in matched categories
        all_matches = []
        for category in matched_categories:
            invoices = get_invoices_by_category(category)
            all_matches.extend(invoices)
        
        # Remove duplicates by invoice_no
        seen = set()
        unique_invoices = []
        for invoice in all_matches:
            inv_no = invoice.get("invoice_no")
            if inv_no not in seen:
                seen.add(inv_no)
                unique_invoices.append(invoice)
        
        if not unique_invoices:
            category_display = " / ".join(set(matched_categories))
            return f"No invoices found for category '{category_display}'."
        
        # Format results
        category_display = matched_categories[0]
        lines = [f"{category_display} Invoices Found ({len(unique_invoices)} total):", ""]
        for index, invoice in enumerate(unique_invoices, start=1):
            lines.append(f"{index}. {invoice.get('invoice_no', 'UNKNOWN')} - {invoice.get('vendor', 'N/A')} - {_display_status(invoice.get('audit_status'))}")
        
        return "\n".join(lines)

    if any(
        phrase in question_text
        for phrase in [
            "how many invoices are stored",
            "how many invoices are in database",
            "how many invoices in database",
            "how many invoices",
            "how many are stored",
            "total invoices",
            "invoice count",
            "count invoices",
        ]
    ):
        count = get_invoice_count()
        return f"There are {count} invoices stored in the database."

    if any(
        phrase in question_text
        for phrase in [
            "show duplicate invoices",
            "duplicate invoices",
            "list duplicate invoices",
            "list duplicates",
        ]
    ):
        duplicates = get_duplicate_invoices()
        if not duplicates:
            return "No duplicate invoices were found in the database."

        return format_invoice_list(duplicates, "Duplicate invoices in database:")

    if any(
        phrase in question_text
        for phrase in [
            "show failed invoices",
            "failed invoices",
            "list failed invoices",
            "warning invoices",
            "invoices with warnings",
        ]
    ):
        invoices = get_failed_invoices()
        if not invoices:
            return "No failed invoices are stored in the database."

        return format_invoice_list(invoices, "Failed or warning invoices in database:")

    if any(
        phrase in question_text
        for phrase in [
            "show passed invoices",
            "passed invoices",
            "list passed invoices",
            "approved invoices",
            "give me invoices that passed",
            "invoices that passed",
        ]
    ):
        invoices = [
            invoice
            for invoice in get_all_invoices()
            if str(invoice.get("audit_status", "")).lower() == "passed"
        ]
        if not invoices:
            return "No passed invoices are stored in the database."

        return format_invoice_list(invoices, "Passed invoices in database:")

    if any(
        phrase in question_text
        for phrase in [
            "show all invoice numbers",
            "show all invoice number",
            "list all invoice numbers",
            "list all invoice number",
            "all invoice numbers",
            "all invoice number",
            "invoice numbers",
        ]
    ) or _is_invoice_number_collection_question(question_text):
        invoices = get_all_invoices()
        return _format_invoice_numbers(invoices)

    # Handle "all invoices with their categories" / "show categories" queries
    if any(
        phrase in question_text
        for phrase in [
            "with their categories",
            "with categories",
            "and their categories",
            "and categories",
            "show categories",
            "list categories",
            "invoices and category",
            "invoices with category",
        ]
    ):
        invoices = get_all_invoices()
        return format_invoice_list_with_category(invoices)

    if any(
        phrase in question_text
        for phrase in [
            "list all invoices",
            "list the invoices",
            "list invoices",
            "all invoices",
            "show all invoices",
            "show the invoices",
            "invoices in database",
            "stored invoices",
        ]
    ) or (
        _is_invoice_collection_question(question_text)
        and not matched_categories  # don't override a category match
    ):
        invoices = get_all_invoices()
        return format_invoice_list(invoices)

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

    if "duplicate" in question_text:
        invoice_no = _extract_invoice_no(question)
        if invoice_no:
            exists = check_duplicate_invoice(invoice_no)
            return (
                f"Invoice {invoice_no} is a duplicate." if exists else f"Invoice {invoice_no} is not a duplicate."
            )

    return ""


def _is_audit_specific_question(question: str) -> bool:
    """
    Detect if a question is about the current invoice's audit findings.
    These questions should use the loaded invoice_data and audit_report,
    not MongoDB.
    """
    question_text = question.lower()
    audit_keywords = [
        "why did",
        "why did this invoice fail",
        "why did the invoice fail",
        "what issues",
        "what issues are in",
        "what problems",
        "explain the audit",
        "explain the findings",
        "explain audit findings",
        "is this invoice safe",
        "is it safe to approve",
        "should i approve",
        "audit findings",
        "audit result",
        "audit issues",
        "audit problems",
        "risk score",
        "risk level",
        "what's the risk",
        "audit status",
    ]
    return any(phrase in question_text for phrase in audit_keywords)


def _answer_with_db(question: str) -> str:
    return answer_database_question(question)


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

    # Check if this is an audit-specific question about the currently loaded invoice
    is_audit_specific = _is_audit_specific_question(question)

    if not is_audit_specific:
        # For general invoice queries, search MongoDB first
        db_answer = _answer_with_db(question)
        if db_answer:
            return db_answer
        # If MongoDB search returns nothing for a general query, say so
        return "No invoice data found matching your query. Try asking about the currently loaded invoice's audit findings."

    # For audit-specific questions, use current invoice context
    if not invoice_data or not audit_result:
        return (
            "I cannot find extracted data or an audit report yet for invoice-specific questions. "
            "Run the OCR and audit first or ask a general invoice query like 'List all invoices'."
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

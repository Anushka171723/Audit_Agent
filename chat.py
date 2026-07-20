"""chat.py — AI Invoice Audit Agent chat with three-mode intent classification.

Mode 1: CURRENT_INVOICE  — questions about the currently loaded invoice / audit report
Mode 2: DB_SEARCH        — find / filter invoices in MongoDB
Mode 3: DB_ANALYTICS     — aggregate statistics across all invoices
"""

import json
import os
import re
from collections import defaultdict
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
    search_invoices,
    _get_collection,
)
from ocr import extract_text_from_image
from parser import parse_invoice_text, save_csv, save_json

GROQ_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_raw_text(text: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_json(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_chat_context() -> dict:
    invoice_data = _load_json("outputs/extracted_data.json")
    audit_result  = _load_json("outputs/audit_report.json")
    audit_report_path = Path("outputs/audit_report.txt")
    audit_report = audit_report_path.read_text(encoding="utf-8") if audit_report_path.exists() else ""
    return {"invoice_data": invoice_data, "audit_result": audit_result, "audit_report": audit_report}


def _display_status(status: object) -> str:
    s = str(status or "unknown").strip().lower()
    return {"passed": "🟢 Passed", "warning": "🟡 Warning", "failed": "🔴 Failed",
            "waiting": "⏳ Waiting"}.get(s, s.title())


def _fmt(value: object) -> str:
    """Format a number as Indian Rupees."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "₹0"
    negative = n < 0
    s = f"{abs(n):.0f}"
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3]
        groups: list[str] = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        s = ",".join(groups) + "," + last3
    return ("−₹" if negative else "₹") + s


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _table(invoices: list[dict], title: str = "") -> str:
    """Return a compact markdown table for a list of invoices."""
    if not invoices:
        return "No invoices matched your search."
    lines: list[str] = []
    if title:
        lines.append(f"**{title}**\n")
    lines.append("| # | Invoice No | Vendor | Customer | Amount | Status | Date |")
    lines.append("|---|-----------|--------|----------|--------|--------|------|")
    for i, inv in enumerate(invoices, 1):
        lines.append(
            f"| {i} "
            f"| {inv.get('invoice_no','—')} "
            f"| {inv.get('vendor','—')} "
            f"| {inv.get('customer_name','—')} "
            f"| {_fmt(inv.get('total',0))} "
            f"| {_display_status(inv.get('audit_status'))} "
            f"| {inv.get('date','—')} |"
        )
    lines.append(f"\n*{len(invoices)} invoice(s) found.*")
    return "\n".join(lines)


def _detail(inv: dict) -> str:
    """Return a detailed markdown block for a single invoice."""
    issues = inv.get("issues") or []
    issue_lines = "\n".join(
        f"  - [{i.get('severity','?').upper()}] {i.get('message','')}" for i in issues
    ) or "  None"
    return (
        f"**Invoice No:** `{inv.get('invoice_no','—')}`\n"
        f"**Vendor:** {inv.get('vendor','—')}\n"
        f"**Customer:** {inv.get('customer_name','—')}\n"
        f"**Date:** {inv.get('date','—')}\n"
        f"**Category:** {inv.get('category','—')}\n"
        f"**Classification:** {inv.get('classification','—')}\n"
        f"**Payment:** {inv.get('payment_method','—')}\n"
        f"**Subtotal:** {_fmt(inv.get('amount',0))}\n"
        f"**Tax / GST:** {_fmt(inv.get('tax',0))}\n"
        f"**Grand Total:** {_fmt(inv.get('total',0))}\n"
        f"**Audit Status:** {_display_status(inv.get('audit_status'))}\n"
        f"**Risk Score:** {inv.get('risk_score',100)}/100\n"
        f"**Issues:**\n{issue_lines}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Intent classifier
# ─────────────────────────────────────────────────────────────────────────────

_CURRENT_INVOICE_PHRASES = [
    "this invoice", "current invoice", "the invoice", "this audit",
    "explain audit", "explain findings", "explain the", "why was this",
    "why did this", "summarize this", "is this", "should i approve",
    "safe to approve", "audit findings", "audit result", "risk score",
    "tax calculation", "gst calculation", "total calculation",
]

_ANALYTICS_PHRASES = [
    "total revenue", "highest invoice", "lowest invoice", "average invoice",
    "average amount", "how many invoices", "invoice count", "total amount",
    "which vendor", "top customer", "top vendor", "monthly revenue",
    "revenue by", "how many failed", "how many passed", "how many warning",
    "number of invoices", "statistics", "analytics",
]


def _classify_intent(question: str) -> str:
    """Return 'current', 'analytics', or 'search'."""
    q = question.lower()
    if any(p in q for p in _CURRENT_INVOICE_PHRASES):
        return "current"
    if any(p in q for p in _ANALYTICS_PHRASES):
        return "analytics"
    return "search"


# ─────────────────────────────────────────────────────────────────────────────
# MongoDB search helpers
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_ALIASES: dict[str, str] = {
    "passed": "passed", "approved": "passed", "successful": "passed", "success": "passed",
    "warning": "warning", "needs review": "warning", "review": "warning",
    "failed": "failed", "rejected": "failed", "critical": "failed", "flagged": "failed",
}

_PAYMENT_ALIASES: dict[str, str] = {
    "upi": "upi", "rtgs": "rtgs", "neft": "neft", "imps": "imps",
    "bank transfer": "bank transfer", "credit card": "credit card",
    "debit card": "debit card", "cash": "cash", "net banking": "net banking",
    "wallet": "wallet", "cheque": "cheque", "check": "cheque",
}


def _build_mongo_query(question: str) -> dict:
    """Convert a natural language search question into a MongoDB filter dict."""
    q = question.lower()
    query: dict = {}

    # Status
    for alias, status in _STATUS_ALIASES.items():
        if alias in q:
            query["audit_status"] = status
            break

    # Payment method
    for alias, method in _PAYMENT_ALIASES.items():
        if alias in q:
            query["payment_method"] = {"$regex": re.escape(method), "$options": "i"}
            break

    # Amount thresholds
    above = re.search(r"above\s+[₹rs\.]*\s*([0-9][0-9,]*)", q)
    below = re.search(r"below\s+[₹rs\.]*\s*([0-9][0-9,]*)", q)
    if above:
        query["total"] = {"$gt": float(above.group(1).replace(",", ""))}
    elif below:
        query["total"] = {"$lt": float(below.group(1).replace(",", ""))}

    # Date / month
    today_str = datetime.today().strftime("%Y-%m-%d")
    if "today" in q:
        query["$or"] = [
            {"date": {"$regex": re.escape(today_str), "$options": "i"}},
            {"created_at": {"$gte": datetime.combine(date.today(), datetime.min.time())}},
        ]

    month_map = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    for month_name, month_num in month_map.items():
        if month_name in q:
            query["date"] = {"$regex": f"-{month_num}-", "$options": "i"}
            break

    # Duplicate
    if "duplicate" in q:
        query["audit_flags"] = "duplicate_candidate"

    return query


def _run_db_search(question: str) -> str:
    """Execute a MongoDB search and return a formatted response."""
    q = question.lower()

    # ── Specific invoice number lookup ───────────────────────────────
    inv_no_match = re.search(
        r"(?:invoice\s*(?:number|no\.?|#)?)\s+([A-Za-z0-9][A-Za-z0-9\-/_.]+)",
        question, re.IGNORECASE,
    )
    if inv_no_match:
        inv = get_invoice_by_number(inv_no_match.group(1).strip())
        if inv:
            return _detail(inv)
        return f"No invoice found with number **{inv_no_match.group(1).strip()}**."

    # ── Latest invoice ────────────────────────────────────────────────
    if "latest" in q or "most recent" in q or "last invoice" in q:
        all_inv = get_all_invoices()
        if all_inv:
            return _detail(all_inv[0])
        return "No invoices are stored yet."

    # ── Duplicate invoices ────────────────────────────────────────────
    if "duplicate" in q:
        dups = get_duplicate_invoices()
        return _table(dups, "Duplicate Invoices") if dups else "No duplicate invoices found."

    # ── Vendor / customer keyword search ─────────────────────────────
    vendor_match = re.search(
        r"(?:from|by|vendor|seller|invoices?\s+(?:from|by)|for\s+vendor)\s+([A-Za-z][A-Za-z0-9 .&,\-]+?)(?:\s+invoices?|\s*$|\s+(?:in|on|above|below|with|where))",
        question, re.IGNORECASE,
    )
    customer_match = re.search(
        r"(?:for|customer|buyer|client|to)\s+([A-Za-z][A-Za-z0-9 .&,\-]+?)(?:\s+invoices?|\s*$|\s+(?:in|on|above|below|with|where))",
        question, re.IGNORECASE,
    )

    if vendor_match:
        results = search_invoices({"vendor": vendor_match.group(1).strip()})
        return _table(results, f"Invoices from '{vendor_match.group(1).strip()}'")

    if customer_match:
        cust = customer_match.group(1).strip()
        if cust.lower() not in {"vendor", "customer", "buyer", "client"}:
            results = search_invoices({"customer_name": cust})
            return _table(results, f"Invoices for customer '{cust}'")

    # ── Category / classification search ─────────────────────────────
    cat_keywords = {
        "retail": "Retail", "electronics": "Electronics", "healthcare": "Healthcare",
        "medical": "Healthcare", "food": "Food", "restaurant": "Food",
        "hospitality": "Hospitality", "hotel": "Hospitality",
        "travel": "Travel", "services": "Services", "utilities": "Utilities",
        "office supplies": "Office Supplies",
    }
    for kw, cat in cat_keywords.items():
        if kw in q:
            results = get_invoices_by_category(cat)
            if results:
                return _table(results, f"{cat} Invoices")

    # ── Build and execute a general query ────────────────────────────
    mongo_query = _build_mongo_query(question)
    try:
        coll = _get_collection()
        results = list(coll.find(mongo_query).sort("created_at", -1).limit(50))
    except Exception:
        results = get_all_invoices()

    if mongo_query and not results:
        return "No invoices matched your search."

    # ── Fallback: show all ────────────────────────────────────────────
    if not mongo_query:
        results = get_all_invoices()

    title = "All Invoices" if not mongo_query else "Matching Invoices"
    return _table(results, title)


# ─────────────────────────────────────────────────────────────────────────────
# Analytics mode
# ─────────────────────────────────────────────────────────────────────────────

def _run_analytics(question: str) -> str:
    """Compute aggregated statistics from MongoDB and return a markdown response."""
    q = question.lower()
    invoices = get_all_invoices()

    if not invoices:
        return "No invoices are stored yet — nothing to analyse."

    totals = [float(inv.get("total") or 0) for inv in invoices]
    count  = len(invoices)

    # Count by status
    by_status: dict[str, int] = defaultdict(int)
    for inv in invoices:
        by_status[str(inv.get("audit_status") or "unknown").lower()] += 1

    # Revenue by vendor
    by_vendor: dict[str, float] = defaultdict(float)
    for inv in invoices:
        by_vendor[inv.get("vendor") or "Unknown"] += float(inv.get("total") or 0)

    # Revenue by customer
    by_customer: dict[str, float] = defaultdict(float)
    for inv in invoices:
        by_customer[inv.get("customer_name") or "Unknown"] += float(inv.get("total") or 0)

    top_vendor   = max(by_vendor, key=by_vendor.__getitem__) if by_vendor else "—"
    top_customer = max(by_customer, key=by_customer.__getitem__) if by_customer else "—"

    if any(p in q for p in ["highest", "maximum", "max invoice", "largest"]):
        inv = max(invoices, key=lambda x: float(x.get("total") or 0))
        return f"**Highest invoice:** {_detail(inv)}"

    if any(p in q for p in ["lowest", "minimum", "min invoice", "smallest"]):
        inv = min((i for i in invoices if float(i.get("total") or 0) > 0),
                  key=lambda x: float(x.get("total") or 0), default=invoices[-1])
        return f"**Lowest invoice:** {_detail(inv)}"

    if "average" in q or "mean" in q:
        avg = sum(totals) / count if count else 0
        return f"**Average invoice amount:** {_fmt(avg)} across {count} invoices."

    if "total revenue" in q or "total amount" in q or "sum" in q:
        return f"**Total revenue across all invoices:** {_fmt(sum(totals))} ({count} invoices)."

    if "top vendor" in q or "highest vendor" in q or "which vendor" in q:
        rows = sorted(by_vendor.items(), key=lambda x: x[1], reverse=True)[:5]
        lines = ["**Top Vendors by Revenue:**", "| Vendor | Total |", "|--------|-------|"]
        lines += [f"| {v} | {_fmt(t)} |" for v, t in rows]
        return "\n".join(lines)

    if "top customer" in q or "which customer" in q:
        rows = sorted(by_customer.items(), key=lambda x: x[1], reverse=True)[:5]
        lines = ["**Top Customers by Revenue:**", "| Customer | Total |", "|----------|-------|"]
        lines += [f"| {c} | {_fmt(t)} |" for c, t in rows]
        return "\n".join(lines)

    if "monthly" in q or "month" in q:
        by_month: dict[str, float] = defaultdict(float)
        for inv in invoices:
            d = str(inv.get("date") or inv.get("created_at") or "")
            m = d[:7] if len(d) >= 7 else "Unknown"
            by_month[m] += float(inv.get("total") or 0)
        rows = sorted(by_month.items())
        lines = ["**Monthly Revenue:**", "| Month | Total |", "|-------|-------|"]
        lines += [f"| {m} | {_fmt(t)} |" for m, t in rows]
        return "\n".join(lines)

    # Default: full summary
    return (
        f"**Invoice Analytics Summary**\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| Total Invoices | {count} |\n"
        f"| Total Revenue | {_fmt(sum(totals))} |\n"
        f"| Average Invoice | {_fmt(sum(totals)/count if count else 0)} |\n"
        f"| Highest Invoice | {_fmt(max(totals) if totals else 0)} |\n"
        f"| Lowest Invoice | {_fmt(min(t for t in totals if t > 0) if any(t > 0 for t in totals) else 0)} |\n"
        f"| Passed | {by_status.get('passed', 0)} |\n"
        f"| Warning | {by_status.get('warning', 0)} |\n"
        f"| Failed | {by_status.get('failed', 0)} |\n"
        f"| Top Vendor | {top_vendor} |\n"
        f"| Top Customer | {top_customer} |"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Current invoice mode
# ─────────────────────────────────────────────────────────────────────────────

AUDIT_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert invoice auditor. Answer only from the invoice data and audit "
        "report provided. Be concise and precise. Use markdown formatting.",
    ),
    (
        "human",
        "Invoice data:\n{invoice_data}\n\nAudit report:\n{audit_report}\n\nQuestion: {question}",
    ),
])


def _answer_current_invoice_llm(question: str, invoice_data: dict, audit_report: str) -> str:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    messages = AUDIT_CHAT_PROMPT.format_messages(
        invoice_data=json.dumps(invoice_data, indent=2, default=str),
        audit_report=audit_report,
        question=question,
    )
    return llm.invoke(messages).content


def _answer_current_invoice_rules(question: str, invoice_data: dict, audit_result: dict, audit_report: str) -> str:
    q = question.lower()
    issues = audit_result.get("issues", [])
    status = audit_result.get("status", "unknown")

    if any(w in q for w in ["safe", "approve", "approval"]):
        if status == "passed":
            return "✅ Yes. This invoice is safe to approve — no audit issues detected."
        issue_lines = "\n".join(f"- [{i.get('severity','?').upper()}] {i.get('message','')}" for i in issues)
        return f"❌ No. The audit found the following issues:\n{issue_lines}"

    if any(w in q for w in ["issue", "problem", "warning", "flag", "finding"]):
        if not issues:
            return "✅ No issues found. The invoice passed all audit checks."
        lines = "\n".join(f"- [{i.get('severity','?').upper()}] {i.get('message','')}" for i in issues)
        return f"**Audit Status:** {_display_status(status)}\n\n**Issues:**\n{lines}"

    if any(w in q for w in ["total", "tax", "gst", "calculation", "amount", "subtotal"]):
        return (
            f"**Financial Summary:**\n"
            f"- Subtotal: {_fmt(invoice_data.get('amount'))}\n"
            f"- Discount: {_fmt(invoice_data.get('discount', 0))}\n"
            f"- GST / Tax: {_fmt(invoice_data.get('tax'))}\n"
            f"- Grand Total: {_fmt(invoice_data.get('total'))}"
        )

    if any(w in q for w in ["summarize", "summary", "details", "about"]):
        return _detail({**invoice_data, "audit_status": status,
                        "risk_score": audit_result.get("risk_score", 100), "issues": issues})

    return audit_report or "No audit report is available for the current invoice."


# ─────────────────────────────────────────────────────────────────────────────
# LLM-powered intent router for ambiguous queries
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_PROMPT = """You are an invoice assistant intent classifier.
Classify the user question into exactly one of: CURRENT_INVOICE, DB_SEARCH, DB_ANALYTICS.

CURRENT_INVOICE: questions about the currently loaded invoice or its audit findings.
  Examples: "Why did this invoice fail?", "Summarize this invoice.", "Explain the GST."

DB_SEARCH: find, filter, or list invoices from the database.
  Examples: "Show all passed invoices.", "Find invoices from TechMart.", "UPI invoices."

DB_ANALYTICS: statistics, totals, averages, counts across all invoices.
  Examples: "Total revenue?", "Which vendor has the highest sales?", "How many failed?"

Reply with ONLY one word: CURRENT_INVOICE, DB_SEARCH, or DB_ANALYTICS.

Question: {question}"""


def _llm_classify_intent(question: str) -> str:
    """Use Groq to classify intent when heuristics are ambiguous."""
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=GROQ_MODEL, temperature=0)
        response = llm.invoke(_INTENT_PROMPT.format(question=question))
        result = response.content.strip().upper()
        if "CURRENT" in result:
            return "current"
        if "ANALYTIC" in result:
            return "analytics"
        return "search"
    except Exception:
        return "search"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def answer_audit_question(question: str, use_llm: bool = True) -> str:
    """Route the question to the correct mode and return a markdown-formatted answer."""
    context      = load_chat_context()
    invoice_data = context["invoice_data"]
    audit_result = context["audit_result"]
    audit_report = context["audit_report"]

    # 1. Heuristic classification (fast, no LLM call)
    intent = _classify_intent(question)

    # 2. If heuristic is ambiguous ("search"), refine with LLM when available
    if intent == "search" and use_llm and os.getenv("GROQ_API_KEY"):
        intent = _llm_classify_intent(question)

    # ── Mode 1: Current invoice ───────────────────────────────────────
    if intent == "current":
        if not invoice_data:
            return (
                "No invoice is currently loaded. Upload or generate an invoice first, "
                "or ask me to search the database — e.g. *'Show all invoices'*."
            )
        if use_llm and os.getenv("GROQ_API_KEY"):
            try:
                return _answer_current_invoice_llm(question, invoice_data, audit_report)
            except Exception as err:
                return _answer_current_invoice_rules(question, invoice_data, audit_result, audit_report)
        return _answer_current_invoice_rules(question, invoice_data, audit_result, audit_report)

    # ── Mode 2: Analytics ────────────────────────────────────────────
    if intent == "analytics":
        return _run_analytics(question)

    # ── Mode 3: DB search (default) ──────────────────────────────────
    return _run_db_search(question)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers kept for compatibility with app.py and other callers
# ─────────────────────────────────────────────────────────────────────────────

def format_invoice_list(invoices: list[dict], title: str = "Invoices in database:") -> str:
    return _table(invoices, title)


def format_invoice_list_with_category(invoices: list[dict], title: str = "Invoices:") -> str:
    return _table(invoices, title)


def answer_database_question(question: str) -> str:
    return _run_db_search(question)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def start_chat() -> None:
    print("Audit chat started. Type 'exit' to stop.")
    print("Mode: Groq LLM" if os.getenv("GROQ_API_KEY") else "Mode: local rules")
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit"]:
            print("Agent: Chat closed.")
            break
        if not question:
            continue
        print(f"Agent: {answer_audit_question(question)}")

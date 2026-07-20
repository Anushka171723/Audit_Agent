import re
from datetime import date, datetime


REQUIRED_FIELDS = [
    "invoice_no",
    "date",
    "vendor",
    "customer_name",
    "gstin",
    "classification",
    "document_type",
    "amount",
    "total",
]
HIGH_AMOUNT_LIMIT = 100_000
TOTAL_TOLERANCE = 1.0
GSTIN_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")
ISSUE_PENALTIES = {
    "high": 25,
    "medium": 10,
    "low": 5,
}
AUDIT_FLAG_ISSUES = {
    "missing_gstin": ("gstin", "GSTIN is missing", "medium"),
    "duplicate_candidate": ("duplicate", "Duplicate candidate detected", "high"),
    "poor_ocr": ("ocr_quality", "Poor OCR quality detected", "medium"),
    "future_date": ("date", "Future invoice date detected", "high"),
}
CRITICAL_FLAGS = {
    "future_date",
    "missing_invoice_no",
    "missing_vendor",
    "duplicate_candidate",
}


def _add_issue(issues: list[dict], field: str, message: str, severity: str) -> None:
    if any(issue.get("field") == field and issue.get("message") == message for issue in issues):
        return

    issues.append(
        {
            "field": field,
            "message": message,
            "severity": severity,
        }
    )


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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


def calculate_risk_score(issues: list[dict]) -> int:
    score = 100

    for issue in issues:
        score -= ISSUE_PENALTIES.get(issue.get("severity"), 5)

    return max(score, 0)


def status_from_score(score: int, force_warning: bool = False, issue_count: int = 0, critical_issues: bool = False) -> str:
    # If there are critical issues, status can never be "passed"
    if critical_issues:
        if score >= 70:
            return "warning"
        return "failed"
    
    # If there are any issues, status cannot be "passed"
    if issue_count > 0 and score >= 90:
        return "warning"

    if force_warning and score >= 90:
        return "warning"

    if score >= 90:
        return "passed"

    if score >= 70:
        return "warning"

    return "failed"


def audit_invoice(invoice_data: dict) -> dict:
    issues = []
    audit_flags = invoice_data.get("audit_flags", [])

    if not isinstance(audit_flags, list):
        audit_flags = []

    # Check critical fields with high severity
    if not invoice_data.get("invoice_no"):
        _add_issue(issues, "invoice_no", "Invoice number is missing", "high")
    
    if not invoice_data.get("vendor"):
        _add_issue(issues, "vendor", "Vendor name is missing", "high")
    
    # Check other required fields
    for field in REQUIRED_FIELDS:
        if field in ["invoice_no", "vendor"]:
            continue  # Already checked above
        if not invoice_data.get(field):
            severity = "high" if field in ["date", "total"] else "medium"
            _add_issue(issues, field, f"{field} is missing", severity)

    for flag in audit_flags:
        issue = AUDIT_FLAG_ISSUES.get(flag)
        if issue:
            _add_issue(issues, *issue)

    invoice_date = _parse_invoice_date(invoice_data.get("date"))
    if invoice_date and invoice_date > date.today():
        _add_issue(issues, "date", "Future invoice date detected", "high")

    gstin = str(invoice_data.get("gstin") or "").strip().upper()
    if gstin and not GSTIN_PATTERN.match(gstin):
        _add_issue(issues, "gstin", "GSTIN format is invalid", "high")

    # Extract and normalize all amounts
    amount = _to_float(invoice_data.get("amount"))
    discount = _to_float(invoice_data.get("discount"))
    tax = _to_float(invoice_data.get("tax"))
    total = _to_float(invoice_data.get("total"))
    items = invoice_data.get("items") if isinstance(invoice_data.get("items"), list) else []

    # ── Financial reconciliation ──────────────────────────────────────
    # Rule: taxable = subtotal(items) - discount
    #       total   = taxable + GST
    line_items_total = sum(_to_float(item.get("amount")) for item in items if isinstance(item, dict))

    if items and line_items_total > 0:
        calculated_taxable = round(line_items_total - discount, 2)
    else:
        calculated_taxable = round(amount - discount, 2)

    calculated_total = round(calculated_taxable + tax, 2)

    debugging_info = {
        "calculated_line_items_total": round(line_items_total, 2),
        "calculated_taxable_amount":   round(calculated_taxable, 2),
        "calculated_total":            round(calculated_total, 2),
        "extracted_amount":            round(amount, 2),
        "extracted_discount":          round(discount, 2),
        "extracted_tax":               round(tax, 2),
        "extracted_total":             round(total, 2),
        "difference":                  round(abs(calculated_total - total), 2),
    }

    # Line-item subtotal must match stored amount (pre-discount)
    if items and line_items_total > 0:
        subtotal_diff = abs(line_items_total - amount)
        if subtotal_diff > TOTAL_TOLERANCE:
            _add_issue(
                issues,
                "items",
                f"Line item subtotal {line_items_total:.2f} does not match stored amount {amount:.2f}",
                "medium",
            )

    if amount <= 0:
        _add_issue(issues, "amount", "Amount should be greater than zero", "high")

    if total <= 0:
        _add_issue(issues, "total", "Total should be greater than zero", "high")

    # Grand total must equal taxable + GST
    if total > 0 and abs(calculated_total - total) > TOTAL_TOLERANCE:
        _add_issue(
            issues,
            "total_mismatch",
            f"Total mismatch: expected {calculated_total:.2f} (taxable {calculated_taxable:.2f} + GST {tax:.2f}), found {total:.2f}",
            "high",
        )

    if str(invoice_data.get("ocr_quality") or "").strip().lower() == "poor":
        _add_issue(issues, "ocr_quality", "Poor OCR quality detected", "medium")

    if amount > HIGH_AMOUNT_LIMIT:
        _add_issue(
            issues,
            "high_value",
            (
                f"High Value Invoice | "
                f"Invoice Amount: ₹{amount:,.0f} | "
                f"Review Threshold: ₹{HIGH_AMOUNT_LIMIT:,.0f} | "
                f"Invoices above this limit require manual approval. "
                f"Verify purchase order and obtain manager approval before processing."
            ),
            "medium",
        )

    score = calculate_risk_score(issues)
    
    # Check if there are critical issues that prevent "passed" status
    has_critical_issues = any(flag in CRITICAL_FLAGS for flag in audit_flags)
    if not has_critical_issues:
        # Also check if any issue relates to critical fields
        critical_issue_fields = {"invoice_no", "vendor", "date"}
        for issue in issues:
            if issue.get("field") in critical_issue_fields and issue.get("severity") == "high":
                has_critical_issues = True
                break
    
    status = status_from_score(score, force_warning=False, issue_count=len(issues), critical_issues=has_critical_issues)

    return {
        "status": status,
        "risk_score": score,
        "issue_count": len(issues),
        "issues": issues,
        "debugging_info": debugging_info,
    }


def create_audit_summary(invoice_data: dict, audit_result: dict) -> str:
    lines = [
        "AUDIT REPORT",
        "============",
        f"Invoice No: {invoice_data.get('invoice_no') or 'Missing'}",
        f"Document Type: {invoice_data.get('document_type') or 'Missing'}",
        f"Date: {invoice_data.get('date') or 'Missing'}",
        f"Vendor: {invoice_data.get('vendor') or 'Missing'}",
        f"Customer: {invoice_data.get('customer_name') or 'Missing'}",
        f"GSTIN: {invoice_data.get('gstin') or 'Missing'}",
        f"Document Category: {invoice_data.get('category') or 'Missing'}",
        f"Business Classification: {invoice_data.get('classification') or 'Missing'}",
        f"Amount: {invoice_data.get('amount')}",
        f"Discount: {invoice_data.get('discount', 0)}",
        f"Tax: {invoice_data.get('tax')}",
        f"Total: {invoice_data.get('total')}",
        f"Payment Method: {invoice_data.get('payment_method') or 'Missing'}",
        f"OCR Quality: {invoice_data.get('ocr_quality') or 'Missing'}",
        f"Extraction Audit Flags: {', '.join(invoice_data.get('audit_flags', [])) or 'None'}",
        "",
    ]

    items = invoice_data.get("items", [])
    if items:
        lines.append("Line Items:")
        for index, item in enumerate(items, start=1):
            lines.append(
                f"{index}. Product: {item.get('product') or 'Item'} | "
                f"Description: {item.get('description') or 'N/A'} | "
                f"HSN/SAC: {item.get('hsn_sac') or 'N/A'} | "
                f"Qty: {item.get('quantity', 0)} | "
                f"Unit Price: {item.get('unit_price', 0)} | "
                f"Tax: {item.get('tax', 0)} | "
                f"Amount: {item.get('amount', 0)}"
            )
        lines.append("")

    lines.extend(
        [
            f"Audit Status: {audit_result['status'].upper()}",
            f"Risk Score: {audit_result.get('risk_score', 100)}/100",
            f"Issues Found: {audit_result['issue_count']}",
            "",
        ]
    )

    issue_count = audit_result.get('issue_count', 0)
    if issue_count == 0:
        lines.append("The invoice was successfully validated. The total amount matches the tax calculation and all required fields are present.")
    else:
        lines.append(f"The audit found {issue_count} issue(s). Review the highlighted findings before approval.")
        lines.append("")
        lines.append("Issues:")

        for index, issue in enumerate(audit_result["issues"], start=1):
            lines.append(
                f"{index}. [{issue['severity'].upper()}] {issue['field']}: {issue['message']}"
            )

    # Add debugging info if present
    if "debugging_info" in audit_result:
        debug = audit_result["debugging_info"]
        lines.extend([
            "",
            "Financial Reconciliation (Debug Info):",
            f"Calculated Line Items Total: {debug.get('calculated_line_items_total', 0):.2f}",
            f"Calculated Taxable Amount: {debug.get('calculated_taxable_amount', 0):.2f}",
            f"Extracted Amount: {debug.get('extracted_amount', 0):.2f}",
            f"Discount: {debug.get('extracted_discount', 0):.2f}",
            f"Extracted Tax: {debug.get('extracted_tax', 0):.2f}",
            f"Calculated Total: {debug.get('calculated_total', 0):.2f}",
            f"Extracted Total: {debug.get('extracted_total', 0):.2f}",
            f"Difference: {debug.get('difference', 0):.2f}",
        ])

    return "\n".join(lines)

REQUIRED_FIELDS = ["invoice_no", "date", "vendor", "amount", "total"]
HIGH_AMOUNT_LIMIT = 100000
TOTAL_TOLERANCE = 1.0


def _add_issue(issues: list[dict], field: str, message: str, severity: str) -> None:
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


def audit_invoice(invoice_data: dict) -> dict:
    issues = []

    for field in REQUIRED_FIELDS:
        if not invoice_data.get(field):
            _add_issue(issues, field, f"{field} is missing", "high")

    amount = _to_float(invoice_data.get("amount"))
    tax = _to_float(invoice_data.get("tax"))
    total = _to_float(invoice_data.get("total"))

    expected_total = amount + tax
    difference = abs(total - expected_total)

    if amount <= 0:
        _add_issue(issues, "amount", "Amount should be greater than zero", "high")

    if total <= 0:
        _add_issue(issues, "total", "Total should be greater than zero", "high")

    if total > 0 and difference > TOTAL_TOLERANCE:
        _add_issue(
            issues,
            "total",
            f"Total mismatch: expected {expected_total:.2f}, found {total:.2f}",
            "high",
        )

    if amount > HIGH_AMOUNT_LIMIT:
        _add_issue(
            issues,
            "amount",
            f"Amount is above the review limit of {HIGH_AMOUNT_LIMIT}",
            "medium",
        )

    if issues:
        status = "warning"
    else:
        status = "passed"

    return {
        "status": status,
        "issue_count": len(issues),
        "issues": issues,
    }


def create_audit_summary(invoice_data: dict, audit_result: dict) -> str:
    lines = [
        "AUDIT REPORT",
        "============",
        f"Invoice No: {invoice_data.get('invoice_no') or 'Missing'}",
        f"Date: {invoice_data.get('date') or 'Missing'}",
        f"Vendor: {invoice_data.get('vendor') or 'Missing'}",
        f"Amount: {invoice_data.get('amount')}",
        f"Tax: {invoice_data.get('tax')}",
        f"Total: {invoice_data.get('total')}",
        "",
        f"Audit Status: {audit_result['status'].upper()}",
        f"Issues Found: {audit_result['issue_count']}",
        "",
    ]

    if not audit_result["issues"]:
        lines.append("No audit issues were found.")
    else:
        lines.append("Issues:")

        for index, issue in enumerate(audit_result["issues"], start=1):
            lines.append(
                f"{index}. [{issue['severity'].upper()}] {issue['field']}: {issue['message']}"
            )

    return "\n".join(lines)

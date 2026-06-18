import os
import re
from datetime import datetime
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection


_client: Optional[MongoClient] = None


def get_database():
    """Return the configured MongoDB database instance."""
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set in environment")

    client = MongoClient(uri)
    db_name = os.getenv("MONGODB_DB", "audit_agent")
    return client[db_name]


def _get_collection() -> Collection:
    db = get_database()
    coll_name = os.getenv("MONGODB_COLLECTION", "invoices")
    return db[coll_name]


def save_invoice_record(record: dict) -> str:
    """Insert an invoice record into MongoDB and return the inserted id as a string."""
    coll = _get_collection()

    if "created_at" not in record:
        record["created_at"] = datetime.utcnow()

    result = coll.insert_one(record)
    return str(result.inserted_id)


def get_invoice_by_number(invoice_no: str) -> Optional[dict]:
    """Return the invoice document matching `invoice_no` or None. Case-insensitive search."""
    if not invoice_no:
        return None
    coll = _get_collection()
    # Use case-insensitive regex search
    return coll.find_one({"invoice_no": {"$regex": f"^{re.escape(invoice_no)}$", "$options": "i"}})


def check_duplicate_invoice(invoice_no: str) -> bool:
    """Return True if an invoice with `invoice_no` already exists."""
    if not invoice_no:
        return False
    coll = _get_collection()
    return coll.count_documents({"invoice_no": invoice_no}, limit=1) > 0


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def find_duplicate_invoice(record: dict) -> Optional[dict]:
    """Return a likely duplicate invoice by invoice number or by core invoice fields."""
    coll = _get_collection()
    invoice_no = str(record.get("invoice_no") or "").strip()

    if invoice_no:
        duplicate = coll.find_one({"invoice_no": invoice_no})
        if duplicate:
            return duplicate

    vendor = str(record.get("vendor") or "").strip()
    invoice_date = str(record.get("date") or "").strip()
    amount = _to_float(record.get("amount"))
    total = _to_float(record.get("total"))

    if not vendor or not invoice_date or amount <= 0 or total <= 0:
        return None

    return coll.find_one(
        {
            "vendor": vendor,
            "date": invoice_date,
            "amount": amount,
            "total": total,
        }
    )


def check_duplicate_invoice_record(record: dict) -> bool:
    """Return True if an invoice is duplicated by number or by vendor/date/amount/total."""
    return find_duplicate_invoice(record) is not None


def get_all_invoices() -> list[dict]:
    """Return all invoice records."""
    coll = _get_collection()
    return list(coll.find().sort("created_at", -1))


def get_failed_invoices() -> list[dict]:
    """Return all invoices that failed or have warnings."""
    coll = _get_collection()
    return list(coll.find({"audit_status": {"$ne": "passed"}}).sort("created_at", -1))


def get_invoice_count() -> int:
    """Return the total number of invoices stored in the database."""
    coll = _get_collection()
    return coll.count_documents({})


def get_duplicate_invoices() -> list[dict]:
    """Return records that share an invoice number or core invoice fields."""
    all_invoices = get_all_invoices()
    seen_numbers = {}
    seen_core_fields = {}

    for invoice in all_invoices:
        invoice_no = invoice.get("invoice_no")
        if invoice_no:
            seen_numbers.setdefault(invoice_no, []).append(invoice)

        core_key = (
            invoice.get("vendor"),
            invoice.get("date"),
            _to_float(invoice.get("amount")),
            _to_float(invoice.get("total")),
        )
        if all(core_key):
            seen_core_fields.setdefault(core_key, []).append(invoice)

    duplicate_ids = set()
    duplicates = []

    for group in list(seen_numbers.values()) + list(seen_core_fields.values()):
        if len(group) <= 1:
            continue

        for record in group:
            record_id = record.get("_id")
            identity = str(record_id) if record_id is not None else id(record)
            if identity in duplicate_ids:
                continue
            duplicate_ids.add(identity)
            duplicates.append(record)

    return duplicates


def get_invoices_by_category(category: str) -> list[dict]:
    """Return all invoices matching the given category."""
    if not category:
        return []
    coll = _get_collection()
    return list(coll.find({"category": {"$regex": f"^{category}$", "$options": "i"}}).sort("created_at", -1))


def update_invoice_record(invoice_no: str, updates: dict) -> bool:
    """Update an invoice record by invoice_no. Returns True if successful."""
    if not invoice_no:
        return False
    coll = _get_collection()
    result = coll.update_one({"invoice_no": invoice_no}, {"$set": updates})
    return result.matched_count > 0

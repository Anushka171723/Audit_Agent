import os
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
    """Return the invoice document matching `invoice_no` or None."""
    if not invoice_no:
        return None
    coll = _get_collection()
    return coll.find_one({"invoice_no": invoice_no})


def check_duplicate_invoice(invoice_no: str) -> bool:
    """Return True if an invoice with `invoice_no` already exists."""
    if not invoice_no:
        return False
    coll = _get_collection()
    return coll.count_documents({"invoice_no": invoice_no}, limit=1) > 0


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
    """Return records that share an invoice number with another record."""
    all_invoices = get_all_invoices()
    seen = {}

    for invoice in all_invoices:
        invoice_no = invoice.get("invoice_no")
        if not invoice_no:
            continue
        seen.setdefault(invoice_no, []).append(invoice)

    duplicates = [record for records in seen.values() if len(records) > 1 for record in records]
    return duplicates

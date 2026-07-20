import os
import re
from datetime import datetime
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


_client: Optional[MongoClient] = None
_last_database_error = ""


def _remember_database_error(error: Exception) -> None:
    global _last_database_error
    _last_database_error = str(error)


def clear_database_error() -> None:
    global _last_database_error
    _last_database_error = ""


def get_database_error() -> str:
    return _last_database_error


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
    try:
        coll = _get_collection()

        if "created_at" not in record:
            record["created_at"] = datetime.utcnow()

        result = coll.insert_one(record)
        clear_database_error()
        return str(result.inserted_id)
    except PyMongoError as error:
        _remember_database_error(error)
        raise RuntimeError("MongoDB connection failed. Check MONGODB_URI credentials and Atlas access settings.") from error


def get_invoice_by_number(invoice_no: str) -> Optional[dict]:
    """Return the invoice document matching `invoice_no` or None. Case-insensitive search."""
    if not invoice_no:
        return None
    try:
        coll = _get_collection()
        # Use case-insensitive regex search
        result = coll.find_one({"invoice_no": {"$regex": f"^{re.escape(invoice_no)}$", "$options": "i"}})
        clear_database_error()
        return result
    except PyMongoError as error:
        _remember_database_error(error)
        return None


def check_duplicate_invoice(invoice_no: str) -> bool:
    """Return True if an invoice with `invoice_no` already exists."""
    if not invoice_no:
        return False
    try:
        coll = _get_collection()
        result = coll.count_documents({"invoice_no": invoice_no}, limit=1) > 0
        clear_database_error()
        return result
    except PyMongoError as error:
        _remember_database_error(error)
        return False


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _item_fingerprint(items: list) -> str:
    """Create a canonical string fingerprint from line items for duplicate detection.

    Each item is represented as "product|qty|price", sorted so order doesn't matter.
    Two invoices with the same vendor/customer/items match regardless of invoice number or date.
    """
    if not items or not isinstance(items, list):
        return ""
    parts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        product = re.sub(r"\s+", " ", str(item.get("product") or item.get("description") or "").strip().lower())
        qty     = round(_to_float(item.get("quantity")), 2)
        price   = round(_to_float(item.get("unit_price")), 2)
        if product:
            parts.append(f"{product}|{qty}|{price}")
    return ";;".join(sorted(parts))


def find_duplicate_by_content(record: dict) -> Optional[dict]:
    """Detect a duplicate by vendor + customer + item fingerprint.

    Matches even when invoice number, date, or totals differ — catches
    re-submitted transactions and fraud attempts that change only metadata.
    """
    try:
        coll = _get_collection()
        vendor   = str(record.get("vendor") or "").strip()
        customer = str(record.get("customer_name") or "").strip()
        items    = record.get("items") or []
        fp       = _item_fingerprint(items)

        if not vendor or not fp:
            return None

        # Build candidate query — vendor must match (case-insensitive)
        query: dict = {"vendor": {"$regex": f"^{re.escape(vendor)}$", "$options": "i"}}
        if customer:
            query["customer_name"] = {"$regex": f"^{re.escape(customer)}$", "$options": "i"}

        candidates = list(coll.find(query))
        for candidate in candidates:
            candidate_fp = _item_fingerprint(candidate.get("items") or [])
            if candidate_fp and candidate_fp == fp:
                clear_database_error()
                return candidate

        clear_database_error()
        return None
    except PyMongoError as error:
        _remember_database_error(error)
        return None


def find_duplicate_invoice(record: dict) -> Optional[dict]:
    """Return a likely duplicate invoice by invoice number, core fields, or content fingerprint."""
    try:
        coll = _get_collection()
        invoice_no = str(record.get("invoice_no") or "").strip()

        # 1. Exact invoice number match
        if invoice_no:
            duplicate = coll.find_one({"invoice_no": invoice_no})
            if duplicate:
                clear_database_error()
                return duplicate

        # 2. Vendor + date + amount + total match (original logic)
        vendor = str(record.get("vendor") or "").strip()
        invoice_date = str(record.get("date") or "").strip()
        amount = _to_float(record.get("amount"))
        total  = _to_float(record.get("total"))

        if vendor and invoice_date and amount > 0 and total > 0:
            result = coll.find_one(
                {
                    "vendor": vendor,
                    "date":   invoice_date,
                    "amount": amount,
                    "total":  total,
                }
            )
            if result:
                clear_database_error()
                return result

        # 3. Content fingerprint match — same vendor + customer + line items
        content_match = find_duplicate_by_content(record)
        if content_match:
            return content_match

        clear_database_error()
        return None
    except PyMongoError as error:
        _remember_database_error(error)
        return None


def check_duplicate_invoice_record(record: dict) -> bool:
    """Return True if an invoice is duplicated by number or by vendor/date/amount/total."""
    return find_duplicate_invoice(record) is not None


def get_all_invoices() -> list[dict]:
    """Return all invoice records."""
    try:
        coll = _get_collection()
        results = list(coll.find().sort("created_at", -1))
        clear_database_error()
        return results
    except PyMongoError as error:
        _remember_database_error(error)
        return []


def get_failed_invoices() -> list[dict]:
    """Return all invoices that failed or have warnings."""
    try:
        coll = _get_collection()
        results = list(coll.find({"audit_status": {"$ne": "passed"}}).sort("created_at", -1))
        clear_database_error()
        return results
    except PyMongoError as error:
        _remember_database_error(error)
        return []


def get_invoice_count() -> int:
    """Return the total number of invoices stored in the database."""
    try:
        coll = _get_collection()
        result = coll.count_documents({})
        clear_database_error()
        return result
    except PyMongoError as error:
        _remember_database_error(error)
        return 0


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
    """Return all invoices matching the given category (exact field first, then keyword fallback)."""
    if not category:
        return []
    try:
        coll = _get_collection()
        # 1. Try exact category field match first (case-insensitive)
        results = list(coll.find({"category": {"$regex": f"^{re.escape(category)}$", "$options": "i"}}).sort("created_at", -1))
        if results:
            clear_database_error()
            return results
        # 2. Fallback: partial match on category, vendor, or description fields
        keyword_pattern = re.escape(category)
        results = list(coll.find({
            "$or": [
                {"category": {"$regex": keyword_pattern, "$options": "i"}},
                {"vendor": {"$regex": keyword_pattern, "$options": "i"}},
                {"description": {"$regex": keyword_pattern, "$options": "i"}},
            ]
        }).sort("created_at", -1))
        clear_database_error()
        return results
    except PyMongoError as error:
        _remember_database_error(error)
        return []


def get_invoices_by_date(date_str: str) -> list[dict]:
    """Return all invoices matching the given date string (YYYY-MM-DD or partial)."""
    if not date_str:
        return []
    try:
        coll = _get_collection()
        results = list(coll.find({"date": {"$regex": re.escape(date_str), "$options": "i"}}).sort("created_at", -1))
        clear_database_error()
        return results
    except PyMongoError as error:
        _remember_database_error(error)
        return []


def search_invoices(query_params: dict) -> list[dict]:
    """Search for invoices by multiple criteria."""
    try:
        coll = _get_collection()
        mongo_query = {}
    
        for field, val in query_params.items():
            if not val:
                continue
            val_str = str(val).strip()
            if field in ["invoice_no", "vendor", "customer_name", "gstin", "category"]:
                mongo_query[field] = {"$regex": re.escape(val_str), "$options": "i"}
            elif field == "date":
                mongo_query[field] = {"$regex": re.escape(val_str), "$options": "i"}
            elif field in ["audit_status", "status"]:
                mongo_query["audit_status"] = {"$regex": f"^{re.escape(val_str)}$", "$options": "i"}
            elif field == "classification":
                mongo_query["classification"] = {"$regex": f"^{re.escape(val_str)}$", "$options": "i"}

        results = list(coll.find(mongo_query).sort("created_at", -1))
        clear_database_error()
        return results
    except PyMongoError as error:
        _remember_database_error(error)
        return []


def update_invoice_record(invoice_no: str, updates: dict) -> bool:
    """Update an invoice record by invoice_no. Returns True if successful."""
    if not invoice_no:
        return False
    try:
        coll = _get_collection()
        result = coll.update_one({"invoice_no": invoice_no}, {"$set": updates})
        clear_database_error()
        return result.matched_count > 0
    except PyMongoError as error:
        _remember_database_error(error)
        return False

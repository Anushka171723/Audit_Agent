from dotenv import load_dotenv
import uuid
import os
from pprint import pprint

load_dotenv()

from database import save_invoice_record, get_invoice_by_number, check_duplicate_invoice


def main():
    invoice_no = f"TEST-{uuid.uuid4()}"
    record = {
        "invoice_no": invoice_no,
        "vendor": "Test Vendor Ltd",
        "category": "services",
        "amount": 100.0,
        "tax": 10.0,
        "total": 110.0,
        "audit_status": "passed",
        "issue_count": 0,
    }

    print("MONGODB_URI present:", bool(os.getenv("MONGODB_URI")))

    print("Checking duplicate before save:")
    print(check_duplicate_invoice(invoice_no))

    print("Saving record...")
    inserted_id = save_invoice_record(record)
    print("Inserted id:", inserted_id)

    print("Retrieving by invoice_no...")
    doc = get_invoice_by_number(invoice_no)
    pprint(doc)


if __name__ == "__main__":
    main()

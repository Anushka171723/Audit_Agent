import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from audit import audit_invoice, create_audit_summary
from chat import start_chat
from ocr import extract_text_from_image
from parser import parse_invoice_text, save_csv, save_json


def save_raw_text(text: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_command(args: argparse.Namespace) -> None:
    image_path = args.file

    print("Reading image and running OCR...")
    raw_text = extract_text_from_image(image_path)

    parser_mode = "Groq structured extraction" if os.getenv("GROQ_API_KEY") else "rule parser"
    print(f"Converting OCR text into structured data using {parser_mode}...")
    invoice_data = parse_invoice_text(raw_text)

    save_raw_text(raw_text, "outputs/raw_ocr_text.txt")
    save_json(invoice_data, "outputs/extracted_data.json")
    save_csv(invoice_data, "outputs/extracted_data.csv")

    print("Extraction completed.")
    print("Raw OCR text saved to outputs/raw_ocr_text.txt")
    print("Structured JSON saved to outputs/extracted_data.json")
    print("Structured CSV saved to outputs/extracted_data.csv")


def audit_command(args: argparse.Namespace) -> None:
    input_path = Path(args.file)

    if not input_path.exists():
        raise FileNotFoundError(f"Extracted data file not found: {args.file}")

    invoice_data = json.loads(input_path.read_text(encoding="utf-8"))

    print("Running audit checks...")
    audit_result = audit_invoice(invoice_data)
    audit_summary = create_audit_summary(invoice_data, audit_result)

    save_json(audit_result, "outputs/audit_report.json")
    save_raw_text(audit_summary, "outputs/audit_report.txt")

    print("Audit completed.")
    print(f"Status: {audit_result['status']}")
    print(f"Issues found: {audit_result['issue_count']}")
    print("Audit report saved to outputs/audit_report.json")
    print("Audit summary saved to outputs/audit_report.txt")


def run_command(args: argparse.Namespace) -> None:
    image_path = args.file

    print("Step 1: Reading image and running OCR...")
    raw_text = extract_text_from_image(image_path)

    parser_mode = "Groq structured extraction" if os.getenv("GROQ_API_KEY") else "rule parser"
    print(f"Step 2: Converting OCR text into structured data using {parser_mode}...")
    invoice_data = parse_invoice_text(raw_text)

    print("Step 3: Running audit checks...")
    audit_result = audit_invoice(invoice_data)
    audit_summary = create_audit_summary(invoice_data, audit_result)

    save_raw_text(raw_text, "outputs/raw_ocr_text.txt")
    save_json(invoice_data, "outputs/extracted_data.json")
    save_csv(invoice_data, "outputs/extracted_data.csv")
    save_json(audit_result, "outputs/audit_report.json")
    save_raw_text(audit_summary, "outputs/audit_report.txt")

    print("Full pipeline completed.")
    print(f"Audit Status: {audit_result['status']}")
    print(f"Issues found: {audit_result['issue_count']}")
    print("Files saved in outputs/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI Audit Agent - Day 1 OCR extraction")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract", help="Extract invoice data from an image")
    extract_parser.add_argument("--file", required=True, help="Path to invoice image file")
    extract_parser.set_defaults(func=extract_command)

    audit_parser = subparsers.add_parser("audit", help="Audit extracted invoice data")
    audit_parser.add_argument("--file", required=True, help="Path to extracted JSON data")
    audit_parser.set_defaults(func=audit_command)

    run_parser = subparsers.add_parser("run", help="Extract and audit invoice data from an image")
    run_parser.add_argument("--file", required=True, help="Path to invoice image file")
    run_parser.set_defaults(func=run_command)

    chat_parser = subparsers.add_parser("chat", help="Chat with the audit report")
    chat_parser.set_defaults(func=chat_command)

    return parser


def chat_command(args: argparse.Namespace) -> None:
    start_chat()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()

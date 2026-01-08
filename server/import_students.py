#!/usr/bin/env python3
"""Bulk import students from a CSV file."""

import argparse
import csv
import sys
from pathlib import Path

import requests


def import_students(
    csv_path: Path,
    server_url: str,
    admin_key: str,
    output_path: Path | None = None,
) -> list[dict]:
    """
    Import students from a CSV file.

    Args:
        csv_path: Path to CSV file with 'email' column
        server_url: Base URL of the evaluation server
        admin_key: Admin API key
        output_path: Optional path to write results CSV

    Returns:
        List of created student records with API keys
    """
    # Read emails from CSV
    emails = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        if "email" not in reader.fieldnames:
            # Try reading as single column without header
            f.seek(0)
            for line in f:
                email = line.strip()
                if email and "@" in email:
                    emails.append(email)
        else:
            for row in reader:
                email = row["email"].strip()
                if email:
                    emails.append(email)

    if not emails:
        print("No valid emails found in CSV")
        return []

    print(f"Found {len(emails)} emails to import")

    # Create students via API
    results = []
    server_url = server_url.rstrip("/")

    for email in emails:
        try:
            response = requests.post(
                f"{server_url}/admin/students",
                json={"email": email},
                headers={"X-Admin-Key": admin_key},
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                results.append({
                    "email": data["email"],
                    "api_key": data["api_key"],
                    "student_id": data["id"],
                    "status": "created",
                })
                print(f"  ✓ Created: {email}")
            elif response.status_code == 400:
                results.append({
                    "email": email,
                    "api_key": "",
                    "student_id": "",
                    "status": "already_exists",
                })
                print(f"  - Exists: {email}")
            else:
                results.append({
                    "email": email,
                    "api_key": "",
                    "student_id": "",
                    "status": f"error_{response.status_code}",
                })
                print(f"  ✗ Error: {email} - {response.text}")

        except requests.RequestException as e:
            results.append({
                "email": email,
                "api_key": "",
                "student_id": "",
                "status": f"error: {e}",
            })
            print(f"  ✗ Error: {email} - {e}")

    # Write results to output CSV
    if output_path is None:
        output_path = csv_path.with_name(csv_path.stem + "_with_keys.csv")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["email", "api_key", "student_id", "status"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to: {output_path}")

    created = sum(1 for r in results if r["status"] == "created")
    exists = sum(1 for r in results if r["status"] == "already_exists")
    errors = len(results) - created - exists

    print(f"Summary: {created} created, {exists} already existed, {errors} errors")

    return results


def main():
    parser = argparse.ArgumentParser(description="Bulk import students from CSV")
    parser.add_argument("csv_file", type=Path, help="CSV file with student emails")
    parser.add_argument(
        "--server",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--admin-key",
        required=True,
        help="Admin API key",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path (default: <input>_with_keys.csv)",
    )

    args = parser.parse_args()

    if not args.csv_file.exists():
        print(f"Error: {args.csv_file} not found")
        sys.exit(1)

    import_students(
        args.csv_file,
        args.server,
        args.admin_key,
        args.output,
    )


if __name__ == "__main__":
    main()

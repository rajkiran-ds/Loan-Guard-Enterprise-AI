"""
src/utils/validators.py
------------------------
Validation helpers for every field collected during the AI customer
interview or extracted via OCR. Each function returns (is_valid, message)
so the caller (Streamlit form or the conversational interview) can surface
a clear, specific error instead of a generic "invalid input".
"""

from __future__ import annotations

import re
from datetime import date

PAN_REGEX = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
AADHAAR_REGEX = re.compile(r"^\d{12}$")
NAME_REGEX = re.compile(r"^[A-Za-z][A-Za-z\s.'-]{1,59}$")

ValidationResult = tuple[bool, str]


def validate_pan(pan: str) -> ValidationResult:
    pan = (pan or "").strip().upper()
    if not PAN_REGEX.match(pan):
        return False, "PAN must follow the format AAAAA9999A (5 letters, 4 digits, 1 letter)."
    return True, "Valid PAN."


def validate_aadhaar(aadhaar: str) -> ValidationResult:
    aadhaar = re.sub(r"\s|-", "", aadhaar or "")
    if not AADHAAR_REGEX.match(aadhaar):
        return False, "Aadhaar must be exactly 12 digits."
    return True, "Valid Aadhaar."


def validate_name(name: str) -> ValidationResult:
    name = (name or "").strip()
    if not NAME_REGEX.match(name):
        return False, "Name must be 2-60 letters and may include spaces, apostrophes, or hyphens."
    return True, "Valid name."


def validate_age(age: int | str) -> ValidationResult:
    try:
        age_int = int(age)
    except (TypeError, ValueError):
        return False, "Age must be a whole number."
    if age_int < 21 or age_int > 65:
        return False, "Applicant age must be between 21 and 65 for a personal loan."
    return True, "Valid age."


def validate_income(monthly_income: float | str) -> ValidationResult:
    try:
        income = float(monthly_income)
    except (TypeError, ValueError):
        return False, "Monthly income must be a number."
    if income < 15000:
        return False, "Monthly income must be at least ₹15,000 to qualify."
    if income > 10_000_000:
        return False, "Monthly income value looks unrealistic — please re-check."
    return True, "Valid income."


def validate_employment_type(value: str) -> ValidationResult:
    allowed = {"salaried", "self-employed", "business owner", "unemployed"}
    if (value or "").strip().lower() not in allowed:
        return False, f"Employment type must be one of: {', '.join(sorted(allowed))}."
    return True, "Valid employment type."


def validate_loan_amount(amount: float | str, monthly_income: float | None = None) -> ValidationResult:
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return False, "Loan amount must be a number."
    if amt <= 0:
        return False, "Loan amount must be greater than zero."
    if monthly_income and amt > monthly_income * 15:
        return False, "Requested loan amount is unusually high relative to income."
    return True, "Valid loan amount."


def validate_dob(dob: str) -> ValidationResult:
    try:
        year, month, day = (int(x) for x in dob.split("-"))
        parsed = date(year, month, day)
    except (ValueError, AttributeError):
        return False, "Date of birth must be in YYYY-MM-DD format."
    age = (date.today() - parsed).days // 365
    if age < 21 or age > 65:
        return False, "Applicant must be between 21 and 65 years old."
    return True, "Valid date of birth."

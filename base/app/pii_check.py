import re

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(
        r"\b(?:(?:\+|00)[1-9]\d{0,2}[ \-])?(?:\(\d{3}\)[ \-])?\d{3}[ \-]?\d{4}(?:[ \-]?\d+)?\b",
        re.IGNORECASE,
    ),
    "ssn_usa": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "aadhaar_india": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    "pan_india": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"),
    "passport": re.compile(r"\b(?!.*\b(?!PASSPORT)\b)[A-Z]{1,2}\d{6,9}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "dob": re.compile(
        r"\b(?:\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}|\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b"
    ),
    "address": re.compile(
        r"\b\d{1,5}\s+(?:[A-Za-z0-9#]+\s?)+(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b",
        re.IGNORECASE,
    ),
    "cherokee_nation_id": re.compile(r"\b\d{6}\b"),
}


def detect_and_mask_pii(text: str) -> tuple[bool, str]:
    if not text:
        return False, ""

    pii_detected = False

    for pattern in PII_PATTERNS.values():
        if pattern.search(text):
            pii_detected = True
            break  # Exit the loop early, we already know PII exists.

    if not pii_detected:
        return False, text

    masked_text = text
    for label, pattern in PII_PATTERNS.items():
        masked_text = pattern.sub(f"[MASKED_{label.upper()}]", masked_text)

    return True, masked_text

"""Input validation and PII masking applied before a message reaches the LLM, the DB or the logs."""

import re

MAX_MESSAGE_CHARS = 1000
MAX_NAME_CHARS = 60
DEFAULT_CUSTOMER_NAME = "Valued Customer"


class InputRejected(ValueError):
    """Message failed validation; str(exc) is safe to show the customer."""


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NAME_DISALLOWED_RE = re.compile(r"[^\w .'-]")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Card numbers: 13-19 digits, optionally separated by spaces or dashes.
_CARD_RE = re.compile(r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)")
# Indian mobile numbers, with optional +91 prefix.
_PHONE_RE = re.compile(r"(?<![\d+])(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)")
# Account numbers and other long identifiers. Ticket IDs are 6 digits and stay intact.
_LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{9,}(?!\d)")
_INJECTION_RE = re.compile(
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?)"
    r"|(reveal|show|print) (your |the )?(system )?prompt"
    r"|you are now|disregard (your|all) (rules|instructions)",
    re.IGNORECASE,
)


def clean_message(message: str) -> str:
    text = _CONTROL_RE.sub("", message or "").strip()
    if not text:
        raise InputRejected("Please type a message so we can help.")
    if len(text) > MAX_MESSAGE_CHARS:
        raise InputRejected(f"Your message is too long. Please keep it under {MAX_MESSAGE_CHARS} characters.")
    return text


def clean_name(name: str | None) -> str:
    cleaned = " ".join(_NAME_DISALLOWED_RE.sub("", name or "").split())[:MAX_NAME_CHARS].strip()
    return cleaned or DEFAULT_CUSTOMER_NAME


def mask_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _CARD_RE.sub("[CARD]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return _LONG_NUMBER_RE.sub("[NUMBER]", text)


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))

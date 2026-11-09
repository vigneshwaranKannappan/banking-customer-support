import re
import secrets
from collections.abc import Container

_TICKET_ID_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def extract_ticket_id(message: str) -> str | None:
    """Return the first standalone 6-digit number in the message, if any."""
    match = _TICKET_ID_RE.search(message)
    return match.group(1) if match else None


def generate_ticket_id(existing: Container[str], max_attempts: int = 20) -> str:
    """Random 6-digit ticket ID (100000-999999) not present in `existing`."""
    for _ in range(max_attempts):
        candidate = str(secrets.randbelow(900_000) + 100_000)
        if candidate not in existing:
            return candidate
    raise RuntimeError("Could not generate a unique ticket ID")

"""Demo tickets so the spec's query examples (e.g. #650932 → Resolved) work out of the box."""

from banking_customer_support_ai.models.schemas import TicketStatus

SEED_TICKETS = [
    ("650932", "Priya", "Net banking login keeps failing.", TicketStatus.RESOLVED),
    ("123456", "Arjun", "Credit card charged twice for one purchase.", TicketStatus.OPEN),
    ("784521", "Meena", "Debit card replacement has not arrived.", TicketStatus.IN_PROGRESS),
]

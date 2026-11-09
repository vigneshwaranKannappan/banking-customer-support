"""Labeled test messages per agent role. Shared by the UI scenario tester and the evaluation runner."""

from dataclasses import dataclass

from banking_customer_support_ai.models.schemas import Classification


@dataclass(frozen=True)
class Scenario:
    message: str
    expected: Classification


SCENARIOS: list[Scenario] = [
    # Positive feedback
    Scenario("Thanks for sorting out my net banking login issue.", Classification.POSITIVE_FEEDBACK),
    Scenario("Thanks for resolving my credit card issue.", Classification.POSITIVE_FEEDBACK),
    Scenario("The branch staff were really helpful today, great service!", Classification.POSITIVE_FEEDBACK),
    # Negative feedback
    Scenario("My debit card replacement still hasn't arrived.", Classification.NEGATIVE_FEEDBACK),
    Scenario("I was charged twice for the same transaction.", Classification.NEGATIVE_FEEDBACK),
    Scenario("The mobile app crashes every time I open it.", Classification.NEGATIVE_FEEDBACK),
    # Queries
    Scenario("Could you check the status of ticket 650932?", Classification.QUERY),
    Scenario("Any update on 784521?", Classification.QUERY),
    Scenario("What is the status of my complaint?", Classification.QUERY),
    Scenario("Where is ticket 999999?", Classification.QUERY),
]

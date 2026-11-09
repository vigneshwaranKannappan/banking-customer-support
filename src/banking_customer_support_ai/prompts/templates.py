"""Fixed customer-facing reply formats from the capstone spec."""

POSITIVE_REPLY = "Thank you for your kind words, {customer_name}! We're delighted to assist you."

NEGATIVE_REPLY = (
    "We apologize for the inconvenience. A new ticket #{ticket_id} has been generated, "
    "and our team will follow up shortly."
)

STATUS_REPLY = "Your ticket #{ticket_id} is currently marked as: {status}."

TICKET_NOT_FOUND_REPLY = (
    "We couldn't find a ticket #{ticket_id}. Please check the number and try again."
)

MISSING_TICKET_ID_REPLY = (
    "Happy to help! Please share your 6-digit ticket number so I can check its status."
)

ERROR_REPLY = "Sorry, something went wrong while handling your request. Please try again shortly."

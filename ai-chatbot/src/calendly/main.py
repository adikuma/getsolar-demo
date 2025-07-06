from langchain.tools import tool

@tool
def get_booking_link() -> str:
    """
    Return Aditya's 30-minute meeting scheduling URL and prompt the user to book.

    When you need to invite someone to schedule a 30-minute meeting with Aditya,
    call this function. It prints a friendly prompt asking the user to book
    and returns the shareable Calendly URL.

    Args:
        None

    Returns:
        str: The Calendly scheduling URL for a 30-minute meeting.
    """
    url = "https://calendly.com/adi-kumar/30min"
    # print(f"Please book a 30-minute appointment here: {url}")
    return url
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pydantic import BaseModel, Field
from typing import Dict, Optional, Annotated, List

class ContactInfo(BaseModel):
    name: Optional[str] = Field(None, description="Person's full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")


@tool
def extract_contact_info(user_message: str) -> Dict[str, str]:
    """
    Extract name, email, and phone number from user's natural language response.

    Args:
        user_message: User's message containing contact details

    Returns:
        dict: Extracted contact information with keys: name, email, phone
    """
    print(f"[extract_contact_info] Processing: {user_message[:100]}")

    # create extraction LLM (separate from main conversation LLM)
    extraction_llm = ChatAnthropic(
        model="claude-3-5-sonnet-20240620",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1,
    ).with_structured_output(ContactInfo)

    extraction_prompt = f"""
    Extract the contact information from this user message. 
    Look for:
    - Full name (e.g., "John Smith", "Jane Doe")  
    - Email address (e.g., "john@email.com")
    - Phone number (e.g., "+65-1234-5678", "91234567")
    
    User message: {user_message}
    
    If any information is missing or unclear, set that field to null.
    """

    try:
        contact_info = extraction_llm.invoke(extraction_prompt)
        result = {
            "name": contact_info.name or "",
            "email": contact_info.email or "",
            "phone": contact_info.phone or "",
        }
        print(f"[extract_contact_info] Extracted: {result}")
        return result
    except Exception as e:
        print(f"[extract_contact_info] Error: {e}")
        return {"name": "", "email": "", "phone": ""}


@tool
def send_consultation_confirmation(
    name: str, email: str, phone: str, calendly_link: str
) -> str:
    """
    Send consultation booking confirmation email using SMTP.

    Args:
        name: User's full name
        email: User's email address
        phone: User's phone number
        calendly_link: The Calendly booking URL

    Returns:
        str: Success/failure message
    """
    print(f"[send_confirmation_email] Sending to {name} <{email}>")

    # email configuration (using Gmail SMTP as example)
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_user = os.getenv("EMAIL_USER")  # 
    smtp_password = os.getenv("EMAIL_PASSWORD")  # 

    if not smtp_user or not smtp_password:
        return "Email configuration missing. Please set EMAIL_USER and EMAIL_PASSWORD environment variables."

    try:
        # Create email content
        subject = "GetSolar Consultation - Booking Confirmation"

        html_content = f"""
        <html>
        <body>
            <h2>🌞 Thank you for your interest in solar energy!</h2>
            
            <p>Hi {name},</p>
            
            <p>This confirms that you're ready to schedule your free solar consultation with GetSolar.</p>
            
            <h3>Next Steps:</h3>
            <ol>
                <li><strong>Book your appointment:</strong> <a href="{calendly_link}">Click here to schedule</a></li>
                <li><strong>Prepare for your consultation:</strong> Have your recent electricity bills ready</li>
                <li><strong>Site assessment:</strong> Our expert will evaluate your property's solar potential</li>
            </ol>
            
            <h3>Your Details:</h3>
            <ul>
                <li><strong>Name:</strong> {name}</li>
                <li><strong>Email:</strong> {email}</li>
                <li><strong>Phone:</strong> {phone}</li>
            </ul>
            
            <p>Questions? Reply to this email or call us at +65-6000-0000.</p>
            
            <p>Best regards,<br>
            The GetSolar Team<br>
            <em>Powering Singapore's solar future</em></p>
        </body>
        </html>
        """

        # create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = email

        # add HTML content
        html_part = MIMEText(html_content, "html")
        msg.attach(html_part)

        # send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)

        print(f"[send_confirmation_email] Successfully sent to {email}")
        return f"Confirmation email sent to {email}! Please check your inbox."

    except Exception as e:
        print(f"[send_confirmation_email] Error: {e}")
        return f"Failed to send confirmation email. Please contact us directly at hello@getsolar.ai"

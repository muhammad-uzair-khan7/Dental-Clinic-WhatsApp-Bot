import datetime as dt
import os
import gspread
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()
CLINIC_BASE_URL = os.getenv("CLINIC_BASE_URL")

gc = gspread.service_account(filename="/etc/secrets/credentials.json")
sh = gc.open("PatientComplaints")  # exact title must match your actual Google Sheet
worksheet = sh.sheet1


@tool
def generate_ticket(name: str, email: str, phone_number: str, complaint: str):
    """
    Generate a support ticket for a patient complaint the AI can't resolve itself.
    Logs the complaint to the clinic system and appends it to the Google Sheet.
    The AI should NOT ask the patient for a timestamp — it's generated here.
    """
    date_time = dt.datetime.now(dt.timezone.utc).isoformat()

    try:
        resp = requests.post(
            url=f"{CLINIC_BASE_URL}/api/complaint-ticket",
            json={
                "name": name,
                "email": email,
                "phone_number": phone_number,
                "complaint": complaint,
                "date_time": date_time,
            },
            timeout=5,
        )

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", {})
            except ValueError:
                detail = resp.text
            if isinstance(detail, dict):
                return f"Sorry, I couldn't log your complaint: {detail.get('message', 'unknown error')}"
            return f"Sorry, I couldn't log your complaint: {detail}"

        data = resp.json()
        worksheet.append_row([name, email, phone_number, complaint, date_time])

        ticket_id = data.get("ticket_id", "N/A")
        return f"Your complaint has been logged (ref: {ticket_id}). Our clinic team will follow up within 24 hours."

    except requests.exceptions.RequestException as e:
        return f"Sorry, I couldn't log your complaint right now: {e}"
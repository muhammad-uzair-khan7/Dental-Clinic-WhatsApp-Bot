from langchain_core.tools import tool
import requests
import os
from dotenv import load_dotenv

load_dotenv()
CLINIC_BASE_URL = os.getenv("CLINIC_BASE_URL")
TIMEOUT_SECONDS = 5

# Must match INTERNAL_API_KEY set on the clinic_appointment_api.py service.
# Only needed on calls that hit routes gated with verify_internal_key —
# right now that's just get_appointment_status(), since /api/availability
# and /api/appointments/book were left open (see clinic_appointment_api.py).
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
_AUTH_HEADERS = {"X-API-Key": INTERNAL_API_KEY}


def get_doctor_availability(doctor_name: str, date: str) -> dict:
    """
    Check available appointment slots for a doctor on a given date (YYYY-MM-DD).

    Returns a dict like:
        {"found": True, "doctor_name": "Dr. Ayesha Khan", "available_slots": ["09:00", "09:30", ...]}
    or on failure:
        {"found": False, "error": "'Dr. Unknown' is not a doctor at this clinic"}
    """
    try:
        resp = requests.get(
            f"{CLINIC_BASE_URL}/api/availability/{doctor_name}",
            params={"date": date},
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code == 404:
            return {"found": False, "error": resp.json().get("detail", "Doctor not found")}
        if resp.status_code in (400, 422):
            detail = resp.json().get("detail", "Invalid request")
            return {"found": False, "error": detail if isinstance(detail, str) else str(detail)}
        resp.raise_for_status()
        data = resp.json()
        data["found"] = True
        return data
    except requests.exceptions.RequestException as e:
        return {"found": False, "error": f"Clinic system unreachable: {e}"}


def book_new_appointment(patient_name: str, phone: str, doctor_name: str, appt_date: str, appt_time: str) -> dict:
    """
    Book a new appointment.

    Returns a dict like:
        {"success": True, "appointment_id": 1002, "doctor_name": "...", "fee_pkr": 2000, ...}
    or on failure (slot taken / invalid input):
        {"success": False, "error": "That slot is not available...", "available_slots": [...]}
    """
    try:
        resp = requests.post(
            f"{CLINIC_BASE_URL}/api/appointments/book",
            json={
                "patient_name": patient_name,
                "phone": phone,
                "doctor_name": doctor_name,
                "appt_date": appt_date,
                "appt_time": appt_time,
            },
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code == 400:
            detail = resp.json().get("detail", {})
            if isinstance(detail, dict):
                return {
                    "success": False,
                    "error": detail.get("message"),
                    "available_slots": detail.get("available_slots", []),
                }
            return {"success": False, "error": str(detail)}
        if resp.status_code == 422:
            return {"success": False, "error": f"Invalid appointment request: {resp.json().get('detail')}"}
        resp.raise_for_status()
        data = resp.json()
        data["success"] = True
        return data
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Clinic system unreachable: {e}"}


def get_appointment_status(appointment_id: int) -> dict:
    """
    Fetch live status of an existing appointment.

    Returns a dict like:
        {"found": True, "appointment_id": 1002, "status": "Scheduled", ...}
    or on failure:
        {"found": False, "error": "Appointment 9999 not found"}
    """
    try:
        # This route (GET /api/appointments/{id}) is gated with
        # verify_internal_key on the API side, so it needs the header.
        resp = requests.get(
            f"{CLINIC_BASE_URL}/api/appointments/{appointment_id}",
            headers=_AUTH_HEADERS,
            timeout=TIMEOUT_SECONDS,
        )
        if resp.status_code == 404:
            return {"found": False, "error": resp.json().get("detail", "Appointment not found")}
        if resp.status_code == 401:
            return {"found": False, "error": "Clinic system authentication failed — check INTERNAL_API_KEY."}
        resp.raise_for_status()
        data = resp.json()
        data["found"] = True
        return data
    except requests.exceptions.RequestException as e:
        return {"found": False, "error": f"Clinic system unreachable: {e}"}


@tool
def check_doctor_availability(doctor_name: str, date: str):
    """Check which appointment slots are available for a doctor on a given date (date format: YYYY-MM-DD)."""
    result = get_doctor_availability(doctor_name, date)
    if not result["found"]:
        return f"Sorry, I couldn't check availability: {result['error']}"
    slots = result.get("available_slots", [])
    if not slots:
        return f"Sorry, {doctor_name} has no available slots on {date}. Would you like to try a different date?"
    return f"{doctor_name} ({result.get('specialty', '')}) has these slots open on {date}: {', '.join(slots)}."


@tool
def book_appointment(patient_name: str, phone: str, doctor_name: str, appt_date: str, appt_time: str):
    """Book an appointment for a patient with a doctor at a specific date (YYYY-MM-DD) and time (HH:MM, 24h)."""
    result = book_new_appointment(patient_name, phone, doctor_name, appt_date, appt_time)
    if not result["success"]:
        slots = result.get("available_slots")
        if slots:
            return f"Sorry, that slot isn't available. Open slots that day: {', '.join(slots)}."
        return f"Sorry, there was a problem booking the appointment: {result.get('error', 'unknown error')}."
    return (
        f"Appointment confirmed! Appointment ID #{result['appointment_id']} with {result['doctor_name']} "
        f"({result['specialty']}) on {result['appt_date']} at {result['appt_time']}. "
        f"Consultation fee: PKR {result['fee_pkr']}."
    )


@tool
def check_appointment_status(appointment_id: int):
    """Look up the current status of an existing appointment by its appointment ID."""
    result = get_appointment_status(appointment_id)
    if not result["found"]:
        return f"Sorry, I couldn't find that appointment: {result['error']}"
    return (
        f"Appointment #{result['appointment_id']} for {result['patient_name']} with {result['doctor_name']} "
        f"is currently '{result['status']}', scheduled for {result['appt_date']} at {result['appt_time']}."
    )
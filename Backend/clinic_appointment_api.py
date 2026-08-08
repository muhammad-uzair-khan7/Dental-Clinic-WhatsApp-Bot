"""
Clinic Appointment API Server
==============================
Simulates a clinic's scheduling backend so the LangGraph appointment_management
agent can check doctor availability, book appointments, and check appointment
status over HTTP — same architecture as the café's POS server.

Persistence: SQLite file (clinic_data.db) — survives restarts.

Run:
    uvicorn clinic_appointment_api:app --port 8001

Endpoints:
    GET  /api/doctors
    GET  /api/availability/{doctor_name}?date=YYYY-MM-DD
    POST /api/appointments/book
    GET  /api/appointments/{appointment_id}
    POST /api/appointments/{appointment_id}/cancel
    GET  /api/appointments               (filterable list — see dashboard section)
    GET  /api/complaints                 (dashboard)
    GET  /api/dashboard/summary          (dashboard)
"""

import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, field_validator

load_dotenv()

DB_PATH = "clinic_data.db"

# --------------------------------------------------------------------------
# Doctors + schedules
# --------------------------------------------------------------------------
DOCTORS = {
    "Dr. Muhammad Zubair Yousuf": {
        "specialty": "Dental Surgeon (BDS, RDS, MScDS, PDG-HP)",
        "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
        "start": "19:30",  # 7:30 PM
        "end": "23:30",    # 11:30 PM
        "slot_minutes": 30,
        "fee_pkr": 500,
    },
}

WEEKDAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
VALID_STATUSES = ["Scheduled", "Completed", "Cancelled", "No-Show"]


# --------------------------------------------------------------------------
# Database helpers
# --------------------------------------------------------------------------
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                appointment_id INTEGER PRIMARY KEY,
                patient_name TEXT NOT NULL,
                phone TEXT NOT NULL,
                doctor_name TEXT NOT NULL,
                appt_date TEXT NOT NULL,   -- 'YYYY-MM-DD'
                appt_time TEXT NOT NULL,   -- 'HH:MM'
                status TEXT NOT NULL,
                fee_pkr INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        cur.execute("SELECT value FROM meta WHERE key = 'next_appointment_id'")
        if cur.fetchone() is None:
            cur.execute("INSERT INTO meta (key, value) VALUES ('next_appointment_id', '1001')")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS complaints (
                ticket_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                complaint TEXT NOT NULL,
                date_time TEXT NOT NULL
            )
            """
        )


def get_next_appointment_id(cur) -> int:
    cur.execute("SELECT value FROM meta WHERE key = 'next_appointment_id'")
    next_id = int(cur.fetchone()["value"])
    cur.execute("UPDATE meta SET value = ? WHERE key = 'next_appointment_id'", (str(next_id + 1),))
    return next_id


# --------------------------------------------------------------------------
# Slot generation
# --------------------------------------------------------------------------
def generate_slots(doctor: dict, target_date: date_cls) -> list[str]:
    """Generate all possible HH:MM slot start times for a doctor on a given date."""
    weekday = WEEKDAY_ABBR[target_date.weekday()]
    if weekday not in doctor["days"]:
        return []

    start_h, start_m = map(int, doctor["start"].split(":"))
    end_h, end_m = map(int, doctor["end"].split(":"))
    slot_minutes = doctor["slot_minutes"]

    slots = []
    current = datetime.combine(target_date, datetime.min.time()).replace(hour=start_h, minute=start_m)
    end = datetime.combine(target_date, datetime.min.time()).replace(hour=end_h, minute=end_m)

    while current + timedelta(minutes=slot_minutes) <= end:
        slots.append(current.strftime("%H:%M"))
        current += timedelta(minutes=slot_minutes)

    return slots


def get_available_slots(doctor_name: str, target_date: date_cls) -> list[str]:
    doctor = DOCTORS[doctor_name]
    all_slots = generate_slots(doctor, target_date)
    if not all_slots:
        return []

    with get_conn() as conn:
        booked = conn.execute(
            """SELECT appt_time FROM appointments
               WHERE doctor_name = ? AND appt_date = ? AND status = 'Scheduled'""",
            (doctor_name, target_date.isoformat()),
        ).fetchall()
    booked_times = {row["appt_time"] for row in booked}

    return [s for s in all_slots if s not in booked_times]


# --------------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------------
class BookAppointmentRequest(BaseModel):
    patient_name: str = Field(..., min_length=1)
    phone: str = Field(..., min_length=5)
    doctor_name: str
    appt_date: str = Field(..., description="Date in YYYY-MM-DD format")
    appt_time: str = Field(..., description="Time in HH:MM 24h format")

    @field_validator("doctor_name")
    @classmethod
    def doctor_must_exist(cls, v):
        if v not in DOCTORS:
            raise ValueError(f"Unknown doctor '{v}'. Valid doctors: {list(DOCTORS.keys())}")
        return v

    @field_validator("patient_name")
    @classmethod
    def name_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("patient_name must not be blank")
        return v.strip()

    @field_validator("appt_date")
    @classmethod
    def date_must_be_valid(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("appt_date must be in YYYY-MM-DD format")
        return v

    @field_validator("appt_time")
    @classmethod
    def time_must_be_valid(cls, v):
        try:
            datetime.strptime(v, "%H:%M")
        except ValueError:
            raise ValueError("appt_time must be in HH:MM 24h format")
        return v


class AppointmentResponse(BaseModel):
    appointment_id: int
    patient_name: str
    phone: str
    doctor_name: str
    specialty: str
    appt_date: str
    appt_time: str
    status: str
    fee_pkr: int


class AvailabilityResponse(BaseModel):
    doctor_name: str
    specialty: str
    date: str
    available_slots: list[str]


class ComplaintSchema(BaseModel):
    name: str
    email: EmailStr
    phone_number: str
    complaint: str
    date_time: str


class ComplaintTicketResponse(BaseModel):
    success: bool
    ticket_id: str
    name: str


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = FastAPI(
    title="Clinic Appointment API",
    description="Backend for a WhatsApp clinic chatbot's appointment_management agent, plus a read-only dashboard API.",
    version="1.1.0",
)

# Dashboard is a separate static HTML file loaded straight from the browser
# (file:// or a different port), so it needs CORS. Tighten allow_origins to
# your actual dashboard host once it's deployed somewhere fixed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# This API is public on Render's free tier (private services require a paid
# plan), so any route that returns patient data (names, phone numbers,
# emails, complaint text) needs its own gate — the URL being unlisted isn't
# protection. INTERNAL_API_KEY must be set to the same value on this service
# and on the bot service that calls it.
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")


def verify_internal_key(x_api_key: str = Header(None)):
    if not INTERNAL_API_KEY or x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
def on_startup():
    init_db()


# ---- Complaints -----------------------------------------------------------
@app.post("/api/complaint-ticket", response_model=ComplaintTicketResponse, tags=["Complaints"])
def complaint_request(payload: ComplaintSchema):
    """Create a new patient complaint ticket and persist it."""
    ticket_id = f"TCK-{int(time.time())}"
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO complaints (ticket_id, name, email, phone_number, complaint, date_time)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ticket_id, payload.name, payload.email, payload.phone_number, payload.complaint, payload.date_time),
        )
    return ComplaintTicketResponse(success=True, ticket_id=ticket_id, name=payload.name)


@app.get("/api/complaints", tags=["Complaints"], dependencies=[Depends(verify_internal_key)])
def list_complaints(limit: int = Query(200, ge=1, le=1000)):
    """
    List logged complaints, most recent first. This reads the local SQLite
    mirror that /api/complaint-ticket writes to — the same complaints your
    generate_ticket tool also appends to the Google Sheet — so the dashboard
    doesn't need direct Google Sheets access.
    """
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM complaints ORDER BY date_time DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# ---- Doctors -----------------------------------------------------------
@app.get("/api/doctors", tags=["Doctors"])
def list_doctors():
    """List all doctors, specialties, working days, and fees."""
    return {
        name: {
            "specialty": d["specialty"],
            "days": d["days"],
            "hours": f"{d['start']} - {d['end']}",
            "slot_minutes": d["slot_minutes"],
            "fee_pkr": d["fee_pkr"],
        }
        for name, d in DOCTORS.items()
    }


# ---- Availability --------------------------------------------------------
@app.get("/api/availability/{doctor_name}", response_model=AvailabilityResponse, tags=["Appointments"])
def check_availability(doctor_name: str, date: str):
    """
    Check available appointment slots for a doctor on a given date.
    `date` query param must be YYYY-MM-DD.
    """
    if doctor_name not in DOCTORS:
        raise HTTPException(status_code=404, detail=f"'{doctor_name}' is not a doctor at this clinic")

    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(status_code=422, detail="date must be in YYYY-MM-DD format")

    if target_date < datetime.now(timezone.utc).date():
        raise HTTPException(status_code=400, detail="Cannot check availability for a past date")

    slots = get_available_slots(doctor_name, target_date)
    return AvailabilityResponse(
        doctor_name=doctor_name,
        specialty=DOCTORS[doctor_name]["specialty"],
        date=date,
        available_slots=slots,
    )


# ---- Book appointment ------------------------------------------------------
@app.post("/api/appointments/book", response_model=AppointmentResponse, status_code=201, tags=["Appointments"])
def book_appointment(payload: BookAppointmentRequest):
    """
    Book a new appointment. Validates the doctor works that day and the
    requested slot is actually open before committing.
    """
    target_date = datetime.strptime(payload.appt_date, "%Y-%m-%d").date()
    if target_date < datetime.now(timezone.utc).date():
        raise HTTPException(status_code=400, detail="Cannot book an appointment in the past")

    available_slots = get_available_slots(payload.doctor_name, target_date)
    if payload.appt_time not in available_slots:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"That slot is not available for {payload.doctor_name} on {payload.appt_date}",
                "available_slots": available_slots,
            },
        )

    doctor = DOCTORS[payload.doctor_name]
    now = datetime.now(timezone.utc).isoformat()

    with get_conn() as conn:
        cur = conn.cursor()
        new_id = get_next_appointment_id(cur)
        cur.execute(
            """INSERT INTO appointments
               (appointment_id, patient_name, phone, doctor_name, appt_date, appt_time, status, fee_pkr, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                new_id,
                payload.patient_name,
                payload.phone,
                payload.doctor_name,
                payload.appt_date,
                payload.appt_time,
                "Scheduled",
                doctor["fee_pkr"],
                now,
            ),
        )

    return AppointmentResponse(
        appointment_id=new_id,
        patient_name=payload.patient_name,
        phone=payload.phone,
        doctor_name=payload.doctor_name,
        specialty=doctor["specialty"],
        appt_date=payload.appt_date,
        appt_time=payload.appt_time,
        status="Scheduled",
        fee_pkr=doctor["fee_pkr"],
    )


# ---- Appointment status / lookup -----------------------------------------
@app.get("/api/appointments/{appointment_id}", response_model=AppointmentResponse, tags=["Appointments"], dependencies=[Depends(verify_internal_key)])
def get_appointment(appointment_id: int):
    """Look up an appointment's current status by ID."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Appointment {appointment_id} not found")

    return AppointmentResponse(
        appointment_id=row["appointment_id"],
        patient_name=row["patient_name"],
        phone=row["phone"],
        doctor_name=row["doctor_name"],
        specialty=DOCTORS.get(row["doctor_name"], {}).get("specialty", "Unknown"),
        appt_date=row["appt_date"],
        appt_time=row["appt_time"],
        status=row["status"],
        fee_pkr=row["fee_pkr"],
    )


@app.post("/api/appointments/{appointment_id}/cancel", response_model=AppointmentResponse, tags=["Appointments"])
def cancel_appointment(appointment_id: int):
    """Cancel an existing appointment, freeing up its slot."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Appointment {appointment_id} not found")
        if row["status"] != "Scheduled":
            raise HTTPException(status_code=400, detail=f"Appointment is already '{row['status']}', cannot cancel")

        conn.execute(
            "UPDATE appointments SET status = 'Cancelled' WHERE appointment_id = ?", (appointment_id,)
        )

    return AppointmentResponse(
        appointment_id=row["appointment_id"],
        patient_name=row["patient_name"],
        phone=row["phone"],
        doctor_name=row["doctor_name"],
        specialty=DOCTORS.get(row["doctor_name"], {}).get("specialty", "Unknown"),
        appt_date=row["appt_date"],
        appt_time=row["appt_time"],
        status="Cancelled",
        fee_pkr=row["fee_pkr"],
    )


@app.get("/api/appointments", tags=["Appointments"], dependencies=[Depends(verify_internal_key)])
def list_appointments(
    doctor_name: Optional[str] = None,
    date: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = Query(None, description="Free-text search on patient name or phone"),
):
    """
    List appointments, optionally filtered. Powers the dashboard's
    appointments table:
      - date              exact YYYY-MM-DD
      - start / end       inclusive YYYY-MM-DD range (e.g. this week)
      - doctor_name        exact match
      - status            Scheduled | Completed | Cancelled | No-Show
      - q                  substring match on patient_name or phone
    With no filters, returns every appointment (debug helper, as before).
    """
    query = "SELECT * FROM appointments WHERE 1=1"
    params: list = []

    if doctor_name:
        query += " AND doctor_name = ?"
        params.append(doctor_name)
    if date:
        query += " AND appt_date = ?"
        params.append(date)
    if start:
        query += " AND appt_date >= ?"
        params.append(start)
    if end:
        query += " AND appt_date <= ?"
        params.append(end)
    if status:
        if status not in VALID_STATUSES:
            raise HTTPException(status_code=422, detail=f"status must be one of {VALID_STATUSES}")
        query += " AND status = ?"
        params.append(status)
    if q:
        query += " AND (patient_name LIKE ? OR phone LIKE ?)"
        like = f"%{q}%"
        params.extend([like, like])

    query += " ORDER BY appt_date, appt_time"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


# ---- Dashboard -------------------------------------------------------------
@app.get("/api/dashboard/summary", tags=["Dashboard"], dependencies=[Depends(verify_internal_key)])
def dashboard_summary(doctor_name: str = "Dr. Muhammad Zubair Yousuf"):
    """
    One bundled call for the dashboard's top-of-page view: today's schedule,
    this week's count, a status breakdown, the next few upcoming appointments,
    and a complaints snapshot.
    """
    if doctor_name not in DOCTORS:
        raise HTTPException(status_code=404, detail=f"'{doctor_name}' is not a doctor at this clinic")

    doctor = DOCTORS[doctor_name]
    today = datetime.now(timezone.utc).date()
    week_start = today - timedelta(days=today.weekday())  # Monday
    week_end = week_start + timedelta(days=6)

    with get_conn() as conn:
        today_rows = conn.execute(
            """SELECT * FROM appointments WHERE doctor_name = ? AND appt_date = ?
               ORDER BY appt_time""",
            (doctor_name, today.isoformat()),
        ).fetchall()

        week_count_row = conn.execute(
            """SELECT COUNT(*) as c FROM appointments
               WHERE doctor_name = ? AND appt_date BETWEEN ? AND ?""",
            (doctor_name, week_start.isoformat(), week_end.isoformat()),
        ).fetchone()

        status_rows = conn.execute(
            """SELECT status, COUNT(*) as c FROM appointments
               WHERE doctor_name = ? GROUP BY status""",
            (doctor_name,),
        ).fetchall()

        upcoming_rows = conn.execute(
            """SELECT * FROM appointments
               WHERE doctor_name = ? AND status = 'Scheduled' AND appt_date >= ?
               ORDER BY appt_date, appt_time LIMIT 5""",
            (doctor_name, today.isoformat()),
        ).fetchall()

        complaints_total_row = conn.execute("SELECT COUNT(*) as c FROM complaints").fetchone()
        recent_complaints = conn.execute(
            "SELECT * FROM complaints ORDER BY date_time DESC LIMIT 5"
        ).fetchall()

    status_counts = {s: 0 for s in VALID_STATUSES}
    for row in status_rows:
        status_counts[row["status"]] = row["c"]

    return {
        "doctor": {
            "name": doctor_name,
            "specialty": doctor["specialty"],
            "days": doctor["days"],
            "hours": f"{doctor['start']} - {doctor['end']}",
            "slot_minutes": doctor["slot_minutes"],
            "fee_pkr": doctor["fee_pkr"],
        },
        "today": {
            "date": today.isoformat(),
            "count": len(today_rows),
            "appointments": [dict(r) for r in today_rows],
        },
        "week": {
            "start": week_start.isoformat(),
            "end": week_end.isoformat(),
            "count": week_count_row["c"],
        },
        "status_counts": status_counts,
        "upcoming": [dict(r) for r in upcoming_rows],
        "complaints": {
            "total": complaints_total_row["c"],
            "recent": [dict(r) for r in recent_complaints],
        },
    }


# ---- Health --------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Clinic Appointment API", "docs": "/docs"}


@app.exception_handler(ValueError)
def value_error_handler(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
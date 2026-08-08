import base64
import os
import sqlite3
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Annotated, Literal, Sequence, TypedDict
from zoneinfo import ZoneInfo

import redis
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field
from pywa import WhatsApp, types
from pywa.types import MessageType

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ---------------- Credentials ----------------
load_dotenv()
PHONENUMBER_ID = os.getenv("PHONENUMBER_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
APP_ID = os.getenv("APP_ID")
APP_SECRET = os.getenv("APP_SECRET")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
GOOGLE_GENERATIVE_AI = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLINIC_BASE_URL = os.getenv("CLINIC_BASE_URL")  # used inside appointment_management.py / patient_complaint.py

openrouter_model = OpenAI(api_key=OPENAI_API_KEY, base_url="https://openrouter.ai/api/v1")

app = FastAPI(title="Clinic WhatsApp API Server")
wa = WhatsApp(
    phone_id=PHONENUMBER_ID,
    token=ACCESS_TOKEN,
    app_id=APP_ID,
    app_secret=APP_SECRET,
    callback_url="https://overvaliant-waneta-optometrical.ngrok-free.dev",  # your stable domain, e.g. https://clinicbot.yourdomain.com
    server=app,
    webhook_endpoint="/whatsapp/webhook",
    verify_token=VERIFY_TOKEN,
)

app.mount("/dashboard", StaticFiles(directory="dashboard_public", html=True), name="dashboard")

def get_chat_model():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, api_key=GOOGLE_GENERATIVE_AI)


CLINIC_TIMEZONE = ZoneInfo("Asia/Karachi")


def get_current_date_context() -> str:
    """Human-readable 'today' context to inject into date-sensitive prompts."""
    now = datetime.now(CLINIC_TIMEZONE)
    return f"{now.strftime('%A, %Y-%m-%d')} (current time: {now.strftime('%I:%M %p')})"


# ---------------- State ----------------
class BotState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    incoming_message: str
    wa_id: int
    classifier: Literal["appointment_management", "patient_query", "patient_complaint"]
    response_to_user: str
    flow_active: bool  # True while mid-task with the current agent; skips reclassification


# ---------------- Shared helpers ----------------
REROUTE_SENTINEL = "REROUTE"

REROUTE_INSTRUCTION = (
    "\n\nIMPORTANT: If the patient's latest message is clearly unrelated to your current task "
    f"(e.g. they ask something from a totally different topic), respond with EXACTLY the single "
    f"word {REROUTE_SENTINEL} and nothing else. Do not use this for anything else."
)


def extract_text(ai_message: AIMessage) -> str:
    """
    Gemini sometimes returns content as a string, sometimes as a list of
    content blocks (e.g. [{'type': 'text', 'text': '...'}]). Normalize both.
    """
    content = ai_message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


def sanitize_history_for_agent(history: list[BaseMessage]) -> list[BaseMessage]:
    """
    Remove tool-call AIMessages and their corresponding ToolMessage results
    from history before handing it to a different agent. Prevents Gemini
    from seeing a function-call turn for a tool that isn't bound in the
    current agent's tool set (e.g. book_appointment showing up mid-complaint-flow).
    """
    cleaned: list[BaseMessage] = []
    skip_tool_results = False
    for msg in history:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            skip_tool_results = True
            continue
        if isinstance(msg, ToolMessage):
            if skip_tool_results:
                continue
        else:
            skip_tool_results = False
        cleaned.append(msg)
    return cleaned


# ---------------- Media -> text preprocessing ----------------
def extract_text_from_message(message: types.Message) -> str:
    """
    Converts any incoming message (text, voice note, image) into a single
    plain-text string that can be fed into the graph as `incoming_message`.
    Runs synchronously in the webhook handler, so failures here are caught
    and degrade to an empty string rather than crashing the handler.
    """
    try:
        if message.type == MessageType.TEXT:
            return message.text

        elif message.type == MessageType.AUDIO and message.audio.voice:
            print("Downloading voice note...")
            audio_bytes = message.audio.get_bytes()
            base64audio = base64.b64encode(audio_bytes).decode("utf-8")
            try:
                response = openrouter_model.chat.completions.create(
                    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Transcribe this audio precisely. Output ONLY the transcription text, nothing else.",
                                },
                                {"type": "input_audio", "input_audio": {"data": base64audio, "format": "ogg"}},
                            ],
                        }
                    ],
                    timeout=30,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"OpenRouter error in STT: {e}")
                return ""

        elif message.type == MessageType.IMAGE:
            image_bytes = message.image.get_bytes()
            mime_type = getattr(message.image, "mime_type", None) or "image/jpeg"
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            caption = message.image.caption or ""
            try:
                response = openrouter_model.chat.completions.create(
                    model="nvidia/nemotron-nano-12b-2-vl:free",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract all text from this image or describe the image clearly so a search engine can index it. If it's not text, analyze the image and describe what it is.",
                                },
                                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                            ],
                        }
                    ],
                    timeout=30,
                )
                extracted_visual_text = response.choices[0].message.content or ""
                return f"Patient Caption: {caption} | Image content: {extracted_visual_text}".strip()
            except Exception as e:
                print(f"OpenRouter image processing error: {e}")
                return caption

    except Exception as e:
        print(f"Failed to download/process media: {e}")
        return ""

    return ""


# ---------------- Classifier ----------------
class IncomingMessageParser(BaseModel):
    category: Literal["appointment_management", "patient_query", "patient_complaint"] = Field(
        description="The category of incoming message."
    )


def classifier_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that classifies incoming WhatsApp messages sent to a clinic into one "
                "of the following categories: appointment_management, patient_query, or patient_complaint.\n\n"
                "appointment_management: the patient wants to BOOK, SCHEDULE, RESERVE, RESCHEDULE, or CANCEL an "
                "appointment, or check the status of an existing one — even if phrased casually, combined with a "
                "greeting, or not using the word 'appointment' explicitly (e.g. 'I want to come in today', "
                "'can I get a slot at 8:30', 'I'd like to see the doctor tomorrow'). Any expressed intent to "
                "actually book/visit takes priority over this being classified as a general question.\n\n"
                "patient_query: general informational questions with no booking intent — doctor's qualifications, "
                "clinic timings, fees, services offered, location, parking, payment options.\n\n"
                "patient_complaint: a complaint or negative feedback about their experience.\n\n"
                "You should only respond with the category name, and nothing else.",
            ),
            ("human", "Classify the following message: {message}"),
        ]
    )


# Deterministic pre-check: if the message unambiguously signals booking intent,
# skip the LLM classifier entirely rather than trusting its judgment. This is a
# defense-in-depth backstop — the classifier prompt above should already catch
# these, but a keyword match costs nothing and can't be talked out of routing
# correctly the way an LLM occasionally can be.
_BOOKING_INTENT_KEYWORDS = [
    "book", "booking", "appoitment", "appointent", "appoint",  # common misspellings included on purpose
    "appointment", "schedule", "reschedule", "cancel my", "slot",
    "reserve", "come at", "come in", "come today", "visit today",
    "i'll come", "ill come", "aana hai", "aa jau", "milna hai",
]


def looks_like_booking_intent(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _BOOKING_INTENT_KEYWORDS)


def structured_llm_model(chat_model: ChatGoogleGenerativeAI = None):
    chat_model = chat_model or get_chat_model()
    return chat_model.with_structured_output(IncomingMessageParser)


def classifier_agent(state: BotState):
    message = state["incoming_message"]

    if looks_like_booking_intent(message):
        return {"classifier": "appointment_management", "flow_active": True}

    prompt = classifier_prompt()
    structured_model = structured_llm_model()
    chain = prompt | structured_model
    response = chain.invoke({"message": message})
    return {"classifier": response.category, "flow_active": True}


# ---------------- RAG tool ----------------
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders.text import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

_VECTORSTORE_PATH = "vectorstore"
_CLINIC_DETAILS_PATH = "./clinic_details.txt"
_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_or_load_vectorstore():
    if os.path.exists(_VECTORSTORE_PATH):
        return FAISS.load_local(_VECTORSTORE_PATH, _embeddings, allow_dangerous_deserialization=True)

    loader = TextLoader(_CLINIC_DETAILS_PATH)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=60)
    chunks = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, _embeddings)
    vectorstore.save_local(_VECTORSTORE_PATH)
    return vectorstore


_vectorstore = build_or_load_vectorstore()


def RAG_tool():
    return _vectorstore.as_retriever(search_kwargs={"k": 3})


# ---------------- Agent: patient query ----------------
def patient_query_agent(state: BotState):
    chat_model = get_chat_model()
    retriever = RAG_tool()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a friendly, professional receptionist assistant for Muhammad Yousuf Dental Clinic "
                "(Nazimabad No. 1, near UC Office and Aga Juice, Karachi). Answer patient questions about "
                "Dr. Muhammad Zubair Yousuf, clinic timings, fees, services, parking, and payment options "
                "using the context below. Always ground your answer in the context first.\n\n"
                "Context:\n{context}\n\n"
                "If the patient sends a greeting, greet them back respectfully. Be precise and concise. "
                "You may receive messages in Roman English (Urdu written in Latin script) — accommodate that. "
                "Remind patients that both walk-in and appointment-booking are available if they ask how to visit. "
                "NEVER provide dental/medical advice, diagnose symptoms, or recommend treatment — you only provide "
                "clinic/scheduling information. If a patient describes symptoms or dental pain and asks for advice, "
                "gently redirect them to book an appointment or come in for a checkup instead. "
                "If a patient describes what sounds like a dental emergency, let them know emergency visits are "
                "available (PKR 1,000) and they can walk in or book an emergency slot. If it sounds like a "
                "non-dental, life-threatening emergency, tell them to call the local emergency number or go to "
                "the nearest emergency room immediately."
                + REROUTE_INSTRUCTION.replace(
                    "respond with EXACTLY the single word",
                    "— including if they want to BOOK, RESCHEDULE, or CANCEL an appointment, or check an "
                    "existing appointment's status, since that is handled by a different assistant — "
                    "respond with EXACTLY the single word",
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm=chat_model, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=question_answer_chain)

    history = sanitize_history_for_agent(state.get("messages", [])[:-1])
    current_user_message = state["incoming_message"]
    response = rag_chain.invoke({"chat_history": history, "input": current_user_message})
    answer = response["answer"].strip()

    if answer == REROUTE_SENTINEL:
        return {
            "messages": [],
            "response_to_user": "",
            "flow_active": False,
        }

    return {
        "messages": [AIMessage(content=answer)],
        "response_to_user": answer,
        "flow_active": False,  # queries are single-turn; always reclassify next message
    }


# ---------------- Agent: appointment management ----------------
from appointment_management import book_appointment, check_appointment_status, check_doctor_availability

_APPOINTMENT_TOOLS = [check_doctor_availability, book_appointment, check_appointment_status]
_APPOINTMENT_TOOLS_BY_NAME = {t.name: t for t in _APPOINTMENT_TOOLS}


def appointment_management_agent(state: BotState):
    chat_model = get_chat_model().bind_tools(_APPOINTMENT_TOOLS)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the appointment booking assistant for Muhammad Yousuf Dental Clinic (Nazimabad No. 1, near UC Office and Aga Juice, Karachi), booking patients in with Dr. Muhammad Zubair Yousuf over WhatsApp and reporting appointment status.
# TODAY'S DATE: {current_date}
# CLINIC FACTS:
* Only one doctor: Dr. Muhammad Zubair Yousuf (BDS, RDS, MScDS, PDG-HP).
* Clinic hours: Monday - Saturday, 7:30 PM - 11:30 PM. Closed Sunday.
* Standard checkup fee: PKR 300-500. Emergency visit: PKR 1,000. Home service: PKR 1,000 per visit.
* You may receive messages in Roman English (Urdu written in Latin script) — accommodate that.
* Both walk-in and prior appointment booking are available — mention this if the patient seems unsure whether they need to book ahead.
* Payment: cash or online payment, both accepted at the clinic.
* Parking is available on-site.

# GUIDELINES:
* Ask for a preferred date and, if relevant, roughly what time within clinic hours (7:30 PM - 11:30 PM) works for them.
* If the patient says a relative date like "today", "tomorrow", "day after tomorrow", or a weekday name (e.g. "this Friday", "next Monday"), YOU must resolve it into the correct YYYY-MM-DD date yourself using TODAY'S DATE above. Do NOT ask the patient to type the date in YYYY-MM-DD format themselves — that's your job, not theirs.
* Use 'check_doctor_availability' with doctor_name="Dr. Muhammad Zubair Yousuf" and the resolved YYYY-MM-DD date to show open slots.
* Once the patient picks a slot, collect their full name and phone number.
* Confirm all details back to the patient (date, time, name, phone) before booking, and remind them of the checkup fee (PKR 300-500, payable at the clinic).
* Use 'book_appointment' to finalize the booking only after confirmation.
* If the patient describes urgent dental pain or asks for an emergency visit, let them know the emergency fee is PKR 1,000 and they can either book an emergency slot the same way, or walk in directly during clinic hours — no need to wait for a booking if it's urgent.
* If the patient asks about a home visit, let them know it's available for PKR 1,000 per visit, and help them book a slot for it the same way.
* If the patient wants to check an existing appointment, ask for their appointment ID and use 'check_appointment_status'.
* Never provide dental/medical advice, diagnose symptoms, or discuss treatment options — only help with scheduling.
* If a patient describes what sounds like a non-dental, life-threatening emergency, tell them to call the local emergency number or go to the nearest emergency room immediately, and do not proceed with booking."""
                .replace("{current_date}", get_current_date_context())
                + REROUTE_INSTRUCTION,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    appointment_llm = prompt | chat_model

    history = sanitize_history_for_agent(state.get("messages", [])[:-1])
    current_user_message = state["incoming_message"]

    response = appointment_llm.invoke({"chat_history": history, "input": current_user_message})
    new_messages: list[BaseMessage] = [response]
    tool_results: list[str] = []

    while response.tool_calls:
        tool_messages = []
        for call in response.tool_calls:
            tool_fn = _APPOINTMENT_TOOLS_BY_NAME[call["name"]]
            tool_result = tool_fn.invoke(call["args"])
            tool_results.append(str(tool_result))
            tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))
        new_messages.extend(tool_messages)

        response = chat_model.invoke(history + [HumanMessage(content=current_user_message)] + new_messages)
        new_messages.append(response)

    answer = extract_text(response).strip()

    if answer == REROUTE_SENTINEL:
        return {
            "messages": [],
            "response_to_user": "",
            "flow_active": False,
        }

    task_complete = any(r.startswith("Appointment confirmed") for r in tool_results)

    return {
        "messages": new_messages,
        "response_to_user": answer,
        "flow_active": not task_complete,
    }


# ---------------- Agent: patient complaint ----------------
from patient_complaint import generate_ticket

_COMPLAINT_TOOLS = [generate_ticket]
_COMPLAINT_TOOLS_BY_NAME = {t.name: t for t in _COMPLAINT_TOOLS}


def patient_complaint_agent(state: BotState):
    chat_model = get_chat_model().bind_tools(_COMPLAINT_TOOLS)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an experienced clinic patient-relations assistant, who responds to complaints about a patient's experience — long wait times, billing issues, staff conduct, or any other non-medical issue.
You can answer questions based on the clinic context/details, but if the problem goes beyond that, proceed with the 'generate_ticket' tool.
# You can log a complaint by using the 'generate_ticket' tool.
* First, ask for the patient's name.
* Next, get a valid active email from the patient.
* After that, ask for a phone number.
* Then analyze the problem the patient is facing, and write it up in the format of a complaint so it can be actioned later.
* Do NOT attempt to resolve medical complaints or comment on clinical care — only log the ticket for the clinic staff to review.
After the tool call is successful, tell the patient they'll be accommodated within 24 hours.
You may receive messages in Roman English (Urdu written in Latin script) — accommodate that."""
                + REROUTE_INSTRUCTION,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    ticket_llm = prompt | chat_model

    history = sanitize_history_for_agent(state.get("messages", [])[:-1])
    current_user_message = state["incoming_message"]

    response = ticket_llm.invoke({"chat_history": history, "input": current_user_message})
    new_messages: list[BaseMessage] = [response]
    tool_results: list[str] = []

    while response.tool_calls:
        tool_messages = []
        for call in response.tool_calls:
            tool_fn = _COMPLAINT_TOOLS_BY_NAME[call["name"]]
            tool_result = tool_fn.invoke(call["args"])
            tool_results.append(str(tool_result))
            tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=call["id"]))
        new_messages.extend(tool_messages)

        response = chat_model.invoke(history + [HumanMessage(content=current_user_message)] + new_messages)
        new_messages.append(response)

    answer = extract_text(response).strip()

    if answer == REROUTE_SENTINEL:
        return {
            "messages": [],
            "response_to_user": "",
            "flow_active": False,
        }

    task_complete = any("has been logged" in r for r in tool_results)

    return {
        "messages": new_messages,
        "response_to_user": answer,
        "flow_active": not task_complete,
    }


# ---------------- Routing ----------------
def entry_router(
    state: BotState,
) -> Literal["classifier_agent", "appointment_management_agent", "patient_query_agent", "patient_complaint_agent"]:
    """
    Sticky routing: if we're mid-task with a known agent, skip reclassification
    and go straight back to it. Otherwise (fresh conversation, or the previous
    agent finished/rerouted), reclassify.
    """
    if state.get("flow_active") and state.get("classifier"):
        return f"{state['classifier']}_agent"
    return "classifier_agent"


def routing_agent(
    state: BotState,
) -> Literal["patient_complaint_agent", "patient_query_agent", "appointment_management_agent"]:
    if state["classifier"] == "patient_complaint":
        return "patient_complaint_agent"
    if state["classifier"] == "patient_query":
        return "patient_query_agent"
    return "appointment_management_agent"


def after_agent_router(state: BotState) -> Literal["classifier_agent", "__end__"]:
    """If an agent hit the REROUTE sentinel, loop back through the classifier once."""
    if state.get("response_to_user", "") == "":
        return "classifier_agent"
    return END


# ---------------- Build the graph ONCE at module load ----------------
_conn = sqlite3.connect("conversations.db", check_same_thread=False)
checkpointer = SqliteSaver(_conn)


def build_graph():
    graph = StateGraph(BotState)
    graph.add_node("classifier_agent", classifier_agent)
    graph.add_node("patient_query_agent", patient_query_agent)
    graph.add_node("patient_complaint_agent", patient_complaint_agent)
    graph.add_node("appointment_management_agent", appointment_management_agent)

    graph.add_conditional_edges(START, entry_router)
    graph.add_conditional_edges("classifier_agent", routing_agent)
    graph.add_conditional_edges("patient_query_agent", after_agent_router)
    graph.add_conditional_edges("appointment_management_agent", after_agent_router)
    graph.add_conditional_edges("patient_complaint_agent", after_agent_router)

    return graph.compile(checkpointer=checkpointer)

compiled_graph = build_graph()

# ---------------- Background processing ----------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

_executor = ThreadPoolExecutor(max_workers=8)

_DEDUPE_TTL_SECONDS = 60 * 60 * 24  # 24h — long enough to cover any realistic Meta webhook retry window
_LOCK_TIMEOUT_SECONDS = 120  # auto-expires if a worker crashes mid-processing, so a lock can't get stuck forever
_LOCK_BLOCKING_TIMEOUT_SECONDS = 130  # how long a second message from the same patient waits for the first to finish


def is_duplicate_message(message_id: str) -> bool:
    """
    Atomically marks a message ID as seen. Returns True if it was ALREADY
    seen (duplicate — skip it), False if this is the first time (proceed).
    Replaces the old in-memory `set()`, which reset on every restart and
    only worked within a single process.
    """
    was_newly_set = redis_client.set(f"processed_msg:{message_id}", "1", nx=True, ex=_DEDUPE_TTL_SECONDS)
    return not was_newly_set


def get_user_lock(wa_id: str):
    """
    Redis-backed distributed lock, one per patient. Replaces the old
    threading.Lock()-per-user dict, which only serialized messages within
    a single process — this version works correctly even across multiple
    server instances, and auto-releases if a worker crashes mid-processing.
    """
    return redis_client.lock(
        f"lock:user:{wa_id}",
        timeout=_LOCK_TIMEOUT_SECONDS,
        blocking_timeout=_LOCK_BLOCKING_TIMEOUT_SECONDS,
    )


def process_message(message: types.Message, incoming_text: str):
    wa_id = str(message.from_user.wa_id)
    lock = get_user_lock(wa_id)

    acquired = False
    try:
        acquired = lock.acquire(blocking=True)
    except Exception as e:
        print(f"Redis lock error for {wa_id}: {e}", flush=True)

    if not acquired:
        # Either Redis is unreachable, or this patient already has a message
        # being processed and we waited the full blocking_timeout for nothing.
        print(f"Could not acquire lock for {wa_id} — dropping this turn", flush=True)
        wa.send_message(to=message.from_user.wa_id, text="Sorry, I'm still processing your previous message. Please wait a moment and try again.")
        return

    try:
        start = time.time()
        config = {"configurable": {"thread_id": wa_id}, "recursion_limit": 15}

        try:
            result = compiled_graph.invoke(
                {
                    "incoming_message": incoming_text,
                    "wa_id": wa_id,
                    "messages": [HumanMessage(content=incoming_text)],
                },
                config=config,
            )
            ai_reply = result.get("response_to_user") or "Sorry, I didn't quite catch that. Could you rephrase?"
        except Exception:
            traceback.print_exc()
            ai_reply = "Sorry, I'm having trouble right now. Please try again in a moment."

        print(f"Processed in {time.time() - start:.2f}s", flush=True)
        wa.send_message(to=message.from_user.wa_id, text=ai_reply)
    finally:
        try:
            lock.release()
        except Exception as e:
            # Lock may have already auto-expired (timeout) — non-fatal either way.
            print(f"Non-fatal: lock release for {wa_id} failed: {e}", flush=True)


@app.get("/")
def health():
    return {"status": "ok"}


@wa.on_message()
def handle_text_message(client: WhatsApp, message: types.Message):
    if is_duplicate_message(message.id):
        return

    incoming_text = extract_text_from_message(message)
    if not incoming_text:
        wa.send_message(to=message.from_user.wa_id, text="Sorry, I couldn't understand that message. Could you try again?")
        return

    try:
        message.mark_as_read()  # verify exact method name against your pywa version's docs
    except Exception as e:
        print(f"mark_as_read failed (non-fatal): {e}")

    _executor.submit(process_message, message, incoming_text)
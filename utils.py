import json
import os
import re
from datetime import datetime

CALL_RECORD_FILE = "logs/call_records.json"
LEAD_LOG_FILE = "logs/lead_logs.txt"

def init_call_record(session_id):
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(CALL_RECORD_FILE):
        with open(CALL_RECORD_FILE, "w") as f:
            json.dump({}, f)

    with open(CALL_RECORD_FILE, "r") as f:
        data = json.load(f)

    data[session_id] = {
        "start_time": str(datetime.now()),
        "turns": [],
        "escalated": False,
        "phone": None,
        "last_unanswered_query": None
    }

    with open(CALL_RECORD_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_call_record(session_id, user, agent, escalated=False, phone=None, unanswered_query=None):
    with open(CALL_RECORD_FILE, "r") as f:
        data = json.load(f)

    if session_id not in data:
        init_call_record(session_id)
        with open(CALL_RECORD_FILE, "r") as f:
            data = json.load(f)

    data[session_id]["turns"].append({
        "user": user,
        "agent": agent
    })

    if escalated:
        data[session_id]["escalated"] = True

    if phone:
        data[session_id]["phone"] = phone

    if unanswered_query is not None:
        data[session_id]["last_unanswered_query"] = unanswered_query

    with open(CALL_RECORD_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_last_unanswered_query(session_id):
    try:
        with open(CALL_RECORD_FILE, "r") as f:
            data = json.load(f)
        return data.get(session_id, {}).get("last_unanswered_query")
    except Exception:
        return None

def end_call_record(session_id):
    with open(CALL_RECORD_FILE, "r") as f:
        data = json.load(f)

    data[session_id]["end_time"] = str(datetime.now())

    with open(CALL_RECORD_FILE, "w") as f:
        json.dump(data, f, indent=2)

def append_lead_log(session_id, phone, unanswered_query):
    os.makedirs("logs", exist_ok=True)
    with open(LEAD_LOG_FILE, "a") as f:
        f.write(f"{datetime.now()} | call_id={session_id} | phone={phone} | unanswered_query={unanswered_query}\n")

def detect_phone_number(text):
    match = re.search(r"(03\d{9})", text)
    return match.group(1) if match else None
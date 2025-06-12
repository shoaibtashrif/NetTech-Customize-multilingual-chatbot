import openai
import os
import json
import uuid
import threading
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template
import requests
from dotenv import load_dotenv
import codecs

# ─── Setup & Logging ─────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEYY")

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ─── Route Dump on Startup & First Request ──────────────────
@app.before_first_request
def log_routes_first():
    logging.info("==== Registered Flask routes (first request) ====")
    for rule in app.url_map.iter_rules():
        logging.info(f"{rule.endpoint:20s} -> {rule.rule}")
    logging.info("==============================================")

if __name__ == "__main__":
    logging.info("==== Registered Flask routes (at startup) ====")
    for rule in app.url_map.iter_rules():
        logging.info(f"{rule.endpoint:20s} -> {rule.rule}")
    logging.info("===========================================")
    app.run(debug=True)

# ─── In-Memory Session Stores ─────────────────────────────────
session_files = {}  # sid → { 'prompt': str, 'knowledge': str }

class ConversationManager:
    def __init__(self):
        self.history = {}      # sid → [messages]
        self.locks   = {}      # sid → threading.Lock
        self.global_lock = threading.Lock()
        self.last_activity = {}  # sid → datetime

    def _get_lock(self, sid):
        with self.global_lock:
            if sid not in self.locks:
                self.locks[sid] = threading.Lock()
                self.last_activity[sid] = datetime.now()
            return self.locks[sid]

    def get(self, sid):
        with self._get_lock(sid):
            self.last_activity[sid] = datetime.now()
            return self.history.get(sid, [])

    def append(self, sid, msgs):
        with self._get_lock(sid):
            self.history.setdefault(sid, []).extend(msgs)
            self.last_activity[sid] = datetime.now()

    def clear(self, sid):
        with self._get_lock(sid):
            self.history.pop(sid, None)
            self.locks.pop(sid, None)
            self.last_activity.pop(sid, None)
            session_files.pop(sid, None)
            logging.info(f"[{sid}] Cleared session data.")

    def cleanup(self, max_minutes=60):
        cutoff = datetime.now() - timedelta(minutes=max_minutes)
        with self.global_lock:
            for sid, last in list(self.last_activity.items()):
                if last < cutoff:
                    self.clear(sid)
                    logging.info(f"[{sid}] Auto-cleared inactive session.")

conv_mgr = ConversationManager()

# ─── Helpers ────────────────────────────────────────────────
def extract_text(file_storage):
    try:
        return file_storage.read().decode("utf-8", errors="ignore")
    except:
        return ""

def load_system_prompt(sid):
    return session_files.get(sid, {}).get("prompt") or (
        codecs.open("system_prompt.txt", "r", "utf-8").read().strip()
        if os.path.exists("system_prompt.txt")
        else "Default system prompt."
    )

def load_custom_knowledge(sid):
    return session_files.get(sid, {}).get("knowledge", "")

def summarize_conv(history, sid):
    text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system", "content":"Summarize this Urdu conversation."},
                {"role":"user",   "content": text}
            ],
            temperature=0.3
        )
        summary = resp.choices[0].message.content
        logging.info(f"[{sid}] Summary generated.")
        return summary
    except Exception as e:
        logging.warning(f"[{sid}] Summarization error: {e}")
        return "Summary not available."

def detect_type(history, sid):
    text = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system", "content":"Classify complaint: timing info, timing complaint, behaviour complaint, or other."},
                {"role":"user",   "content": text}
            ],
            temperature=0.3
        )
        ctype = resp.choices[0].message.content.lower()
        logging.info(f"[{sid}] Complaint type detected: {ctype}")
        return ctype
    except Exception as e:
        logging.warning(f"[{sid}] Classification error: {e}")
        return "unknown"

# ─── Routes ────────────────────────────────────────────────

# 1) Home → serves templates/index.html
@app.route("/")
def home():
    return render_template("index.html")

# 2) Start a new session
@app.route("/start_session", methods=["POST"])
def start_session():
    sid = str(uuid.uuid4())
    conv_mgr.clear(sid)
    session_files[sid] = {}
    logging.info(f"[{sid}] Session started.")
    return jsonify({"session_id": sid})

# 3) Upload prompt & knowledge files
@app.route("/upload", methods=["POST"])
def upload():
    sid = request.form.get("session_id")
    if not sid or sid not in session_files:
        return jsonify({"status":"error","message":"Invalid session_id."}), 400

    if "promptFile" in request.files:
        session_files[sid]["prompt"] = extract_text(request.files["promptFile"])
        logging.info(f"[{sid}] Prompt file uploaded.")
    if "knowledgeFile" in request.files:
        session_files[sid]["knowledge"] = extract_text(request.files["knowledgeFile"])
        logging.info(f"[{sid}] Knowledge file uploaded.")

    return jsonify({"status":"success","message":"Files uploaded."})

# 4) Chatbot
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    prompt = data.get("prompt")
    sid    = data.get("session_id")
    if not sid:
        return jsonify({"response":"Session ID missing."}), 400

    sys_p = load_system_prompt(sid)
    know  = load_custom_knowledge(sid)
    dyn   = f"اضافی معلومات:\n{know}"

    history = conv_mgr.get(sid)
    if not history:
        history = [{"role":"system","content":sys_p},
                   {"role":"assistant","content":dyn}]

    query = history + [{"role":"user","content":prompt}]
    limited = query[-20:]

    try:
        resp  = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=limited,
            temperature=0.4
        )
        reply = resp.choices[0].message.content

        conv_mgr.append(sid, [
            {"role":"user",     "content":prompt},
            {"role":"assistant","content":reply}
        ])

        if len(conv_mgr.last_activity) % 10 == 0:
            conv_mgr.cleanup()

        logging.info(f"[{sid}] User → {prompt}")
        logging.info(f"[{sid}] Bot  → {reply}")
        return jsonify({"response": reply})

    except Exception as e:
        logging.error(f"[{sid}] OpenAI error: {e}")
        return jsonify({"response":"معاف کیجیے، کوشش کریں دوبارہ۔"}), 500

# 5) End session
@app.route("/end_session", methods=["POST"])
def end_session():
    data = request.get_json()
    sid  = data.get("session_id")
    if not sid:
        return jsonify({"status":"error","message":"Session ID missing."}), 400

    history = conv_mgr.get(sid)
    if not history:
        return jsonify({"status":"error","message":"No history found."}), 404

    summary = summarize_conv(history, sid)
    ctype   = detect_type(history, sid)

    payload = {
        "transcript": history,
        "summary": summary,
        "complaint_type": ctype
    }

    try:
        logging.info(f"[{sid}] Uploading JSON payload...")
        res = requests.post("http://localhost:5001/save", json=payload)
        logging.info(f"[{sid}] External server responded {res.status_code}")
    except Exception as e:
        logging.error(f"[{sid}] Upload failed: {e}")
        return jsonify({"status":"error","message":"Upload failed."}), 500

    conv_mgr.clear(sid)
    return jsonify({"status":"success"})

import os
import json
import uuid
import logging
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import codecs
import openai
from zoneinfo import ZoneInfo

# ─── Setup ─────────────────────────────────────────────────────────────
load_dotenv(override=True)

# OpenAI creds
openai.api_key = os.getenv("OPENAI_API_KEY")

# HF creds & model (also used by Groq)
HF_API_KEY   = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL_ID  = os.getenv("HUGGINGFACE_MODEL_ID", "gemma2-9b-it")

# Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Admin‐controlled toggle
current_model = "openai"

# Flask app init
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── Global Overrides ──────────────────────────────────────────────────
global_prompt    = None
global_knowledge = None

# ─── In‐Memory Stores ──────────────────────────────────────────────────
histories   = {}   # sid → list of {role,content}
last_active = {}   # sid → datetime

# ─── Helpers ───────────────────────────────────────────────────────────
def now_str():
    return datetime.now(ZoneInfo("Asia/Karachi")).isoformat(sep=" ")

def extract_text(f):
    return f.read().decode("utf-8", errors="ignore")

def load_prompt(sid=None):
    if global_prompt:
        return global_prompt
    if os.path.exists("system_prompt.txt"):
        return codecs.open("system_prompt.txt", "r", "utf8").read()
    return "Default system prompt."

def load_knowledge(sid=None):
    return global_knowledge or ""

# ─── Groq helper ───────────────────────────────────────────────────────
def groq_chat(messages):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }
    payload = {"model": HF_MODEL_ID, "messages": messages}
    logging.info(f"Groq API request → model: {HF_MODEL_ID}, messages: {messages}")
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    logging.info(f"Groq API response status: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ─── AI utilities ──────────────────────────────────────────────────────
# Summarization

def ai_summarize_openai(chat):
    system_msg = "Summarize this conversation in Urdu."
    user_msg   = "\n".join(f"{m['role']}: {m['content']}" for m in chat)
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",  "content": system_msg},
            {"role": "user",    "content": user_msg}
        ]
    )
    return resp.choices[0].message.content


def ai_summarize_groq(chat):
    system_msg = "Summarize this conversation in Urdu."
    user_msg   = "\n".join(f"{m['role']}: {m['content']}" for m in chat)
    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg}
    ]
    return groq_chat(msgs)


def ai_summarize(chat):
    if current_model == "huggingface":
        try:
            return ai_summarize_groq(chat)
        except Exception as e:
            logging.error(f"Groq summarize failed, falling back: {e}")
    return ai_summarize_openai(chat)

# Type detection

def ai_detect_types_openai(chat):
    system_msg = (
        "Analyze this conversation and return a JSON object with two arrays: "
        "'complaints' and 'info_requests'. Categorize each into: "
        "'money', 'district', 'eligibility', 'ingredients', or 'other'."
    )
    user_msg = "\n".join(f"{m['role']}: {m['content']}" for m in chat)
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}
        ]
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return {"complaints": ["other"], "info_requests": ["other"]}


def ai_detect_types_groq(chat):
    system_msg = (
        "Analyze this conversation and return a JSON object with two arrays: "
        "'complaints' and 'info_requests'. Categorize each into: "
        "'money', 'district', 'eligibility', 'ingredients', or 'other'."
    )
    user_msg = "\n".join(f"{m['role']}: {m['content']}" for m in chat)
    msgs = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg}
    ]
    text = groq_chat(msgs)
    try:
        return json.loads(text)
    except Exception as e:
        logging.error(f"Groq detect_types parse failed: {e}")
        return {"complaints": ["other"], "info_requests": ["other"]}


def ai_detect_types(chat):
    if current_model == "huggingface":
        try:
            return ai_detect_types_groq(chat)
        except Exception as e:
            logging.error(f"Groq detect failed, fallback: {e}")
    return ai_detect_types_openai(chat)

# ─── Admin endpoints ───────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload_override():
    global global_prompt, global_knowledge
    if "promptFile" in request.files:
        global_prompt = extract_text(request.files["promptFile"])
        logging.info("Global prompt overridden.")
    if "knowledgeFile" in request.files:
        global_knowledge = extract_text(request.files["knowledgeFile"])
        logging.info("Global knowledge overridden.")
    return jsonify({"status": "override_uploaded"})

@app.route("/set_model", methods=["POST"])
def set_model():
    global current_model
    data = request.get_json() or {}
    m = data.get("model")
    if m not in ("openai", "huggingface"):
        return jsonify(status="error", message="Invalid model"), 400
    current_model = m
    logging.info(f"Global model switched to: {current_model}")
    return jsonify(status="ok", model=current_model)

# ─── UI routes ─────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/client")
def client():
    return render_template("client.html")

# ─── Session start/end ─────────────────────────────────────────────────
@app.route("/start_toggle", methods=["POST"])
def start_toggle():
    data = request.get_json() or {}
    sid  = data.get("session_id")
    logging.info(f"Toggle session called with session_id={sid}")

    if not sid or sid not in histories:
        sid = str(uuid.uuid4())
        histories[sid] = []
        last_active[sid] = datetime.now()
        return jsonify({"started": True, "session_id": sid})

    chat     = histories.pop(sid, [])
    last_active.pop(sid, None)
    summary  = ai_summarize(chat)
    types    = ai_detect_types(chat)
    duration = str(len(chat))

    payload = {
        "tdatetime": now_str(),
        "sessionid": sid,
        "summary":   summary,
        "duration":  duration,
        "types":     json.dumps({
            "complaints":    types.get("complaints", []),
            "info_requests": types.get("info_requests", [])
        }, ensure_ascii=False),
        "chat":      json.dumps(chat, ensure_ascii=False)
    }
    logging.info(f"Posting transcript payload: {payload}")

    try:
        res = requests.post(
            "https://urdubot.nettechltd.com/api/transcripts",
            json=payload, timeout=10
        )
        res.raise_for_status()
        status = "uploaded"
        logging.info(f"Transcript uploaded: {res.status_code}")
    except Exception as e:
        logging.error(f"Transcripts POST failed: {e}")
        os.makedirs("Interactions", exist_ok=True)
        path = f"Interactions/session_{sid}_{now_str()}.json"
        with open(path, "w", encoding="utf8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        status = "saved_locally"

    return jsonify({"ended": True, "status": status})

# ─── Chat route ────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data   = request.get_json() or {}
    sid    = data.get("session_id")
    prompt = data.get("prompt", "")
    if sid not in histories:
        return jsonify(response="Please start a session first."), 400

    histories[sid].append({"role": "user", "content": prompt})

    if current_model == "huggingface":
        # Use Groq's endpoint
        system_p  = load_prompt(sid)
        knowledge = load_knowledge(sid)
        msgs = [{"role": "system", "content": system_p}]
        if knowledge:
            msgs.append({"role": "system", "content": knowledge})
        msgs.append({"role": "user",   "content": prompt})
        try:
            reply = groq_chat(msgs)
        except Exception as e:
            logging.error(f"Groq chat failed: {e}")
            reply = "[Error] Could not get response from Groq"
    else:
        # Use OpenAI
        system_p = load_prompt(sid)
        msgs     = [{"role": "system", "content": system_p}]
        msgs    += histories[sid][-20:]
        oa_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=msgs,
            temperature=0.4
        )
        reply = oa_resp.choices[0].message.content

    histories[sid].append({"role": "assistant", "content": reply})
    return jsonify(response=reply)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

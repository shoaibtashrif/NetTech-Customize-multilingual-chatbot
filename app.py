import os
import json
import uuid
import logging
import requests
import urllib3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import codecs
import openai

# ─── Setup ─────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── In-Memory Stores ─────────────────────────────────────────────────
session_files = {}    # sid → { 'prompt':…, 'knowledge':… }
histories    = {}     # sid → [ {role,content} ]
last_active  = {}     # sid → datetime

# ─── Helpers ─────────────────────────────────────────────────────────
def now_str():
    return datetime.now().strftime("%Y-%m-%d")

def extract_text(f):
    return f.read().decode("utf-8", errors="ignore")

def load_prompt(sid):
    return session_files.get(sid, {}).get("prompt") or (
        codecs.open("system_prompt.txt","r","utf8").read()
    )

def load_knowledge(sid):
    return session_files.get(sid, {}).get("knowledge","")

def ai_summarize(chat):
    resp = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role":"system","content":"Summarize this conversation in Urdu."},
        {"role":"user","content": "\n".join(f"{m['role']}: {m['content']}" for m in chat)}
      ]
    )
    return resp.choices[0].message.content

def ai_detect_types(chat):
    resp = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role":"system","content":(
          "Classify each complaint into any of these types: "
          "money complaints, ingredients related, eligibility, districts, other. "
          "Return a JSON array of all types mentioned."
        )},
        {"role":"user","content": "\n".join(f"{m['role']}: {m['content']}" for m in chat)}
      ]
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except:
        return [t.strip() for t in resp.choices[0].message.content.split(",")]

# ─── Routes ───────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start_toggle", methods=["POST"])
def start_toggle():
    data = request.get_json() or {}
    sid  = data.get("session_id")
    logging.info(f"TOGGLE called with session_id={sid}")
    # Start new session
    if not sid or sid not in histories:
        sid = str(uuid.uuid4())
        histories[sid] = []
        session_files[sid] = {}
        last_active[sid] = datetime.now()
        return jsonify({"started": True, "session_id": sid})
    # End session
    chat = histories.pop(sid, [])
    session_files.pop(sid, None)
    last_active.pop(sid, None)

    summary = ai_summarize(chat)
    types   = ai_detect_types(chat)
    duration = str(len(chat))

    payload = {
      "tdatetime": now_str(),
      "summary":   summary,
      "duration":  duration,
      "types":     {"types": types},
      "chat":      chat
    }
    logging.info(f"Posting transcript payload: {json.dumps(payload)}")
    try:
        res = requests.post(
          "https://urdubot.nettechltd.com/api/transcripts",
          json=payload, timeout=10
        )
        logging.info(f"Transcripts API responded: {res.status_code} {res.text}")
        res.raise_for_status()
        status = "uploaded"
    except Exception as e:
        logging.error(f"Transcripts POST failed: {e}")
        os.makedirs("Interactions", exist_ok=True)
        path = f"Interactions/session_{sid}_{now_str()}.json"
        with open(path, "w", encoding="utf8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        status = "saved_locally"

    return jsonify({"ended": True, "status": status})

@app.route("/upload", methods=["POST"])
def upload():
    sid = request.form.get("session_id")
    if sid not in session_files:
        return jsonify({"error":"Invalid session"}),400
    if "promptFile" in request.files:
        session_files[sid]["prompt"] = extract_text(request.files["promptFile"])
    if "knowledgeFile" in request.files:
        session_files[sid]["knowledge"] = extract_text(request.files["knowledgeFile"])
    return jsonify({"status":"files_uploaded"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    sid  = data.get("session_id")
    msg  = data.get("prompt")
    if sid not in histories:
        return jsonify({"response":"Please start a session first."}),400

    histories[sid].append({"role":"user","content":msg})
    last_active[sid] = datetime.now()

    sys  = load_prompt(sid)
    know = load_knowledge(sid)
    context = [{"role":"system","content":sys},
               {"role":"assistant","content":know}]
    context += histories[sid][-20:]

    resp = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=context,
      temperature=0.4
    )
    reply = resp.choices[0].message.content
    histories[sid].append({"role":"assistant","content":reply})

    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
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
openai.api_key = os.getenv("OPENAI_API_KEY1")
print(f"KEY PRINTED : {openai.api_key}")
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── Global Overrides ──────────────────────────────────────────────────
global_prompt   = None
global_knowledge = None

# ─── In-Memory Stores ─────────────────────────────────────────────────
histories   = {}   # sid → list of {role,content}
last_active = {}   # sid → datetime

# ─── Helpers ─────────────────────────────────────────────────────────
def now_str():
    return datetime.now().strftime("%Y-%m-%d")

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

def ai_summarize(chat):
    resp = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role":"system","content":"Summarize this conversation in Urdu."},
        {"role":"user","content":"\n".join(f"{m['role']}: {m['content']}" for m in chat)}
      ]
    )
    return resp.choices[0].message.content

def ai_detect_types(chat):
    resp = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role":"system","content":(
           "Analyze this conversation and return a JSON object with two arrays: "
           "'complaints' (for any complaints mentioned) and 'info_requests' (for information requests). "
           "Categorize each item into one of these categories: 'money', 'district', 'eligibility', 'ingredients', or 'other'. "
           "Example format: {'complaints': ['other', 'money'], 'info_requests': ['other', 'district']}"
        )},
        {"role":"user","content":"\n".join(f"{m['role']}: {m['content']}" for m in chat)}
      ]
    )
    try:
        result = json.loads(resp.choices[0].message.content)
        
        # Process complaints
        processed_complaints = []
        for item in result.get("complaints", []):
            lower_item = item.lower()
            if 'money' in lower_item or 'payment' in lower_item or 'fund' in lower_item:
                processed_complaints.append("money")
            elif 'district' in lower_item:
                processed_complaints.append("district")
            elif 'eligibility' in lower_item or 'eligible' in lower_item or 'qualification' in lower_item:
                processed_complaints.append("eligibility")
            elif 'ingredient' in lower_item or 'material' in lower_item:
                processed_complaints.append("ingredients")
            else:
                processed_complaints.append("other")
        
        # Process info_requests
        processed_info_requests = []
        for item in result.get("info_requests", []):
            lower_item = item.lower()
            if 'money' in lower_item or 'payment' in lower_item or 'fund' in lower_item:
                processed_info_requests.append("money")
            elif 'district' in lower_item:
                processed_info_requests.append("district")
            elif 'eligibility' in lower_item or 'eligible' in lower_item or 'qualification' in lower_item:
                processed_info_requests.append("eligibility")
            elif 'ingredient' in lower_item or 'material' in lower_item:
                processed_info_requests.append("ingredients")
            else:
                processed_info_requests.append("other")
        
        return {
            "complaints": processed_complaints,
            "info_requests": processed_info_requests
        }
    except:
        return {"complaints": ["other"], "info_requests": ["other"]}

# ─── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/client")
def client_chat():
    return render_template("client.html")

@app.route("/upload", methods=["POST"])
def upload_override():
    global global_prompt, global_knowledge
    if "promptFile" in request.files:
        global_prompt = extract_text(request.files["promptFile"])
        logging.info("Global prompt overridden.")
    if "knowledgeFile" in request.files:
        global_knowledge = extract_text(request.files["knowledgeFile"])
        logging.info("Global knowledge overridden.")
    return jsonify({"status":"override_uploaded"})

@app.route("/start_toggle", methods=["POST"])
def start_toggle():
    data = request.get_json() or {}
    sid  = data.get("session_id")
    logging.info(f"Toggle session called with session_id={sid}")

    if not sid or sid not in histories:
        sid = str(uuid.uuid4())
        histories[sid] = []
        last_active[sid] = datetime.now()
        return jsonify({"started":True, "session_id":sid})

    chat = histories.pop(sid, [])
    last_active.pop(sid, None)

    summary  = ai_summarize(chat)
    types    = ai_detect_types(chat)
    duration = str(len(chat))

    payload = {
      "tdatetime": now_str(),
      "session_id": sid,  # Added session_id to payload
      "summary": summary,
      "duration": duration,
      "types": json.dumps({
          "complaints": types.get("complaints", []),
          "info_requests": types.get("info_requests", [])
      }, ensure_ascii=False),
      "chat": json.dumps(chat, ensure_ascii=False)
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

    return jsonify({"ended":True, "status":status})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    sid  = data.get("session_id")
    msg  = data.get("prompt")
    if sid not in histories:
        return jsonify({"response":"Please start a session first."}), 400

    histories[sid].append({"role":"user","content":msg})
    last_active[sid] = datetime.now()

    system_p  = load_prompt(sid)
    knowledge = load_knowledge(sid)
    context = [
      {"role":"system","content":system_p},
      {"role":"assistant","content":knowledge}
    ] + histories[sid][-20:]

    resp = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=context,
      temperature=0.4
    )
    reply = resp.choices[0].message.content
    histories[sid].append({"role":"assistant","content":reply})
    return jsonify({"response":reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
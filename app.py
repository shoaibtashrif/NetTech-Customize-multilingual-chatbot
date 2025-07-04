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

# HF creds & model
HF_API_KEY   = os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL_ID  = os.getenv("HUGGINGFACE_MODEL_ID", "gpt2")  # default to gpt2

# Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Admin‐controlled toggle
current_model = "openai"

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── Global Overrides ──────────────────────────────────────────────────
global_prompt    = None
global_knowledge = None

# ─── In‐Memory Stores ──────────────────────────────────────────────────
histories   = {}   # sid → list of {role,content}
last_active = {}   # sid → datetime
session_types = {} # sid → 'complaint' or 'info'

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
    if global_knowledge:
        return global_knowledge
    if os.path.exists("custom_knowledge.txt"):
        try:
            return codecs.open("custom_knowledge.txt", "r", "utf8").read()
        except Exception as e:
            logging.error(f"Error loading knowledge base: {str(e)}")
            return ""
    return "KNowledge base not found. pls upload knowledge base."

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
           "Analyze this conversation and return a JSON object with: "
           "'complaints' array (categorized as 'money', 'district', 'eligibility', 'ingredients', or 'other'). "
           "If there are any complaints, ignore all information requests and just return the complaint types. "
           "Only if there are no complaints, return information requests as 'other'."
        )},
        {"role":"user","content":"\n".join(f"{m['role']}: {m['content']}" for m in chat)}
      ],
      temperature=0.3
    )
    try:
        result = json.loads(resp.choices[0].message.content)
        valid_types = ['money', 'district', 'eligibility', 'ingredients', 'other']
        # Prioritize complaints - return them if they exist
        if 'complaints' in result and result['complaints']:
            types = [c if c in valid_types else 'other' for c in result['complaints']]
            if not types:
                types = ['other']
            return {'types': types}
        # Fallback to info requests if no complaints
        return {'types': ['other']}
    except:
        return {'types': ['other']}  # Safe default

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
    return jsonify({"status":"override_uploaded"})

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
        return jsonify({"started":True, "session_id":sid})

    chat     = histories.pop(sid, [])
    last_active.pop(sid, None)
    summary  = ai_summarize(chat)
    # Determine session type
    stype = session_types.get(sid, "info")
    if stype == "complaint":
        types = ai_detect_types(chat)
        complaints = types.get("types", ["other"])
        if not complaints:
            complaints = ["other"]
        info_requests = []
    else:
        complaints = []
        info_requests = ["info"]
    duration = str(len(chat))

    payload = {
      "tdatetime": now_str(),
      "sessionid": sid,
      "summary":   summary,
      "duration":  duration,
      "types":     json.dumps({
                       "complaints":    complaints,
                       "info_requests": info_requests
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

    return jsonify({"ended":True, "status":status})

# ─── Chat route ────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    data   = request.get_json() or {}
    sid    = data.get("session_id")
    prompt = data.get("prompt", "")
    session_type = data.get("session_type")
    if sid not in histories:
        return jsonify(response="Please start a session first."), 400

    # Store session type if provided and not already set
    if session_type and sid not in session_types:
        session_types[sid] = session_type

    # Default to info if not set (for backward compatibility)
    stype = session_types.get(sid, "info")

    histories[sid].append({"role":"user","content":prompt})

    # Always load the full knowledge base
    try:
        with open("custom_knowledge.txt", "r", encoding="utf8") as f:
            full_knowledge = f.read()
    except Exception as e:
        full_knowledge = ""

    if stype == "complaint":
        # Extract complaint type
        complaint_types = ai_detect_types(histories[sid]).get("types", ["other"])
        if not complaint_types:
            complaint_types = ["other"]
        # POST complaint type to backend endpoint
        payload = {
            "session_id": sid,
            "complaint_types": complaint_types,
            "message": prompt
        }
        try:
            res = requests.post(
                "https://urdubot.nettechltd.com/api/complaints",
                json=payload, timeout=10
            )
            res.raise_for_status()
        except Exception as e:
            logging.error(f"Complaint POST failed: {e}")
        # Do not reply to complaint
        return jsonify(response=None)

    # Info session: proceed as normal
    if current_model == "huggingface":
        groq_url     = "https://api.groq.com/openai/v1/chat/completions"
        headers      = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        system_p  = load_prompt(sid)
        knowledge = full_knowledge
        msgs = [{"role": "system", "content": system_p}]
        if knowledge:
            msgs.append({"role": "system", "content": knowledge})
        msgs.append({"role": "user",   "content": prompt})
        payload = {
            "model": "gemma2-9b-it",
            "messages": msgs
        }
        logging.info(f"Groq request → URL: {groq_url}, model: {HF_MODEL_ID}")
        groq_resp = requests.post(groq_url, headers=headers, json=payload, timeout=30)
        logging.info(f"Groq response status: {groq_resp.status_code}")
        try:
            groq_resp.raise_for_status()
            data  = groq_resp.json()
            reply = data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError:
            code  = groq_resp.status_code
            reply = f"[Groq error {code}] {groq_resp.text}"
    else:
        system_p = load_prompt()
        # Always include full knowledge base as a system message
        context  = [
            {"role":"system","content":system_p},
            {"role":"system","content":full_knowledge}
        ] + histories[sid][-20:]
        oa_resp  = openai.ChatCompletion.create(
                     model="gpt-3.5-turbo",
                     messages=context,
                     temperature=0.4
                   )
        reply    = oa_resp.choices[0].message.content
    histories[sid].append({"role":"assistant","content":reply})

    # 2. Get relevant knowledge base excerpt
    kb_excerpt = ""
    if full_knowledge.strip():
        try:
            kb_prompt = (
                "Given the following knowledge base and user question, extract the most relevant section or paragraph from the knowledge base that answers or relates to the user's question. "
                "Return only the relevant excerpt, not the whole knowledge base."
            )
            kb_resp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": kb_prompt},
                    {"role": "user", "content": f"Knowledge Base:\n{full_knowledge}\nUser Question: {prompt}"}
                ],
                temperature=0.2
            )
            kb_excerpt = kb_resp.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Knowledge base excerpt error: {e}")
            kb_excerpt = ""

    return jsonify(response=reply, kb_excerpt=kb_excerpt)

@app.route("/suggestions", methods=["POST"])
def suggestions():
    data = request.get_json() or {}
    user_input = data.get("input", "")
    # Load knowledge base
    try:
        with open("custom_knowledge.txt", "r", encoding="utf8") as f:
            knowledge = f.read()
    except Exception as e:
        knowledge = ""
    # Use OpenAI to generate 2-3 related questions based on the input and knowledge
    if not user_input.strip():
        system_prompt = (
            "Given the following knowledge base, suggest 2-3 common questions or issues a user might ask. "
            "Return ONLY a plain English list of questions, no code, no JSON, no brackets."
        )
        user_content = knowledge
    else:
        system_prompt = (
            "Given the following knowledge base and user input, suggest 2-3 related, common questions or issues that might be relevant. "
            "Return ONLY a plain English list of questions, no code, no JSON, no brackets."
        )
        user_content = f"Knowledge Base:\n{knowledge}\nUser Input: {user_input}"
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.5
        )
        # Parse as plain text lines
        text = resp.choices[0].message.content.strip()
        suggestions = [line.strip('-* 1234567890.') for line in text.split('\n') if line.strip()][:3]
        if not suggestions:
            suggestions = [
                "What is the purpose of the BISP Nashonuma program?",
                "Who is eligible for the program?",
                "How can I receive cash transfers?"
            ]
        return jsonify({"suggestions": suggestions[:3]})
    except Exception as e:
        logging.error(f"Suggestion generation failed: {e}")
        # Always return some default suggestions in English
        return jsonify({"suggestions": [
            "What is the purpose of the BISP Nashonuma program?",
            "Who is eligible for the program?",
            "How can I receive cash transfers?"
        ]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

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
import re
import pprint

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
session_type_history = {} # sid → set of types used in session

RECENT_LOGS = []
MAX_LOGS = 100

# ─── Program Q&A Knowledge Base ───────────────────────────────────────
PROGRAM_QA = {
    # Eligibility questions
    "age requirement": {
        "answer": "Age for this program is 13-19 years girls ",
        "variations": ["what is the age limit", "how old should the girl be", "age criteria"]
    },
    "registration criteria": {
        "answer": "Requirements: Mother CNIC , girl's B-form , and age 13-19 years ",
        "variations": ["what documents needed", "registration requirements", "what is needed to enroll"]
    },
    
    # Process questions
    "first visit process": {
        "answer": "First visit steps: 1) Register , 2) Health check , 3) Awareness session , 4) Receive IFA ",
        "variations": ["what happens in first visit", "initial visit process", "first time procedure"]
    },
    "cash transfer amount": {
        "answer": "Cash transfer: Rs. 2000/- per quarter  when compliance met",
        "variations": ["how much money", "payment amount", "cash benefit"]
    },
    
    # Special cases
    "disabled girl registration": {
        "answer": "Disabled girls: Can register  if they have B-form and visit FC with mother",
        "variations": ["what if disabled", "handicapped girl", "special needs registration"]
    },
    "multiple daughters": {
        "answer": "Multiple daughters: All eligible  if they have B-forms (no limit per family)",
        "variations": ["two daughters", "sisters enrollment", "more than one girl"]
    },
    "mother not present": {
        "answer": "Mother must be present  - required for cash transfer and compliance",
        "variations": ["father brings girl", "mother not available", "can father come instead"]
    },
    "district requirement": {
        "answer": "Only these districts: Neelam, Lasbela, Hub, Qambar Shahdadkot, Swat, Rajanpur, Jampur, Ghizer ",
        "variations": ["which areas eligible", "my district included", "location requirements"]
    },
    "ifa supplementation": {
        "answer": "IFA protocol: 60mg iron + 2800μg folic acid , once weekly  for 3 months",
        "variations": ["what tablets given", "supplement details", "medicine protocol"]
    },
    "missed dose": {
        "answer": "Missed dose: Take next day if remembered , otherwise wait for next week's dose",
        "variations": ["forgot tablet", "missed medicine", "what if not taken"]
    },
    "side effects": {
        "answer": "Side effects: Consult doctor  if severe issues, take with water after meals",
        "variations": ["nausea from tablets", "constipation issues", "medicine problems"]
    }
}

# ─── Helpers ───────────────────────────────────────────────────────────
def now_str():
    return datetime.now(ZoneInfo("Asia/Karachi")).strftime("%Y-%m-%d %H:%M:%S")

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
    return "Knowledge base not found. Please upload knowledge base."

def ai_summarize(chat):
    resp = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {"role":"system","content":"Summarize this conversation in English."},
        {"role":"user","content":"\n".join(f"{m['role']}: {m['content']}" for m in chat)}
      ]
    )
    return resp.choices[0].message.content

def log_to_ui(msg, level='info'):
    global RECENT_LOGS
    RECENT_LOGS.append(msg)
    if len(RECENT_LOGS) > MAX_LOGS:
        RECENT_LOGS = RECENT_LOGS[-MAX_LOGS:]
    if level == 'info':
        logging.info(msg)
    elif level == 'warning':
        logging.warning(msg)
    elif level == 'error':
        logging.error(msg)
    else:
        logging.info(msg)

def ai_detect_types(chat, session_types_map=None):
    system_prompt = (
        "Analyze the following conversation. Each user message is labeled as either (complaint) or (info). "
        "For each, categorize it as one or more of: 'money', 'eligibility', 'district', 'ingredients', or 'other'. "
        "If a message is about money (e.g., uses words like 'pasy', 'paise', 'money', 'amount', 'rupees'), always include 'money' as the first category. "
        "Return a JSON object with two arrays: 'complaints' and 'info', each listing the categories (no duplicates)."
    )
    lines = []
    if session_types_map is None:
        session_types_map = {}
    complaint_indices = []
    for idx, m in enumerate(chat):
        role = m.get('role')
        content = m.get('content', '')
        if role == 'user':
            stype = session_types_map.get(idx, 'info')
            lines.append(f"user ({stype}): {content}")
            if stype == 'complaint':
                complaint_indices.append(idx)
        else:
            lines.append(f"{role}: {content}")
    
    resp = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {"role":"system","content": system_prompt},
        {"role":"user","content":"\n".join(lines)}
      ],
      temperature=0.3
    )
    try:
        result = json.loads(resp.choices[0].message.content)
        allowed = ['money', 'eligibility', 'district', 'ingredients', 'other']
        def dedup_and_order(lst):
            seen = set()
            ordered = []
            if 'money' in lst:
                ordered.append('money')
                seen.add('money')
            for i in lst:
                if i not in seen and i in allowed:
                    ordered.append(i)
                    seen.add(i)
            return ordered
        complaints = dedup_and_order(result.get('complaints', []))
        info = dedup_and_order(result.get('info', []))
        if not complaints and complaint_indices:
            money_keywords = ['pasy', 'paise', 'money', 'amount', 'rupees']
            for idx in complaint_indices:
                content = chat[idx].get('content', '').lower()
                if any(word in content for word in money_keywords):
                    complaints = ['money']
                    break
            if not complaints:
                complaints = ['other']
        return {'complaints': complaints, 'info': info}
    except Exception as e:
        if chat and any(m.get('role') == 'user' and m.get('content','').strip() for m in chat):
            return {'complaints': [], 'info': ['other']}
        return {'complaints': [], 'info': []}

def find_matching_question(user_input):
    user_input = user_input.lower().strip()
    
    # First check direct matches
    for q, data in PROGRAM_QA.items():
        if q in user_input:
            return data["answer"]
        # Check variations
        for variation in data.get("variations", []):
            if variation in user_input:
                return data["answer"]
    
    # Then check if any question is contained in the input
    for q, data in PROGRAM_QA.items():
        if any(word in user_input for word in q.split()):
            return data["answer"]
    
    return None

# ─── Admin endpoints ───────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload_override():
    global global_prompt, global_knowledge
    if "promptFile" in request.files:
        file = request.files["promptFile"]
        text = extract_text(file)
        global_prompt = text
        try:
            with open("system_prompt.txt", "w", encoding="utf8") as f:
                f.write(text)
            logging.info("Global prompt overridden and saved to system_prompt.txt.")
        except Exception as e:
            logging.error(f"Failed to save system_prompt.txt: {e}")
    if "knowledgeFile" in request.files:
        file = request.files["knowledgeFile"]
        text = extract_text(file)
        global_knowledge = text
        try:
            with open("custom_knowledge.txt", "w", encoding="utf8") as f:
                f.write(text)
            logging.info("Global knowledge overridden and saved to custom_knowledge.txt.")
        except Exception as e:
            logging.error(f"Failed to save custom_knowledge.txt: {e}")
    return jsonify({"status":"override_uploaded"})

@app.route("/set_model", methods=["POST"])
def set_model():
    global current_model
    data = request.get_json() or {}
    m = data.get("model")
    if m not in ("openai", "huggingface", "groq"):
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
        session_type_history[sid] = set()
        return jsonify({"started":True, "session_id":sid})

    chat     = histories.pop(sid, [])
    last_active.pop(sid, None)
    summary  = ai_summarize(chat)
    session_types_map = {}
    for idx, m in enumerate(chat):
        if m.get('role') == 'user':
            session_types_map[idx] = m.get('session_type', 'info')
    detected_types = ai_detect_types(chat, session_types_map)
    complaints = detected_types.get("complaints", [])
    info = detected_types.get("info", [])
    if not complaints and not info and chat:
        if any(m.get('role') == 'user' and m.get('content','').strip() for m in chat):
            info = ['other']
    allowed = ['money', 'eligibility', 'district', 'ingredients', 'other']
    complaints = [c if c in allowed else 'other' for c in complaints]
    info = [i if i in allowed else 'other' for i in info]
    if 'money' in complaints:
        complaints = ['money'] + [c for c in complaints if c != 'money']
    if 'money' in info:
        info = ['money'] + [i for i in info if i != 'money']
    duration = str(len(chat))

    payload = {
      "tdatetime": now_str(),
      "sessionid": sid,
      "summary":   summary,
      "duration":  duration,
      "types":     json.dumps({
                       "complaints":    complaints,
                       "info": info
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

    session_type_history.pop(sid, None)
    session_types.pop(sid, None)
    return jsonify({"ended":True, "status":status})

@app.route("/chat", methods=["POST"])
def chat():
    data   = request.get_json() or {}
    sid    = data.get("session_id")
    prompt = data.get("prompt", "")
    session_type = data.get("session_type")
    if sid not in histories:
        return jsonify(response="Please start a session first."), 400

    # Check for matching question first
    matched_answer = find_matching_question(prompt)
    if matched_answer:
        histories[sid].append({"role":"assistant","content":matched_answer})
        return jsonify(response=matched_answer, kb_excerpt="")

    # Store session type if provided
    if session_type:
        session_types[sid] = session_type
        if sid not in session_type_history:
            session_type_history[sid] = set()
        session_type_history[sid].add(session_type)

    stype = session_types.get(sid, "info")

    # Store the session_type with each user message
    histories[sid].append({"role":"user","content":prompt, "session_type": session_type or stype})

    # Load knowledge base
    try:
        with open("custom_knowledge.txt", "r", encoding="utf8") as f:
            full_knowledge = f.read()
    except Exception as e:
        full_knowledge = ""

    # Build session_types_map for ai_detect_types
    session_types_map = {}
    for idx, m in enumerate(histories[sid]):
        if m.get('role') == 'user':
            session_types_map[idx] = m.get('session_type', 'info')
    
    detected_types = ai_detect_types(histories[sid], session_types_map)
    complaints = detected_types.get("complaints", [])
    info = detected_types.get("info", [])
    print(f"detected types: {detected_types}")
    
    if not complaints and not info and prompt.strip():
        info = ['other']
    
    allowed = ['money', 'eligibility', 'district', 'ingredients', 'other']
    complaints = [c if c in allowed else 'other' for c in complaints]
    info = [i if i in allowed else 'other' for i in info]
    
    if 'money' in complaints:
        complaints = ['money'] + [c for c in complaints if c != 'money']
    if 'money' in info:
        info = ['money'] + [i for i in info if i != 'money']

    if stype == "complaint":
        payload = {
            "session_id": sid,
            "complaint_types": complaints,
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
        return jsonify(response=None)

    # System prompt for language and direct answers
    system_p = (
        "You are an assistant for the BISP Nashonuma program. Always answer user questions concisely using the provided knowledge base. "
        "Detect the language of the user's input (English, Urdu, or Roman Urdu) and reply in the same language. "
        "Keep answers brief and highlight key information with ** on both sides."
    )
    
    if current_model == "huggingface":
        # ... (keep existing HF implementation)
        pass
    elif current_model == "groq":
        # ... (keep existing Groq implementation)
        pass
    else:
        context  = [
            {"role":"system","content":system_p},
            {"role":"system","content":full_knowledge}
        ] + histories[sid][-20:]
        oa_resp  = openai.ChatCompletion.create(
                     model="gpt-4",
                     messages=context,
                     temperature=0.4
                   )
        reply    = oa_resp.choices[0].message.content
        # Convert markdown highlights to our hilight tags
        reply = reply.replace("**", "", 1).replace("**", " ", 1)
    
    histories[sid].append({"role":"assistant","content":reply})

    # Get relevant knowledge base excerpt
    kb_excerpt = ""
    if full_knowledge.strip():
        try:
            kb_prompt = "Extract the most relevant section from the knowledge base that answers the user's question."
            kb_resp = openai.ChatCompletion.create(
                model="gpt-4",
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
    user_input = data.get("input", "").lower()
    
    # Get suggestions from predefined questions first
    suggestions = []
    for q in PROGRAM_QA:
        if not user_input or any(word in user_input for word in q.split()):
            suggestions.append(q.capitalize() + "?")
        if len(suggestions) >= 3:
            break
    
    # If we need more suggestions, use AI
    if len(suggestions) < 3:
        try:
            with open("custom_knowledge.txt", "r", encoding="utf8") as f:
                knowledge = f.read()
            
            system_prompt = (
                "Given the following knowledge base and user input, suggest 2-3 related questions. "
                "Return ONLY a plain English list of questions, no code, no JSON."
            )
            user_content = f"Knowledge Base:\n{knowledge}\nUser Input: {user_input}"
            
            resp = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.5
            )
            text = resp.choices[0].message.content.strip()
            ai_suggestions = [line.strip('-* 1234567890.') for line in text.split('\n') if line.strip()][:3]
            suggestions.extend(ai_suggestions[:3-len(suggestions)])
        except Exception as e:
            logging.error(f"Suggestion generation failed: {e}")
    
    # Ensure we always return 3 suggestions
    default_suggestions = [
        "What is the age requirement?",
        "How much is the cash transfer?",
        "What documents are needed?"
    ]
    while len(suggestions) < 3:
        suggestions.append(default_suggestions[len(suggestions)])
    
    return jsonify({"suggestions": suggestions[:3]})

@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify({'logs': RECENT_LOGS[-MAX_LOGS:]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
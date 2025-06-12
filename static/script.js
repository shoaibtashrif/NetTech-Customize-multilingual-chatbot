// static/script.js
console.log("script.js loaded");

const ENDPOINT = {
  TOGGLE: "/start_toggle",
  UPLOAD: "/upload",
  CHAT:   "/chat",
  SETMDL: "/set_model"
};

let sessionId = null;
const btn     = document.getElementById("sessionBtn");
const sendBtn = document.getElementById("sendBtn");

function addMessage(sender, content) {
  const chat = document.getElementById("chat-box");
  const msg  = document.createElement("div");
  msg.className = `message ${sender}`;
  msg.textContent = content;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

btn.addEventListener("click", async () => {
  btn.disabled     = true;
  sendBtn.disabled = true;
  btn.textContent  = sessionId ? "Ending session..." : "Starting session...";

  const resp = await fetch(ENDPOINT.TOGGLE, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId })
  });
  const data = await resp.json();

  if (data.started) {
    sessionId = data.session_id;
    addMessage("system", `Session started: ${sessionId}`);
    btn.textContent = "End Session";
    sendBtn.disabled= false;
  } else if (data.ended) {
    addMessage("system", `Session ended (${data.status})`);
    document.getElementById("chat-box").innerHTML = "";
    sessionId = null;
    btn.textContent = "Start Session";
  }

  btn.disabled = false;
});

async function uploadFiles() {
  const fd = new FormData();
  const pf = document.getElementById("promptFile").files[0];
  const kf = document.getElementById("knowledgeFile").files[0];
  if (pf) fd.append("promptFile", pf);
  if (kf) fd.append("knowledgeFile", kf);
  const resp = await fetch(ENDPOINT.UPLOAD, { method: "POST", body: fd });
  const data = await resp.json();
  alert(data.status);
}

async function applyModel() {
  const m = document.getElementById("model-picker").value;
  const resp = await fetch(ENDPOINT.SETMDL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: m })
  });
  const data = await resp.json();
  if (data.status === "ok") {
    alert("Model switched to: " + data.model);
  } else {
    alert("Error: " + data.message);
  }
}

async function sendMessage() {
  const input = document.getElementById("user-input");
  const text  = input.value.trim();
  if (!text) return;
  if (!sessionId) {
    addMessage("bot","Please start a session first.");
    return;
  }
  addMessage("user", text);
  input.value = "";

  const resp = await fetch(ENDPOINT.CHAT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: text, session_id: sessionId })
  });
  const data = await resp.json();
  addMessage("bot", data.response || "No response.");
}

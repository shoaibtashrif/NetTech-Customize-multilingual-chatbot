console.log("script.js loaded");

const ENDPOINT = {
  START:       "/start_session",
  UPLOAD:      "/upload",
  CHAT:        "/chatbot",
  END_SESSION: "/end_session"
};

let sessionId = null;

function addMessage(sender, content) {
  const chat = document.getElementById("chat-box");
  const msg  = document.createElement("div");
  msg.className = `message ${sender}`;
  msg.textContent = content;
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

// — Start Session —
async function startSession() {
  console.log("▶️ POST", ENDPOINT.START);
  try {
    const resp = await fetch(ENDPOINT.START, { method: "POST" });
    console.log("Raw response:", resp);
    const data = await resp.json();
    console.log("Parsed JSON:", data);

    if (!data.session_id) {
      console.error("❌ No session_id in response");
      alert("Failed to start session");
      return;
    }
    sessionId = data.session_id;
    document.getElementById("chat-box").innerHTML = "";
    addMessage("system", `Session started: ${sessionId}`);
    document.getElementById("sendBtn").disabled = false;
    console.log(`✅ Session initialized: ${sessionId}`);
  } catch (e) {
    console.error("❌ startSession error:", e);
    alert("Error starting session. Check console.");
  }
}

// — Upload Files —
async function uploadFiles() {
  if (!sessionId) return alert("Start a session first.");
  const fd = new FormData();
  fd.append("session_id", sessionId);
  const pf = document.getElementById("promptFile").files[0];
  const kf = document.getElementById("knowledgeFile").files[0];
  if (pf) fd.append("promptFile", pf);
  if (kf) fd.append("knowledgeFile", kf);

  console.log(`▶️ POST ${ENDPOINT.UPLOAD} with files`);
  try {
    const resp = await fetch(ENDPOINT.UPLOAD, { method: "POST", body: fd });
    const data = await resp.json();
    console.log("Upload response:", data);
    alert(data.message);
  } catch (e) {
    console.error("❌ uploadFiles error:", e);
    alert("File upload failed.");
  }
}

// — Send a Message —
async function sendMessage() {
  const input = document.getElementById("user-input");
  const text  = input.value.trim();
  if (!text) return;
  if (!sessionId) await startSession();  // auto-start

  addMessage("user", text);
  input.value = "";

  console.log(`▶️ POST ${ENDPOINT.CHAT} →`, text);
  try {
    const resp = await fetch(ENDPOINT.CHAT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text, session_id: sessionId })
    });
    const data = await resp.json();
    console.log("Chat response:", data);
    addMessage("bot", data.response || "No response.");
  } catch (e) {
    console.error("❌ sendMessage error:", e);
    addMessage("bot", "Error processing your request.");
  }
}

// — End Session —
async function endSession() {
  if (!sessionId) return alert("No session to end.");
  console.log("▶️ POST", ENDPOINT.END_SESSION, sessionId);
  try {
    const resp = await fetch(ENDPOINT.END_SESSION, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    });
    const data = await resp.json();
    console.log("End session response:", data);

    if (data.status === "success") {
      addMessage("system", `Session ended: ${sessionId}`);
      console.log(`✅ Session ended: ${sessionId}`);
      sessionId = null;
      document.getElementById("sendBtn").disabled = true;
    } else {
      alert("End session error: " + data.message);
    }
  } catch (e) {
    console.error("❌ endSession error:", e);
    alert("Failed to end session.");
  }
}

// static/script.js
console.log("script.js loaded");

const ENDPOINT = {
  TOGGLE: "/start_toggle",
  UPLOAD: "/upload",
  CHAT:   "/chat",
  SETMDL: "/set_model"
};

let sessionId = null;
let sessionType = null; // 'complaint' or 'info'
let hasAskedSessionType = false;
let suggestionDebounce = null;
const btn     = document.getElementById("sessionBtn");
const sendBtn = document.getElementById("sendBtn");

function addMessage(sender, content) {
  const chat = document.getElementById("chat-box");
  const msg  = document.createElement("div");
  msg.className = sender === "user" ? "message user-message" : sender === "bot" ? "message bot-message" : "message system-message";
  msg.textContent = content;
  // Side-by-side alignment
  msg.style.display = 'inline-block';
  msg.style.maxWidth = '70%';
  msg.style.margin = '8px';
  msg.style.padding = '12px 18px';
  msg.style.borderRadius = '20px';
  msg.style.fontSize = '15px';
  msg.style.lineHeight = '1.6';
  msg.style.wordBreak = 'break-word';
  if (sender === "user") {
    msg.style.background = 'linear-gradient(90deg, #007bff 60%, #00c6ff 100%)';
    msg.style.color = '#fff';
    msg.style.alignSelf = 'flex-end';
    msg.style.textAlign = 'right';
    msg.style.boxShadow = '0 2px 8px rgba(0,0,0,0.04)';
  } else if (sender === "bot") {
    msg.style.background = '#e9ecef';
    msg.style.color = '#333';
    msg.style.alignSelf = 'flex-start';
    msg.style.textAlign = 'left';
    msg.style.boxShadow = '0 2px 8px rgba(0,0,0,0.04)';
  } else {
    msg.style.background = '#f7fafd';
    msg.style.color = '#888';
    msg.style.alignSelf = 'center';
    msg.style.textAlign = 'center';
    msg.style.fontStyle = 'italic';
  }
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
    // Reset session type for new session
    sessionType = null;
    hasAskedSessionType = false;
    addMessage("system", `Session started: ${sessionId}`);
    btn.textContent = "End Session";
    sendBtn.disabled= false;
  } else if (data.ended) {
    addMessage("system", `Session ended (${data.status})`);
    document.getElementById("chat-box").innerHTML = "";
    sessionId = null;
    // Reset session type when session ends
    sessionType = null;
    hasAskedSessionType = false;
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

function showSessionTypeModal(callback) {
  // Create modal
  const modal = document.createElement('div');
  modal.id = 'session-type-modal';
  modal.style.position = 'fixed';
  modal.style.top = '0';
  modal.style.left = '0';
  modal.style.width = '100vw';
  modal.style.height = '100vh';
  modal.style.background = 'rgba(0,0,0,0.4)';
  modal.style.display = 'flex';
  modal.style.justifyContent = 'center';
  modal.style.alignItems = 'center';
  modal.style.zIndex = '9999';

  const box = document.createElement('div');
  box.style.background = '#fff';
  box.style.padding = '24px 24px 20px 24px';
  box.style.borderRadius = '16px';
  box.style.boxShadow = '0 4px 32px rgba(0,0,0,0.18)';
  box.style.textAlign = 'center';
  box.style.minWidth = '220px';
  box.style.minHeight = '80px';
  box.style.display = 'flex';
  box.style.flexDirection = 'column';
  box.style.alignItems = 'center';
  box.style.justifyContent = 'center';

  const label = document.createElement('div');
  label.textContent = 'Is this a Complaint or Info?';
  label.style.fontSize = '17px';
  label.style.marginBottom = '18px';
  label.style.fontWeight = '600';
  box.appendChild(label);

  const btnComplaint = document.createElement('button');
  btnComplaint.textContent = 'Complaint';
  btnComplaint.className = 'type-btn complaint';
  btnComplaint.onclick = () => {
    sessionType = 'complaint';
    hasAskedSessionType = true;
    document.body.removeChild(modal);
    renderSwitchTypeBtn();
    if (callback) callback('complaint');
  };
  const btnInfo = document.createElement('button');
  btnInfo.textContent = 'Info';
  btnInfo.className = 'type-btn info';
  btnInfo.onclick = () => {
    sessionType = 'info';
    hasAskedSessionType = true;
    document.body.removeChild(modal);
    renderSwitchTypeBtn();
    if (callback) callback('info');
  };
  box.appendChild(btnComplaint);
  box.appendChild(btnInfo);

  // Add a close button
  const closeBtn = document.createElement('span');
  closeBtn.textContent = '×';
  closeBtn.style.position = 'absolute';
  closeBtn.style.top = '10px';
  closeBtn.style.right = '18px';
  closeBtn.style.fontSize = '22px';
  closeBtn.style.cursor = 'pointer';
  closeBtn.onclick = () => {
    document.body.removeChild(modal);
  };
  box.appendChild(closeBtn);

  modal.appendChild(box);
  document.body.appendChild(modal);
}

function renderSwitchTypeBtn() {
  let btn = document.getElementById('switchTypeBtn');
  const header = document.querySelector('.header-container');
  if (!btn) {
    btn = document.createElement('button');
    btn.id = 'switchTypeBtn';
    header.appendChild(btn);
  }
  if (sessionType === 'info') {
    btn.textContent = 'Switch to Complaint';
    btn.className = 'complaint';
    btn.onclick = () => {
      sessionType = 'complaint';
      renderSwitchTypeBtn();
    };
  } else {
    btn.textContent = 'Switch to Info';
    btn.className = 'info';
    btn.onclick = () => {
      sessionType = 'info';
      renderSwitchTypeBtn();
    };
  }
}

function ensureSessionType(cb) {
  if (sessionType) return cb(sessionType);
  if (!hasAskedSessionType) {
    showSessionTypeModal(cb);
  } else {
    // fallback
    cb('info');
  }
}

function addSessionTypeSwitch() {
  let switchDiv = document.getElementById('session-type-switch');
  if (!switchDiv) {
    switchDiv = document.createElement('div');
    switchDiv.id = 'session-type-switch';
    document.querySelector('.chat-container').insertBefore(switchDiv, document.querySelector('.chat-container').firstChild.nextSibling);
  }
  switchDiv.innerHTML = '';
  const label = document.createElement('span');
  label.textContent = 'Session type:';
  label.style.fontSize = '13px';
  label.style.marginRight = '6px';
  switchDiv.appendChild(label);
  ['info', 'complaint'].forEach(type => {
    const btn = document.createElement('button');
    btn.textContent = type.charAt(0).toUpperCase() + type.slice(1);
    btn.className = sessionType === type ? 'selected' : '';
    btn.style.padding = '4px 14px';
    btn.style.borderRadius = '14px';
    btn.style.border = sessionType === type ? '2px solid #007bff' : '1.5px solid #bbb';
    btn.style.background = sessionType === type ? '#e3f0ff' : '#f5f7fa';
    btn.style.color = sessionType === type ? '#007bff' : '#333';
    btn.style.fontSize = '13px';
    btn.style.cursor = 'pointer';
    btn.style.marginRight = '6px';
    btn.onclick = () => {
      sessionType = type;
      hasAskedSessionType = true;
      addSessionTypeSwitch();
    };
    switchDiv.appendChild(btn);
  });
  switchDiv.style.borderBottom = '1px solid #e0eafc';
  switchDiv.style.paddingBottom = '6px';
  switchDiv.style.marginBottom = '8px';
}

function renderSuggestions(suggestions) {
  let sugDiv = document.getElementById('suggestions');
  if (!sugDiv) {
    sugDiv = document.createElement('div');
    sugDiv.id = 'suggestions';
    // Insert suggestions just after the last user message
    const chatBox = document.getElementById('chat-box');
    let lastUserMsg = null;
    for (let i = chatBox.children.length - 1; i >= 0; i--) {
      if (chatBox.children[i].classList.contains('user-message')) {
        lastUserMsg = chatBox.children[i];
        break;
      }
    }
    if (lastUserMsg && lastUserMsg.nextSibling) {
      chatBox.insertBefore(sugDiv, lastUserMsg.nextSibling);
    } else if (lastUserMsg) {
      chatBox.appendChild(sugDiv);
    } else {
      chatBox.appendChild(sugDiv);
    }
  }
  sugDiv.innerHTML = '';
  if (!suggestions || suggestions.length === 0) {
    const msg = document.createElement('div');
    msg.textContent = 'No quick questions available.';
    msg.style.color = '#888';
    msg.style.fontSize = '14px';
    msg.style.margin = '6px 0 0 8px';
    sugDiv.appendChild(msg);
    return;
  }
  (suggestions || []).slice(0,3).forEach(s => {
    let clean = s.replace(/[`\[\]{}"']/g, '').trim();
    const btn = document.createElement('button');
    btn.textContent = clean;
    btn.onclick = () => {
      document.getElementById('user-input').value = clean;
      document.getElementById('user-input').focus();
    };
    sugDiv.appendChild(btn);
  });
}

async function fetchSuggestions(input) {
  const resp = await fetch('/suggestions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input })
  });
  const data = await resp.json();
  renderSuggestions(data.suggestions);
}

function setupSuggestionInputListener() {
  const input = document.getElementById('user-input');
  input.addEventListener('input', function() {
    if (suggestionDebounce) clearTimeout(suggestionDebounce);
    suggestionDebounce = setTimeout(() => {
      fetchSuggestions(input.value);
    }, 350);
  });
}

window.addEventListener('DOMContentLoaded', () => {
  fetchSuggestions('');
  setupSuggestionInputListener();
  renderSwitchTypeBtn();
  addSessionTypeSwitch();
});

async function sendMessage() {
  const input = document.getElementById("user-input");
  const text  = input.value.trim();
  if (!text) return;
  if (!sessionId) {
    addMessage("bot","Please start a session first.");
    return;
  }
  // Always ask for session type on first message of each session
  if (!sessionType || !hasAskedSessionType) {
    showSessionTypeModal(() => sendMessage());
    return;
  }
  addMessage("user", text);
  input.value = "";

  const resp = await fetch(ENDPOINT.CHAT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: text, session_id: sessionId, session_type: sessionType })
  });
  const data = await resp.json();
  if (sessionType === 'info') {
    addBotMessageWithDetails(data.response || "No response.", data.kb_excerpt);
    fetchSuggestions(text);
  } else {
    // Complaint mode: do not show any bot message
    fetchSuggestions(text);
  }
}

function addBotMessageWithDetails(response, kbExcerpt) {
  const chat = document.getElementById("chat-box");
  const msg  = document.createElement("div");
  msg.className = "message bot-message";
  msg.style.display = 'inline-block';
  msg.style.maxWidth = '70%';
  msg.style.margin = '8px';
  msg.style.padding = '12px 18px';
  msg.style.borderRadius = '20px';
  msg.style.fontSize = '15px';
  msg.style.lineHeight = '1.6';
  msg.style.wordBreak = 'break-word';
  msg.style.background = '#e9ecef';
  msg.style.color = '#333';
  msg.style.alignSelf = 'flex-start';
  msg.style.textAlign = 'left';
  msg.style.boxShadow = '0 2px 8px rgba(0,0,0,0.04)';

  // Main response text
  const mainText = document.createElement('div');
  mainText.textContent = response;
  msg.appendChild(mainText);

  if (kbExcerpt && kbExcerpt.length > 10) {
    const detailsBtn = document.createElement('button');
    detailsBtn.textContent = 'Read Details';
    detailsBtn.style.marginTop = '8px';
    detailsBtn.style.background = '#e6f0ff';
    detailsBtn.style.color = '#007bff';
    detailsBtn.style.border = '1.5px solid #007bff';
    detailsBtn.style.borderRadius = '14px';
    detailsBtn.style.padding = '7px 18px';
    detailsBtn.style.fontWeight = '600';
    detailsBtn.style.fontSize = '14px';
    detailsBtn.style.cursor = 'pointer';
    detailsBtn.style.transition = 'background 0.2s, color 0.2s';
    detailsBtn.onmouseover = () => detailsBtn.style.background = '#007bff', detailsBtn.style.color = '#fff';
    detailsBtn.onmouseout = () => detailsBtn.style.background = '#e6f0ff', detailsBtn.style.color = '#007bff';
    let detailsBox = null;
    detailsBtn.onclick = () => {
      // Close any open details box
      document.querySelectorAll('.kb-details-box').forEach(box => box.remove());
      if (!detailsBox) {
        detailsBox = document.createElement('div');
        detailsBox.className = 'kb-details-box';
        detailsBox.textContent = kbExcerpt;
        detailsBox.style.background = 'rgba(255,255,255,0.95)';
        detailsBox.style.border = '1.5px solid #007bff';
        detailsBox.style.borderRadius = '16px';
        detailsBox.style.marginTop = '10px';
        detailsBox.style.padding = '16px 18px';
        detailsBox.style.fontSize = '15px';
        detailsBox.style.color = '#222';
        detailsBox.style.boxShadow = '0 2px 12px rgba(0,123,255,0.08)';
        detailsBox.style.maxWidth = '90%';
        detailsBox.style.whiteSpace = 'pre-line';
        msg.appendChild(detailsBox);
      } else {
        detailsBox.remove();
        detailsBox = null;
      }
    };
    msg.appendChild(detailsBtn);
  }
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

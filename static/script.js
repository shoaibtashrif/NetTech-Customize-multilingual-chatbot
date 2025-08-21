// static/script.js
console.log("script.js loaded");

const ENDPOINT = {
  TOGGLE: "/start_toggle",
  UPLOAD: "/upload",
  CHAT:   "/chat",
  SETMDL: "/set_model"
};

let sessionId = null;
let sessionType = "info"; // Always 'info' - no complaints
let hasAskedSessionType = false;
let suggestionDebounce = null;
let totalInputTokens = 0;
let totalOutputTokens = 0;
const btn     = document.getElementById("sessionBtn");
const sendBtn = document.getElementById("sendBtn");

function updateTokenDisplay(inputTokens, outputTokens, modelName) {
  if (inputTokens) totalInputTokens += inputTokens;
  if (outputTokens) totalOutputTokens += outputTokens;
  
  // Update display
  const totalInputElement = document.getElementById('total-input-tokens');
  const totalOutputElement = document.getElementById('total-output-tokens');
  const lastApiElement = document.getElementById('last-api-tokens');
  
  if (totalInputElement) totalInputElement.textContent = totalInputTokens.toLocaleString();
  if (totalOutputElement) totalOutputElement.textContent = totalOutputTokens.toLocaleString();
  if (lastApiElement) {
    lastApiElement.textContent = `${inputTokens || 0} → ${outputTokens || 0}`;
    lastApiElement.title = `Model: ${modelName}`;
  }
  
  // Log to console for debugging
  console.log(`Token Usage: ${inputTokens || 0} input, ${outputTokens || 0} output (${modelName})`);
}

function monitorTokenUsage() {
  // Check logs every 2 seconds for token usage
  setInterval(async () => {
    try {
      const response = await fetch('/logs');
      const data = await response.json();
      
      // Look for the latest token usage log
      const tokenLogs = data.logs.filter(log => log.includes('TOKEN_USAGE'));
      if (tokenLogs.length > 0) {
        const latestTokenLog = tokenLogs[tokenLogs.length - 1];
        
        // Parse token usage from log format: "TOKEN_USAGE [timestamp] Model: input=X, output=Y"
        const match = latestTokenLog.match(/input=(\d+), output=(\d+)/);
        if (match) {
          const inputTokens = parseInt(match[1]);
          const outputTokens = parseInt(match[2]);
          
          // Extract model name
          const modelMatch = latestTokenLog.match(/TOKEN_USAGE \[.*?\] (.*?): input=/);
          const modelName = modelMatch ? modelMatch[1] : 'Unknown';
          
          updateTokenDisplay(inputTokens, outputTokens, modelName);
        }
      }
    } catch (error) {
      console.log('Error monitoring token usage:', error);
    }
  }, 2000);
}

// Function to convert markdown to HTML
function markdownToHtml(text) {
  // Convert **text** to <strong>text</strong>
  return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
}

function addMessage(sender, content) {
  const chat = document.getElementById("chat-box");
  const msg  = document.createElement("div");
  msg.className = sender === "user" ? "message user-message" : sender === "bot" ? "message bot-message" : "message system-message";
  
  // Use innerHTML to render markdown formatting
  if (sender === "bot") {
    msg.innerHTML = markdownToHtml(content);
  } else {
    msg.textContent = content;
  }
  
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
    sessionType = "info";
    hasAskedSessionType = false;
    addMessage("system", `Session started: ${sessionId}`);
    btn.textContent = "End Session";
    sendBtn.disabled= false;
  } else if (data.ended) {
    addMessage("system", `Session ended (${data.status})`);
    document.getElementById("chat-box").innerHTML = "";
    sessionId = null;
    // Reset session type when session ends
    sessionType = "info";
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

function showSessionTypeModal(callback) {
  // All sessions are now info sessions - no modal needed
  if (callback) callback('info');
}

function renderSwitchTypeBtn() {
  // No switching needed - always info
  return;
}

function ensureSessionType(cb) {
  // Always return info
  return cb('info');
}

function addSessionTypeSwitch() {
  // No session type switching needed
  return;
}

function renderSuggestions(suggestions) {
  console.log('renderSuggestions called with:', suggestions);
  let sugDiv = document.getElementById('suggestions');
  if (!sugDiv) {
    console.error('Suggestions container not found');
    return;
  }
  
  console.log('Found suggestions container, rendering...');
  sugDiv.innerHTML = '';
  if (!suggestions || suggestions.length === 0) {
    const msg = document.createElement('div');
    msg.textContent = 'No quick questions available.';
    msg.style.color = '#888';
    msg.style.fontSize = '14px';
    msg.style.textAlign = 'center';
    msg.style.padding = '20px';
    sugDiv.appendChild(msg);
    return;
  }
  
  console.log(`Rendering ${suggestions.length} suggestions`);
  (suggestions || []).slice(0,6).forEach((s, index) => {
    let clean = s.replace(/[`\[\]{}"']/g, '').trim();
    const btn = document.createElement('button');
    btn.textContent = clean;
    btn.onclick = () => {
      document.getElementById('user-input').value = clean;
      document.getElementById('user-input').focus();
    };
    sugDiv.appendChild(btn);
    console.log(`Added suggestion ${index + 1}:`, clean);
  });
  
  console.log('Suggestions rendering complete');
}

async function fetchSuggestions(input) {
  console.log('fetchSuggestions called with input:', input);
  try {
    const resp = await fetch('/suggestions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ input })
    });
    const data = await resp.json();
    console.log('Suggestions API response:', data);
    renderSuggestions(data.suggestions);
  } catch (error) {
    console.error('Error fetching suggestions:', error);
  }
}

function setupSuggestionInputListener() {
  const input = document.getElementById('user-input');
  input.addEventListener('input', function() {
    if (suggestionDebounce) clearTimeout(suggestionDebounce);
    suggestionDebounce = setTimeout(() => {
      // Only fetch suggestions if user has typed at least 3 characters
      // This reduces unnecessary API calls and saves tokens
      if (input.value.length >= 3) {
        fetchSuggestions(input.value);
      }
    }, 1000); // Increased delay from 350ms to 1000ms to reduce API calls
  });
}

window.addEventListener('DOMContentLoaded', () => {
  // Don't fetch suggestions on page load - save tokens!
  // fetchSuggestions('');  // REMOVED - was costing 2400+ tokens per page load
  
  // Show default suggestions without API calls
  showDefaultSuggestions();
  
  setupSuggestionInputListener();
  monitorTokenUsage(); // Start monitoring token usage
});

function showDefaultSuggestions() {
  const defaultSuggestions = [
    "What is the age requirement?",
    "How much is the cash transfer?",
    "What documents are needed?"
  ];
  renderSuggestions(defaultSuggestions);
}

async function sendMessage() {
  const input = document.getElementById("user-input");
  const text  = input.value.trim();
  if (!text) return;
  if (!sessionId) {
    addMessage("bot","Please start a session first.");
    return;
  }
  
  // All sessions are info sessions
  sessionType = "info";
  hasAskedSessionType = true;
  
  addMessage("user", text);
  input.value = "";

  const resp = await fetch(ENDPOINT.CHAT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt: text, session_id: sessionId, session_type: sessionType })
  });
  const data = await resp.json();
  
  // Always show bot message since all sessions are info
  addBotMessageWithDetails(data.response || "No response.", data.kb_excerpt);
  fetchSuggestions(text);
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
  mainText.innerHTML = markdownToHtml(response);
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



function formatResponse(text) {
  // Convert hilight tags to spans
  return text.replace(/<hilight>(.*?)<\/hilight>/g, 
      '<span class="hilight">$1</span>');
}
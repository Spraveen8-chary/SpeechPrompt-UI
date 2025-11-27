const api = {
  upload: "/api/upload"
};

const elements = {
  form: document.getElementById("mainForm"),
  runBtn: document.getElementById("runBtn"),
  resultDot: document.getElementById("resultIndicator"),
  promptMicBtn: document.getElementById("promptMicBtn"),
  promptMicStatus: document.getElementById("promptMicStatus"),
  promptAudioField: document.getElementById("promptAudioField"),
  liveMicBtn: document.getElementById("liveMicBtn"),
  liveStatus: document.getElementById("liveStatus"),
  liveAudioField: document.getElementById("liveAudioField"),
  themeToggle: document.getElementById("themeToggle"),
  themeLabel: document.getElementById("themeLabel")
};

let promptRecorder;
let promptChunks = [];
let liveRecorder;
let liveChunks = [];

async function getAudioStream() {
  return navigator.mediaDevices.getUserMedia({ audio: true });
}

async function ensurePromptRecorder() {
  if (promptRecorder) return;
  const stream = await getAudioStream();
  promptRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

  promptRecorder.ondataavailable = (e) => {
    if (e.data.size) promptChunks.push(e.data);
  };

  promptRecorder.onstop = () => {
    handleBlobUpload(
      promptChunks,
      "prompt_",
      elements.promptAudioField,
      elements.promptMicStatus
    );
  };
}

async function ensureLiveRecorder() {
  if (liveRecorder) return;
  const stream = await getAudioStream();
  liveRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

  liveRecorder.ondataavailable = (e) => {
    if (e.data.size) liveChunks.push(e.data);
  };

  liveRecorder.onstop = () => {
    handleBlobUpload(
      liveChunks,
      "live_",
      elements.liveAudioField,
      elements.liveStatus
    );
  };
}

async function handleBlobUpload(chunks, prefix, hiddenField, statusEl) {
  const blob = new Blob(chunks, { type: "audio/webm" });
  chunks.length = 0;

  const fd = new FormData();
  fd.append("file", blob, prefix + "clip.webm");

  if (statusEl) statusEl.textContent = "Uploading...";

  try {
    const res = await fetch(api.upload, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok || !data.filename) {
      throw new Error("upload failed");
    }
    hiddenField.value = data.filename;
    if (statusEl) statusEl.textContent = "Ready · will be used on next run.";
  } catch (err) {
    if (statusEl) statusEl.textContent = "Upload failed. Try again.";
  }
}

// prompt mic
elements.promptMicBtn?.addEventListener("click", async () => {
  try {
    await ensurePromptRecorder();
  } catch (e) {
    elements.promptMicStatus.textContent = "Mic permission denied.";
    return;
  }

  if (promptRecorder.state === "inactive") {
    promptRecorder.start();
    elements.promptMicBtn.classList.add("active");
    elements.promptMicStatus.textContent =
      "Recording prompt… click again to stop.";
  } else {
    promptRecorder.stop();
    elements.promptMicBtn.classList.remove("active");
  }
});

// live mic
elements.liveMicBtn?.addEventListener("click", async () => {
  try {
    await ensureLiveRecorder();
  } catch (e) {
    elements.liveStatus.textContent = "Mic permission denied.";
    return;
  }

  if (liveRecorder.state === "inactive") {
    liveRecorder.start();
    elements.liveMicBtn.classList.add("active");
    elements.liveStatus.textContent = "Recording… tap again to stop.";
  } else {
    liveRecorder.stop();
    elements.liveMicBtn.classList.remove("active");
    elements.liveStatus.textContent = "Processing recording…";
  }
});

// form submit visual feedback
elements.form?.addEventListener("submit", () => {
  elements.runBtn?.classList.add("loading");
  elements.resultDot?.classList.add("live");
});

// theme toggle
(function setupThemeToggle() {
  if (!elements.themeToggle || !elements.themeLabel) return;

  const stored = localStorage.getItem("theme");
  if (stored === "light" || stored === "dark") {
    document.documentElement.setAttribute("data-theme", stored);
    elements.themeLabel.textContent =
      stored === "dark" ? "Dark" : "Light";
  } else {
    // default: dark
    document.documentElement.setAttribute("data-theme", "dark");
    elements.themeLabel.textContent = "Dark";
  }

  elements.themeToggle.addEventListener("click", () => {
    const current =
      document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    elements.themeLabel.textContent =
      next === "dark" ? "Dark" : "Light";
  });
})();

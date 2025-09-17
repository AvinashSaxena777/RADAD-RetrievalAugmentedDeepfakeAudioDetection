// static/app.js
let callLog = [];
let selectedFromLog = null;      // { file, url, ... }
let uploadedFile = null;         // File object from <input>
let recordedBlob = null;         // Blob captured via MediaRecorder
let mediaRecorder = null;
let recChunks = [];
let recInterval = null;
let recSeconds = 0;

const $ = (id) => document.getElementById(id);

function fmtProb(p) {
  if (p === null || p === undefined || isNaN(p)) return "—";
  return (p * 100).toFixed(2) + "%";
}

async function fetchCallLog() {
  const res = await fetch("/api/list");
  const data = await res.json();
  callLog = data.items || [];
  renderCallLog(callLog);
}

function renderCallLog(items) {
  const tbody = $("callLogBody");
  tbody.innerHTML = "";
  for (const it of items) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${it.file}</td>
      <td>${it.speaker}</td>
      <td>${it.duration}</td>
      <td><button class="btn" data-url="${it.url}" data-file="${it.file}">Play</button></td>
    `;
    // click row to "select"
    tr.addEventListener("click", (e) => {
      selectedFromLog = it;
      $("selectedInfo").textContent = `Selected: ${it.file} (${it.speaker}, ${it.duration})`;
    });
    // play button
    tr.querySelector("button").addEventListener("click", (e) => {
      e.stopPropagation();
      $("callLogPlayer").src = e.target.getAttribute("data-url");
      $("callLogPlayer").play();
    });
    tbody.appendChild(tr);
  }
}

$("searchInput").addEventListener("input", (e) => {
  const q = e.target.value.toLowerCase();
  const filtered = callLog.filter(it =>
    it.file.toLowerCase().includes(q) ||
    String(it.speaker).toLowerCase().includes(q)
  );
  renderCallLog(filtered);
});

$("fileInput").addEventListener("change", (e) => {
  recordedBlob = null;
  uploadedFile = e.target.files[0] || null;
  $("fileName").textContent = uploadedFile ? uploadedFile.name : "No file chosen";
  if (uploadedFile) {
    const url = URL.createObjectURL(uploadedFile);
    $("recPreview").src = url;
    $("recPreview").style.display = "block";
  } else {
    $("recPreview").style.display = "none";
  }
});

$("useSelected").addEventListener("click", () => {
  uploadedFile = null;
  recordedBlob = null;
  $("fileName").textContent = "Using selected from call log";
  $("status").textContent = "";
});

$("recStart").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recChunks = [];
    recordedBlob = null;
    uploadedFile = null;

    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(recChunks, { type: "audio/webm" });
      const url = URL.createObjectURL(recordedBlob);
      $("recPreview").src = url;
      $("recPreview").style.display = "block";
      // stop all tracks
      stream.getTracks().forEach(t => t.stop());
    };

    mediaRecorder.start();
    $("recStart").disabled = true;
    $("recStop").disabled = false;
    recSeconds = 0;
    $("recTimer").textContent = "00:00";
    recInterval = setInterval(() => {
      recSeconds++;
      const m = String(Math.floor(recSeconds / 60)).padStart(2,"0");
      const s = String(recSeconds % 60).padStart(2,"0");
      $("recTimer").textContent = `${m}:${s}`;
    }, 1000);
  } catch (e) {
    $("status").textContent = "Mic permission denied or not available.";
  }
});

$("recStop").addEventListener("click", () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    $("recStart").disabled = false;
    $("recStop").disabled = true;
    clearInterval(recInterval);
  }
});

$("predictBtn").addEventListener("click", async () => {
  $("status").textContent = "Predicting…";
  $("neighborsBody").innerHTML = "";
  $("predBadge").className = "badge";
  $("predBadge").textContent = "—";
  $("prob").textContent = "Probability: —";
  $("srcMeta").textContent = "Source: —";

  const fd = new FormData();

  // Priority: recorded -> uploaded -> selectedFromLog
  if (recordedBlob) {
    fd.append("file", recordedBlob, "recording.webm");
  } else if (uploadedFile) {
    fd.append("file", uploadedFile, uploadedFile.name || "upload.wav");
  } else if (selectedFromLog) {
    fd.append("filename", selectedFromLog.file);
  } else {
    $("status").textContent = "Choose/record a file or select from the call log.";
    return;
  }

  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await res.json();
    if (!data.ok) {
      $("status").textContent = data.error || "Prediction failed.";
      return;
    }

    // Col4
    $("status").textContent = "";
    $("predBadge").textContent = data.prediction || "—";
    const good = (data.prediction || "").toLowerCase() === "bona-fide";
    $("predBadge").className = "badge " + (good ? "good" : "bad");
    $("prob").textContent = `Probability (spoof): ${fmtProb(data.probability)}`;
    $("srcMeta").textContent = `Source: ${data.source?.path || ""}`;

    // Col3
    const nb = data.neighbors || [];
    const tbody = $("neighborsBody");
    tbody.innerHTML = "";
    for (const r of nb) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.file || ""}</td>
        <td>${r.speaker || ""}</td>
        <td>${r.label || ""}</td>
        <td>${r.duration || ""}</td>
        <td>${(r.distance == null || isNaN(r.distance)) ? "" : Number(r.distance).toFixed(3)}</td>
        <td>${r.url ? `<button class="btn" data-url="${r.url}">Play</button>` : ""}</td>
      `;
      if (r.url) {
        tr.querySelector("button").addEventListener("click", (e) => {
          $("neighborPlayer").src = r.url;
          $("neighborPlayer").play();
        });
      }
      tbody.appendChild(tr);
    }
  } catch (e) {
    $("status").textContent = "Error: " + (e.message || e);
  }
});

window.addEventListener("DOMContentLoaded", fetchCallLog);
